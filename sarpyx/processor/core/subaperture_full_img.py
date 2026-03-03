import os
import re
import glob
import logging
from typing import List, Optional, Union
import warnings

import numpy as np
import dask
import dask.array as da
from dask.array import fft as dafft
import rasterio
from rasterio.windows import Window

from .meta import Handler
from . import utilis as ut
from .dim_updater import update_dim_add_bands_from_data_dir

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

_IQ_SWATH_POL_RE = re.compile(
    r"(^|_)(?P<iq>[iq])_(?:(?P<swath>[A-Z]{2}\d)_)?(?P<pol>[A-Z]{2})(?:$|_)",
    re.IGNORECASE,
)
_SA_TOKEN_RE = re.compile(r"(^|_)SA\d+($|_)", re.IGNORECASE)


def _stem_contains_sa_token(stem: str) -> bool:
    """Return True when *stem* looks like a generated subaperture band."""
    return _SA_TOKEN_RE.search(stem) is not None


def _extract_iq_swath_pol_from_stem(stem: str):
    """
    Extract (iq, swath, pol) from stems that contain:
      - i_<POL> / q_<POL>
      - i_<SWATH>_<POL> / q_<SWATH>_<POL>   (e.g. i_IW1_VV)
    Accepts extra prefixes/suffixes (e.g. CAL_i_IW1_VH or foo_i_VV_bar).
    """
    m = _IQ_SWATH_POL_RE.search(stem)
    if not m:
        return None
    iq = m.group("iq").lower()
    swath = m.group("swath")
    pol = m.group("pol")
    swath = swath.upper() if swath else None
    pol = pol.upper()
    return iq, swath, pol


def _candidate_priority(stem: str, iq: str, tag: str):
    """Lower tuples are preferred when several i/q files exist for one tag."""
    stem_u = stem.upper()
    canonical = f"{iq.upper()}_{tag.upper()}"
    if stem_u == canonical:
        return (0, len(stem_u), stem_u)
    if stem_u.endswith("_" + canonical):
        return (1, len(stem_u), stem_u)
    if ("_" + canonical + "_") in ("_" + stem_u + "_"):
        return (2, len(stem_u), stem_u)
    return (3, len(stem_u), stem_u)



def estimate_central_freq_from_spectrum(
    spectrum_az_fft: Union[np.ndarray, "da.Array"],
    freq_vect: np.ndarray,
    valid_frac: float = 0.8,
    smooth_win: int = 9,
) -> float:
    """Estimate azimuth central frequency (Hz) from the spectrum peak.

    The estimate is the argmax of the mean power spectrum over range,
    optionally smoothed and restricted to the central portion of the band.

    Notes
    -----
    * Works with both numpy and dask arrays. For dask, the 1D profile is
      computed explicitly (small) to keep memory bounded.
    * `freq_vect` is expected to match the frequency axis of `spectrum_az_fft`.
    """
    # Mean power per frequency bin (average over range dimension).
    if "dask.array" in str(type(spectrum_az_fft)):
        E = da.mean(da.absolute(spectrum_az_fft) ** 2, axis=1).astype(np.float64).compute()
    else:
        E = np.mean(np.abs(spectrum_az_fft) ** 2, axis=1).astype(np.float64)

    # Smooth via moving average (odd window).
    w = int(smooth_win) if smooth_win else 0
    if w > 1:
        if w % 2 == 0:
            w += 1
        kernel = np.ones(w, dtype=np.float64) / w
        E = np.convolve(E, kernel, mode="same")

    n = int(E.size)
    if n == 0:
        raise ValueError("Empty spectrum profile for central frequency estimation.")

    valid_frac = float(valid_frac)
    valid_frac = min(max(valid_frac, 0.2), 1.0)
    half = int((1.0 - valid_frac) * n / 2.0)
    lo, hi = half, n - half
    if hi <= lo + 1:
        lo, hi = 0, n

    idx = int(np.argmax(E[lo:hi])) + lo
    return float(freq_vect[idx])

def find_base_iq_pairs_in_dim_data(data_dir: str):
    """
    Detect base complex i/q pairs from a DIM ``.data`` folder.

    - Supports SM-like: i_VV.img / q_VV.img
    - Supports IW/EW swaths: i_IW1_VV.img / q_IW1_VV.img, etc.
    - Accepts non-canonical prefixes/suffixes (e.g. CAL_i_IW1_VH.img).
    - Ignores generated subaperture outputs containing ``SA<k>`` in the stem.

    Returns
    -------
    dict
        key: tag (e.g. 'VV' or 'IW1_VV')
        value: (i_path, q_path)
    """
    candidates = {}  # tag -> iq -> [(stem, fp), ...]
    for fp in glob.glob(os.path.join(data_dir, "*.img")):
        stem = os.path.splitext(os.path.basename(fp))[0]
        if _stem_contains_sa_token(stem):
            continue

        parsed = _extract_iq_swath_pol_from_stem(stem)
        if parsed is None:
            continue
        iq, swath, pol = parsed
        tag = pol if swath is None else f"{swath}_{pol}"

        candidates.setdefault(tag, {}).setdefault(iq, []).append((stem, fp))

    pairs = {}
    for tag, by_iq in candidates.items():
        if "i" not in by_iq or "q" not in by_iq:
            continue
        best_i = min(by_iq["i"], key=lambda t: _candidate_priority(t[0], "i", tag))
        best_q = min(by_iq["q"], key=lambda t: _candidate_priority(t[0], "q", tag))
        pairs[tag] = (best_i[1], best_q[1])
    return pairs

# ---------------------------------------------------------------------------
# Memory threshold helpers
# ---------------------------------------------------------------------------
_DEFAULT_MAX_MEMORY_GB = 32  # fallback if psutil is unavailable

def _available_memory_bytes():
    """Best-effort query for available system RAM."""
    try:
        import psutil
        return psutil.virtual_memory().available
    except ImportError:
        return _DEFAULT_MAX_MEMORY_GB * 1024**3

def _estimate_peak_bytes(nRows, nCols, numberOfLooks):
    """
    Conservative estimate of peak memory (bytes) for the numpy eager path.
    Counts: Box + SpectrumOneDim + SpectrumOneDimNorm + SpectrumOneDimNormDeWe
           + 3 cubes in Generation (LookSpectr, LookSpectrCentered, Looks)
    All complex64 (8 bytes per element).
    """
    elem = nRows * nCols
    n_full_arrays = 4 + 3 * numberOfLooks  # 4 image-size + 3 × N cubes
    return elem * 8 * n_full_arrays


def DeHammWin(signal, coeff):
    """
    Vectorized Hamming de-weighting.

    Divides *signal* by the Hamming window built from *coeff*, then conjugates
    the imaginary part (equivalent to the legacy ``changeImag``).

    Parameters
    ----------
    signal : np.ndarray, 1-D complex
    coeff  : float  (Hamming coefficient, typically 0.5–1.0)

    Returns
    -------
    np.ndarray – de-weighted signal (complex64).
    """
    n = len(signal)
    alpha = np.linspace(0, 2 * np.pi * (n - 1) / n, n, dtype=np.float32)
    w = np.float32(coeff) - np.float32(1 - coeff) * np.cos(alpha)
    divided = signal / w
    # changeImag: negate imaginary part  →  conjugate
    return np.conj(divided).astype(np.complex64)


def DeHammWin_2d(signals_2d, coeff):
    """
    Apply Hamming de-weighting to every column (axis-1 slice) of a 2-D array.

    Parameters
    ----------
    signals_2d : np.ndarray, shape (nGood, nCols), complex
    coeff      : float

    Returns
    -------
    np.ndarray – same shape, de-weighted and conjugated (complex64).
    """
    n = signals_2d.shape[0]
    alpha = np.linspace(0, 2 * np.pi * (n - 1) / n, n, dtype=np.float32)
    w = np.float32(coeff) - np.float32(1 - coeff) * np.cos(alpha)
    divided = signals_2d / w[:, None]  # broadcast (n,1) over (n, nCols)
    return np.conj(divided).astype(np.complex64)

def write_envi_bsq_float32(path_img, path_hdr, arr2d, band_name, byte_order=1, type_="real"):
    """
    Write a single-band ENVI BSQ float32 image + header.

    Parameters
    ----------
    path_img : str
        Output binary file path (.img).
    path_hdr : str
        Output ENVI header path (.hdr).
    arr2d : array-like
        2D array (lines, samples).
    band_name : str
        Band name to write into the ENVI header (must match your naming convention).
    byte_order : int
        ENVI byte order: 0 = little endian, 1 = big endian.
    """
    arr = np.asarray(arr2d, dtype=np.float32)

    # Force explicit endianness
    arr_out = arr.astype(">f4", copy=False) if byte_order == 1 else arr.astype("<f4", copy=False)

    # Write binary (BSQ, 1 band)
    arr_out.tofile(path_img)

    lines, samples = arr.shape
    hdr = f"""ENVI
description = {{Sentinel-1 SM Level-1 SLC Product - Unit: {type_}}}
samples = {samples}
lines = {lines}
bands = 1
header offset = 0
file type = ENVI Standard
data type = 4
interleave = bsq
byte order = {byte_order}
band names = {{ {band_name} }}
data gain values = {{1.0}}
data offset values = {{0.0}}
"""
    with open(path_hdr, "w", encoding="ascii") as f:
        f.write(hdr)


# ---------------------------------------------------------------------------
# Dask lazy-loading helpers
# ---------------------------------------------------------------------------

def _read_envi_block(i_path, q_path, row_off, col_off, height, width):
    """
    Read a spatial window from paired I/Q ENVI images and return complex64.
    Used as the ``dask.delayed`` building block.
    """
    win = Window(col_off, row_off, width, height)
    with rasterio.open(i_path) as si:
        img_i = si.read(1, window=win).astype(np.float32, copy=False)
    with rasterio.open(q_path) as sq:
        img_q = sq.read(1, window=win).astype(np.float32, copy=False)

    out = np.empty(img_i.shape, dtype=np.complex64)
    out.real = img_i
    out.imag = img_q
    return out


def _load_complex_dask(i_path, q_path, chunk_cols=2048):
    """
    Build a *lazy* ``dask.array`` of dtype complex64 from paired ENVI I/Q
    images, chunked along columns (axis-1).

    Each chunk covers **all rows** so that an FFT along axis-0 (azimuth)
    stays within a single chunk.
    """
    nRows, nCols = ut.get_image_shape_from_file(i_path)
    chunk_cols = min(chunk_cols, nCols)

    chunks_list = []
    col = 0
    while col < nCols:
        w = min(chunk_cols, nCols - col)
        delayed_block = dask.delayed(_read_envi_block)(
            i_path, q_path, 0, col, nRows, w
        )
        arr = da.from_delayed(delayed_block, shape=(nRows, w), dtype=np.complex64)
        chunks_list.append(arr)
        col += w

    return da.concatenate(chunks_list, axis=1)


def _choose_chunk_cols(nRows, nCols, numberOfLooks, target_mem_bytes=2 * 1024**3):
    """
    Pick a column-chunk width so that the working set for one chunk
    (through all pipeline stages) fits in *target_mem_bytes*.

    Working set per chunk ≈ chunk_cols × nRows × 8 (complex64) × (4 arrays + 3×N cubes-slices).
    We keep at most 2 sublooks materialised per chunk at once so use factor 6+2N.
    """
    per_col_bytes = nRows * 8 * (6 + 2 * numberOfLooks)
    chunk_cols = max(64, int(target_mem_bytes / per_col_bytes))
    chunk_cols = min(chunk_cols, nCols)
    return chunk_cols


class CombinedSublooking:
    """
    Subaperture decomposition of a SAR SLC image.

    Supports two execution backends selected automatically based on image
    size (or overridden with ``force_dask``):

    * **numpy** – eager, in-memory.  Fast for moderate images.
    * **dask**  – lazy, chunked along columns.  Handles arbitrarily large
      images with bounded RAM.

    Parameters
    ----------
    metadata_pointer_safe : str
        Path to the .SAFE product (metadata source).
    numberofLooks : int
        Number of sublooks to generate (≥ 2).
    i_image, q_image : str | np.ndarray | None
        In-phase / quadrature inputs (file paths or arrays).
    DownSample : bool
        Reserved for legacy compatibility.
    assetMetadata : dict | None
        Optional pre-extracted metadata dictionary.
    force_dask : bool | None
        ``True`` → always use dask.  ``False`` → always numpy.
        ``None`` (default) → auto-detect from image size vs available RAM.
    chunk_cols : int | None
        Column-chunk width for the dask path (auto-computed if *None*).
    """

    def __init__(
        self,
        metadata_pointer_safe: str,
        numberofLooks: int = 3,
        i_image=None,
        q_image=None,
        DownSample: bool = True,
        assetMetadata=None,
        force_dask: Optional[bool] = None,
        chunk_cols: Optional[int] = None,
        estimate_center_from_spectrum: bool = True,
        center_valid_frac: float = 0.8,
        center_smooth_win: int = 9,
    ):
        # ---- tunable knobs ----
        self.choice = 1  # Range == 0 | Azimuth == 1
        self.numberOfLooks = numberofLooks

        # ---- optional: estimate azimuth central frequency from spectrum ----
        self.estimate_center_from_spectrum = bool(estimate_center_from_spectrum)
        self.center_valid_frac = float(center_valid_frac)
        self.center_smooth_win = int(center_smooth_win)
        self.CentralFreqRange = 0
        self.CentralFreqAzim = 0
        self.AzimRes = 5
        self.RangeRes = 5
        self.WeightFunctAzim = "HAMMING"
        self.WeightFunctRange = "HAMMING"
        self.DownSample = DownSample

        self.metadata_pointer_safe = metadata_pointer_safe
        self.i_img = i_image
        self.q_img = q_image

        # ---- deweighting mode ----
        self.choiceDeWe = 0  # 0 = ancillary (theoretical), 1 = average

        # ---- determine image size before loading ----
        if isinstance(self.i_img, str):
            self.nRows, self.nCols = ut.get_image_shape_from_file(self.i_img)
        elif isinstance(self.i_img, np.ndarray):
            self.nRows, self.nCols = self.i_img.shape[:2]
        else:
            raise ValueError("Complex image creation unsuccessful.")

        # ---- decide backend ----
        if force_dask is None:
            peak = _estimate_peak_bytes(self.nRows, self.nCols, self.numberOfLooks)
            avail = _available_memory_bytes()
            self._use_dask = peak > 0.5 * avail
        else:
            self._use_dask = bool(force_dask)

        if self._use_dask:
            self._chunk_cols = chunk_cols or _choose_chunk_cols(
                self.nRows, self.nCols, self.numberOfLooks
            )
            logger.info(
                "Dask backend active  (image %d×%d, chunk_cols=%d)",
                self.nRows, self.nCols, self._chunk_cols,
            )
        else:
            self._chunk_cols = None
            logger.info(
                "NumPy backend active (image %d×%d)", self.nRows, self.nCols,
            )

        # ---- load complex image ----
        if isinstance(self.i_img, str):
            if self._use_dask:
                self.Box = _load_complex_dask(
                    self.i_img, self.q_img, chunk_cols=self._chunk_cols
                )
            else:
                self.Box = ut.create_complex_image_from_file(self.i_img, self.q_img)
        elif isinstance(self.i_img, np.ndarray):
            box_np = ut.create_complex_image_from_array(self.i_img, self.q_img)
            if self._use_dask:
                self.Box = da.from_array(box_np, chunks=(self.nRows, self._chunk_cols))
            else:
                self.Box = box_np

        # ---- metadata ----
        if assetMetadata is None:
            metadata_pointer = self.metadata_pointer_safe
        elif isinstance(assetMetadata, dict):
            metadata_pointer = assetMetadata
        else:
            raise ValueError("Supplied asset metadata in unrecognized format.")

        meta = Handler(metadata_pointer)
        meta.chain()

        self.PRF = meta.PRF
        self.AzimBand = meta.AzimBand
        self.ChirpBand = meta.ChirpBand
        self.RangeBand = meta.RangeBand
        self.WeightFunctRangeParams = meta.WeightFunctRangeParams
        self.WeightFunctAzimParams = meta.WeightFunctAzimParams
        self.AzimSpacing = meta.AzimSpacing
        self.RangeSpacing = meta.RangeSpacing

        # derivative parameters
        self.centroidSeparations = (self.AzimBand - 1) / self.numberOfLooks
        self.subLookBandwidth = (self.AzimBand - 1) / self.numberOfLooks

        # central frequency
        if self.choice == 0:
            self.centralFreq = self.CentralFreqRange
        else:
            self.centralFreq = self.CentralFreqAzim

    # ------------------------------------------------------------------
    # Frequency computation  (pure scalars — identical for both backends)
    # ------------------------------------------------------------------
    def FrequencyComputation(self, VERBOSE=False):
        """Calculate the min, max and central frequencies of each sublook."""
        self.freqCentr = np.empty(self.numberOfLooks)
        self.freqMin = np.empty(self.numberOfLooks)
        self.freqMax = np.empty(self.numberOfLooks)

        if self.numberOfLooks % 2 == 0:
            for k in range(self.numberOfLooks):
                if k < 2:
                    self.freqCentr[k] = (-1) ** k * self.centroidSeparations / 2
                else:
                    multiple = 0.5 + (k // 2)
                    self.freqCentr[k] = (-1) ** k * multiple * self.centroidSeparations
        else:
            for k in range(self.numberOfLooks):
                multiple = (k + 1) // 2
                self.freqCentr[k] = (-1) ** k * multiple * self.centroidSeparations

        for k in range(self.numberOfLooks):
            self.freqMin[k] = round(self.freqCentr[k] - self.subLookBandwidth / 2, 4)
            self.freqMax[k] = round(self.freqCentr[k] + self.subLookBandwidth / 2, 4)

        if VERBOSE:
            if self.centroidSeparations < self.subLookBandwidth:
                print("Sub-looking with spectral overlap")
            else:
                print("Execution without overlapping sublooks \n")
            print(f"Available spectral range:[{-self.AzimBand/2}, {self.AzimBand/2}] \n")
            for idx in range(self.numberOfLooks):
                print(f"Sub{idx+1}: [{self.freqMin[idx]}, {self.freqMax[idx]}]")

        min_freq = np.min(self.freqMin)
        max_freq = np.max(self.freqMax)
        if self.choice == 0:
            band_limit = self.ChirpBand / 2 * 1e-6
        else:
            band_limit = self.AzimBand / 2

        if min_freq < -band_limit or max_freq > band_limit:
            raise ValueError("Sub-look spectrum outside the available bandwidth")

        if VERBOSE:
            print("\n Frequency computation successfully ended.")

    def CalcFrequencyVectors(self, VERBOSE=False):
        """Zero-centred frequency vector for the analysis axis."""
        if self.choice == 0:
            self.nSample = self.nCols
            seq = np.arange(-self.nSample / 2, self.nSample / 2)
            self.freqVect = self.RangeBand / self.nSample * seq
        else:
            self.nSample = self.nRows
            seq = np.arange(-self.nSample / 2, self.nSample / 2)
            self.freqVect = self.PRF / self.nSample * seq

        if VERBOSE:
            print(f"Frequency vector: {self.freqVect[:3]} ... {self.freqVect[-3:]}")

    # ------------------------------------------------------------------
    # Spectrum computation
    # ------------------------------------------------------------------
    def SpectrumComputation(self, VERBOSE=False):
        """FFT along the chosen axis (azimuth=0 or range=1), with fftshift."""
        if self._use_dask:
            axis = 1 if self.choice == 0 else 0
            raw = dafft.fftshift(dafft.fft(self.Box, axis=axis), axes=axis)
            # Cast to complex64 to match numpy path and halve memory
            self.SpectrumOneDim = raw.astype(np.complex64)
        else:
            axis = 1 if self.choice == 0 else 0
            raw = np.fft.fft(self.Box, axis=axis)
            self.SpectrumOneDim = np.fft.fftshift(raw, axes=axis).astype(np.complex64)

        # Free the raw image — no longer needed
        del self.Box

        if VERBOSE:
            _spec = self.SpectrumOneDim if isinstance(self.SpectrumOneDim, np.ndarray) else self.SpectrumOneDim.compute()
            import matplotlib.pyplot as plt
            plt.figure(figsize=(5, 5))
            col = min(300, self.nCols - 1)
            plt.plot(_spec[:, col])
            plt.title("Spectrum Computed in " + ("Range" if self.choice == 0 else "Azimuth"))
            plt.show()

    # ------------------------------------------------------------------
    # Spectrum normalisation
    # ------------------------------------------------------------------
    def SpectrumNormalization(self, VERBOSE=False):
        """Normalise spectrum to the average magnitude profile."""
        S = self.SpectrumOneDim
        xp = da if self._use_dask else np  # array backend

        if self.choice == 0:  # RANGE
            target_average = xp.sum(S) / S.shape[1]
            dim_average_int = xp.mean(xp.abs(S), axis=1).astype(np.float32)
            self.SpectrumOneDimNorm = (S / dim_average_int[:, None] * target_average).astype(np.complex64)
        else:  # AZIMUTH
            dim_average_int = xp.mean(xp.abs(S), axis=1).astype(np.float32)
            if self._use_dask:
                # Compute median explicitly to match numpy path exactly
                dim_average_int = da.where(dim_average_int == 0, np.float32(1e-6), dim_average_int)
                med = da.percentile(dim_average_int, 50)
            else:
                dim_average_int[dim_average_int == 0] = np.float32(1e-6)
                med = np.float32(np.median(dim_average_int))
            self.SpectrumOneDimNorm = ((S / dim_average_int[:, None]) * med).astype(np.complex64)

        # Optionally re-center using a spectrum-based estimate (useful for IW/TOPS variability)
        if self.choice != 0 and getattr(self, "estimate_center_from_spectrum", False):
            try:
                _meta = float(self.centralFreq)
                _est = estimate_central_freq_from_spectrum(
                    spectrum_az_fft=S,
                    freq_vect=self.freqVect,
                    valid_frac=getattr(self, "center_valid_frac", 0.8),
                    smooth_win=getattr(self, "center_smooth_win", 9),
                )
                self.centralFreq = _est
                logger.info(
                    "CentralFreqAzim adjusted from spectrum: est=%.6f Hz",
                    _meta,
                    _est,
                    _est - _meta,
                )
            except Exception as exc:
                logger.warning("Spectrum-based central frequency estimation skipped: %s", exc)

        # Free raw spectrum
        del self.SpectrumOneDim

        if VERBOSE:
            _norm = self.SpectrumOneDimNorm if isinstance(self.SpectrumOneDimNorm, np.ndarray) else self.SpectrumOneDimNorm.compute()
            import matplotlib.pyplot as plt
            col = min(300, self.nCols - 1)
            plt.subplot(1, 2, 2)
            plt.plot(_norm[:, col])
            plt.title("Normalized frequency Spectrum")
            plt.show()

    # ------------------------------------------------------------------
    # De-weighting
    # ------------------------------------------------------------------
    def SpectrumDeWeighting(self):
        if self.choiceDeWe == 0:
            self.AncillaryDeWe()
        else:
            self.AverageDeWe()

    def AverageDeWe(self, VERBOSE=False):
        S = self.SpectrumOneDimNorm
        xp = da if self._use_dask else np

        AverSpectrNorm = xp.sum(xp.abs(S))
        deWe = S / AverSpectrNorm
        scale = 1.0 / xp.max(xp.abs(deWe))
        self.SpectrumOneDimNormDeWe = deWe * scale

        # Free normalised spectrum
        del self.SpectrumOneDimNorm

    # ---- Ancillary (Hamming) de-weighting ----

    def _ancillary_dewe_numpy(self):
        """Vectorised ancillary de-weighting for the numpy path."""
        bHamm = self.RangeBand if self.choice == 0 else self.AzimBand
        coeff = self.WeightFunctRangeParams if self.choice == 0 else self.WeightFunctAzimParams

        indexGoodHamm = np.where(np.abs(self.freqVect) <= bHamm / 2)[0]
        nSampleGoodHamm = len(indexGoodHamm)
        if nSampleGoodHamm % 2 != 0:
            nSampleGoodHamm -= 1
            indexGoodHamm = indexGoodHamm[:-1]

        diff = np.abs(self.freqVect - self.centralFreq)
        indexCentroid = np.where(diff == diff.min())[0][0] + 1
        indexGoodOrdHamm = np.arange(
            indexCentroid - nSampleGoodHamm / 2,
            indexCentroid + nSampleGoodHamm / 2,
        ).astype(int) - 1

        # Wrap-around index correction
        indexGoodOrdHamm[indexGoodOrdHamm > self.nSample - 2] -= self.nSample - 1
        indexGoodOrdHamm[indexGoodOrdHamm <= 0] += self.nSample - 1

        roll_shift = int(self.nSample / 2 - indexCentroid)

        self.SpectrumOneDimNormDeWe = np.zeros(
            (self.nRows, self.nCols), dtype=np.complex64
        )

        if self.choice == 0:  # RANGE — iterate over rows
            for kt in range(self.nRows):
                curr = self.SpectrumOneDimNorm[kt, :]
                currApp = curr[indexGoodOrdHamm]
                currApp = np.roll(currApp, roll_shift)
                DeHammSign = DeHammWin(currApp, coeff)
                DeHammSignOrd = np.roll(DeHammSign, -roll_shift)
                self.SpectrumOneDimNormDeWe[kt, :] = DeHammSignOrd
        else:  # AZIMUTH — vectorised 2-D path over all columns at once
            # Gather: (nGoodHamm, nCols)
            gathered = self.SpectrumOneDimNorm[indexGoodOrdHamm, :]
            gathered = np.roll(gathered, roll_shift, axis=0)
            dehamm = DeHammWin_2d(gathered, coeff)
            dehamm = np.roll(dehamm, -roll_shift, axis=0)
            self.SpectrumOneDimNormDeWe[indexGoodHamm, :] = dehamm

        # Free normalised spectrum
        del self.SpectrumOneDimNorm

    def _ancillary_dewe_block(self, block, block_info=None):
        """
        ``dask.map_blocks`` kernel for ancillary de-weighting.

        Each *block* is a 2-D column-strip ``(nRows, chunk_cols)`` of the
        normalised spectrum.  We apply the Hamming de-weighting exactly as
        in the numpy path but only to this strip.
        """
        bHamm = self.RangeBand if self.choice == 0 else self.AzimBand
        coeff = self.WeightFunctRangeParams if self.choice == 0 else self.WeightFunctAzimParams

        indexGoodHamm = np.where(np.abs(self.freqVect) <= bHamm / 2)[0]
        nSampleGoodHamm = len(indexGoodHamm)
        if nSampleGoodHamm % 2 != 0:
            nSampleGoodHamm -= 1
            indexGoodHamm = indexGoodHamm[:-1]

        diff = np.abs(self.freqVect - self.centralFreq)
        indexCentroid = np.where(diff == diff.min())[0][0] + 1
        indexGoodOrdHamm = np.arange(
            indexCentroid - nSampleGoodHamm / 2,
            indexCentroid + nSampleGoodHamm / 2,
        ).astype(int) - 1

        indexGoodOrdHamm[indexGoodOrdHamm > self.nSample - 2] -= self.nSample - 1
        indexGoodOrdHamm[indexGoodOrdHamm <= 0] += self.nSample - 1

        roll_shift = int(self.nSample / 2 - indexCentroid)
        out = np.zeros_like(block)

        if self.choice == 0:  # RANGE
            for kt in range(block.shape[0]):
                curr = block[kt, :]
                currApp = curr[indexGoodOrdHamm]
                currApp = np.roll(currApp, roll_shift)
                d = DeHammWin(currApp, coeff)
                d = np.roll(d, -roll_shift)
                out[kt, :] = d
        else:  # AZIMUTH — vectorised across columns in this chunk
            gathered = block[indexGoodOrdHamm, :]
            gathered = np.roll(gathered, roll_shift, axis=0)
            dehamm = DeHammWin_2d(gathered, coeff)
            dehamm = np.roll(dehamm, -roll_shift, axis=0)
            out[indexGoodHamm, :] = dehamm

        return out

    def AncillaryDeWe(self, VERBOSE=False):
        if self._use_dask:
            self.SpectrumOneDimNormDeWe = self.SpectrumOneDimNorm.map_blocks(
                self._ancillary_dewe_block,
                dtype=np.complex64,
            )
            del self.SpectrumOneDimNorm
        else:
            self._ancillary_dewe_numpy()

    # ------------------------------------------------------------------
    # Sublook generation
    # ------------------------------------------------------------------

    def _generation_numpy(self, VERBOSE=False):
        """Vectorised sublook generation — numpy path."""
        indexLooks = np.empty((self.numberOfLooks, 2), dtype=int)
        for it in range(self.numberOfLooks):
            startIndex = np.abs(self.freqVect - self.freqMin[it]).argmin()
            endIndex = np.abs(self.freqVect - self.freqMax[it]).argmin() - 1
            indexLooks[it] = [startIndex, endIndex]

        nPixLook = int(np.min(indexLooks[:, 1] - indexLooks[:, 0] + 1))
        # FFT/IFFT axis: range → columns (1), azimuth → rows (0)
        fft_axis = 1 if self.choice == 0 else 0
        self.Looks = []

        for it in range(self.numberOfLooks):
            si, ei = indexLooks[it]
            # Reverse spectral slice [si:ei+1], trim to nPixLook rows
            reversed_spec = self.SpectrumOneDimNormDeWe[si:ei + 1, :][::-1, :]
            window = reversed_spec[:nPixLook, :].copy()

            # Place into zero-padded array
            padded = np.zeros((self.nRows, self.nCols), dtype=np.complex64)
            padded[:nPixLook, :] = window

            # Vectorised IFFT along the analysis axis
            look = np.fft.ifft(padded, axis=fft_axis).astype(np.complex64)
            del padded

            # Keep native pixel alignment with the full-aperture grid.
            self.Looks.append(look)

        # Free de-weighted spectrum
        del self.SpectrumOneDimNormDeWe

        if VERBOSE:
            print(f"{self.numberOfLooks} sublooks created successfully.")

    def _generation_dask_block(self, block, si, ei, nPixLook, nRows, block_info=None):
        """map_blocks kernel: extract spectral window, zero-pad, and IFFT."""
        _, chunk_cols = block.shape
        # FFT/IFFT axis: range → 1, azimuth → 0
        fft_axis = 1 if self.choice == 0 else 0
        reversed_spec = block[si:ei + 1, :][::-1, :]
        window = reversed_spec[:nPixLook, :].copy()
        padded = np.zeros((nRows, chunk_cols), dtype=np.complex64)
        padded[:nPixLook, :] = window
        look = np.fft.ifft(padded, axis=fft_axis).astype(np.complex64)
        return look

    def _generation_dask(self, VERBOSE=False):
        """Lazy sublook generation — dask path.  Each sublook stays lazy."""
        indexLooks = np.empty((self.numberOfLooks, 2), dtype=int)
        for it in range(self.numberOfLooks):
            startIndex = np.abs(self.freqVect - self.freqMin[it]).argmin()
            endIndex = np.abs(self.freqVect - self.freqMax[it]).argmin() - 1
            indexLooks[it] = [startIndex, endIndex]

        nPixLook = int(np.min(indexLooks[:, 1] - indexLooks[:, 0] + 1))
        self.Looks = []
        fft_axis = 1 if self.choice == 0 else 0
        for it in range(self.numberOfLooks):
            si = int(indexLooks[it, 0])
            ei = int(indexLooks[it, 1])
            look = self.SpectrumOneDimNormDeWe.map_blocks(
                self._generation_dask_block,
                si=si,
                ei=ei,
                nPixLook=nPixLook,
                nRows=self.nRows,
                dtype=np.complex64,
            )
            self.Looks.append(look)

        del self.SpectrumOneDimNormDeWe

        if VERBOSE:
            print(f"{self.numberOfLooks} sublooks created (lazy dask graphs).")

    def Generation(self, VERBOSE=False):
        if self._use_dask:
            self._generation_dask(VERBOSE=VERBOSE)
        else:
            self._generation_numpy(VERBOSE=VERBOSE)

    # ------------------------------------------------------------------
    # Save sublooks
    # ------------------------------------------------------------------
    def save_sublooks_envi(self, out_dir, pol, prefix="", byte_order=1):
        """
        Save i/q of each sublook as ENVI BSQ float32, forcing endianness.
        For the dask path, each sublook is computed and written one at a time
        so only one full-image array is in RAM at any moment.
        """
        os.makedirs(out_dir, exist_ok=True)
        if prefix and not prefix.endswith("_"):
            prefix = prefix + "_"

        for idx_slook in range(self.numberOfLooks):
            sa = idx_slook + 1
            z = self.Looks[idx_slook]

            # Materialise if dask
            if isinstance(z, da.Array):
                z = z.compute()

            i = np.asarray(z.real, dtype=np.float32)
            q = np.asarray(z.imag, dtype=np.float32)
            del z  # free the complex copy

            img_i = os.path.join(out_dir, f"{prefix}i_{pol}_SA{sa}.img")
            hdr_i = os.path.join(out_dir, f"{prefix}i_{pol}_SA{sa}.hdr")
            img_q = os.path.join(out_dir, f"{prefix}q_{pol}_SA{sa}.img")
            hdr_q = os.path.join(out_dir, f"{prefix}q_{pol}_SA{sa}.hdr")

            band_name_i = f"{prefix}i_{pol}_SA{sa}" if prefix else f"i_{pol}_SA{sa}"
            band_name_q = f"{prefix}q_{pol}_SA{sa}" if prefix else f"q_{pol}_SA{sa}"

            write_envi_bsq_float32(img_i, hdr_i, i, band_name=band_name_i, byte_order=byte_order, type_="real")
            write_envi_bsq_float32(img_q, hdr_q, q, band_name=band_name_q, byte_order=byte_order, type_="imaginary")
            del i, q  # free before next sublook

    # ------------------------------------------------------------------
    # Pipeline entry-points
    # ------------------------------------------------------------------
    def chain(self, VERBOSE=False):
        self.FrequencyComputation(VERBOSE=VERBOSE)
        self.CalcFrequencyVectors(VERBOSE=VERBOSE)
        self.SpectrumComputation(VERBOSE=VERBOSE)
        self.SpectrumNormalization(VERBOSE=VERBOSE)
        self.SpectrumDeWeighting()
        self.Generation(VERBOSE=VERBOSE)
        return self.Looks

    def run_and_save(self, out_dir, pol, prefix="", byte_order=1, VERBOSE=False):
        self.chain(VERBOSE=VERBOSE)
        self.save_sublooks_envi(
            out_dir=out_dir,
            pol=pol,
            prefix=prefix,
            byte_order=byte_order,
        )
        return self.Looks

def find_pols_in_dim_data(data_dir: str):
    """
    Search for base polarization i/q pairs in the product .data folder.

    Notes
    -----
    - Ignores previously generated subaperture outputs that contain ``SA<k>``.
    - Accepts non-canonical prefixes/suffixes (e.g. ``CAL_i_VH.img``).
    - Returns a sorted list of detected polarizations (e.g. ['VV', 'VH']).
    """
    return sorted(find_base_iq_pairs_in_dim_data(data_dir).keys())

def do_subaps(
    dim_path: str,
    safe_path: str,
    numberofLooks: int = 3,
    n_decompositions: Optional[Union[int, List[int]]] = None,
    DownSample: bool = True,
    byte_order: int = 1,
    prefix: str = "",
    VERBOSE: bool = False,
    force_dask: Optional[bool] = None,
    chunk_cols: Optional[int] = None,
):
    """
    Orchestrator:
    - Takes a DIMAP product (.dim) to locate the .data folder (i/q bands)
    - Takes the original .SAFE product to extract metadata (PRF, bandwidths, window coeffs, etc.)
    - Detects available base polarizations in .data (i_<POL>.img + q_<POL>.img)
    - Generates subapertures for each pol and writes them into the same .data folder

    Parameters
    ----------
    n_decompositions :
        * None -> uses numberofLooks (backward compatible)
        * int  -> runs a single decomposition with that N
        * list -> runs all decompositions in the list (e.g. [2,3,5,7])
    force_dask : bool | None
        ``True`` → always use dask lazy backend (bounded memory).
        ``False`` → always use numpy eager backend.
        ``None`` (default) → auto-detect based on image size vs available RAM.
    chunk_cols : int | None
        Column-chunk width for dask.  ``None`` → auto-computed.
    """

    # -------------------------
    # Validate input paths
    # -------------------------
    base, ext = os.path.splitext(dim_path)
    if ext.lower() != ".dim":
        raise ValueError(f"Expected a .dim file, got: {dim_path}")

    data_dir = base + ".data"
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Expected .data directory not found: {data_dir}")

    if not os.path.isdir(safe_path):
        raise FileNotFoundError(f"SAFE directory not found: {safe_path}")

    # -------------------------
    # Normalize decompositions list
    # -------------------------
    if n_decompositions is None:
        decomps = [int(numberofLooks)]
    else:
        if isinstance(n_decompositions, int):
            decomps = [int(n_decompositions)]
        else:
            decomps = [int(x) for x in n_decompositions]

        # Remove duplicates and sort
        decomps = sorted(set(decomps))

    if not decomps:
        raise ValueError("n_decompositions ended up empty.")
    if any(n < 2 for n in decomps):
        raise ValueError(f"Invalid n_decompositions: {decomps}. Use values >= 2.")

    # -------------------------
    # Detect base polarizations from complex i/q pairs.
    # This detection is robust to prefixes/suffixes and ignores generated *_SA* files.
    # -------------------------
    base_pairs = find_base_iq_pairs_in_dim_data(data_dir)
    pols = sorted(base_pairs.keys())

    if not pols:
        raise RuntimeError(f"No i_<POL>.img / q_<POL>.img pairs found in: {data_dir}")

    if VERBOSE:
        print(f"Base polarizations detected in {data_dir}: {pols}")
        for pol in pols:
            i_fp, q_fp = base_pairs.get(pol, (None, None))
            print(f"  {pol}: i={i_fp}, q={q_fp}")
        print(f"Metadata source SAFE: {safe_path}")
        print(f"Decompositions to run: {decomps}")

    # -------------------------
    # Run one or multiple decompositions
    # -------------------------
    for nlooks in decomps:
        # If multiple decompositions, add an N-based prefix to avoid overwriting files
        prefix_n = f"{prefix}L{nlooks}_" if len(decomps) > 1 else prefix

        if VERBOSE:
            print(f"\n=== Running decomposition N={nlooks} (prefix='{prefix_n}') ===")

        for pol in pols:
            i_fp, q_fp = base_pairs.get(pol, (None, None))

            if not i_fp or not q_fp or not (os.path.exists(i_fp) and os.path.exists(q_fp)):
                if VERBOSE:
                    print(f"Skipping POL={pol} (missing i/q): i={i_fp}, q={q_fp}")
                continue

            if VERBOSE:
                print(f"\nProcessing subapertures for POL={pol} (N={nlooks})")
                print(f"  i: {i_fp}")
                print(f"  q: {q_fp}")

            sub = CombinedSublooking(
                metadata_pointer_safe=safe_path,  # SAFE used for metadata
                numberofLooks=nlooks,
                i_image=i_fp,
                q_image=q_fp,
                DownSample=DownSample,
                assetMetadata=None,
                force_dask=force_dask,
                chunk_cols=chunk_cols,
            )

            sub.run_and_save(
                out_dir=data_dir,
                pol=pol,
                prefix=prefix_n,
                byte_order=byte_order,
                VERBOSE=VERBOSE,
            )

    # -------------------------
    # Update the .dim ONCE at the end
    # (at this point all new bands exist in .data)
    # -------------------------
    try:
        dim_updated = update_dim_add_bands_from_data_dir(
            dim_in=dim_path,
            dim_out=None,  # writes "<original>_subaps.dim"
            verbose=VERBOSE,
        )
        if VERBOSE:
            print(f"\nDIM updated: {dim_updated}")
    except Exception as e:
        # Do not fail the pipeline if the DIM update fails
        if VERBOSE:
            print(f"\nWARNING: could not auto-update the .dim file: {e}")

    # return pols

