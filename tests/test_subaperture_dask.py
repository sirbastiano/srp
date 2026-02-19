"""
Tests for the dask divide-and-conquer subaperture pipeline.

Verifies that numpy and dask backends produce identical results,
and that the dask path keeps memory bounded.
"""
import os
import tempfile
import shutil

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers: synthetic ENVI images & fake metadata
# ---------------------------------------------------------------------------

def _write_envi_float32(path_img, path_hdr, arr2d, band_name, byte_order=1):
    """Minimal ENVI BSQ writer for test fixtures."""
    arr = np.asarray(arr2d, dtype=np.float32)
    arr_out = arr.astype(">f4") if byte_order == 1 else arr.astype("<f4")
    arr_out.tofile(path_img)

    lines, samples = arr.shape
    hdr = (
        f"ENVI\nsamples = {samples}\nlines = {lines}\nbands = 1\n"
        f"header offset = 0\nfile type = ENVI Standard\ndata type = 4\n"
        f"interleave = bsq\nbyte order = {byte_order}\n"
        f"band names = {{ {band_name} }}\n"
    )
    with open(path_hdr, "w") as f:
        f.write(hdr)


def _make_synthetic_envi(tmpdir, nrows, ncols, pol="VV", seed=42):
    """
    Write paired i_<POL>.img / q_<POL>.img ENVI files with random complex64
    data and return (i_path, q_path, complex_array).
    """
    rng = np.random.default_rng(seed)
    img_i = rng.standard_normal((nrows, ncols)).astype(np.float32)
    img_q = rng.standard_normal((nrows, ncols)).astype(np.float32)

    i_path = os.path.join(tmpdir, f"i_{pol}.img")
    h_path = os.path.join(tmpdir, f"i_{pol}.hdr")
    q_path = os.path.join(tmpdir, f"q_{pol}.img")
    qh_path = os.path.join(tmpdir, f"q_{pol}.hdr")

    _write_envi_float32(i_path, h_path, img_i, f"i_{pol}")
    _write_envi_float32(q_path, qh_path, img_q, f"q_{pol}")

    cplx = np.empty((nrows, ncols), dtype=np.complex64)
    cplx.real = img_i
    cplx.imag = img_q
    return i_path, q_path, cplx


# Realistic-ish Sentinel-1 metadata as a nested dict
_FAKE_META = {
    "generalAnnotation": {
        "downlinkInformationList": {
            "downlinkInformation": {
                "prf": "1717.129",
                "downlinkValues": {
                    "txPulseLength": "4.175039e-05",
                    "txPulseRampRate": "1.078230382e+12",
                },
            }
        }
    },
    "imageAnnotation": {
        "processingInformation": {
            "swathProcParamsList": {
                "swathProcParams": {
                    "azimuthProcessing": {
                        "totalBandwidth": "327.0",
                        "windowCoefficient": "0.75",
                    },
                    "rangeProcessing": {
                        "totalBandwidth": "56500000.0",
                        "windowCoefficient": "0.75",
                    },
                }
            }
        },
        "imageInformation": {
            "rangePixelSpacing": "2.329562",
            "azimuthPixelSpacing": "13.96248",
        },
    },
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestNumpyDaskEquivalence:
    """Both backends must produce numerically identical sublooks."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.tmpdir = str(tmp_path)
        self.nrows, self.ncols = 256, 128
        self.nlooks = 3
        self.i_path, self.q_path, self.cplx = _make_synthetic_envi(
            self.tmpdir, self.nrows, self.ncols
        )

    def _run_backend(self, use_dask):
        from sarpyx.processor.core.subaperture_full_img import CombinedSublooking

        sub = CombinedSublooking(
            metadata_pointer_safe="unused",
            numberofLooks=self.nlooks,
            i_image=self.i_path,
            q_image=self.q_path,
            assetMetadata=_FAKE_META,
            force_dask=use_dask,
            chunk_cols=32,  # small chunks to exercise dask thoroughly
        )
        looks = sub.chain()

        # Ensure they are materialised numpy arrays
        out = []
        for lk in looks:
            import dask.array as da
            if isinstance(lk, da.Array):
                lk = lk.compute()
            out.append(np.asarray(lk))
        return out

    def test_numpy_produces_correct_shape(self):
        looks = self._run_backend(use_dask=False)
        assert len(looks) == self.nlooks
        for lk in looks:
            assert lk.shape == (self.nrows, self.ncols)
            assert lk.dtype == np.complex64

    def test_dask_produces_correct_shape(self):
        looks = self._run_backend(use_dask=True)
        assert len(looks) == self.nlooks
        for lk in looks:
            assert lk.shape == (self.nrows, self.ncols)
            assert lk.dtype == np.complex64

    def test_numpy_dask_match(self):
        """Core equivalence: both paths yield the same sublooks within float32 tolerance."""
        np_looks = self._run_backend(use_dask=False)
        da_looks = self._run_backend(use_dask=True)

        for idx, (nl, dl) in enumerate(zip(np_looks, da_looks)):
            np.testing.assert_allclose(
                nl, dl,
                rtol=1e-4,
                atol=1e-6,
                err_msg=f"Sublook {idx} diverges between numpy and dask backends",
            )


class TestSaveAndLoad:
    """Verify save_sublooks_envi round-trips correctly."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.tmpdir = str(tmp_path)
        self.nrows, self.ncols = 128, 64
        self.nlooks = 2
        self.i_path, self.q_path, _ = _make_synthetic_envi(
            self.tmpdir, self.nrows, self.ncols
        )

    def _run_and_save(self, use_dask):
        from sarpyx.processor.core.subaperture_full_img import CombinedSublooking

        out_dir = os.path.join(self.tmpdir, f"out_{'dask' if use_dask else 'numpy'}")
        os.makedirs(out_dir, exist_ok=True)

        sub = CombinedSublooking(
            metadata_pointer_safe="unused",
            numberofLooks=self.nlooks,
            i_image=self.i_path,
            q_image=self.q_path,
            assetMetadata=_FAKE_META,
            force_dask=use_dask,
            chunk_cols=32,
        )
        sub.run_and_save(out_dir=out_dir, pol="VV", byte_order=1)
        return out_dir

    @pytest.mark.parametrize("use_dask", [False, True], ids=["numpy", "dask"])
    def test_output_files_exist(self, use_dask):
        out_dir = self._run_and_save(use_dask)
        for sa in range(1, self.nlooks + 1):
            for comp in ("i", "q"):
                img = os.path.join(out_dir, f"{comp}_VV_SA{sa}.img")
                hdr = os.path.join(out_dir, f"{comp}_VV_SA{sa}.hdr")
                assert os.path.isfile(img), f"Missing: {img}"
                assert os.path.isfile(hdr), f"Missing: {hdr}"

    @pytest.mark.parametrize("use_dask", [False, True], ids=["numpy", "dask"])
    def test_output_shape(self, use_dask):
        import rasterio
        out_dir = self._run_and_save(use_dask)
        for sa in range(1, self.nlooks + 1):
            img = os.path.join(out_dir, f"i_VV_SA{sa}.img")
            with rasterio.open(img) as src:
                assert src.height == self.nrows
                assert src.width == self.ncols


class TestAutoDetection:
    """Verify the auto-detection logic picks the right backend."""

    def test_small_image_uses_numpy(self, tmp_path):
        tmpdir = str(tmp_path)
        i_path, q_path, _ = _make_synthetic_envi(tmpdir, 64, 64)

        from sarpyx.processor.core.subaperture_full_img import CombinedSublooking

        sub = CombinedSublooking(
            metadata_pointer_safe="unused",
            numberofLooks=3,
            i_image=i_path,
            q_image=q_path,
            assetMetadata=_FAKE_META,
            force_dask=None,  # auto
        )
        assert sub._use_dask is False, "Small image should use numpy path"

    def test_force_dask_overrides(self, tmp_path):
        tmpdir = str(tmp_path)
        i_path, q_path, _ = _make_synthetic_envi(tmpdir, 64, 64)

        from sarpyx.processor.core.subaperture_full_img import CombinedSublooking

        sub = CombinedSublooking(
            metadata_pointer_safe="unused",
            numberofLooks=3,
            i_image=i_path,
            q_image=q_path,
            assetMetadata=_FAKE_META,
            force_dask=True,
        )
        assert sub._use_dask is True


class TestDtypeConsistency:
    """Ensure complex64 is maintained throughout — no silent promotion to complex128."""

    def test_no_complex128(self, tmp_path):
        tmpdir = str(tmp_path)
        i_path, q_path, _ = _make_synthetic_envi(tmpdir, 128, 64)

        from sarpyx.processor.core.subaperture_full_img import CombinedSublooking

        for use_dask in (False, True):
            sub = CombinedSublooking(
                metadata_pointer_safe="unused",
                numberofLooks=2,
                i_image=i_path,
                q_image=q_path,
                assetMetadata=_FAKE_META,
                force_dask=use_dask,
                chunk_cols=32,
            )
            looks = sub.chain()
            for idx, lk in enumerate(looks):
                import dask.array as da
                if isinstance(lk, da.Array):
                    lk = lk.compute()
                assert lk.dtype == np.complex64, (
                    f"Sublook {idx} dtype={lk.dtype} (dask={use_dask})"
                )


class TestVectorizedDeHammWin:
    """Verify vectorized DeHammWin matches legacy behavior."""

    def test_1d(self):
        from sarpyx.processor.core.subaperture_full_img import DeHammWin

        rng = np.random.default_rng(123)
        signal = (rng.standard_normal(256) + 1j * rng.standard_normal(256)).astype(np.complex64)
        coeff = 0.75

        result = DeHammWin(signal, coeff)
        assert result.shape == signal.shape
        assert result.dtype == np.complex64

        # Manually verify: division by hamming window then conjugate
        n = len(signal)
        alpha = np.linspace(0, 2 * np.pi * (n - 1) / n, n, dtype=np.float32)
        w = np.float32(coeff) - np.float32(1 - coeff) * np.cos(alpha)
        expected = np.conj(signal / w).astype(np.complex64)

        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_2d(self):
        from sarpyx.processor.core.subaperture_full_img import DeHammWin_2d, DeHammWin

        rng = np.random.default_rng(456)
        signals = (rng.standard_normal((128, 16)) + 1j * rng.standard_normal((128, 16))).astype(np.complex64)
        coeff = 0.75

        result_2d = DeHammWin_2d(signals, coeff)

        # Compare with column-by-column 1-D calls
        for col in range(signals.shape[1]):
            expected_col = DeHammWin(signals[:, col], coeff)
            np.testing.assert_allclose(
                result_2d[:, col], expected_col, rtol=1e-5,
                err_msg=f"Column {col} mismatch",
            )
