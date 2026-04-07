#!/usr/bin/env python3
"""
Compute azimuth-subaperture features from WorldSAR SM TC products.

For each *_TC.data directory, this script looks for:
    i_<POL>_SA1.img, q_<POL>_SA1.img
    i_<POL>_SA2.img, q_<POL>_SA2.img
    i_<POL>_SA3.img, q_<POL>_SA3.img

and computes, per polarization:
    1) Inter-look coherence:
        - gamma12
        - gamma13
        - gamma23
        - gamma_mean

    2) Covariance terms:
        - C11
        - C22
        - C33
        - Re(C12), Im(C12)
        - Re(C13), Im(C13)
        - Re(C23), Im(C23)

    3) Phase variance across looks:
        - phase_variance = 1 - |mean(exp(j*phi_k))|

Outputs are written under:
    <output_root>/<product_name>/...

The script processes images block-wise with halo padding, so it is safer for
large products.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import rasterio
from rasterio.windows import Window
from scipy.ndimage import uniform_filter


POLS = ("VV", "VH")
SUBAPS = ("SA1", "SA2", "SA3")


def local_mean(arr: np.ndarray, size: int) -> np.ndarray:
    """
    Local mean filter for real or complex arrays.
    """
    if np.iscomplexobj(arr):
        real = uniform_filter(arr.real, size=size, mode="nearest")
        imag = uniform_filter(arr.imag, size=size, mode="nearest")
        return real + 1j * imag
    return uniform_filter(arr, size=size, mode="nearest")


def coherence(si: np.ndarray, sj: np.ndarray, win: int, eps: float = 1e-8) -> np.ndarray:
    """
    Magnitude coherence between two complex SLC-like looks.
    """
    num = np.abs(local_mean(si * np.conj(sj), size=win))
    den = np.sqrt(
        local_mean(np.abs(si) ** 2, size=win) *
        local_mean(np.abs(sj) ** 2, size=win)
    )
    return num / (den + eps)


def phase_variance(stack: np.ndarray) -> np.ndarray:
    """
    Circular phase variance across looks.
    stack shape: (nlooks, rows, cols), complex
    """
    phases = np.angle(stack)
    coh_phase = np.abs(np.mean(np.exp(1j * phases), axis=0))
    return 1.0 - coh_phase


def covariance_terms(s1: np.ndarray, s2: np.ndarray, s3: np.ndarray, win: int) -> Dict[str, np.ndarray]:
    """
    Compute local covariance terms from 3 complex looks.
    """
    c11 = local_mean(s1 * np.conj(s1), size=win).real
    c22 = local_mean(s2 * np.conj(s2), size=win).real
    c33 = local_mean(s3 * np.conj(s3), size=win).real

    c12 = local_mean(s1 * np.conj(s2), size=win)
    c13 = local_mean(s1 * np.conj(s3), size=win)
    c23 = local_mean(s2 * np.conj(s3), size=win)

    return {
        "C11": c11.astype(np.float32),
        "C22": c22.astype(np.float32),
        "C33": c33.astype(np.float32),
        "ReC12": c12.real.astype(np.float32),
        "ImC12": c12.imag.astype(np.float32),
        "ReC13": c13.real.astype(np.float32),
        "ImC13": c13.imag.astype(np.float32),
        "ReC23": c23.real.astype(np.float32),
        "ImC23": c23.imag.astype(np.float32),
    }


def build_output_profile(src_profile: dict, count: int) -> dict:
    """
    Convert source profile to a GeoTIFF output profile.
    """
    profile = src_profile.copy()
    profile.update(
        driver="GTiff",
        dtype="float32",
        count=count,
        compress="lzw",
        BIGTIFF="IF_SAFER",
    )
    return profile


def band_paths(product_dir: Path, pol: str) -> Dict[str, Path]:
    """
    Build expected file paths for a polarization.
    """
    return {
        "i_SA1": product_dir / f"i_{pol}_SA1.img",
        "q_SA1": product_dir / f"q_{pol}_SA1.img",
        "i_SA2": product_dir / f"i_{pol}_SA2.img",
        "q_SA2": product_dir / f"q_{pol}_SA2.img",
        "i_SA3": product_dir / f"i_{pol}_SA3.img",
        "q_SA3": product_dir / f"q_{pol}_SA3.img",
    }


def all_exist(paths: Dict[str, Path]) -> bool:
    return all(p.exists() for p in paths.values())


def clamp_window(col_off: int, row_off: int, width: int, height: int, max_width: int, max_height: int) -> Window:
    """
    Clamp a window to image bounds.
    """
    col_off = max(0, col_off)
    row_off = max(0, row_off)
    width = min(width, max_width - col_off)
    height = min(height, max_height - row_off)
    return Window(col_off, row_off, width, height)


def read_complex_window(
    src_i: rasterio.io.DatasetReader,
    src_q: rasterio.io.DatasetReader,
    window: Window,
) -> np.ndarray:
    """
    Read I/Q and return complex array.
    """
    i = src_i.read(1, window=window).astype(np.float32)
    q = src_q.read(1, window=window).astype(np.float32)
    return i + 1j * q


def crop_center(arr: np.ndarray, top: int, left: int, height: int, width: int) -> np.ndarray:
    """
    Crop central valid area from an expanded block.
    Works for 2D arrays only.
    """
    return arr[top:top + height, left:left + width]


def process_product(product_dir: Path, out_root: Path, win_size: int, verbose: bool = True) -> None:
    """
    Process one *_TC.data product directory.
    """
    product_out_dir = out_root / product_dir.name
    product_out_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\nProcessing product: {product_dir}")

    for pol in POLS:
        paths = band_paths(product_dir, pol)
        if not all_exist(paths):
            if verbose:
                print(f"  [{pol}] Missing one or more SA files. Skipping.")
            continue

        if verbose:
            print(f"  [{pol}] Found SA1/SA2/SA3 I/Q files.")

        with rasterio.open(paths["i_SA1"]) as src_ref, \
             rasterio.open(paths["i_SA1"]) as src_i1, rasterio.open(paths["q_SA1"]) as src_q1, \
             rasterio.open(paths["i_SA2"]) as src_i2, rasterio.open(paths["q_SA2"]) as src_q2, \
             rasterio.open(paths["i_SA3"]) as src_i3, rasterio.open(paths["q_SA3"]) as src_q3:

            height = src_ref.height
            width = src_ref.width

            coh_out = product_out_dir / f"coherence_{pol}.tif"
            cov_out = product_out_dir / f"covariance_{pol}.tif"
            phs_out = product_out_dir / f"phase_variance_{pol}.tif"

            coh_profile = build_output_profile(src_ref.profile, count=4)
            cov_profile = build_output_profile(src_ref.profile, count=9)
            phs_profile = build_output_profile(src_ref.profile, count=1)

            halo = win_size // 2

            with rasterio.open(coh_out, "w", **coh_profile) as dst_coh, \
                 rasterio.open(cov_out, "w", **cov_profile) as dst_cov, \
                 rasterio.open(phs_out, "w", **phs_profile) as dst_phs:

                dst_coh.set_band_description(1, "gamma12")
                dst_coh.set_band_description(2, "gamma13")
                dst_coh.set_band_description(3, "gamma23")
                dst_coh.set_band_description(4, "gamma_mean")

                cov_names = ["C11", "C22", "C33", "ReC12", "ImC12", "ReC13", "ImC13", "ReC23", "ImC23"]
                for b, name in enumerate(cov_names, start=1):
                    dst_cov.set_band_description(b, name)

                dst_phs.set_band_description(1, "phase_variance")

                for _, core_window in src_ref.block_windows(1):
                    ext_window = clamp_window(
                        col_off=int(core_window.col_off) - halo,
                        row_off=int(core_window.row_off) - halo,
                        width=int(core_window.width) + 2 * halo,
                        height=int(core_window.height) + 2 * halo,
                        max_width=width,
                        max_height=height,
                    )

                    s1_ext = read_complex_window(src_i1, src_q1, ext_window)
                    s2_ext = read_complex_window(src_i2, src_q2, ext_window)
                    s3_ext = read_complex_window(src_i3, src_q3, ext_window)

                    g12_ext = coherence(s1_ext, s2_ext, win=win_size)
                    g13_ext = coherence(s1_ext, s3_ext, win=win_size)
                    g23_ext = coherence(s2_ext, s3_ext, win=win_size)
                    gmean_ext = (g12_ext + g13_ext + g23_ext) / 3.0

                    cov_ext = covariance_terms(s1_ext, s2_ext, s3_ext, win=win_size)

                    stack_ext = np.stack([s1_ext, s2_ext, s3_ext], axis=0)
                    phv_ext = phase_variance(stack_ext).astype(np.float32)

                    top = int(core_window.row_off - ext_window.row_off)
                    left = int(core_window.col_off - ext_window.col_off)
                    h = int(core_window.height)
                    w = int(core_window.width)

                    g12 = crop_center(g12_ext, top, left, h, w).astype(np.float32)
                    g13 = crop_center(g13_ext, top, left, h, w).astype(np.float32)
                    g23 = crop_center(g23_ext, top, left, h, w).astype(np.float32)
                    gmean = crop_center(gmean_ext, top, left, h, w).astype(np.float32)

                    phv = crop_center(phv_ext, top, left, h, w).astype(np.float32)

                    cov_core = {
                        key: crop_center(val, top, left, h, w).astype(np.float32)
                        for key, val in cov_ext.items()
                    }

                    dst_coh.write(g12, 1, window=core_window)
                    dst_coh.write(g13, 2, window=core_window)
                    dst_coh.write(g23, 3, window=core_window)
                    dst_coh.write(gmean, 4, window=core_window)

                    for b, key in enumerate(cov_names, start=1):
                        dst_cov.write(cov_core[key], b, window=core_window)

                    dst_phs.write(phv, 1, window=core_window)

        if verbose:
            print(f"  [{pol}] Written:")
            print(f"       {coh_out}")
            print(f"       {cov_out}")
            print(f"       {phs_out}")


def find_tc_products(root: Path) -> List[Path]:
    """
    If root itself is a *_TC.data directory, process only that product.
    Otherwise, recursively find all *_TC.data directories under root.
    """
    if root.is_dir() and root.name.endswith("_TC.data"):
        return [root]
    return sorted([p for p in root.rglob("*_TC.data") if p.is_dir()])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute coherence, covariance, and phase variance from WorldSAR SM TC products."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Root directory containing worldsar_output products.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Directory where output feature folders will be created.",
    )
    parser.add_argument(
        "--win-size",
        type=int,
        default=5,
        help="Local averaging window size. Recommended: 5 or 7.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce console output.",
    )

    args = parser.parse_args()

    if args.win_size < 1 or args.win_size % 2 == 0:
        raise ValueError("--win-size must be a positive odd integer.")

    input_root = args.input_root
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    products = find_tc_products(input_root)
    if not products:
        print(f"No *_TC.data directories found under: {input_root}")
        return

    print(f"Found {len(products)} *_TC.data products.")

    for product_dir in products:
        process_product(
            product_dir=product_dir,
            out_root=output_root,
            win_size=args.win_size,
            verbose=not args.quiet,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()