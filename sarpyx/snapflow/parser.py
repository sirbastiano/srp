"""Argument parsing for the WorldSAR CLI."""

from __future__ import annotations

import argparse

from sarpyx.snapflow.config import DEFAULT_ORBIT_TYPE, DEFAULT_ZARR_CHUNK_SIZE


class SentinelSubapAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        decompositions = values if isinstance(values, list) else [values]
        setattr(namespace, self.dest, decompositions)
        setattr(namespace, "sentinel_subaps", decompositions[0] if len(decompositions) == 1 else None)
        setattr(namespace, "_sentinel_subap_option", option_string)


def add_worldsar_arguments(parser: argparse.ArgumentParser) -> None:
    parser.set_defaults(
        sentinel_subaps=None,
        sentinel_subap_decompositions=None,
        _sentinel_subap_option=None,
    )
    parser.add_argument("--input", "-i", dest="product_path", type=str, required=True, help="Path to the input SAR product.")
    parser.add_argument(
        "--output",
        "-o",
        dest="output_dir",
        type=str,
        default=None,
        help="Processed output directory. Defaults to <input parent>/output, or target .zarr path in --h5-to-zarr-only mode.",
    )
    parser.add_argument(
        "--cuts-outdir",
        "--cuts_outdir",
        dest="cuts_outdir",
        type=str,
        default=None,
        help="Where to store the tiles after extraction. Defaults to <output>/tiles.",
    )
    parser.add_argument(
        "--product-wkt",
        "--product_wkt",
        dest="product_wkt",
        type=str,
        default=None,
        help="WKT string defining the product region of interest.",
    )
    parser.add_argument(
        "--h5-to-zarr-only",
        dest="h5_to_zarr_only",
        action="store_true",
        help="Skip preprocessing/tiling and convert an existing .h5 tile into a Zarr v3 store.",
    )
    parser.add_argument(
        "--zarr-chunk-size",
        dest="zarr_chunk_size",
        type=int,
        nargs=2,
        metavar=("ROWS", "COLS"),
        default=DEFAULT_ZARR_CHUNK_SIZE,
        help="Chunk size for Zarr outputs (default: 128 128).",
    )
    parser.add_argument("--overwrite-zarr", dest="overwrite_zarr", action="store_true")
    parser.add_argument(
        "--preprocessing-root-name",
        dest="preprocessing_root_name",
        type=str,
        default=None,
        help=(
            "Use the deterministic directory worldsar_tmp_<name> below --output, "
            "replacing it if it already exists. By default a random suffix is used."
        ),
    )
    parser.add_argument("--tile-writer", dest="tile_writer", type=str, default="zarr", help="Tile artifact writer: h5, zarr, numpy/npz, npy, geotiff/tif, or pickle.")
    parser.add_argument(
        "--gpt-path",
        dest="gpt_path",
        type=str,
        default=None,
        help="SNAP GPT executable override. Defaults to the active conda env SNAP install.",
    )
    parser.add_argument("--grid-path", dest="grid_path", type=str, default=None)
    parser.add_argument("--db-dir", dest="db_dir", type=str, default=None, help="Tile database directory. Defaults to <output>/db.")
    parser.add_argument("--gpt-memory", dest="gpt_memory", type=str, default="16G")
    parser.add_argument("--gpt-cache-size", dest="gpt_cache_size", type=str, default="8G")
    parser.add_argument("--gpt-parallelism", dest="gpt_parallelism", type=int, default=6)
    parser.add_argument("--gpt-timeout", dest="gpt_timeout", type=int, default=None)
    parser.add_argument(
        "--lock-timeout",
        dest="lock_timeout",
        type=float,
        default=0.0,
        help="Seconds to wait for a matching WorldSAR product/output lock. Defaults to fail-fast.",
    )
    parser.add_argument("--snap-userdir", dest="snap_userdir", type=str, default=None)
    parser.add_argument(
        "--keep-intermediate",
        "--keep_intermediate",
        dest="keep_intermediate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep preprocessing intermediate products. By default WorldSAR keeps the final tiles, DB, and PDFs only.",
    )
    parser.add_argument("--orbit-type", dest="orbit_type", type=str, default=DEFAULT_ORBIT_TYPE)
    parser.add_argument("--orbit-continue-on-fail", dest="orbit_continue_on_fail", action="store_true")
    parser.add_argument(
        "--sentinel-swath",
        dest="sentinel_swath",
        choices=("IW1", "IW2", "IW3"),
        default=None,
        help="Limit Sentinel-1 TOPS preprocessing to one IW swath.",
    )
    parser.add_argument("--sentinel-first-burst", dest="sentinel_first_burst", type=int, default=None)
    parser.add_argument("--sentinel-last-burst", dest="sentinel_last_burst", type=int, default=None)
    parser.add_argument("--sentinel-tc-source-band", dest="sentinel_tc_source_band", type=str, default=None)
    parser.add_argument(
        "--sentinel-subap-decompositions",
        "--sentinel-subaps",
        dest="sentinel_subap_decompositions",
        type=int,
        nargs="+",
        action=SentinelSubapAction,
        default=None,
        metavar="N",
        help="Sentinel sub-aperture decomposition count(s). Defaults to 2.",
    )
    parser.add_argument(
        "--sentinel-subap-feature-window-size",
        dest="sentinel_subap_feature_window_size",
        type=int,
        default=5,
        help="Odd local window size for Sentinel sub-aperture Zarr features.",
    )
    parser.add_argument(
        "--skip-preprocessing",
        dest="skip_preprocessing",
        action="store_true",
        help="Skip TC preprocessing and reuse existing BEAM-DIMAP intermediate products for tiling.",
    )


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Process SAR data using SNAP GPT and sarpyx pipelines.")
    add_worldsar_arguments(parser)
    return parser
