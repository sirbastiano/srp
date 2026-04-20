"""WorldSAR CLI pipelines for SAR product preprocessing and tiling.

TODO: metadata reorganization.
TODO: SUBAPERTURE PROCESSING for all missions.
TODO: PolSAR support.
TODO: InSAR support.
"""

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from functools import partial
from pathlib import Path

import h5py
import pandas as pd
import pyproj
from dotenv import load_dotenv

from sarpyx.processor.core.dim_updater import update_dim_add_bands_from_data_dir
from sarpyx.snapflow.engine import GPT
from sarpyx.utils.geos import (
    check_points_in_polygon, grid_cell_utm_bbox, rectangle_to_wkt, rectanglify,
)
from sarpyx.utils.io import read_h5
from sarpyx.utils.meta import normalize_sar_timestamp
from sarpyx.utils.nisar_utils import NISARCutter, NISARReader
from sarpyx.utils.wkt_utils import nisar_wkt_extractor, sentinel1_wkt_extractor_cdse, sentinel1_wkt_extractor_manifest
from sarpyx.utils.worldsar_h5 import (
    DEFAULT_ZARR_CHUNK_SIZE,
    convert_tile_h5_to_zarr,
    enrich_validation_results_with_h5_structure,
    validate_h5_tile as _shared_validate_h5_tile,
    write_h5_validation_report_pdf as _shared_write_h5_validation_report_pdf,
)

load_dotenv()


def check_points_in_polygon(*args, **kwargs):
    from sarpyx.utils.geos import check_points_in_polygon as _impl
    globals()['check_points_in_polygon'] = _impl
    return _impl(*args, **kwargs)


def rectangle_to_wkt(*args, **kwargs):
    from sarpyx.utils.geos import rectangle_to_wkt as _impl
    globals()['rectangle_to_wkt'] = _impl
    return _impl(*args, **kwargs)


def rectanglify(*args, **kwargs):
    from sarpyx.utils.geos import rectanglify as _impl
    globals()['rectanglify'] = _impl
    return _impl(*args, **kwargs)


def sentinel1_swath_wkt_extractor_safe(*args, **kwargs):
    from sarpyx.utils.wkt_utils import sentinel1_swath_wkt_extractor_safe as _impl
    globals()['sentinel1_swath_wkt_extractor_safe'] = _impl
    return _impl(*args, **kwargs)


def sentinel1_wkt_extractor_cdse(*args, **kwargs):
    from sarpyx.utils.wkt_utils import sentinel1_wkt_extractor_cdse as _impl
    globals()['sentinel1_wkt_extractor_cdse'] = _impl
    return _impl(*args, **kwargs)


def sentinel1_wkt_extractor_manifest(*args, **kwargs):
    from sarpyx.utils.wkt_utils import sentinel1_wkt_extractor_manifest as _impl
    globals()['sentinel1_wkt_extractor_manifest'] = _impl
    return _impl(*args, **kwargs)


def merge_iq_into_pdec(*args, **kwargs):
    try:
        from src.merge_iq_into_pdec import merge_iq_into_pdec as _impl
    except ModuleNotFoundError as exc:
        if getattr(exc, 'name', None) != 'merge_iq_into_pdec':
            raise
        module_file = globals().get('__file__')
        candidate = Path(module_file).with_name('merge_iq_into_pdec.py') if module_file else None
        if candidate is None or not candidate.exists():
            raise
        spec = importlib.util.spec_from_file_location('worldsar_merge_iq_into_pdec', candidate)
        if spec is None or spec.loader is None:
            raise ImportError(f'Could not load merge_iq_into_pdec from {candidate}')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _impl = module.merge_iq_into_pdec
    globals()['merge_iq_into_pdec'] = _impl
    return _impl(*args, **kwargs)


# ══════════════════════════════════════════════════════════════════════════════
#  Configuration
# ══════════════════════════════════════════════════════════════════════════════

def _env(*names, default=None):
    """Return the first non-empty environment variable from *names*, or *default*."""
    for n in names:
        v = os.getenv(n)
        if v:
            return v
    return default


def _expand_path(path_value):
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).absolute()
    return path


def _ensure_existing_path(path_value, label):
    path = _expand_path(path_value)
    if not path.exists():
        raise FileNotFoundError(f'{label} does not exist: {path}')
    return path


GPT_PATH     = _env('gpt_path', 'GPT_PATH')
GRID_PATH    = _env('grid_path', 'GRID_PATH')
DB_DIR       = _env('db_dir', 'DB_DIR')
CUTS_OUTDIR  = _env('cuts_outdir', 'OUTPUT_CUTS_DIR')
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_PATH    = _env('base_path', 'BASE_PATH', default=str(PROJECT_ROOT))
SNAP_USERDIR = _env('SNAP_USERDIR', 'snap_userdir', default=str(PROJECT_ROOT / '.snap'))
os.environ.setdefault('SNAP_USERDIR', SNAP_USERDIR)

prepro      = True
tiling      = True
db_indexing = True
CORE_METADATA_KEYS = (
    'MISSION',
    'ACQUISITION_MODE',
    'PRODUCT_TYPE',
    'radar_frequency',
    'pulse_repetition_frequency',
    'range_spacing',
    'azimuth_spacing',
    'range_bandwidth',
    'azimuth_bandwidth',
    'antenna_pointing',
    'PASS',
    'avg_scene_height',
    'PRODUCT',
    'mds1_tx_rx_polar',
    'mds2_tx_rx_polar',
    'first_line_time',
)
REQUIRED_BAND_ATTRS = (
    'CLASS',
    'IMAGE_VERSION',
    'log10_scaled',
    'raster_height',
    'raster_width',
    'scaling_factor',
    'scaling_offset',
    'unit',
)


# ══════════════════════════════════════════════════════════════════════════════
#  Pipelines  –  one per mission family, dispatched via ROUTER[mode]
# ══════════════════════════════════════════════════════════════════════════════

def _apply_sentinel_orbit_file(
    op,
    orbit_type='Sentinel Precise (Auto Download)',
    orbit_continue_on_fail=False,
):
    orbit_product = op.ApplyOrbitFile(
        orbit_type=orbit_type,
        continue_on_fail=orbit_continue_on_fail,
    )
    if orbit_product is not None:
        return orbit_product

    error_summary = op.last_error_summary()
    normalized_error = error_summary.lower()
    offline_orbit_failure_markers = (
        'network is unreachable',
        'unable to connect to http://step.esa.int/auxdata/orbits/',
        'unable to connect to https://step.esa.int/auxdata/orbits/',
    )
    missing_orbit_file_markers = (
        'no valid orbit file found',
        'orbit files may be downloaded from copernicus dataspaces',
    )
    recoverable_offline_failure = (
        any(marker in normalized_error for marker in offline_orbit_failure_markers)
        or all(marker in normalized_error for marker in missing_orbit_file_markers)
    )
    if orbit_continue_on_fail or recoverable_offline_failure:
        print(f'WARNING: Apply-Orbit-File failed but continuing without orbit correction: {error_summary}')
        return op.prod_path

    raise RuntimeError(f'Apply-Orbit-File failed: {error_summary}')


def _sentinel_post_chain(
    op,
    product_path,
    orbit_type='Sentinel Precise (Auto Download)',
    orbit_continue_on_fail=False,
):
    """Calibration → DerampDemod → Deburst → PolDecomp → TC  (shared by each swath)."""
    fp_orb = _apply_sentinel_orbit_file(
        op,
        orbit_type=orbit_type,
        orbit_continue_on_fail=orbit_continue_on_fail,
    )
    fp_cal = op.Calibration(output_complex=True)
    if fp_cal is None:
        raise RuntimeError(f'Calibration failed: {op.last_error_summary()}')
    fp_deramp = op.TopsarDerampDemod()
    if fp_deramp is None:
        raise RuntimeError(f'TOPSAR-DerampDemod failed: {op.last_error_summary()}')
    fp_deb = op.Deburst()
    if fp_deb is None:
        raise RuntimeError(f'TOPSAR-Deburst failed: {op.last_error_summary()}')

    op.do_subaps(
        dim_path=op.prod_path,
        safe_path=product_path,
        n_decompositions=[2],
        byte_order=1,
        VERBOSE=False,
        update_dim=False,
        tops_iw_mode=True,
        iw_apply_spectrum_normalization=False,
        iw_energy_compensation=True,
        iw_flip_output=True,
        iw_row_equalization=False,
        iw_doppler_centroid_correction=True,
        iw_dc_smooth_win=129,
        iw_equal_energy_split=True,
        iw_crosslook_row_balance=True,
        iw_crosslook_row_balance_smooth_win=257,
        iw_crosslook_row_balance_clip=1.5,
    )

    fp_pdec = op.polarimetric_decomposition(decomposition="H-Alpha Dual Pol Decomposition", window_size=5)
    if fp_pdec is None:
        raise RuntimeError(f'Polarimetric decomposition failed: {op.last_error_summary()}')
    try:
        merge_iq_into_pdec(
            src_dim=fp_deb,
            pdec_dim=fp_pdec,
            is_tops=True,
            overwrite_copied_files=False,
            backup=False,
        )
    except ModuleNotFoundError as exc:
        if getattr(exc, 'name', None) != 'merge_iq_into_pdec':
            raise
        raise RuntimeError(
            'merge_iq_into_pdec module is required for TOPS flow. '
            'TOPS fallback to DIM metadata rewrite is disabled to avoid malformed DEB metadata.'
        )
    fp_tc = op.TerrainCorrection(
        map_projection='AUTO:42001',
        pixel_spacing_in_meter=10.0,
    )
    if fp_tc is None:
        raise RuntimeError(f'Terrain Correction failed: {op.last_error_summary()}')
    return op.prod_path


def pipeline_sentinel(
    product_path, output_dir, is_TOPS=False,
    gpt_memory=None, gpt_parallelism=None, gpt_timeout=None,
    orbit_type='Sentinel Precise (Auto Download)',
    orbit_continue_on_fail=False, **_,
):
    """Sentinel-1 pipeline.

    TOPS mode:  orbit → split IW1/IW2/IW3 → (cal → deramp → deburst → subap → PDEC → merge → TC) per swath.
    STRIP mode: orbit → cal → subap → PDEC → merge → TC.
    """
    gpt_kw = dict(gpt_memory=gpt_memory, gpt_parallelism=gpt_parallelism, gpt_timeout=gpt_timeout)
    op = _create_gpt_operator(product_path, output_dir, 'BEAM-DIMAP', **gpt_kw)

    if is_TOPS:
        results = {}
        for swath in ('IW1', 'IW2', 'IW3'):
            sw_op = _create_gpt_operator(Path(op.prod_path), output_dir / swath, 'BEAM-DIMAP', **gpt_kw)
            split_result = sw_op.TopsarSplit(subswath=swath)  # SPLIT
            if split_result is None:
                raise RuntimeError(f'TOPSAR-Split failed for {swath}: {sw_op.last_error_summary()}')
            if not Path(split_result).exists():
                raise FileNotFoundError(f'TOPSAR-Split output missing for {swath}: {split_result}')
            results[swath] = _sentinel_post_chain(
                op=sw_op,
                product_path=product_path,
                orbit_type=orbit_type,
                orbit_continue_on_fail=orbit_continue_on_fail,
            )
        return results                    # {IW1: path, IW2: path, IW3: path}
    
    # STRIP mode – no split / deburst needed
    orbit_product = _apply_sentinel_orbit_file(
        op,
        orbit_type=orbit_type,
        orbit_continue_on_fail=orbit_continue_on_fail,
    )
    fp_cal = op.Calibration(output_complex=True)
    if fp_cal is None:
        raise RuntimeError(f'Calibration failed: {op.last_error_summary()}')

    op.do_subaps(
        safe_path=product_path,
        dim_path=op.prod_path,
        n_decompositions=[3],
        byte_order=1,
        VERBOSE=False,
        update_dim=False,
    )
    fp_pdec = op.polarimetric_decomposition(decomposition="H-Alpha Dual Pol Decomposition", window_size=5)
    if fp_pdec is None:
        raise RuntimeError(f'Polarimetric decomposition failed: {op.last_error_summary()}')
    try:
        merge_iq_into_pdec(
            src_dim=fp_cal,
            pdec_dim=fp_pdec,
            is_tops=False,
            overwrite_copied_files=False,
            backup=False,
        )
    except ModuleNotFoundError as exc:
        if getattr(exc, 'name', None) != 'merge_iq_into_pdec':
            raise
        fp_cal = update_dim_add_bands_from_data_dir(fp_cal, verbose=False)
        fp_merged = op.BandMerge(
            source_products=[fp_pdec, fp_cal],
            output_name=f'{Path(fp_pdec).stem}_MERGED',
        )
        if fp_merged is None:
            raise RuntimeError(f'BandMerge failed: {op.last_error_summary()}')
    fp_tc = op.TerrainCorrection(
        map_projection='AUTO:42001',
        pixel_spacing_in_meter=10.0,
    )
    if fp_tc is None:
        raise RuntimeError(f'Terrain Correction failed: {op.last_error_summary()}')
    return op.prod_path


def pipeline_tsx_csg(product_path, output_dir, gpt_memory=None, gpt_parallelism=None, gpt_timeout=None, **_):
    """TerraSAR-X / COSMO-SkyMed: calibration → terrain correction."""
    op = _create_gpt_operator(product_path, output_dir, 'BEAM-DIMAP', gpt_memory, gpt_parallelism, gpt_timeout)
    if op.Calibration(output_complex=True) is None:
        raise RuntimeError('Calibration failed.')
    # TODO: Add subaperture.
    if op.TerrainCorrection(map_projection='AUTO:42001', pixel_spacing_in_meter=5.0) is None:
        raise RuntimeError('Terrain Correction failed.')
    return op.prod_path


def pipeline_biomass(product_path, output_dir, gpt_memory=None, gpt_parallelism=None, gpt_timeout=None, **_):
    """BIOMASS: write to GeoTIFF."""
    op = _create_gpt_operator(product_path, output_dir, 'GDAL-GTiff-WRITER', gpt_memory, gpt_parallelism, gpt_timeout)
    if op.Write() is None:
        raise RuntimeError('Write failed.')
    # TODO: Calculate SubApertures with BIOMASS Data.
    return op.prod_path


def pipeline_nisar(product_path, output_dir, **_):
    """NISAR: pass-through (tiling handled downstream by NISARCutter)."""
    if Path(product_path).suffix.lower() != '.h5':
        raise ValueError('NISAR products must be in .h5 format.')
    return product_path


ROUTER = {
    'S1TOPS':  partial(pipeline_sentinel, is_TOPS=True),
    'S1STRIP': partial(pipeline_sentinel, is_TOPS=False),
    'TSX':     pipeline_tsx_csg,
    'CSG':     pipeline_tsx_csg,
    'BM':      pipeline_biomass,
    'NISAR':   pipeline_nisar,
}


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

_PARSER_ARGS = [
    # (['--input', '-i'],                dict(dest='product_path', type=str, required=True, help='Path to the input SAR product.')),
    (['--input', '-i'],                dict(dest='product_path', type=str, default='/shared/home/vmarsocci/S1C_IW_SLC__1SDV_20260130T152608_20260130T152634_006135_00C4FA_664F.SAFE', required=False, help='Path to the input SAR product.')),
    # (['--output', '-o'],               dict(dest='output_dir', type=str, default=None, help='Processed output directory, or target .zarr path in --h5-to-zarr-only mode.')),
    (['--output', '-o'],               dict(dest='output_dir', type=str, default='/shared/home/vmarsocci/WORLDSAR/outputs/worldsar-output', required=False, help='Directory to save the processed output.')),
    # (['--cuts-outdir', '--cuts_outdir'], dict(dest='cuts_outdir', type=str, default=None, help='Where to store the tiles after extraction.')),
    (['--cuts-outdir', '--cuts_outdir'], dict(dest='cuts_outdir', type=str, default='/shared/home/vmarsocci/WORLDSAR/outputs/tiles', help='Where to store the tiles after extraction.')),
    (['--product-wkt', '--product_wkt'], dict(dest='product_wkt', type=str, default=None, help='WKT string defining the product region of interest.')),
    (['--h5-to-zarr-only'],            dict(dest='h5_to_zarr_only', action='store_true', help='Skip preprocessing/tiling and convert an existing .h5 tile into a Zarr v3 store.')),
    (['--zarr-chunk-size'],            dict(dest='zarr_chunk_size', type=int, nargs=2, metavar=('ROWS', 'COLS'), default=DEFAULT_ZARR_CHUNK_SIZE, help='Chunk size for H5-to-Zarr conversion. Defaults to 32 32.')),
    (['--overwrite-zarr'],             dict(dest='overwrite_zarr', action='store_true', help='Replace an existing output Zarr store when converting H5 tiles.')),
    # (['--gpt-path'],                   dict(dest='gpt_path', type=str, default=None, help='Override GPT executable path.')),
    (['--gpt-path'],                   dict(dest='gpt_path', type=str, default='/shared/home/vmarsocci/WORLDSAR/gpt-wrapper.sh', help='Override GPT executable path.')),
    (['--grid-path'],                  dict(dest='grid_path', type=str, default='/shared/home/vmarsocci/WORLDSAR/grid/grid_10km.geojson', help='Override grid GeoJSON path.')),
    # (['--grid-path'],                  dict(dest='grid_path', type=str, default=None, help='Override grid GeoJSON path.')),
    (['--db-dir'],                     dict(dest='db_dir', type=str, default='/shared/home/vmarsocci/WORLDSAR/outputs/DB', help='Override database output directory.')),
    # (['--db-dir'],                     dict(dest='db_dir', type=str, default=None, help='Override database output directory.')),
    (['--gpt-memory'],                 dict(dest='gpt_memory', type=str, default='16G', help='GPT Java heap (e.g., 24G).')),
    (['--gpt-parallelism'],            dict(dest='gpt_parallelism', type=int, default=16, help='GPT parallelism (number of tiles).')),
    (['--gpt-timeout'],                dict(dest='gpt_timeout', type=int, default=0, help='GPT timeout in seconds. Use 0 to disable timeout (recommended for long TOPS flows).')),
    (['--snap-userdir'],               dict(dest='snap_userdir', type=str, default=None, help='Override SNAP user directory.')),
    (['--orbit-type'],                 dict(dest='orbit_type', type=str, default='Sentinel Precise (Auto Download)', help='SNAP Apply-Orbit-File orbitType.')),
    (['--orbit-continue-on-fail'],     dict(dest='orbit_continue_on_fail', action='store_true', help='Continue if orbit file cannot be applied.')),
    (['--skip-preprocessing'],         dict(dest='skip_preprocessing', action='store_true', help='Skip TC preprocessing and reuse existing BEAM-DIMAP intermediate products for tiling.')),
]

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Process SAR data using SNAP GPT and sarpyx pipelines.')
    for flags, kwargs in _PARSER_ARGS:
        parser.add_argument(*flags, **kwargs)
    return parser


# ══════════════════════════════════════════════════════════════════════════════
#  Internals
# ══════════════════════════════════════════════════════════════════════════════

# ── GPT helpers ──────────────────────────────────────────────────────────────

def _build_gpt_kwargs(gpt_memory, gpt_parallelism, gpt_timeout):
    return {k: v for k, v in [('memory', gpt_memory), ('parallelism', gpt_parallelism), ('timeout', gpt_timeout)] if v}


def _create_gpt_operator(product_path, output_dir, output_format, gpt_memory=None, gpt_parallelism=None, gpt_timeout=None):
    return GPT(
        product=product_path, outdir=output_dir, format=output_format,
        gpt_path=GPT_PATH, snap_userdir=SNAP_USERDIR,
        **_build_gpt_kwargs(gpt_memory, gpt_parallelism, gpt_timeout),
    )


def _run_gpt_op(product_path, output_dir, output_format, op_name, gpt_memory=None, gpt_parallelism=None, gpt_timeout=None, **op_kwargs):
    """Run a single GPT operation, validate the result, and return the output path."""
    op = _create_gpt_operator(product_path, output_dir, output_format, gpt_memory, gpt_parallelism, gpt_timeout)
    result = getattr(op, op_name)(**op_kwargs)
    if result is None:
        error_summary = op.last_error_summary()
        timeout_hint = ''
        if 'timed out' in error_summary.lower():
            timeout_hint = ' Increase --gpt-timeout (e.g. 14400) or disable it with --gpt-timeout 0.'
        raise RuntimeError(f'GPT {op_name} failed: {error_summary}{timeout_hint}')
    output_path = Path(result)
    if not output_path.exists():
        raise RuntimeError(f'GPT {op_name} reported {output_path} but output file is missing.')
    return output_path


# ── Product identification ───────────────────────────────────────────────────

def extract_product_id(path: str) -> str | None:
    """Extract product ID from BEAM-DIMAP path."""
    match = re.search(r'/([^/]+?)_[^/_]+\.dim$', path)
    return match.group(1) if match else None


def infer_product_mode(product_path: Path) -> str:
    """Infer product mode from product naming patterns."""
    name = product_path.name.upper()
    stem = product_path.stem.upper()
    as_path = product_path.as_posix().upper()

    if 'NISAR' in as_path or ('GSLC' in as_path and product_path.suffix.lower() == '.h5'):
        return 'NISAR'
    if any(t in as_path for t in ('TSX', 'TDX', 'TERRASAR', 'TANDEMX')):
        return 'TSX'
    if any(t in as_path for t in ('CSG', 'CSK', 'COSMO')):
        return 'CSG'
    if any(t in as_path for t in ('BIOMASS', '/BIO', '_BIO', '-BIO')):
        return 'BM'
    if re.search(r'(?:^|[^A-Z0-9])S1[ABC](?:_|[^A-Z0-9])', as_path):
        mode_match = re.search(r'S1[ABC]_([A-Z0-9]{2})_', stem)
        mode_token = mode_match.group(1) if mode_match else None
        if mode_token in {'IW', 'EW'}:
            return 'S1TOPS'
        if mode_token in {'SM', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6'}:
            return 'S1STRIP'
        if '_IW_' in name or '_EW_' in name or 'TOPS' in name:
            return 'S1TOPS'
        return 'S1TOPS'

    raise ValueError(
        f'Could not infer product mode from input path: {product_path}. '
        'Supported modes: S1TOPS/S1STRIP, BM, NISAR, TSX, CSG.'
    )


# ── Tile cutting ─────────────────────────────────────────────────────────────

def _validate_tile_result(tile_name, output_path, label):
    """Return a success/failure dict by checking whether *output_path* exists and is non-empty."""
    if not output_path.exists():
        return {'tile': tile_name, 'status': 'failed', 'reason': f'output missing after {label}', 'output_path': str(output_path)}
    size = output_path.stat().st_size
    if size == 0:
        return {'tile': tile_name, 'status': 'failed', 'reason': f'output file is empty after {label}', 'output_path': str(output_path)}
    return {'tile': tile_name, 'status': 'success', 'output_path': str(output_path), 'size_bytes': size}


def to_geotiff(product_path, output_dir, geo_region=None, output_name=None, gpt_memory=None, gpt_parallelism=None, gpt_timeout=None):
    if geo_region is None:
        raise ValueError('Geo region WKT string must be provided.')
    return _run_gpt_op(product_path, output_dir, 'GDAL-GTiff-WRITER', 'Write',
                       gpt_memory=gpt_memory, gpt_parallelism=gpt_parallelism, gpt_timeout=gpt_timeout)


def subset(product_path, output_dir, geo_region=None, region=None, output_name=None, gpt_memory=None, gpt_parallelism=None, gpt_timeout=None):
    assert geo_region is not None or region is not None, \
        'Either geo_region (WKT) or region (pixel coords "x,y,width,height") must be provided.'
    kwargs = {'copy_metadata': True, 'output_name': output_name}
    if geo_region is not None:
        kwargs['geo_region'] = geo_region
    if region is not None:
        kwargs['region'] = region
    return _run_gpt_op(
        product_path, output_dir, 'HDF5', 'Subset',
        gpt_memory=gpt_memory, gpt_parallelism=gpt_parallelism, gpt_timeout=gpt_timeout,
        **kwargs,
    )


def swath_splitter(swath, product_path, output_dir, gpt_memory=None, gpt_parallelism=None, gpt_timeout=None, **extra):
    """Split a Sentinel-1 TOPS product by subswath (1, 2, or 3)."""
    return _run_gpt_op(
        product_path, output_dir, 'BEAM-DIMAP', 'topsar_split',
        gpt_memory=gpt_memory, gpt_parallelism=gpt_parallelism, gpt_timeout=gpt_timeout,
        subswath=f'IW{swath}', **extra,
    )


def _read_geotransform(dim_path: Path) -> tuple:
    """Read a GDAL-style geotransform from a BEAM-DIMAP .dim file.

    Returns (x_origin, x_pixel_size, 0, y_origin, 0, y_pixel_size) where
    x_origin/y_origin are the UTM coordinates of the top-left pixel corner and
    y_pixel_size is negative (northing decreases going down rows).

    BEAM-DIMAP IMAGE_TO_MODEL_TRANSFORM stores values in Java AffineTransform order:
    (m00, m10, m01, m11, m02, m12) = (x_scale, y_shear, x_shear, y_scale, x_translate, y_translate)
    GDAL geotransform order: (x_origin, x_scale, x_rot, y_origin, y_rot, y_scale)
    Mapping: GDAL = (m02, m00, m01, m12, m10, m11) = (values[4], values[0], values[2], values[5], values[1], values[3])
    Note: GDAL's own BEAM-DIMAP driver misparses this transform, so we always use the XML directly.
    """
    tree = ET.parse(dim_path)
    root = tree.getroot()
    elem = root.find('.//IMAGE_TO_MODEL_TRANSFORM')
    if elem is not None and elem.text is not None:
        # Java AffineTransform order: (m00, m10, m01, m11, m02, m12)
        v = [float(x.strip()) for x in elem.text.split(',')]
        m00, m10, m01, m11, m02, m12 = v
        return (m02, m00, m01, m12, m10, m11)
    ulx = root.find('.//ULXMAP')
    uly = root.find('.//ULYMAP')
    xdim = root.find('.//XDIM')
    ydim = root.find('.//YDIM')
    if ulx is not None and uly is not None and xdim is not None and ydim is not None:
        return (float(ulx.text), float(xdim.text), 0.0, float(uly.text), 0.0, -float(ydim.text))  # type: ignore[arg-type]
    raise RuntimeError(f'Could not extract geotransform from {dim_path}')


def _utm_bbox_to_pixel_region(utm_bbox: tuple, geotransform: tuple) -> str:
    """Convert a UTM bounding box to a SNAP Subset region string 'x,y,width,height'.

    Args:
        utm_bbox: (x_min, y_min, x_max, y_max) in UTM metres.
        geotransform: GDAL-style 6-tuple from _read_geotransform.

    Returns:
        SNAP region string suitable for the Subset operator's -Pregion parameter.
    """
    x_min, y_min, x_max, y_max = utm_bbox
    orig_x, px_w, _, orig_y, _, px_h = geotransform  # px_h is negative

    col_start = int(round((x_min - orig_x) / px_w))
    row_start = int(round((y_max - orig_y) / px_h))  # px_h < 0 and y_max <= orig_y → positive
    width     = int(round((x_max - x_min) / px_w))
    height    = int(round((y_max - y_min) / abs(px_h)))

    return f'{col_start},{row_start},{width},{height}'


def _update_h5_corners(h5_path: Path, utm_bbox: tuple, epsg: int) -> None:
    """Overwrite Abstracted_Metadata corner coordinates with the actual TC-output WGS84 corners.

    SNAP preserves pre-TC SAR acquisition corners in the HDF5 metadata even after
    terrain correction and subsetting.  This function replaces them with the correct
    values derived from the UTM bounding box used to produce the tile.
    """
    x_min, y_min, x_max, y_max = utm_bbox
    t = pyproj.Transformer.from_crs(epsg, 4326, always_xy=True)

    bl_lon, bl_lat = t.transform(x_min, y_min)   # bottom-left  (last_near)
    tl_lon, tl_lat = t.transform(x_min, y_max)   # top-left     (first_near)
    tr_lon, tr_lat = t.transform(x_max, y_max)   # top-right    (first_far)
    br_lon, br_lat = t.transform(x_max, y_min)   # bottom-right (last_far)
    cx_lon, cx_lat = t.transform((x_min + x_max) / 2, (y_min + y_max) / 2)

    with h5py.File(h5_path, 'r+') as f:
        am = f['metadata/Abstracted_Metadata'].attrs
        am['last_near_long']  = bl_lon
        am['last_near_lat']   = bl_lat
        am['first_near_long'] = tl_lon
        am['first_near_lat']  = tl_lat
        am['first_far_long']  = tr_lon
        am['first_far_lat']   = tr_lat
        am['last_far_long']   = br_lon
        am['last_far_lat']    = br_lat
        am['centre_lon']      = cx_lon
        am['centre_lat']      = cx_lat


def _cut_single_tile(rect, product_path, cuts_dir, product_mode, gpt_memory, gpt_parallelism, gpt_timeout):
    """Cut one tile from the product and return a result dict."""
    tile_name = rect['BL']['properties']['name']
    tile_path = cuts_dir / f'{tile_name}.h5'
    try:
        if product_mode == 'NISAR':
            epsg = int(rect['BL']['properties']['epsg'].split(':')[1])
            x_min, y_min, x_max, y_max = grid_cell_utm_bbox(rect, epsg)
            reader = NISARReader(str(product_path))
            cutter = NISARCutter(reader)
            cutter.save_subset(cutter.cut_by_bbox(x_min, y_min, x_max, y_max, ['HH', 'HV'], apply_mask=False), tile_path, driver='H5')
        else:
            epsg = int(rect['BL']['properties']['epsg'].split(':')[1])
            utm_bbox = grid_cell_utm_bbox(rect, epsg)
            gt = _read_geotransform(product_path)
            region = _utm_bbox_to_pixel_region(utm_bbox, gt)
            tile_path = Path(subset(
                product_path, cuts_dir,
                output_name=tile_name, region=region,
                gpt_memory=gpt_memory, gpt_parallelism=gpt_parallelism, gpt_timeout=gpt_timeout,
            ))
            _update_h5_corners(tile_path, utm_bbox, epsg)
        return _validate_tile_result(tile_name, tile_path, 'tile cut')
    except Exception as exc:
        reason = f'{type(exc).__name__}: {exc}'
        # Tiles at the edge of the global grid can be outside the product footprint.
        # Treat these as expected skips, not hard failures.
        if 'does not intersect with product bounds' in reason:
            return {'tile': tile_name, 'status': 'skipped', 'reason': reason, 'output_path': str(tile_path)}
        return {'tile': tile_name, 'status': 'failed', 'reason': reason, 'output_path': str(tile_path)}


# ── Reporting ────────────────────────────────────────────────────────────────

def _write_cut_report(
    report_dir, product_name, product_path, intermediate_product,
    product_wkt, expected_tiles, actual_tiles, results,
    missing_tiles, extra_tiles,
):
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
    failed  = [r for r in results if r.get('status') == 'failed']
    skipped = [r for r in results if r.get('status') == 'skipped']
    ok      = [r for r in results if r.get('status') == 'success']
    status = 'SUCCESS' if not failed and not missing_tiles else 'FAILURE'

    lines = [
        'WorldSAR tile cutting report',
        f'Timestamp (UTC): {timestamp}',
        f'Product name: {product_name}',
        f'Product path: {product_path}',
        f'Intermediate product: {intermediate_product}',
        f'Cuts output dir: {report_dir}',
        f'Product WKT: {product_wkt}',
        '',
        f'Expected tiles: {len(expected_tiles)}',
        f'Actual tiles on disk: {len(actual_tiles)}',
        f'Successful tiles (this run): {len(ok)}',
        f'Skipped tiles (outside product bounds): {len(skipped)}',
        f'Failed tiles (this run): {len(failed)}',
        f'Missing tiles: {len(missing_tiles)}',
        f'Unexpected tiles: {len(extra_tiles)}',
    ]
    if skipped:
        lines.extend(['', 'Skipped tiles:'])
        for r in sorted(skipped, key=lambda r: r.get('tile', '')):
            lines.append(f"- {r.get('tile', 'UNKNOWN')}: {r.get('reason', '?')} | {r.get('output_path', '')}")
    if failed:
        lines.extend(['', 'Failed tiles:'])
        for r in sorted(failed, key=lambda r: r.get('tile', '')):
            lines.append(f"- {r.get('tile', 'UNKNOWN')}: {r.get('reason', '?')} | {r.get('output_path', '')}")
    if missing_tiles:
        lines.extend(['', 'Missing tiles (expected but not found on disk):'] + [f'- {t}' for t in missing_tiles])
    if extra_tiles:
        lines.extend(['', 'Unexpected tiles (found on disk but not expected):'] + [f'- {t}' for t in extra_tiles])

    report_path = report_dir / f'{product_name}_cuts_report_{status}.txt'
    report_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return report_path


def _expected_band_names_from_dim(dim_path):
    import xml.etree.ElementTree as ET

    root = ET.parse(dim_path).getroot()
    band_names = []
    for spectral_band in root.findall('./Image_Interpretation/Spectral_Band_Info'):
        band_name = (spectral_band.findtext('BAND_NAME') or '').strip()
        if band_name:
            band_names.append(band_name)
    if not band_names:
        raise RuntimeError(f'No band names found in {dim_path}')
    return sorted(band_names)


def _normalize_attr_value(value):
    if not isinstance(value, (str, bytes, bytearray)) and hasattr(value, 'item'):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, (bytes, bytearray)):
        return value.decode('utf-8', errors='replace')
    return value


def _is_blank_attr_value(value):
    value = _normalize_attr_value(value)
    return value is None or (isinstance(value, str) and not value.strip())


def _format_issue_map(issue_map):
    if not issue_map:
        return []
    lines = []
    for band_name in sorted(issue_map):
        issue = issue_map[band_name]
        parts = []
        if issue.get('missing_attrs'):
            parts.append(f"missing attrs={issue['missing_attrs']}")
        if issue.get('empty_attrs'):
            parts.append(f"empty attrs={issue['empty_attrs']}")
        if issue.get('invalid_shape'):
            parts.append(f"shape={issue.get('shape')}")
        lines.append(f'{band_name}: ' + '; '.join(parts))
    return lines


def _validate_h5_tile(tile_path, expected_bands, swath=None):
    return _shared_validate_h5_tile(tile_path, expected_bands, swath=swath)


def _validate_tile_group(cuts_dir, intermediate_product, swath=None, tiling_result=None):
    cuts_dir = Path(cuts_dir)
    expected_bands = _expected_band_names_from_dim(intermediate_product)
    tile_files = sorted(cuts_dir.glob('*.h5'))
    if not tile_files:
        swath_label = f' for swath {swath}' if swath else ''
        raise FileNotFoundError(f'No H5 tiles found in {cuts_dir}{swath_label}.')
    results = [_validate_h5_tile(tile_file, expected_bands, swath=swath) for tile_file in tile_files]
    structure_summary = enrich_validation_results_with_h5_structure(results)
    rows = [result['quickinfo_row'] for result in results]
    group = {
        'name': cuts_dir.name,
        'swath': swath,
        'cuts_dir': str(cuts_dir),
        'intermediate_product': str(intermediate_product),
        'expected_bands': expected_bands,
        'results': results,
        'rows': rows,
    }
    group.update(structure_summary)
    if tiling_result is not None:
        group.update({
            'expected_tile_count': len(tiling_result['expected_tiles']),
            'actual_tile_count': len(tiling_result['actual_tiles']),
            'cut_failed': tiling_result['cut_failed'],
            'cut_report_path': str(tiling_result['report_path']),
        })
    return group


def _chunked(lines, size):
    for index in range(0, len(lines), size):
        yield lines[index:index + size]


def _write_pdf_text_page(pdf, title, lines):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11.69, 8.27))
    fig.text(0.03, 0.97, title, va='top', ha='left', fontsize=14, fontweight='bold', family='monospace')
    fig.text(0.03, 0.93, '\n'.join(lines), va='top', ha='left', fontsize=8, family='monospace')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def _write_h5_validation_report_pdf(report_path, product_name, validation_groups):
    return _shared_write_h5_validation_report_pdf(report_path, product_name, validation_groups)


def create_tile_database_from_rows(rows, output_db_folder, output_name):
    """Create a parquet database of tile metadata from pre-validated rows."""
    import pandas as pd

    if not rows:
        raise ValueError('No validated tile metadata rows available.')

    db = pd.DataFrame(rows)
    out = Path(output_db_folder)
    out.mkdir(parents=True, exist_ok=True)
    output_file = out / f'{output_name}_core_metadata.parquet'
    db.to_parquet(output_file, index=False)
    print(f'Core metadata saved to {output_file}')
    return db


# ── Database ─────────────────────────────────────────────────────────────────

def create_tile_database(input_folder, output_db_folder):
    """Create a parquet database of tile metadata from h5 files."""
    import pandas as pd

    from sarpyx.utils.io import read_h5

    tile_path = Path(input_folder)
    h5_tiles = list(tile_path.rglob('*.h5'))
    if not h5_tiles:
        raise FileNotFoundError(f'No .h5 tiles found in {tile_path}')
    print(f'Found {len(h5_tiles)} h5 files in {input_folder}')

    db = pd.DataFrame()
    for idx, tile_file in enumerate(h5_tiles):
        print(f'Processing tile {idx + 1}/{len(h5_tiles)}: {tile_file.name}')
        _data, metadata = read_h5(tile_file)
        row = pd.Series(metadata['quickinfo'])
        row['first_line_time'] = normalize_sar_timestamp(row.get('first_line_time'))
        row['ID'] = tile_file.stem
        db = pd.concat([db, pd.DataFrame([row])], ignore_index=True)

    out = Path(output_db_folder)
    out.mkdir(parents=True, exist_ok=True)
    output_file = out / f'{tile_path.name}_core_metadata.parquet'
    db.to_parquet(output_file, index=False)
    print(f'Core metadata saved to {output_file}')
    return db


# ── Orchestration (called by main) ──────────────────────────────────────────

def _apply_runtime_overrides(args):
    global GPT_PATH, GRID_PATH, DB_DIR, SNAP_USERDIR
    if args.gpt_path:
        GPT_PATH = args.gpt_path
    if args.grid_path:
        GRID_PATH = args.grid_path
    if args.db_dir:
        DB_DIR = args.db_dir
    if args.snap_userdir:
        SNAP_USERDIR = args.snap_userdir
        os.environ['SNAP_USERDIR'] = SNAP_USERDIR


def _validate_runtime_args(args):
    if args.gpt_parallelism is not None and args.gpt_parallelism <= 0:
        raise ValueError(f'--gpt-parallelism must be > 0, got {args.gpt_parallelism}')
    if args.gpt_timeout is not None and args.gpt_timeout < 0:
        raise ValueError(f'--gpt-timeout must be >= 0, got {args.gpt_timeout}')
    if len(args.zarr_chunk_size) != 2 or any(size <= 0 for size in args.zarr_chunk_size):
        raise ValueError(f'--zarr-chunk-size must contain two positive integers, got {args.zarr_chunk_size}')


def _resolve_db_dir(cuts_outdir=None):
    global DB_DIR

    if DB_DIR:
        db_dir = _expand_path(DB_DIR)
    else:
        if cuts_outdir is None:
            raise ValueError('db_dir not provided and no default output root is available.')
        db_dir = _expand_path(Path(cuts_outdir) / '_db')
        DB_DIR = str(db_dir)
        print(f'DB_DIR not configured; defaulting to {db_dir}')

    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir


def _ensure_grid_file(grid_path, base_path):
    if grid_path.exists():
        return grid_path
    grid_dir = base_path / 'grid'
    grid_dir.mkdir(parents=True, exist_ok=True)
    print(f'Grid file not found at {grid_path}. Generating grid_10km.geojson in {grid_dir}.')
    subprocess.run([sys.executable, '-m', 'sarpyx.utils.grid'], cwd=grid_dir, check=True)
    generated = grid_dir / 'grid_10km.geojson'
    if not generated.exists():
        raise FileNotFoundError(f'Grid generation completed, but {generated} was not created.')
    return generated


def _find_existing_intermediates(output_dir: Path, product_mode: str) -> dict | Path:
    """Locate existing BEAM-DIMAP intermediate products for --skip-preprocessing."""
    if product_mode == 'S1TOPS':
        result = {}
        for swath in ('IW1', 'IW2', 'IW3'):
            dims = sorted((output_dir / swath).glob('*.dim'), key=lambda p: p.stat().st_mtime, reverse=True)
            if not dims:
                raise FileNotFoundError(f'No .dim intermediate found in {output_dir / swath}')
            if len(dims) > 1:
                print(f'[WARN] Multiple .dim files in {output_dir / swath}, using most recent: {dims[0].name}')
            result[swath] = dims[0]
            print(f'Reusing intermediate {swath}: {dims[0]}')
        return result
    dims = sorted(output_dir.glob('*.dim'), key=lambda p: p.stat().st_mtime, reverse=True)
    if not dims:
        raise FileNotFoundError(f'No .dim intermediate found in {output_dir}')
    print(f'Reusing intermediate: {dims[0]}')
    return dims[0]


def _run_preprocessing(product_path, output_dir, product_mode, orbit_type, orbit_continue_on_fail, gpt_memory, gpt_parallelism, gpt_timeout, skip=False):
    if not prepro or skip:
        if skip:
            return _find_existing_intermediates(output_dir, product_mode)
        return product_path
    result = ROUTER[product_mode](
        product_path, output_dir,
        orbit_type=orbit_type, orbit_continue_on_fail=orbit_continue_on_fail,
        gpt_memory=gpt_memory, gpt_parallelism=gpt_parallelism, gpt_timeout=gpt_timeout,
    )
    # TOPS returns {IW1: path, IW2: path, IW3: path}; others return a single path.
    if isinstance(result, dict):
        for swath, path in result.items():
            print(f'Intermediate {swath}: {path}')
            if path is None:
                raise RuntimeError(f'Intermediate product for {swath} was not returned.')
            if not Path(path).exists():
                raise FileNotFoundError(f'Intermediate product {path} ({swath}) does not exist.')
        return {sw: Path(p) for sw, p in result.items()}
    print(f'Intermediate processed product located at: {result}')
    if result is None:
        raise RuntimeError(f'No intermediate product was returned for mode {product_mode}.')
    if not Path(result).exists():
        raise FileNotFoundError(f'Intermediate product {result} does not exist.')
    return Path(result)


def _run_tiling(product_wkt, grid_geoj_path, source_product, intermediate_product, cuts_outdir, product_mode, gpt_memory, gpt_parallelism, gpt_timeout):
    if grid_geoj_path is None or not Path(grid_geoj_path).exists():
        raise FileNotFoundError(f'grid_10km.geojson does not exist: {grid_geoj_path}')

    contained = check_points_in_polygon(product_wkt, geojson_path=grid_geoj_path)
    if not contained:
        raise ValueError('No grid points contained; check WKT and grid CRS alignment.')

    rectangles = rectanglify(contained)
    if not rectangles:
        raise ValueError('No rectangles formed; check WKT coverage and grid alignment.')

    name = extract_product_id(intermediate_product.as_posix()) if product_mode != 'NISAR' else intermediate_product.stem
    if name is None:
        raise ValueError(f'Could not extract product id from: {intermediate_product}')

    cuts_dir = cuts_outdir / name
    cuts_dir.mkdir(parents=True, exist_ok=True)

    results = [
        _cut_single_tile(rect, intermediate_product, cuts_dir, product_mode, gpt_memory, gpt_parallelism, gpt_timeout)
        for rect in rectangles
    ]

    expected_tiles = sorted({rect['BL']['properties']['name'] for rect in rectangles})
    skipped_tiles  = {r['tile'] for r in results if r.get('status') == 'skipped'}
    required_tiles = sorted(set(expected_tiles) - skipped_tiles)
    actual_tiles   = sorted({p.stem for p in cuts_dir.glob('*.h5')})
    missing_tiles  = sorted(set(required_tiles) - set(actual_tiles))
    extra_tiles    = sorted(set(actual_tiles) - set(expected_tiles))

    report_path = _write_cut_report(
        cuts_dir, name, source_product, intermediate_product, product_wkt,
        expected_tiles, actual_tiles, results, missing_tiles, extra_tiles,
    )

    for res in results:
        if res.get('status') == 'success':
            print(f"Tile saved: {res.get('output_path', '')}")
        elif res.get('status') == 'skipped':
            print(f"Skipped tile {res.get('tile', 'UNKNOWN')}: {res.get('reason', '?')}")
        else:
            print(f"Failed tile {res.get('tile', 'UNKNOWN')}: {res.get('reason', '?')}")

    cut_failed = bool(missing_tiles or any(r.get('status') != 'success' for r in results))
    return {
        'name': name,
        'cuts_dir': cuts_dir,
        'report_path': report_path,
        'cut_failed': cut_failed,
        'expected_tiles': expected_tiles,
        'actual_tiles': actual_tiles,
        'missing_tiles': missing_tiles,
        'extra_tiles': extra_tiles,
    }


def _resolve_tiling_wkt(product_wkt, source_product, intermediate_product, product_mode, swath=None):
    """Prefer the processed product footprint when tiling a single TOPS swath."""
    if product_mode == 'S1TOPS' and swath:
        derived_wkt = sentinel1_swath_wkt_extractor_safe(source_product, swath, display_results=False, verbose=False)
        if derived_wkt:
            return derived_wkt
    return product_wkt


def _run_tops_swath_tiling(product_wkt, grid_geoj_path, product_path, intermediate, cuts_outdir, product_mode, gpt_kwargs):
    swath_tiling_errors = {}
    swath_wkts = {}
    validation_groups = []
    report_name = None

    for swath, swath_product in intermediate.items():
        name = swath_product.stem
        swath_wkt = _resolve_tiling_wkt(product_wkt, product_path, swath_product, product_mode, swath=swath)
        swath_wkts[swath] = swath_wkt

        if tiling:
            tiling_result = _run_tiling(
                swath_wkt, grid_geoj_path, product_path,
                swath_product, cuts_outdir / swath, product_mode, **gpt_kwargs,
            )
            name = tiling_result['name']
            report_name = report_name or name
            if tiling_result['cut_failed']:
                swath_tiling_errors[swath] = RuntimeError(f"Tile cutting failed; report: {tiling_result['report_path']}")
            validation_group = _validate_tile_group(
                tiling_result['cuts_dir'],
                swath_product,
                swath=swath,
                tiling_result=tiling_result,
            )
            validation_groups.append(validation_group)
            if validation_group['rows']:
                _run_db_indexing(validation_group['rows'], name, swath=swath)
        else:
            validation_group = _validate_tile_group(cuts_outdir / swath / name, swath_product, swath=swath)
            validation_groups.append(validation_group)
            if validation_group['rows']:
                _run_db_indexing(validation_group['rows'], name, swath=swath)

    if validation_groups:
        pdf_path = cuts_outdir / f'{report_name or validation_groups[0]["name"]}_h5_validation_report.pdf'
        _write_h5_validation_report_pdf(pdf_path, report_name or validation_groups[0]['name'], validation_groups)

    if swath_tiling_errors:
        _verify_tops_tile_coverage(
            product_wkt, grid_geoj_path, cuts_outdir, intermediate, swath_wkts=swath_wkts,
        )
    if any(result['status'] != 'success' for group in validation_groups for result in group['results']):
        raise RuntimeError(f'H5 validation failed; report: {pdf_path}')


def _verify_tops_tile_coverage(product_wkt, grid_geoj_path, cuts_outdir, swath_products, swath_wkts=None):
    """After TOPS tiling, verify that expected tiles exist across all swaths combined.

    In TOPS mode each subswath covers only part of the full product footprint,
    so per-swath tile failures are expected.  This function checks aggregate
    coverage and raises only if *no* tile could be produced at all.
    """
    contained = check_points_in_polygon(product_wkt, geojson_path=grid_geoj_path)
    rectangles = rectanglify(contained)
    if not rectangles:
        return

    expected_tiles = {rect['BL']['properties']['name'] for rect in rectangles}

    produced_tiles = set()
    for swath in swath_products:
        swath_dir = cuts_outdir / swath
        for h5_file in swath_dir.rglob('*.h5'):
            produced_tiles.add(h5_file.stem)

    missing = sorted(expected_tiles - produced_tiles)
    covered = expected_tiles - set(missing)

    swath_expected_tiles = set()
    if swath_wkts:
        for swath, swath_wkt in swath_wkts.items():
            contained_swath = check_points_in_polygon(swath_wkt, geojson_path=grid_geoj_path)
            swath_rectangles = rectanglify(contained_swath)
            swath_expected_tiles.update(rect['BL']['properties']['name'] for rect in swath_rectangles)

    print(f'\n[TOPS Aggregate Coverage]')
    print(f'  Expected tiles (from full product WKT): {len(expected_tiles)}')
    if swath_expected_tiles:
        print(f'  Expected tiles (union of swath WKTs):  {len(swath_expected_tiles)}')
    print(f'  Produced tiles (across all swaths):     {len(covered)}')
    print(f'  Missing tiles:                          {len(missing)}')

    if missing:
        print(f'  Missing tile names: {missing}')
        print(f'  Note: tiles at subswath boundaries may legitimately fail to be subset from any single swath.')
    if not produced_tiles:
        raise RuntimeError('TOPS tiling produced zero tiles across all swaths.')


def _run_db_indexing(validation_rows, name, swath=None):
    if not db_indexing:
        return
    db_dir = _resolve_db_dir()
    # Backward-compatible behavior: callers may pass either precomputed rows
    # (new path) or a cuts directory path (legacy path).
    if isinstance(validation_rows, (str, Path)):
        db = create_tile_database((Path(validation_rows) / name).as_posix(), db_dir)
    else:
        output_name = f'{swath}_{name}' if swath else name
        db = create_tile_database_from_rows(validation_rows, db_dir, output_name)
    if db.empty:
        raise RuntimeError('Database creation failed, resulting DataFrame is empty.')
    print('Database created successfully.')


def _run_h5_to_zarr_only(product_path, output_path, chunk_size, overwrite):
    converted = convert_tile_h5_to_zarr(
        input_path=product_path,
        output_path=output_path,
        chunk_size=tuple(chunk_size),
        overwrite=overwrite,
    )
    summary = {
        'input': str(Path(product_path).expanduser().absolute()),
        'output': str(converted),
        'chunk_size': list(chunk_size),
        'zarr_format': 3,
    }
    print(json.dumps(summary, indent=2))
    return converted


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = create_parser().parse_args()
    _validate_runtime_args(args)
    _apply_runtime_overrides(args)

    product_path = _ensure_existing_path(args.product_path, 'Input product')
    if args.h5_to_zarr_only:
        _run_h5_to_zarr_only(
            product_path=product_path,
            output_path=args.output_dir,
            chunk_size=args.zarr_chunk_size,
            overwrite=args.overwrite_zarr,
        )
        sys.exit(0)

    if not args.output_dir:
        raise ValueError('--output is required unless --h5-to-zarr-only is set.')
    output_dir = _expand_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if CUTS_OUTDIR is None:
        print('Warning: cuts_outdir env var not found. Set cuts_outdir to avoid passing --cuts-outdir each run.')
    cuts_outdir_value = args.cuts_outdir or CUTS_OUTDIR
    if not cuts_outdir_value:
        raise ValueError('cuts_outdir not provided. Set cuts_outdir env var or pass --cuts-outdir.')
    cuts_outdir = _expand_path(cuts_outdir_value)
    cuts_outdir.mkdir(parents=True, exist_ok=True)
    if db_indexing:
        _resolve_db_dir(cuts_outdir)

    base_path = _expand_path(BASE_PATH)
    grid_geoj_path = _expand_path(GRID_PATH) if GRID_PATH else base_path / 'grid' / 'grid_10km.geojson'
    grid_geoj_path = _ensure_grid_file(grid_geoj_path, base_path)

    product_mode = infer_product_mode(product_path)
    print(f'Inferred product mode: {product_mode}')

    if args.product_wkt is not None:
        product_wkt = args.product_wkt.strip()
        if not product_wkt:
            raise ValueError('--product-wkt cannot be blank.')
    elif product_mode in {'S1TOPS', 'S1STRIP'}:
        product_wkt = sentinel1_wkt_extractor_manifest(product_path, display_results=False)
        if product_wkt is None:
            product_wkt = sentinel1_wkt_extractor_cdse(product_path.name, display_results=False)
        if product_wkt is None:
            raise ValueError(f'Failed to extract Sentinel-1 WKT for product: {product_path}')
    elif product_mode == 'NISAR':
        product_wkt = nisar_wkt_extractor(product_path)
    else:
        raise ValueError('No --product-wkt provided and automatic WKT extraction is only available for Sentinel-1 and NISAR.')

    gpt_kwargs = dict(gpt_memory=args.gpt_memory, gpt_parallelism=args.gpt_parallelism, gpt_timeout=args.gpt_timeout)

    intermediate = _run_preprocessing(
        product_path, output_dir, product_mode,
        orbit_type=args.orbit_type,
        orbit_continue_on_fail=args.orbit_continue_on_fail,
        skip=args.skip_preprocessing,
        **gpt_kwargs,
    )

    if isinstance(intermediate, dict):
        _run_tops_swath_tiling(
            product_wkt, grid_geoj_path, product_path,
            intermediate, cuts_outdir, product_mode, gpt_kwargs,
        )
    else:
        name = intermediate.stem
        if tiling:
            tiling_result = _run_tiling(
                product_wkt, grid_geoj_path, product_path,
                intermediate, cuts_outdir, product_mode, **gpt_kwargs,
            )
            name = tiling_result['name']
            validation_group = _validate_tile_group(
                tiling_result['cuts_dir'],
                intermediate,
                tiling_result=tiling_result,
            )
            pdf_path = cuts_outdir / f'{name}_h5_validation_report.pdf'
            _write_h5_validation_report_pdf(pdf_path, name, [validation_group])
            if validation_group['rows']:
                _run_db_indexing(validation_group['rows'], name)
            if tiling_result['cut_failed']:
                raise RuntimeError(f"Tile cutting failed; report: {tiling_result['report_path']}")
            if any(result['status'] != 'success' for result in validation_group['results']):
                raise RuntimeError(f'H5 validation failed; report: {pdf_path}')
        else:
            validation_group = _validate_tile_group(cuts_outdir / name, intermediate)
            pdf_path = cuts_outdir / f'{name}_h5_validation_report.pdf'
            _write_h5_validation_report_pdf(pdf_path, name, [validation_group])
            if validation_group['rows']:
                _run_db_indexing(validation_group['rows'], name)
            if any(result['status'] != 'success' for result in validation_group['results']):
                raise RuntimeError(f'H5 validation failed; report: {pdf_path}')

    sys.exit(0)


if __name__ == '__main__':
    main()
    