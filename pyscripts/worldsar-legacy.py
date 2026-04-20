"""WorldSAR CLI pipelines for SAR product preprocessing and tiling.

TODO: metadata reorganization.
TODO: SUBAPERTURE PROCESSING for all missions.
TODO: PolSAR support.
TODO: InSAR support.
"""

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime
from functools import partial
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from sarpyx.snapflow.engine import GPT
from sarpyx.utils.geos import check_points_in_polygon, rectangle_to_wkt, rectanglify
from sarpyx.utils.io import read_h5
from sarpyx.utils.nisar_utils import NISARCutter, NISARReader
from sarpyx.utils.wkt_utils import sentinel1_wkt_extractor_cdse, sentinel1_wkt_extractor_manifest

from merge_iq_into_pdec import merge_iq_into_pdec

load_dotenv()


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


# ══════════════════════════════════════════════════════════════════════════════
#  Pipelines  –  one per mission family, dispatched via ROUTER[mode]
# ══════════════════════════════════════════════════════════════════════════════

def _sentinel_post_chain(op, product_path):
    """Calibration → DerampDemod → Deburst → PolDecomp → TC  (shared by each swath)."""
    op.ApplyOrbitFile()
    op.Calibration(output_complex=True)
    op.TopsarDerampDemod()
    fp_deb = op.Deburst()

    """
    old:
    
    op.do_subaps(
        safe_path=product_path,
        dim_path=op.prod_path,
        n_decompositions=[2],
        byte_order=1,
        VERBOSE=False,
    )
    """

    op.do_subaps(
        dim_path=op.prod_path,
        safe_path=product_path,
        n_decompositions=[2],
        byte_order=1,
        VERBOSE=False,
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

    op.polarimetric_decomposition(decomposition="H-Alpha Dual Pol Decomposition", window_size=5)
    merge_iq_into_pdec(
        src_dim=fp_deb,
        pdec_dim=op.prod_path,
        is_tops=True,
        overwrite_copied_files=False,
        backup=False,
    )
    op.TerrainCorrection(map_projection='AUTO:42001', pixel_spacing_in_meter=10.0)
    return op.prod_path


def pipeline_sentinel(
    product_path, output_dir, is_TOPS=False,
    gpt_memory=None, gpt_parallelism=None, gpt_timeout=None, **_,
):
    """Sentinel-1 pipeline.

    TOPS mode:  orbit → split IW1/IW2/IW3 → (cal → deramp → deburst → poldecomp → TC) per swath.
    STRIP mode: orbit → cal → poldecomp → TC.
    """
    gpt_kw = dict(gpt_memory=gpt_memory, gpt_parallelism=gpt_parallelism, gpt_timeout=gpt_timeout)
    op = _create_gpt_operator(product_path, output_dir, 'BEAM-DIMAP', **gpt_kw)

    if is_TOPS:
        results = {}
        for swath in ('IW1', 'IW2', 'IW3'):
            sw_op = _create_gpt_operator(Path(op.prod_path), output_dir / swath, 'BEAM-DIMAP', **gpt_kw)
            sw_op.TopsarSplit(subswath=swath) # SPLIT
            results[swath] = _sentinel_post_chain(op=sw_op, product_path=product_path)
        return results                    # {IW1: path, IW2: path, IW3: path}
    
    # STRIP mode – no split / deburst needed
    op.ApplyOrbitFile()
    orbit_product = op.prod_path          # product after orbit correction
    fp_cal = op.Calibration(output_complex=True)

    op.do_subaps(
        safe_path=product_path,
        dim_path=op.prod_path,
        n_decompositions=[3],
        byte_order=1,
        VERBOSE=False,
    )
    op.polarimetric_decomposition(decomposition="H-Alpha Dual Pol Decomposition", window_size=5)
    
    merge_iq_into_pdec(
        src_dim=fp_cal,
        pdec_dim=op.prod_path,
        is_tops=False,
        overwrite_copied_files=False,
        backup=False,
    )

    op.TerrainCorrection(map_projection='AUTO:42001', pixel_spacing_in_meter=10.0)
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
    assert product_path.suffix == '.h5', 'NISAR products must be in .h5 format.'
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
    (['--input', '-i'],                dict(dest='product_path', type=str, required=True, help='Path to the input SAR product.')),
    (['--output', '-o'],               dict(dest='output_dir', type=str, required=True, help='Directory to save the processed output.')),
    (['--cuts-outdir', '--cuts_outdir'], dict(dest='cuts_outdir', type=str, default=None, help='Where to store the tiles after extraction.')),
    (['--product-wkt', '--product_wkt'], dict(dest='product_wkt', type=str, default=None, help='WKT string defining the product region of interest.')),
    (['--gpt-path'],                   dict(dest='gpt_path', type=str, default=None, help='Override GPT executable path.')),
    (['--grid-path'],                  dict(dest='grid_path', type=str, default=None, help='Override grid GeoJSON path.')),
    (['--db-dir'],                     dict(dest='db_dir', type=str, default=None, help='Override database output directory.')),
    (['--gpt-memory'],                 dict(dest='gpt_memory', type=str, default='16G', help='GPT Java heap (e.g., 24G).')),
    (['--gpt-parallelism'],            dict(dest='gpt_parallelism', type=int, default=10, help='GPT parallelism (number of tiles).')),
    (['--gpt-timeout'],                dict(dest='gpt_timeout', type=int, default=None, help='GPT timeout in seconds.')),
    (['--snap-userdir'],               dict(dest='snap_userdir', type=str, default=None, help='Override SNAP user directory.')),
    (['--orbit-type'],                 dict(dest='orbit_type', type=str, default='Sentinel Precise (Auto Download)', help='SNAP Apply-Orbit-File orbitType.')),
    (['--orbit-continue-on-fail'],     dict(dest='orbit_continue_on_fail', action='store_true', help='Continue if orbit file cannot be applied.')),
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
        raise RuntimeError(f'GPT {op_name} failed: {op.last_error_summary()}')
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
    assert geo_region is not None, 'Geo region WKT string must be provided.'
    return _run_gpt_op(product_path, output_dir, 'GDAL-GTiff-WRITER', 'Write',
                       gpt_memory=gpt_memory, gpt_parallelism=gpt_parallelism, gpt_timeout=gpt_timeout)


def subset(product_path, output_dir, geo_region=None, output_name=None, gpt_memory=None, gpt_parallelism=None, gpt_timeout=None):
    assert geo_region is not None, 'Geo region WKT string must be provided.'
    return _run_gpt_op(
        product_path, output_dir, 'HDF5', 'Subset',
        gpt_memory=gpt_memory, gpt_parallelism=gpt_parallelism, gpt_timeout=gpt_timeout,
        copy_metadata=True, output_name=output_name, geo_region=geo_region,
    )


def swath_splitter(swath, product_path, output_dir, gpt_memory=None, gpt_parallelism=None, gpt_timeout=None, **extra):
    """Split a Sentinel-1 TOPS product by subswath (1, 2, or 3)."""
    return _run_gpt_op(
        product_path, output_dir, 'BEAM-DIMAP', 'topsar_split',
        gpt_memory=gpt_memory, gpt_parallelism=gpt_parallelism, gpt_timeout=gpt_timeout,
        subswath=f'IW{swath}', **extra,
    )


def _cut_single_tile(rect, product_path, cuts_dir, product_mode, gpt_memory, gpt_parallelism, gpt_timeout):
    """Cut one tile from the product and return a result dict."""
    geo_region = rectangle_to_wkt(rect)
    tile_name = rect['BL']['properties']['name']
    tile_path = cuts_dir / f'{tile_name}.h5'
    try:
        if product_mode == 'NISAR':
            reader = NISARReader(str(product_path))
            cutter = NISARCutter(reader)
            cutter.save_subset(cutter.cut_by_wkt(geo_region, 'HH', apply_mask=False), tile_path, driver='H5')
        else:
            tile_path = Path(subset(
                product_path, cuts_dir,
                output_name=tile_name, geo_region=geo_region,
                gpt_memory=gpt_memory, gpt_parallelism=gpt_parallelism, gpt_timeout=gpt_timeout,
            ))
        return _validate_tile_result(tile_name, tile_path, 'tile cut')
    except Exception as exc:
        return {'tile': tile_name, 'status': 'failed', 'reason': f'{type(exc).__name__}: {exc}', 'output_path': str(tile_path)}


# ── Reporting ────────────────────────────────────────────────────────────────

def _write_cut_report(
    report_dir, product_name, product_path, intermediate_product,
    product_wkt, expected_tiles, actual_tiles, results,
    missing_tiles, extra_tiles,
):
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
    failed = [r for r in results if r.get('status') != 'success']
    ok     = [r for r in results if r.get('status') == 'success']
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
        f'Failed tiles (this run): {len(failed)}',
        f'Missing tiles: {len(missing_tiles)}',
        f'Unexpected tiles: {len(extra_tiles)}',
    ]
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


# ── Database ─────────────────────────────────────────────────────────────────

def create_tile_database(input_folder, output_db_folder):
    """Create a parquet database of tile metadata from h5 files."""
    tile_path = Path(input_folder)
    h5_tiles = list(tile_path.rglob('*.h5'))
    print(f'Found {len(h5_tiles)} h5 files in {input_folder}')

    db = pd.DataFrame()
    for idx, tile_file in enumerate(h5_tiles):
        print(f'Processing tile {idx + 1}/{len(h5_tiles)}: {tile_file.name}')
        _data, metadata = read_h5(tile_file)
        row = pd.Series(metadata['quickinfo'])
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


def _run_preprocessing(product_path, output_dir, product_mode, orbit_type, orbit_continue_on_fail, gpt_memory, gpt_parallelism, gpt_timeout):
    if not prepro:
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
            assert Path(path).exists(), f'Intermediate product {path} ({swath}) does not exist.'
        return {sw: Path(p) for sw, p in result.items()}
    print(f'Intermediate processed product located at: {result}')
    assert Path(result).exists(), f'Intermediate product {result} does not exist.'
    return Path(result)


def _run_tiling(product_wkt, grid_geoj_path, source_product, intermediate_product, cuts_outdir, product_mode, gpt_memory, gpt_parallelism, gpt_timeout):
    assert grid_geoj_path is not None and grid_geoj_path.exists(), 'grid_10km.geojson does not exist.'

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
    actual_tiles   = sorted({p.stem for p in cuts_dir.glob('*.h5')})
    missing_tiles  = sorted(set(expected_tiles) - set(actual_tiles))
    extra_tiles    = sorted(set(actual_tiles) - set(expected_tiles))

    report_path = _write_cut_report(
        cuts_dir, name, source_product, intermediate_product, product_wkt,
        expected_tiles, actual_tiles, results, missing_tiles, extra_tiles,
    )

    for res in results:
        if res.get('status') == 'success':
            print(f"Tile saved: {res.get('output_path', '')}")
        else:
            print(f"Failed tile {res.get('tile', 'UNKNOWN')}: {res.get('reason', '?')}")

    if missing_tiles or any(r.get('status') != 'success' for r in results):
        raise RuntimeError(f'Tile cutting failed; report: {report_path}')
    return name


def _verify_tops_tile_coverage(product_wkt, grid_geoj_path, cuts_outdir, swath_products):
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

    print(f'\n[TOPS Aggregate Coverage]')
    print(f'  Expected tiles (from full product WKT): {len(expected_tiles)}')
    print(f'  Produced tiles (across all swaths):     {len(covered)}')
    print(f'  Missing tiles:                          {len(missing)}')

    if missing:
        print(f'  Missing tile names: {missing}')
        print(f'  Note: tiles at subswath boundaries may legitimately fail to be subset from any single swath.')
    if not produced_tiles:
        raise RuntimeError('TOPS tiling produced zero tiles across all swaths.')


def _run_db_indexing(cuts_outdir, name):
    if not db_indexing:
        return
    db = create_tile_database((cuts_outdir / name).as_posix(), DB_DIR)  # type: ignore[arg-type]
    assert not db.empty, 'Database creation failed, resulting DataFrame is empty.'
    print('Database created successfully.')


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = create_parser().parse_args()
    _apply_runtime_overrides(args)

    product_path = Path(args.product_path)
    output_dir   = Path(args.output_dir)

    if CUTS_OUTDIR is None:
        print('Warning: cuts_outdir env var not found. Set cuts_outdir to avoid passing --cuts-outdir each run.')
    cuts_outdir_value = args.cuts_outdir or CUTS_OUTDIR
    if not cuts_outdir_value:
        raise ValueError('cuts_outdir not provided. Set cuts_outdir env var or pass --cuts-outdir.')
    cuts_outdir = Path(cuts_outdir_value)

    base_path = Path(BASE_PATH)
    grid_geoj_path = Path(GRID_PATH) if GRID_PATH else base_path / 'grid' / 'grid_10km.geojson'
    grid_geoj_path = _ensure_grid_file(grid_geoj_path, base_path)

    product_mode = infer_product_mode(product_path)
    print(f'Inferred product mode: {product_mode}')

    if args.product_wkt is not None:
        product_wkt = args.product_wkt
    elif product_mode in {'S1TOPS', 'S1STRIP'}:
        product_wkt = sentinel1_wkt_extractor_manifest(product_path, display_results=False)
        if product_wkt is None:
            product_wkt = sentinel1_wkt_extractor_cdse(product_path.name, display_results=False)
        if product_wkt is None:
            raise ValueError(f'Failed to extract Sentinel-1 WKT for product: {product_path}')
    else:
        raise ValueError('No --product-wkt provided and automatic WKT extraction is only available for Sentinel-1.')

    gpt_kwargs = dict(gpt_memory=args.gpt_memory, gpt_parallelism=args.gpt_parallelism, gpt_timeout=args.gpt_timeout)

    intermediate = _run_preprocessing(
        product_path, output_dir, product_mode,
        orbit_type=args.orbit_type,
        orbit_continue_on_fail=args.orbit_continue_on_fail, **gpt_kwargs,
    )

    # TOPS returns a dict of {swath: path} — tile each swath separately.
    # Per-swath failures are expected because the full-scene WKT covers all
    # three subswaths but each processed product covers only its own subswath.
    if isinstance(intermediate, dict):
        swath_tiling_errors = {}
        for swath, swath_product in intermediate.items():
            name = swath_product.stem
            if tiling:
                try:
                    name = _run_tiling(
                        product_wkt, grid_geoj_path, product_path,
                        swath_product, cuts_outdir / swath, product_mode, **gpt_kwargs,
                    )
                except RuntimeError as exc:
                    swath_tiling_errors[swath] = exc
                    name = extract_product_id(swath_product.as_posix()) or swath_product.stem
                    print(f'[WARN] Tiling for {swath} had partial failures (expected in TOPS mode): {exc}')
            try:
                _run_db_indexing(cuts_outdir / swath, name)
            except Exception as exc:
                print(f'[WARN] DB indexing for {swath} skipped: {exc}')

        if swath_tiling_errors:
            _verify_tops_tile_coverage(
                product_wkt, grid_geoj_path, cuts_outdir, intermediate,
            )
    else:
        name = intermediate.stem
        if tiling:
            name = _run_tiling(
                product_wkt, grid_geoj_path, product_path,
                intermediate, cuts_outdir, product_mode, **gpt_kwargs,
            )
        _run_db_indexing(cuts_outdir, name)

    sys.exit(0)


if __name__ == '__main__':
    main()
    