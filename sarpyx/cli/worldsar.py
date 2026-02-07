"""WorldSAR CLI pipelines for SAR product preprocessing and tiling.

TODO: metadate reorganization.
TODO: SUBAPERTURE PROCESSING for all missions.
TODO: PolSAR support.
TODO: InSAR support.
"""

import argparse
import asyncio
import hashlib
import os
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from urllib import request

import pandas as pd
from dotenv import load_dotenv

from sarpyx.snapflow.engine import GPT
from sarpyx.utils.geos import check_points_in_polygon, rectangle_to_wkt, rectanglify
from sarpyx.utils.io import read_h5
from sarpyx.utils.nisar_utils import NISARCutter, NISARReader
from sarpyx.utils.wkt_utils import sentinel1_wkt_extractor_cdse, sentinel1_wkt_extractor_manifest

# Load environment variables from .env file
load_dotenv()

# Read paths from environment variables (accept legacy/uppercase names)
GPT_PATH = os.getenv('gpt_path') or os.getenv('GPT_PATH')
GRID_PATH = os.getenv('grid_path') or os.getenv('GRID_PATH')
DB_DIR = os.getenv('db_dir') or os.getenv('DB_DIR')
CUTS_OUTDIR = os.getenv('cuts_outdir') or os.getenv('OUTPUT_CUTS_DIR')
PROJECT_ROOT = Path(__file__).resolve().parents[3]
BASE_PATH = os.getenv('base_path') or os.getenv('BASE_PATH') or str(PROJECT_ROOT)
SNAP_USERDIR = os.getenv('SNAP_USERDIR') or os.getenv('snap_userdir') or str(PROJECT_ROOT / '.snap')
os.environ.setdefault('SNAP_USERDIR', SNAP_USERDIR)
ORBIT_BASE_URL = os.getenv('orbit_base_url') or os.getenv('ORBIT_BASE_URL') or 'https://step.esa.int/auxdata/orbits/Sentinel-1'

# Processing settings
prepro = True
tiling = True
db_indexing = False
MAX_CUT_WORKERS = 5


# ================================================================================================================================ Parser
def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(description='Process SAR data using SNAP GPT and sarpyx pipelines.')
    parser.add_argument(
        '--input',
        '-i',
        dest='product_path',
        type=str,
        required=True,
        help='Path to the input SAR product.',
    )
    parser.add_argument(
        '--output',
        '-o',
        dest='output_dir',
        type=str,
        required=True,
        help='Directory to save the processed output.',
    )
    parser.add_argument(
        '--cuts-outdir',
        '--cuts_outdir',
        dest='cuts_outdir',
        type=str,
        required=False,
        default=None,
        help='Where to store the tiles after extraction (default: cuts_outdir env var).',
    )
    parser.add_argument(
        '--product-wkt',
        '--product_wkt',
        dest='product_wkt',
        type=str,
        required=False,
        default=None,
        help='WKT string defining the product region of interest.',
    )
    parser.add_argument(
        '--gpt-path',
        dest='gpt_path',
        type=str,
        default=None,
        help='Override GPT executable path (default: gpt_path env var).',
    )
    parser.add_argument(
        '--grid-path',
        dest='grid_path',
        type=str,
        default=None,
        help='Override grid GeoJSON path (default: grid_path env var).',
    )
    parser.add_argument(
        '--db-dir',
        dest='db_dir',
        type=str,
        default=None,
        help='Override database output directory (default: db_dir env var).',
    )
    parser.add_argument(
        '--gpt-memory',
        dest='gpt_memory',
        type=str,
        default="16G",
        help='Override GPT Java heap (e.g., 24G).',
    )
    parser.add_argument(
        '--gpt-parallelism',
        dest='gpt_parallelism',
        type=int,
        default=10,
        help='Override GPT parallelism (number of tiles).',
    )
    parser.add_argument(
        '--gpt-timeout',
        dest='gpt_timeout',
        type=int,
        default=None,
        help='Override GPT timeout in seconds for a single invocation.',
    )
    parser.add_argument(
        '--snap-userdir',
        dest='snap_userdir',
        type=str,
        default=None,
        help='Override SNAP user directory (default: SNAP_USERDIR env or project .snap).',
    )
    parser.add_argument(
        '--orbit-type',
        dest='orbit_type',
        type=str,
        default='Sentinel Precise (Auto Download)',
        help='SNAP Apply-Orbit-File orbitType string.',
    )
    parser.add_argument(
        '--orbit-continue-on-fail',
        dest='orbit_continue_on_fail',
        action='store_true',
        help='Continue processing if orbit file cannot be applied.',
    )
    parser.add_argument(
        '--orbit-download-type',
        dest='orbit_download_type',
        type=str,
        default='POEORB',
        choices=['POEORB', 'RESORB'],
        help='Orbit type to prefetch (POEORB or RESORB).',
    )
    parser.add_argument(
        '--orbit-years',
        dest='orbit_years',
        type=str,
        default=None,
        help='Years to prefetch orbits for (e.g., "2024,2025" or "2020-2026").',
    )
    parser.add_argument(
        '--orbit-satellites',
        dest='orbit_satellites',
        type=str,
        default='S1A,S1B,S1C',
        help='Comma-separated satellites to prefetch (e.g., "S1A,S1C").',
    )
    parser.add_argument(
        '--orbit-base-url',
        dest='orbit_base_url',
        type=str,
        default=None,
        help='Base URL for orbit downloads (default: step.esa.int auxdata).',
    )
    parser.add_argument(
        '--orbit-outdir',
        dest='orbit_outdir',
        type=str,
        default=None,
        help='Override orbit storage directory (default: SNAP_USERDIR/auxdata/Orbits/Sentinel-1).',
    )
    parser.add_argument(
        '--prefetch-orbits',
        dest='prefetch_orbits',
        action='store_true',
        help='Download orbit files in advance for selected years.',
    )
    parser.add_argument(
        '--use-graph',
        dest='use_graph',
        action='store_true',
        help='Use unique GPT graph pipeline instead of op.OperatorCall steps.',
    )
    return parser


# ======================================================================================================================== AUXILIARY
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

    if any(token in as_path for token in ('TSX', 'TDX', 'TERRASAR', 'TANDEMX')):
        return 'TSX'

    if any(token in as_path for token in ('CSG', 'CSK', 'COSMO')):
        return 'CSG'

    if any(token in as_path for token in ('BIOMASS', '/BIO', '_BIO', '-BIO')):
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
        'Supported inferred modes are S1TOPS/S1STRIP, BM, NISAR, TSX, and CSG.'
    )


def _parse_years(years_str: str | None) -> list[int]:
    if not years_str:
        return []
    years: set[int] = set()
    for part in re.split(r'[,\s]+', years_str.strip()):
        if not part:
            continue
        if '-' in part:
            start_s, end_s = part.split('-', 1)
            start = int(start_s)
            end = int(end_s)
            for year in range(min(start, end), max(start, end) + 1):
                years.add(year)
        else:
            years.add(int(part))
    return sorted(years)


def _parse_csv_list(csv_str: str | None) -> list[str]:
    if not csv_str:
        return []
    return [item.strip().upper() for item in csv_str.split(',') if item.strip()]


def _parse_memory_bytes(mem_str: str | None) -> int | None:
    if not mem_str:
        return None
    match = re.match(r'^\s*(\d+(?:\.\d+)?)\s*([KMGTP]?)(?:B)?\s*$', mem_str, re.IGNORECASE)
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2).upper()
    scale = {
        '': 1,
        'K': 1024,
        'M': 1024 ** 2,
        'G': 1024 ** 3,
        'T': 1024 ** 4,
        'P': 1024 ** 5,
    }[unit]
    return int(value * scale)


def _get_total_memory_bytes() -> int | None:
    try:
        page_size = os.sysconf('SC_PAGE_SIZE')
        pages = os.sysconf('SC_PHYS_PAGES')
        if isinstance(page_size, int) and isinstance(pages, int):
            return page_size * pages
    except (ValueError, OSError, AttributeError):
        return None
    return None


def _build_gpt_kwargs(
    gpt_memory: str | None,
    gpt_parallelism: int | None,
    gpt_timeout: int | None,
) -> dict[str, str | int]:
    gpt_kwargs: dict[str, str | int] = {}
    if gpt_memory:
        gpt_kwargs['memory'] = gpt_memory
    if gpt_parallelism:
        gpt_kwargs['parallelism'] = gpt_parallelism
    if gpt_timeout:
        gpt_kwargs['timeout'] = gpt_timeout
    return gpt_kwargs


def _create_gpt_operator(
    product_path: Path,
    output_dir: Path,
    output_format: str,
    gpt_memory: str | None = None,
    gpt_parallelism: int | None = None,
    gpt_timeout: int | None = None,
) -> GPT:
    return GPT(
        product=product_path,
        outdir=output_dir,
        format=output_format,
        gpt_path=GPT_PATH,
        snap_userdir=SNAP_USERDIR,
        **_build_gpt_kwargs(gpt_memory, gpt_parallelism, gpt_timeout),
    )


def _fetch_listing(url: str) -> list[str]:
    with request.urlopen(url, timeout=30) as response:
        html = response.read().decode('utf-8', errors='ignore')
    return re.findall(r'href="([^"]+\\.EOF\\.zip)"', html, flags=re.IGNORECASE)


def prefetch_sentinel_orbits(
    years: list[int],
    orbit_type: str,
    satellites: list[str],
    outdir: Path,
    base_url: str,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for year in years:
        for month in range(1, 13):
            ym_path = f'{year:04d}/{month:02d}'
            for sat in satellites:
                url = f'{base_url}/{orbit_type}/{sat}/{ym_path}/'
                try:
                    files = _fetch_listing(url)
                except Exception as exc:
                    print(f'Warning: failed to list {url}: {exc}')
                    continue
                if not files:
                    continue
                dest_dir = outdir / orbit_type / sat / f'{year:04d}' / f'{month:02d}'
                dest_dir.mkdir(parents=True, exist_ok=True)
                for fname in files:
                    dest_path = dest_dir / fname
                    if dest_path.exists():
                        continue
                    try:
                        print(f'Downloading {fname}...')
                        request.urlretrieve(f'{url}{fname}', dest_path)
                    except Exception as exc:
                        print(f'Warning: failed to download {fname} from {url}: {exc}')


def create_tile_database(input_folder: str, output_db_folder: str) -> pd.DataFrame:
    """Create a database of tile metadata from h5 files."""
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

    output_db_path = Path(output_db_folder)
    output_db_path.mkdir(parents=True, exist_ok=True)

    prod_name = tile_path.name
    output_file = output_db_path / f'{prod_name}_core_metadata.parquet'
    db.to_parquet(output_file, index=False)

    print(f'Core metadata saved to {output_file}')
    return db


def to_geotiff(
    product_path: Path,
    output_dir: Path,
    geo_region: str = None,
    output_name: str = None,
    gpt_memory: str | None = None,
    gpt_parallelism: int | None = None,
    gpt_timeout: int | None = None,
):
    assert geo_region is not None, 'Geo region WKT string must be provided for subsetting.'
    op = _create_gpt_operator(
        product_path=product_path,
        output_dir=output_dir,
        output_format='GDAL-GTiff-WRITER',
        gpt_memory=gpt_memory,
        gpt_parallelism=gpt_parallelism,
        gpt_timeout=gpt_timeout,
    )
    op.Write()
    return op.prod_path


def subset(
    product_path: Path,
    output_dir: Path,
    geo_region: str = None,
    output_name: str = None,
    gpt_memory: str | None = None,
    gpt_parallelism: int | None = None,
    gpt_timeout: int | None = None,
):
    assert geo_region is not None, 'Geo region WKT string must be provided for subsetting.'
    op = _create_gpt_operator(
        product_path=product_path,
        output_dir=output_dir,
        output_format='HDF5',
        gpt_memory=gpt_memory,
        gpt_parallelism=gpt_parallelism,
        gpt_timeout=gpt_timeout,
    )
    op.Subset(
        copy_metadata=True,
        output_name=output_name,
        geo_region=geo_region,
    )
    return op.prod_path


def _cut_rect_worker(
    rect: dict,
    product_path_str: str,
    cuts_outdir_str: str,
    name: str,
    product_mode: str,
    gpt_memory: str | None,
    gpt_parallelism: int | None,
    gpt_timeout: int | None,
):
    geo_region = rectangle_to_wkt(rect)
    tile_name = rect['BL']['properties']['name']
    if product_mode != 'NISAR':
        final_product = subset(
            Path(product_path_str),
            Path(cuts_outdir_str) / name,
            output_name=tile_name,
            geo_region=geo_region,
            gpt_memory=gpt_memory,
            gpt_parallelism=gpt_parallelism,
            gpt_timeout=gpt_timeout,
        )
        return ('product', str(final_product))

    reader = NISARReader(product_path_str)
    cutter = NISARCutter(reader)
    subset_data = cutter.cut_by_wkt(geo_region, 'HH', apply_mask=False)
    nisar_tile_path = Path(cuts_outdir_str) / name / f'{tile_name}.h5'
    cutter.save_subset(subset_data, nisar_tile_path, driver='H5')
    return ('nisar', str(nisar_tile_path))


async def _cut_rectangles_async(
    rectangles: list,
    product_path: Path,
    cuts_outdir: Path,
    name: str,
    product_mode: str,
    gpt_memory: str | None,
    gpt_parallelism: int | None,
    gpt_timeout: int | None,
    max_workers: int | None,
):
    loop = asyncio.get_running_loop()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        tasks = [
            loop.run_in_executor(
                executor,
                _cut_rect_worker,
                rect,
                product_path.as_posix(),
                cuts_outdir.as_posix(),
                name,
                product_mode,
                gpt_memory,
                gpt_parallelism,
                gpt_timeout,
            )
            for rect in rectangles
        ]
        return await asyncio.gather(*tasks)


def _build_sentinel_graph_xml(
    product_path: Path,
    output_path: Path,
    is_TOPS: bool,
    subaperture: bool,
    orbit_type: str,
    orbit_continue_on_fail: bool,
) -> str:
    deramp_node = ''
    deburst_source = 'Apply-Orbit-File'
    if is_TOPS and subaperture:
        deramp_node = """
      <node id="TOPSAR-DerampDemod">
        <operator>TOPSAR-DerampDemod</operator>
        <sources>
          <sourceProduct refid="Apply-Orbit-File"/>
        </sources>
        <parameters class="com.bc.ceres.binding.dom.XppDomElement">
          <outputDerampDemodPhase>false</outputDerampDemodPhase>
        </parameters>
      </node>"""
        deburst_source = 'TOPSAR-DerampDemod'

    return f"""<graph id="Graph">
  <version>1.0</version>
  <node id="Read">
    <operator>Read</operator>
    <sources/>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <file>{product_path.as_posix()}</file>
    </parameters>
  </node>
  <node id="Apply-Orbit-File">
    <operator>Apply-Orbit-File</operator>
    <sources>
      <sourceProduct refid="Read"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <orbitType>{orbit_type}</orbitType>
      <polyDegree>3</polyDegree>
      <continueOnFail>{str(orbit_continue_on_fail).lower()}</continueOnFail>
    </parameters>
  </node>{deramp_node}
  <node id="TOPSAR-Deburst">
    <operator>TOPSAR-Deburst</operator>
    <sources>
      <sourceProduct refid="{deburst_source}"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement"/>
  </node>
  <node id="Calibration">
    <operator>Calibration</operator>
    <sources>
      <sourceProduct refid="TOPSAR-Deburst"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <outputImageInComplex>true</outputImageInComplex>
    </parameters>
  </node>
  <node id="Terrain-Correction">
    <operator>Terrain-Correction</operator>
    <sources>
      <sourceProduct refid="Calibration"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <demName>Copernicus 30m Global DEM</demName>
      <pixelSpacingInMeter>10.0</pixelSpacingInMeter>
      <mapProjection>AUTO:42001</mapProjection>
      <outputComplex>true</outputComplex>
    </parameters>
  </node>
  <node id="Write">
    <operator>Write</operator>
    <sources>
      <sourceProduct refid="Terrain-Correction"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <file>{output_path.as_posix()}</file>
      <formatName>BEAM-DIMAP</formatName>
    </parameters>
  </node>
</graph>"""


def _build_sentinel_graph_signature(
    product_path: Path,
    output_path: Path,
    is_TOPS: bool,
    subaperture: bool,
    orbit_type: str,
    orbit_continue_on_fail: bool,
) -> str:
    sig_payload = '|'.join(
        [
            product_path.as_posix(),
            output_path.as_posix(),
            str(is_TOPS),
            str(subaperture),
            orbit_type,
            str(orbit_continue_on_fail),
            'AUTO:42001',
            '10.0',
        ]
    )
    return hashlib.sha1(sig_payload.encode('utf-8')).hexdigest()[:12]


def _write_text_if_changed(path: Path, content: str) -> None:
    if not path.exists() or path.read_text(encoding='utf-8') != content:
        path.write_text(content, encoding='utf-8')


# ======================================================================================================================== PIPELINES
def pipeline_sentinel(
    product_path: Path,
    output_dir: Path,
    is_TOPS: bool = False,
    subaperture: bool = False,
    use_graph: bool = False,
    orbit_type: str = 'Sentinel Precise (Auto Download)',
    orbit_continue_on_fail: bool = False,
    gpt_memory: str | None = None,
    gpt_parallelism: int | None = None,
    gpt_timeout: int | None = None,
):
    """Sentinel-1 pipeline."""
    op = _create_gpt_operator(
        product_path=product_path,
        output_dir=output_dir,
        output_format='BEAM-DIMAP',
        gpt_memory=gpt_memory,
        gpt_parallelism=gpt_parallelism,
        gpt_timeout=gpt_timeout,
    )

    if use_graph:
        output_path = output_dir / f'{product_path.stem}_TC.dim'
        graph_dir = output_dir / 'graphs'
        graph_dir.mkdir(parents=True, exist_ok=True)

        graph_xml = _build_sentinel_graph_xml(
            product_path=product_path,
            output_path=output_path,
            is_TOPS=is_TOPS,
            subaperture=subaperture,
            orbit_type=orbit_type,
            orbit_continue_on_fail=orbit_continue_on_fail,
        )
        signature = _build_sentinel_graph_signature(
            product_path=product_path,
            output_path=output_path,
            is_TOPS=is_TOPS,
            subaperture=subaperture,
            orbit_type=orbit_type,
            orbit_continue_on_fail=orbit_continue_on_fail,
        )
        graph_path = graph_dir / f'{product_path.stem}_sentinel_{signature}.xml'
        _write_text_if_changed(graph_path, graph_xml)

        result = op.run_graph(graph_path=graph_path, output_path=output_path)
        if result is None:
            raise RuntimeError('Sentinel graph execution failed.')
        return result

    op.ApplyOrbitFile()
    # op.TopsarSplit(subswath="IW1")
    calib_path = op.Calibration(output_complex=True)
    if calib_path is None:
        raise RuntimeError('Calibration failed.')

    deburst_path = op.Deburst()
    if deburst_path is None:
        raise RuntimeError('TOPSAR Deburst failed.')

    # TODO: Add subaperture.
    if is_TOPS and subaperture:
        deramp_path = op.TopsarDerampDemod()
        if deramp_path is None:
            raise RuntimeError('TOPSAR Deramp/Demod failed.')

    tc_path = op.TerrainCorrection(map_projection='AUTO:42001', pixel_spacing_in_meter=10.0)
    if tc_path is None:
        raise RuntimeError('Terrain Correction failed.')
    return op.prod_path


def _calibration_terrain_pipeline(
    product_path: Path,
    output_dir: Path,
    gpt_memory: str | None,
    gpt_parallelism: int | None,
    gpt_timeout: int | None,
) -> Path:
    op = _create_gpt_operator(
        product_path=product_path,
        output_dir=output_dir,
        output_format='BEAM-DIMAP',
        gpt_memory=gpt_memory,
        gpt_parallelism=gpt_parallelism,
        gpt_timeout=gpt_timeout,
    )
    calib_path = op.Calibration(output_complex=True)
    if calib_path is None:
        raise RuntimeError('Calibration failed.')

    # TODO: Add subaperture.
    tc_path = op.TerrainCorrection(map_projection='AUTO:42001', pixel_spacing_in_meter=5.0)
    if tc_path is None:
        raise RuntimeError('Terrain Correction failed.')
    return op.prod_path


def pipeline_terrasar(
    product_path: Path,
    output_dir: Path,
    use_graph: bool = False,
    orbit_type: str = 'Sentinel Precise (Auto Download)',
    orbit_continue_on_fail: bool = False,
    gpt_memory: str | None = None,
    gpt_parallelism: int | None = None,
    gpt_timeout: int | None = None,
):
    """Terrasar-X pipeline."""
    return _calibration_terrain_pipeline(
        product_path=product_path,
        output_dir=output_dir,
        gpt_memory=gpt_memory,
        gpt_parallelism=gpt_parallelism,
        gpt_timeout=gpt_timeout,
    )


def pipeline_cosmo(
    product_path: Path,
    output_dir: Path,
    use_graph: bool = False,
    orbit_type: str = 'Sentinel Precise (Auto Download)',
    orbit_continue_on_fail: bool = False,
    gpt_memory: str | None = None,
    gpt_parallelism: int | None = None,
    gpt_timeout: int | None = None,
):
    """COSMO-SkyMed pipeline."""
    return _calibration_terrain_pipeline(
        product_path=product_path,
        output_dir=output_dir,
        gpt_memory=gpt_memory,
        gpt_parallelism=gpt_parallelism,
        gpt_timeout=gpt_timeout,
    )


def pipeline_biomass(
    product_path: Path,
    output_dir: Path,
    use_graph: bool = False,
    orbit_type: str = 'Sentinel Precise (Auto Download)',
    orbit_continue_on_fail: bool = False,
    gpt_memory: str | None = None,
    gpt_parallelism: int | None = None,
    gpt_timeout: int | None = None,
):
    """BIOMASS pipeline."""
    op = _create_gpt_operator(
        product_path=product_path,
        output_dir=output_dir,
        output_format='GDAL-GTiff-WRITER',
        gpt_memory=gpt_memory,
        gpt_parallelism=gpt_parallelism,
        gpt_timeout=gpt_timeout,
    )
    write_path = op.Write()
    if write_path is None:
        raise RuntimeError('Write failed.')
    # TODO: Calculate SubApertures with BIOMASS Data.
    return op.prod_path


def pipeline_nisar(
    product_path: Path,
    output_dir: Path,
    use_graph: bool = False,
    orbit_type: str = 'Sentinel Precise (Auto Download)',
    orbit_continue_on_fail: bool = False,
    gpt_memory: str | None = None,
    gpt_parallelism: int | None = None,
    gpt_timeout: int | None = None,
):
    """NISAR pipeline."""
    assert product_path.suffix == '.h5', 'NISAR products must be in .h5 format.'
    # Monkey patching for NISAR products
    return product_path


# Router switches between different pipelines based on the product mode.
ROUTER_PIPE = {
    'S1TOPS': partial(pipeline_sentinel, is_TOPS=True),
    'S1STRIP': partial(pipeline_sentinel, is_TOPS=False),
    'BM': pipeline_biomass,
    'TSX': pipeline_terrasar,
    'NISAR': pipeline_nisar,
    'CSG': pipeline_cosmo,
}


def _apply_runtime_overrides(args: argparse.Namespace) -> None:
    """Apply CLI overrides to global runtime settings."""
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


def _maybe_prefetch_orbits(args: argparse.Namespace) -> None:
    """Download orbit files if requested via CLI."""
    if not args.prefetch_orbits:
        return

    orbit_base_url = args.orbit_base_url or ORBIT_BASE_URL
    orbit_outdir = (
        Path(args.orbit_outdir)
        if args.orbit_outdir
        else Path(SNAP_USERDIR) / 'auxdata' / 'Orbits' / 'Sentinel-1'
    )
    years = _parse_years(args.orbit_years)
    satellites = _parse_csv_list(args.orbit_satellites)

    if not years:
        raise ValueError('Orbit prefetch requested but no years were provided.')
    if not satellites:
        raise ValueError('Orbit prefetch requested but no satellites were provided.')

    prefetch_sentinel_orbits(
        years=years,
        orbit_type=args.orbit_download_type,
        satellites=satellites,
        outdir=orbit_outdir,
        base_url=orbit_base_url,
    )


def _compute_cut_workers(rectangles: list, gpt_memory: str | None) -> int:
    """Compute worker count for tile cutting."""
    max_workers = min(len(rectangles), os.cpu_count() or 1)

    if gpt_memory:
        total_mem = _get_total_memory_bytes()
        gpt_mem = _parse_memory_bytes(gpt_memory)
        if total_mem and gpt_mem:
            # Leave headroom: assume each GPT process may use ~1.5x heap with native overhead.
            max_workers_by_mem = max(1, int(total_mem / (gpt_mem * 1.5)))
            if max_workers_by_mem < max_workers:
                print(
                    f'Limiting cut workers to {max_workers_by_mem} '
                    f'based on gpt_memory={gpt_memory} and total_mem={total_mem} bytes.'
                )
            max_workers = min(max_workers, max_workers_by_mem)

    if MAX_CUT_WORKERS and max_workers > MAX_CUT_WORKERS:
        print(f'Limiting cut workers to {MAX_CUT_WORKERS} (MAX_CUT_WORKERS cap).')
        max_workers = MAX_CUT_WORKERS

    return max_workers




def _ensure_grid_file(grid_path: Path, base_path: Path) -> Path:
    if grid_path.exists():
        return grid_path

    grid_dir = base_path / 'grid'
    grid_dir.mkdir(parents=True, exist_ok=True)
    print(f'Grid file not found at {grid_path}. Generating grid_10km.geojson in {grid_dir}.')
    subprocess.run([sys.executable, '-m', 'sarpyx.utils.grid'], cwd=grid_dir, check=True)

    generated = grid_dir / 'grid_10km.geojson'
    if not generated.exists():
        raise FileNotFoundError(
            f'Grid generation completed, but {generated} was not created. Check sarpyx.utils.grid output.'
        )
    return generated


def _run_preprocessing(
    product_path: Path,
    output_dir: Path,
    product_mode: str,
    use_graph: bool,
    orbit_type: str,
    orbit_continue_on_fail: bool,
    gpt_memory: str | None,
    gpt_parallelism: int | None,
    gpt_timeout: int | None,
) -> Path:
    if not prepro:
        return product_path

    intermediate_product = ROUTER_PIPE[product_mode](
        product_path,
        output_dir,
        use_graph=use_graph,
        orbit_type=orbit_type,
        orbit_continue_on_fail=orbit_continue_on_fail,
        gpt_memory=gpt_memory,
        gpt_parallelism=gpt_parallelism,
        gpt_timeout=gpt_timeout,
    )
    print(f'Intermediate processed product located at: {intermediate_product}')
    assert Path(intermediate_product).exists(), f'Intermediate product {intermediate_product} does not exist.'
    return Path(intermediate_product)


def _run_tiling(
    product_wkt: str,
    grid_geoj_path: Path | None,
    intermediate_product: Path,
    cuts_outdir: Path,
    product_mode: str,
    gpt_memory: str | None,
    gpt_parallelism: int | None,
    gpt_timeout: int | None,
) -> str:
    print(f'Checking points within polygon: {product_wkt}')
    assert grid_geoj_path is not None and grid_geoj_path.exists(), 'grid_10km.geojson does not exist.'

    contained = check_points_in_polygon(product_wkt, geojson_path=grid_geoj_path)
    if not contained:
        print('No grid points contained within the provided WKT.')
        raise ValueError('No grid points contained; check WKT and grid CRS alignment.')

    rectangles = rectanglify(contained)
    if not rectangles:
        print('No rectangles could be formed from contained points.')
        raise ValueError('No rectangles formed; check WKT coverage and grid alignment.')

    name = extract_product_id(intermediate_product.as_posix()) if product_mode != 'NISAR' else intermediate_product.stem
    if name is None:
        raise ValueError(f'Could not extract product id from: {intermediate_product}')

    max_workers = _compute_cut_workers(rectangles, gpt_memory)
    results = asyncio.run(
        _cut_rectangles_async(
            rectangles=rectangles,
            product_path=intermediate_product,
            cuts_outdir=cuts_outdir,
            name=name,
            product_mode=product_mode,
            gpt_memory=gpt_memory,
            gpt_parallelism=gpt_parallelism,
            gpt_timeout=gpt_timeout,
            max_workers=max_workers,
        )
    )

    for kind, path in results:
        if kind == 'nisar':
            print(f'Final processed NISAR tile saved at: {path}')
        else:
            print(f'Final processed product located at: {path}')

    total_tiles = len(rectangles)
    num_cuts = list((cuts_outdir / name).rglob('*.h5'))
    assert total_tiles == len(num_cuts), f'Expected {total_tiles} tiles, but found {len(num_cuts)}.'
    return name


def _run_db_indexing(cuts_outdir: Path, name: str) -> None:
    if not db_indexing:
        return

    cuts_folder = cuts_outdir / name
    db = create_tile_database(cuts_folder.as_posix(), DB_DIR)  # type: ignore[arg-type]
    assert not db.empty, 'Database creation failed, resulting DataFrame is empty.'
    print('Database created successfully.')


# =============================================== MAIN =========================================================================
def main():
    parser = create_parser()
    args = parser.parse_args()

    _apply_runtime_overrides(args)

    product_path = Path(args.product_path)
    output_dir = Path(args.output_dir)
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
        raise ValueError(
            'No --product-wkt provided and automatic WKT extraction is only available for Sentinel-1 products.'
        )

    gpt_memory = args.gpt_memory
    gpt_parallelism = args.gpt_parallelism
    gpt_timeout = args.gpt_timeout
    orbit_type = args.orbit_type
    orbit_continue_on_fail = args.orbit_continue_on_fail
    use_graph = args.use_graph

    _maybe_prefetch_orbits(args)

    intermediate_product = _run_preprocessing(
        product_path=product_path,
        output_dir=output_dir,
        product_mode=product_mode,
        use_graph=use_graph,
        orbit_type=orbit_type,
        orbit_continue_on_fail=orbit_continue_on_fail,
        gpt_memory=gpt_memory,
        gpt_parallelism=gpt_parallelism,
        gpt_timeout=gpt_timeout,
    )

    name = intermediate_product.stem
    if tiling:
        name = _run_tiling(
            product_wkt=product_wkt,
            grid_geoj_path=grid_geoj_path,
            intermediate_product=intermediate_product,
            cuts_outdir=cuts_outdir,
            product_mode=product_mode,
            gpt_memory=gpt_memory,
            gpt_parallelism=gpt_parallelism,
            gpt_timeout=gpt_timeout,
        )

    _run_db_indexing(cuts_outdir=cuts_outdir, name=name)
    sys.exit(0)


if __name__ == '__main__':
    main()
