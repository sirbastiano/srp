"""
This script processes satellite SAR data from various missions using the SNAP GPT tool and the sarpyx library. It supports different pipelines for Sentinel-1, Terrasar-X, COSMO-SkyMed, BIOMASS, and NISAR products. The processing steps include debursting, calibration, terrain


TODO: metadate reorganization.
TODO: SUBAPERTURE PROCESSING for all missions.
TODO: PolSAR support.
TODO: InSAR support.



"""


from pathlib import Path
from dotenv import load_dotenv
import hashlib
import re, os, sys
from urllib import request
import pandas as pd
from functools import partial
import argparse
import yaml

from sarpyx.snapflow.engine import GPT
from sarpyx.utils.geos import check_points_in_polygon, rectangle_to_wkt, rectanglify
from sarpyx.utils.io import read_h5
from sarpyx.utils.nisar_utils import NISARReader, NISARCutter, NISARMetadata


# Load environment variables from .env file
load_dotenv()
# Read paths from environment variables
GPT_PATH = os.getenv('gpt_path')
GRID_PATH = os.getenv('grid_path')
DB_DIR = os.getenv('db_dir')
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SNAP_USERDIR = os.getenv('SNAP_USERDIR') or os.getenv('snap_userdir') or str(PROJECT_ROOT / '.snap')
os.environ.setdefault('SNAP_USERDIR', SNAP_USERDIR)
ORBIT_BASE_URL = os.getenv('orbit_base_url') or os.getenv('ORBIT_BASE_URL') or 'https://step.esa.int/auxdata/orbits/Sentinel-1'
# ========================================================================================================================================
# ================================================================================================================================ Parser
# Parse command-line arguments
def create_parser() -> argparse.ArgumentParser:
    """
    Create command-line argument parser.

    Returns:
        argparse.ArgumentParser: Parser for worldsar command.
    """
    parser = argparse.ArgumentParser(
        description='Process SAR data using SNAP GPT and sarpyx pipelines.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple processing with minimal arguments (recommended for beginners)
  worldsar --input product.SAFE --mode S1TOPS --wkt "POLYGON(...)"
  
  # Specify custom output directories
  worldsar --input product.SAFE --mode S1TOPS --wkt "POLYGON(...)" \\
           --output ./processed --tiles ./tiles
  
  # Use a configuration file for complex setups
  worldsar --config worldsar_config.yaml
  
  # Advanced: Custom GPT settings and orbit prefetch
  worldsar --input product.SAFE --mode S1TOPS --wkt "POLYGON(...)" \\
           --gpt-memory 24G --gpt-parallelism 4 --prefetch-orbits

Supported modes: S1TOPS, S1STRIP, BM (BIOMASS), NISAR, TSX (TerraSAR-X), CSG (COSMO-SkyMed)
"""
    )
    
    # ===== ESSENTIAL ARGUMENTS =====
    essential = parser.add_argument_group('Essential Arguments', 'Core arguments required for processing (can be provided via --config)')
    essential.add_argument(
        '--input',
        '-i',
        dest='product_path',
        type=str,
        required=False,  # Can be provided via config file
        help='Path to the input SAR product (e.g., S1A_*.SAFE, *.h5)'
    )
    essential.add_argument(
        '--mode',
        '-m',
        dest='prod_mode',
        type=str,
        required=False,  # Can be provided via config file
        choices=['S1TOPS', 'S1STRIP', 'BM', 'NISAR', 'TSX', 'CSG'],
        help='Processing mode based on satellite/product type'
    )
    essential.add_argument(
        '--wkt',
        '-w',
        dest='product_wkt',
        type=str,
        required=False,  # Can be provided via config file
        help='WKT string defining the region of interest (e.g., "POLYGON((lon lat, ...))")'
    )
    
    # ===== OUTPUT CONFIGURATION =====
    output_group = parser.add_argument_group('Output Configuration', 'Control where processed data is saved')
    output_group.add_argument(
        '--output',
        '-o',
        dest='output_dir',
        type=str,
        default='./worldsar_output',
        help='Directory for intermediate processing outputs (default: ./worldsar_output)'
    )
    output_group.add_argument(
        '--tiles',
        '-t',
        dest='cuts_outdir',
        type=str,
        default='./worldsar_tiles',
        help='Directory for extracted tiles (default: ./worldsar_tiles)'
    )
    output_group.add_argument(
        '--create-db',
        dest='create_database',
        action='store_true',
        help='Create a metadata database from tiles (default: disabled)'
    )
    output_group.add_argument(
        '--db-dir',
        dest='db_dir',
        type=str,
        default=None,
        help='Database output directory (default: from DB_DIR env var)'
    )
    
    # ===== PROCESSING OPTIONS =====
    processing = parser.add_argument_group('Processing Options', 'Control processing behavior')
    processing.add_argument(
        '--skip-preprocessing',
        dest='skip_prepro',
        action='store_true',
        help='Skip preprocessing step (use if product is already preprocessed)'
    )
    processing.add_argument(
        '--skip-tiling',
        dest='skip_tiling',
        action='store_true',
        help='Skip tiling step (only preprocess the product)'
    )
    processing.add_argument(
        '--use-graph',
        dest='use_graph',
        action='store_true',
        help='Use GPT graph XML pipeline (faster for batch processing)'
    )
    
    # ===== SNAP/GPT CONFIGURATION =====
    snap_group = parser.add_argument_group('SNAP/GPT Configuration', 'Configure SNAP processing engine')
    snap_group.add_argument(
        '--gpt-path',
        dest='gpt_path',
        type=str,
        default=None,
        help='Path to GPT executable (default: from gpt_path env var or system PATH)'
    )
    snap_group.add_argument(
        '--gpt-memory',
        dest='gpt_memory',
        type=str,
        default=None,
        help='GPT Java heap size (e.g., "8G", "24G") (default: SNAP defaults)'
    )
    snap_group.add_argument(
        '--gpt-parallelism',
        dest='gpt_parallelism',
        type=int,
        default=None,
        help='Number of parallel tiles to process (default: auto)'
    )
    snap_group.add_argument(
        '--snap-userdir',
        dest='snap_userdir',
        type=str,
        default=None,
        help='SNAP user directory (default: from SNAP_USERDIR env or .snap/)'
    )
    
    # ===== GRID CONFIGURATION =====
    grid_group = parser.add_argument_group('Grid Configuration', 'Configure tiling grid system')
    grid_group.add_argument(
        '--grid-path',
        dest='grid_path',
        type=str,
        default=None,
        help='Path to grid GeoJSON file (default: from grid_path env var)'
    )
    
    # ===== ORBIT CONFIGURATION =====
    orbit_group = parser.add_argument_group('Orbit Configuration', 'Configure orbit file handling (Sentinel-1 only)')
    orbit_group.add_argument(
        '--orbit-type',
        dest='orbit_type',
        type=str,
        default='Sentinel Precise (Auto Download)',
        help='Orbit file type for SNAP Apply-Orbit-File (default: Sentinel Precise Auto Download)'
    )
    orbit_group.add_argument(
        '--orbit-continue-on-fail',
        dest='orbit_continue_on_fail',
        action='store_true',
        help='Continue processing if orbit file cannot be applied'
    )
    orbit_group.add_argument(
        '--prefetch-orbits',
        dest='prefetch_orbits',
        action='store_true',
        help='Prefetch orbit files before processing'
    )
    orbit_group.add_argument(
        '--orbit-years',
        dest='orbit_years',
        type=str,
        default=None,
        help='Years for orbit prefetch (e.g., "2024" or "2020-2026") (required if --prefetch-orbits)'
    )
    orbit_group.add_argument(
        '--orbit-satellites',
        dest='orbit_satellites',
        type=str,
        default='S1A,S1B,S1C',
        help='Satellites for orbit prefetch, comma-separated (default: S1A,S1B,S1C)'
    )
    orbit_group.add_argument(
        '--orbit-type-download',
        dest='orbit_download_type',
        type=str,
        default='POEORB',
        choices=['POEORB', 'RESORB'],
        help='Orbit file type to download: POEORB (precise) or RESORB (restituted) (default: POEORB)'
    )
    orbit_group.add_argument(
        '--orbit-base-url',
        dest='orbit_base_url',
        type=str,
        default=None,
        help='Base URL for orbit downloads (default: https://step.esa.int/auxdata/orbits/Sentinel-1)'
    )
    orbit_group.add_argument(
        '--orbit-outdir',
        dest='orbit_outdir',
        type=str,
        default=None,
        help='Directory to store downloaded orbits (default: SNAP_USERDIR/auxdata/Orbits/Sentinel-1)'
    )
    
    # ===== CONFIGURATION FILE =====
    config_group = parser.add_argument_group('Configuration File', 'Use a YAML file for complex configurations')
    config_group.add_argument(
        '--config',
        '-c',
        dest='config_file',
        type=str,
        default=None,
        help='Load settings from a YAML configuration file (overrides command-line args)'
    )
    
    return parser






# ======================================================================================================================== SETTINGS
""" Processing settings"""

def load_config_file(config_path: str) -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dictionary with configuration values
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config or {}


def merge_config_with_args(args: argparse.Namespace) -> argparse.Namespace:
    """
    Merge configuration file with command-line arguments.
    Command-line arguments take precedence.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Updated arguments with config file values
    """
    if args.config_file:
        config = load_config_file(args.config_file)
        
        # Map config keys to argument names
        config_mapping = {
            'input': 'product_path',
            'output': 'output_dir',
            'tiles': 'cuts_outdir',
            'wkt': 'product_wkt',
            'mode': 'prod_mode',
            'grid_path': 'grid_path',
            'db_dir': 'db_dir',
            'create_database': 'create_database',
            'skip_preprocessing': 'skip_prepro',
            'skip_tiling': 'skip_tiling',
            'use_graph': 'use_graph',
            'gpt_path': 'gpt_path',
            'gpt_memory': 'gpt_memory',
            'gpt_parallelism': 'gpt_parallelism',
            'snap_userdir': 'snap_userdir',
            'orbit_type': 'orbit_type',
            'orbit_continue_on_fail': 'orbit_continue_on_fail',
            'prefetch_orbits': 'prefetch_orbits',
            'orbit_years': 'orbit_years',
            'orbit_satellites': 'orbit_satellites',
            'orbit_download_type': 'orbit_download_type',
            'orbit_base_url': 'orbit_base_url',
            'orbit_outdir': 'orbit_outdir',
        }
        
        # Apply config values only if not provided via command line
        # We use parser defaults to determine if a value was user-provided
        parser_defaults = {
            'output_dir': './worldsar_output',
            'cuts_outdir': './worldsar_tiles',
            'orbit_type': 'Sentinel Precise (Auto Download)',
            'orbit_satellites': 'S1A,S1B,S1C',
            'orbit_download_type': 'POEORB',
        }
        
        for config_key, arg_name in config_mapping.items():
            if config_key in config:
                current_val = getattr(args, arg_name, None)
                # Only set from config if argument is None or at its default value
                if current_val is None or current_val == parser_defaults.get(arg_name):
                    setattr(args, arg_name, config[config_key])
    
    return args


# Processing control flags (can be overridden by arguments)
prepro = True
tiling = True
db_indexing = False

# ======================================================================================================================== AUXILIARY
""" Auxiliary functions for database creation and product subsetting. """
def extract_product_id(path: str) -> str | None:
    m = re.search(r"/([^/]+?)_[^/_]+\.dim$", path)
    return m.group(1) if m else None


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
            for y in range(min(start, end), max(start, end) + 1):
                years.add(y)
        else:
            years.add(int(part))
    return sorted(years)


def _parse_csv_list(csv_str: str | None) -> list[str]:
    if not csv_str:
        return []
    return [s.strip().upper() for s in csv_str.split(',') if s.strip()]


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
                except Exception as e:
                    print(f'Warning: failed to list {url}: {e}')
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
                    except Exception as e:
                        print(f'Warning: failed to download {fname} from {url}: {e}')


def create_tile_database(input_folder: str, output_db_folder: str) -> pd.DataFrame:
    """Create a database of tile metadata from h5 files.
    
    Args:
        input_folder: Path to folder containing h5 tile files
        output_db_folder: Path to folder where database parquet file will be saved
        
    Returns:
        DataFrame containing the metadata for all tiles
    """
    # Find all h5 tiles in the input folder
    tile_path = Path(input_folder)
    h5_tiles = list(tile_path.rglob('*.h5'))
    print(f"Found {len(h5_tiles)} h5 files in {input_folder}")
    
    # Initialize empty database
    db = pd.DataFrame()
    
    # Process each tile
    for idx, tile_file in enumerate(h5_tiles):
        print(f"Processing tile {idx + 1}/{len(h5_tiles)}: {tile_file.name}")
        
        # Read h5 file and extract metadata
        data, metadata = read_h5(tile_file)
        row = pd.Series(metadata['quickinfo'])
        row['ID'] = tile_file.stem  # Add TileID to the row
        
        # Append to database
        db = pd.concat([db, pd.DataFrame([row])], ignore_index=True)
    
    # Save database to parquet file
    output_db_path = Path(output_db_folder)
    output_db_path.mkdir(parents=True, exist_ok=True)
    
    prod_name = tile_path.name
    output_file = output_db_path / f'{prod_name}_core_metadata.parquet'
    db.to_parquet(output_file, index=False)
    
    print(f"Core metadata saved to {output_file}")
    
    return db


def to_geotiff(
    product_path: Path,
    output_dir: Path,
    geo_region: str = None,
    output_name: str = None,
    gpt_memory: str | None = None,
    gpt_parallelism: int | None = None,
):
    assert geo_region is not None, "Geo region WKT string must be provided for subsetting."
    gpt_kwargs = {}
    if gpt_memory:
        gpt_kwargs['memory'] = gpt_memory
    if gpt_parallelism:
        gpt_kwargs['parallelism'] = gpt_parallelism
    
    op = GPT(
        product=product_path,
        outdir=output_dir,
        format='GDAL-GTiff-WRITER',
        gpt_path=GPT_PATH,
        snap_userdir=SNAP_USERDIR,
        **gpt_kwargs,
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
):
    assert geo_region is not None, "Geo region WKT string must be provided for subsetting."
    gpt_kwargs = {}
    if gpt_memory:
        gpt_kwargs['memory'] = gpt_memory
    if gpt_parallelism:
        gpt_kwargs['parallelism'] = gpt_parallelism
    
    op = GPT(
        product=product_path,
        outdir=output_dir,
        format='HDF5',
        gpt_path=GPT_PATH,
        snap_userdir=SNAP_USERDIR,
        **gpt_kwargs,
    )

    op.Subset(
        copy_metadata=True,
        output_name=output_name,
        geo_region=geo_region,
        )

    return op.prod_path

# ======================================================================================================================== PIPELINES
""" Different pipelines for different missions/products. """
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
):
    """A simple test pipeline to validate the GPT wrapper functionality.

    The operations included are:
    - Debursting
    - Calibration to complex

    Args:
        product_path (Path): Path to the input product.
        output_dir (Path): Directory to save the processed output.

    Returns:
        Path: Path to the processed product.
    """
    gpt_kwargs = {}
    if gpt_memory:
        gpt_kwargs['memory'] = gpt_memory
    if gpt_parallelism:
        gpt_kwargs['parallelism'] = gpt_parallelism
    op = GPT(
        product=product_path,
        outdir=output_dir,
        format='BEAM-DIMAP',
        gpt_path=GPT_PATH,
        snap_userdir=SNAP_USERDIR,
        **gpt_kwargs,
    )
    if use_graph:
        output_path = output_dir / f'{product_path.stem}_TC.dim'
        graph_dir = output_dir / 'graphs'
        graph_dir.mkdir(parents=True, exist_ok=True)

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

        graph_xml = f"""<graph id="Graph">
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

        sig_payload = '|'.join([
            product_path.as_posix(),
            output_path.as_posix(),
            str(is_TOPS),
            str(subaperture),
            orbit_type,
            str(orbit_continue_on_fail),
            'AUTO:42001',
            '10.0',
        ])
        signature = hashlib.sha1(sig_payload.encode('utf-8')).hexdigest()[:12]
        graph_path = graph_dir / f'{product_path.stem}_sentinel_{signature}.xml'

        if not graph_path.exists() or graph_path.read_text(encoding='utf-8') != graph_xml:
            graph_path.write_text(graph_xml, encoding='utf-8')

        result = op.run_graph(graph_path=graph_path, output_path=output_path)
        if result is None:
            raise RuntimeError('Sentinel graph execution failed.')
        return result


    else:
        
        op.ApplyOrbitFile()
        if is_TOPS and subaperture:
            op.TopsarDerampDemod()
        op.Deburst()
        op.Calibration(output_complex=True, pols=["VV", "VH"])
        # TODO: Add subaperture.
        op.TerrainCorrection(map_projection='AUTO:42001', pixel_spacing_in_meter=10.0)
        return op.prod_path



def pipeline_terrasar(
    product_path: Path,
    output_dir: Path,
    use_graph: bool = False,
    orbit_type: str = 'Sentinel Precise (Auto Download)',
    orbit_continue_on_fail: bool = False,
    gpt_memory: str | None = None,
    gpt_parallelism: int | None = None,
):
    """Terrasar-X pipeline.

    The operations included are:
    - Calibration, outputting complex data if available.
    - Terrain Correction with automatic map projection and 5m pixel spacing.

    Args:
        product_path (Path): Path to the input product.
        output_dir (Path): Directory to save the processed output.

    Returns:
        Path: Path to the processed product.
    """
    gpt_kwargs = {}
    if gpt_memory:
        gpt_kwargs['memory'] = gpt_memory
    if gpt_parallelism:
        gpt_kwargs['parallelism'] = gpt_parallelism
    op = GPT(
        product=product_path,
        outdir=output_dir,
        format='BEAM-DIMAP',
        gpt_path=GPT_PATH,
        snap_userdir=SNAP_USERDIR,
        **gpt_kwargs,
    )
    op.Calibration(output_complex=True)
    # TODO: Add subaperture.
    op.TerrainCorrection(map_projection='AUTO:42001', pixel_spacing_in_meter=5.0)
    return op.prod_path


def pipeline_cosmo(
    product_path: Path,
    output_dir: Path,
    use_graph: bool = False,
    orbit_type: str = 'Sentinel Precise (Auto Download)',
    orbit_continue_on_fail: bool = False,
    gpt_memory: str | None = None,
    gpt_parallelism: int | None = None,
):
    """COSMO-SkyMed pipeline.

    The operations included are:
    - Calibration, outputting complex data if available.
    - Terrain Correction with automatic map projection and 5m pixel spacing.

    Args:
        product_path (Path): Path to the input product.
        output_dir (Path): Directory to save the processed output.

    Returns:
        Path: Path to the processed product.
    """
    gpt_kwargs = {}
    if gpt_memory:
        gpt_kwargs['memory'] = gpt_memory
    if gpt_parallelism:
        gpt_kwargs['parallelism'] = gpt_parallelism
    op = GPT(
        product=product_path,
        outdir=output_dir,
        format='BEAM-DIMAP',
        gpt_path=GPT_PATH,
        snap_userdir=SNAP_USERDIR,
        **gpt_kwargs,
    )
    op.Calibration(output_complex=True)
    # TODO: Add subaperture.
    op.TerrainCorrection(map_projection='AUTO:42001', pixel_spacing_in_meter=5.0)
    return op.prod_path


def pipeline_biomass(
    product_path: Path,
    output_dir: Path,
    use_graph: bool = False,
    orbit_type: str = 'Sentinel Precise (Auto Download)',
    orbit_continue_on_fail: bool = False,
    gpt_memory: str | None = None,
    gpt_parallelism: int | None = None,
):
    """BIOMASS pipeline.

    Args:
        product_path (Path): Path to the input product.
        output_dir (Path): Directory to save the processed output.

    Returns:
        Path: Path to the processed product.
    """
    gpt_kwargs = {}
    if gpt_memory:
        gpt_kwargs['memory'] = gpt_memory
    if gpt_parallelism:
        gpt_kwargs['parallelism'] = gpt_parallelism
    op = GPT(
        product=product_path,
        outdir=output_dir,
        format='GDAL-GTiff-WRITER',
        gpt_path=GPT_PATH,
        snap_userdir=SNAP_USERDIR,
        **gpt_kwargs,
    )
    op.Write()
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
):
    """ NISAR Pipeline.

    The operations included are:

    Args:
        product_path (Path): Path to the input product.
        output_dir (Path): Directory to save the processed output. [Not used]

    Returns:
        Path: Path to the processed product.
    """
    assert product_path.suffix == '.h5', "NISAR products must be in .h5 format."
    # Monkey patching for NISAR products
    return product_path
# ========================================================================================================================================



# ========================================================================================================================================
""" The router switches between different pipelines based on the product mode. """
ROUTER_PIPE = {
    'S1TOPS': partial(pipeline_sentinel, is_TOPS=True),
    'S1STRIP': partial(pipeline_sentinel, is_TOPS=False),
    'BM': pipeline_biomass,
    'TSX': pipeline_terrasar,
    'NISAR': pipeline_nisar,
    'CSG': pipeline_cosmo,
}
# ========================================================================================================================================




# =============================================== MAIN =========================================================================
def main():
    parser = create_parser()
    args = parser.parse_args()
    
    # Load configuration file if provided
    args = merge_config_with_args(args)
    
    # Validate required arguments
    if not args.product_path:
        parser.error("--input is required")
    if not args.prod_mode:
        parser.error("--mode is required")
    if not args.product_wkt:
        parser.error("--wkt is required")
    
    # Handle orbit prefetch validation
    if args.prefetch_orbits and not args.orbit_years:
        parser.error("--orbit-years is required when using --prefetch-orbits")

    global GPT_PATH, GRID_PATH, DB_DIR, SNAP_USERDIR, prepro, tiling, db_indexing
    
    # Override global settings from arguments
    if args.gpt_path:
        GPT_PATH = args.gpt_path
    if args.grid_path:
        GRID_PATH = args.grid_path
    if args.db_dir:
        DB_DIR = args.db_dir
    if args.snap_userdir:
        SNAP_USERDIR = args.snap_userdir
        os.environ['SNAP_USERDIR'] = SNAP_USERDIR
    
    # Set processing flags
    prepro = not args.skip_prepro
    tiling = not args.skip_tiling
    db_indexing = args.create_database

    # Extract arguments
    product_path = Path(args.product_path)
    output_dir = Path(args.output_dir)
    product_wkt = args.product_wkt
    cuts_outdir = Path(args.cuts_outdir)
    grid_geoj_path = Path(GRID_PATH) if GRID_PATH else None
    product_mode = args.prod_mode
    gpt_memory = args.gpt_memory
    gpt_parallelism = args.gpt_parallelism
    orbit_type = args.orbit_type
    orbit_continue_on_fail = args.orbit_continue_on_fail
    use_graph = args.use_graph

    # Print processing configuration
    print(f"\n{'='*80}")
    print(f"WorldSAR Processing Configuration")
    print(f"{'='*80}")
    print(f"Input Product:     {product_path}")
    print(f"Product Mode:      {product_mode}")
    print(f"Output Directory:  {output_dir}")
    print(f"Tiles Directory:   {cuts_outdir}")
    print(f"Preprocessing:     {'Enabled' if prepro else 'Skipped'}")
    print(f"Tiling:            {'Enabled' if tiling else 'Skipped'}")
    print(f"Database Creation: {'Enabled' if db_indexing else 'Disabled'}")
    if gpt_memory:
        print(f"GPT Memory:        {gpt_memory}")
    if gpt_parallelism:
        print(f"GPT Parallelism:   {gpt_parallelism}")
    print(f"{'='*80}\n")

    orbit_base_url = args.orbit_base_url or ORBIT_BASE_URL
    orbit_outdir = Path(args.orbit_outdir) if args.orbit_outdir else Path(SNAP_USERDIR) / 'auxdata' / 'Orbits' / 'Sentinel-1'
    if args.prefetch_orbits:
        print("Prefetching orbit files...")
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

    # STEP1: Preprocessing
    if prepro:
        print(f"\nStep 1: Preprocessing {product_mode} product...")
        intermediate_product = ROUTER_PIPE[product_mode](
            product_path,
            output_dir,
            use_graph=use_graph,
            orbit_type=orbit_type,
            orbit_continue_on_fail=orbit_continue_on_fail,
            gpt_memory=gpt_memory,
            gpt_parallelism=gpt_parallelism,
        )
        print(f"✓ Preprocessing complete. Output: {intermediate_product}")
        assert Path(intermediate_product).exists(), f"Intermediate product {intermediate_product} does not exist."
    else:
        print("\nStep 1: Preprocessing skipped (using input product directly)")
        intermediate_product = product_path

    # STEP2: Tiling
    if tiling:
        print(f"\nStep 2: Tiling product into grid system...")
        # ------ Cutting according to the tile griding system: UTM / WGS84 Auto ------
        print(f'Checking grid points within polygon...')
        assert grid_geoj_path is not None and grid_geoj_path.exists(), f'Grid GeoJSON not found: {grid_geoj_path}'
        
        # step 1: check the contained grid points in the prod
        contained = check_points_in_polygon(product_wkt, geojson_path=grid_geoj_path)
        if not contained:
            print('⚠ No grid points contained within the provided WKT.')
            raise ValueError('No grid points contained; check WKT and grid CRS alignment.')
        
        print(f"Found {len(contained)} grid points in region")
        
        # step 2: Build the rectangles for cutting
        rectangles = rectanglify(contained)
        if not rectangles:
            print('⚠ No rectangles could be formed from contained points.')
            raise ValueError('No rectangles formed; check WKT coverage and grid alignment.')
        
        print(f"Created {len(rectangles)} tiles to extract")
        
        product_path = Path(intermediate_product)
        name = extract_product_id(product_path.as_posix()) if product_mode != 'NISAR' else product_path.stem
        if name is None:
            raise ValueError(f"Could not extract product id from: {product_path}")
        
        for idx, rect in enumerate(rectangles, 1):
            tile_name = rect['BL']['properties']['name']
            print(f"  Processing tile {idx}/{len(rectangles)}: {tile_name}", end='', flush=True)
            try:
                geo_region = rectangle_to_wkt(rect)
                if product_mode != 'NISAR':
                    final_product = subset(
                        product_path,
                        cuts_outdir / name,
                        output_name=tile_name,
                        geo_region=geo_region,
                        gpt_memory=gpt_memory,
                        gpt_parallelism=gpt_parallelism,
                    )
                else:
                    reader = NISARReader(product_path.as_posix())
                    cutter = NISARCutter(reader)
                    subset_data = cutter.cut_by_wkt(geo_region, "HH", apply_mask=False)
                    nisar_tile_path = cuts_outdir / name / f"{tile_name}.tiff"
                    cutter.save_subset(subset_data, nisar_tile_path)
                print(" ✓")
            except Exception as e:
                print(f" ✗ Error: {e}")
                raise
                
        total_tiles = len(rectangles)
        num_cuts = list(Path(cuts_outdir / name).rglob('*.h5'))
        print(f"✓ Tiling complete. Generated {len(num_cuts)} tiles (expected {total_tiles})")
        assert total_tiles == len(num_cuts), f"Expected {total_tiles} tiles, but found {len(num_cuts)}."
    else:
        print("\nStep 2: Tiling skipped")
            
    # STEP3: Database indexing
    if db_indexing:
        print(f"\nStep 3: Creating metadata database...")
        cuts_folder = cuts_outdir / name
        db = create_tile_database(cuts_folder.as_posix(), DB_DIR) # type: ignore
        assert not db.empty, "Database creation failed, resulting DataFrame is empty."
        print("✓ Database created successfully")
    else:
        print("\nStep 3: Database creation disabled")
    
    print(f"\n{'='*80}")
    print("WorldSAR Processing Complete!")
    print(f"{'='*80}\n")
    
    sys.exit(0)
# ========================================================================================================================================  





if __name__ == "__main__":

    
    main()
