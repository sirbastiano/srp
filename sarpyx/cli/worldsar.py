"""
This script processes satellite SAR data from various missions using the SNAP GPT tool and the sarpyx library. It supports different pipelines for Sentinel-1, Terrasar-X, COSMO-SkyMed, BIOMASS, and NISAR products. The processing steps include debursting, calibration, terrain


TODO: metadate reorganization.
TODO: SUBAPERTURE PROCESSING for all missions.
TODO: PolSAR support.
TODO: InSAR support.



"""


from pathlib import Path
from dotenv import load_dotenv
import re, os, sys
import pandas as pd
from functools import partial
import argparse

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
# ========================================================================================================================================
# ================================================================================================================================ Parser
# Parse command-line arguments
def create_parser() -> argparse.ArgumentParser:
    """
    Create command-line argument parser.

    Returns:
        argparse.ArgumentParser: Parser for worldsar command.
    """
    parser = argparse.ArgumentParser(description='Process SAR data using SNAP GPT and sarpyx pipelines.')
    parser.add_argument(
        '--input',
        '-i',
        dest='product_path',
        type=str,
        required=True,
        help='Path to the input SAR product.'
    )
    parser.add_argument(
        '--output',
        '-o',
        dest='output_dir',
        type=str,
        required=True,
        help='Directory to save the processed output.'
    )
    parser.add_argument(
        '--cuts-outdir',
        '--cuts_outdir',
        dest='cuts_outdir',
        type=str,
        required=True,
        help='Where to store the tiles after extraction.'
    )
    parser.add_argument(
        '--product-wkt',
        '--product_wkt',
        dest='product_wkt',
        type=str,
        required=True,
        help='WKT string defining the product region of interest.'
    )
    parser.add_argument(
        '--prod-mode',
        '--prod_mode',
        dest='prod_mode',
        type=str,
        required=True,
        help='Product mode: ["S1TOPS", "S1STRIP", "BM", "NISAR", "TSX", "CSG", "ICE"].'
    )
    parser.add_argument(
        '--gpt-path',
        dest='gpt_path',
        type=str,
        default=None,
        help='Override GPT executable path (default: gpt_path env var).'
    )
    parser.add_argument(
        '--grid-path',
        dest='grid_path',
        type=str,
        default=None,
        help='Override grid GeoJSON path (default: grid_path env var).'
    )
    parser.add_argument(
        '--db-dir',
        dest='db_dir',
        type=str,
        default=None,
        help='Override database output directory (default: db_dir env var).'
    )
    parser.add_argument(
        '--gpt-memory',
        dest='gpt_memory',
        type=str,
        default=None,
        help='Override GPT Java heap (e.g., 24G).'
    )
    parser.add_argument(
        '--gpt-parallelism',
        dest='gpt_parallelism',
        type=int,
        default=None,
        help='Override GPT parallelism (number of tiles).'
    )
    return parser






# ======================================================================================================================== SETTINGS
""" Processing settings"""
prepro = True
tiling = False
db_indexing = False

# ======================================================================================================================== AUXILIARY
""" Auxiliary functions for database creation and product subsetting. """
def extract_product_id(path: str) -> str | None:
    m = re.search(r"/([^/]+?)_[^/_]+\.dim$", path)
    return m.group(1) if m else None


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
    gpt_memory: str | None = None,
    gpt_parallelism: int | None = None,
):
    """A simple test pipeline to validate the GPT wrapper functionality.

    The operations included are:
    - Debursting
    - Calibration to complex
    - (Optional) Subsetting by geographic coordinates

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
        **gpt_kwargs,
    )
    op.ApplyOrbitFile()
    if is_TOPS and subaperture:
        op.TopsarDerampDemod()
    op.Deburst()
    op.Calibration(output_complex=True)
    # TODO: Add subaperture.
    op.TerrainCorrection(map_projection='AUTO:42001', pixel_spacing_in_meter=10.0)
    return op.prod_path


def pipeline_terrasar(
    product_path: Path,
    output_dir: Path,
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
        **gpt_kwargs,
    )
    op.Calibration(output_complex=True)
    # TODO: Add subaperture.
    op.TerrainCorrection(map_projection='AUTO:42001', pixel_spacing_in_meter=5.0)
    return op.prod_path


def pipeline_cosmo(
    product_path: Path,
    output_dir: Path,
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
        **gpt_kwargs,
    )
    op.Calibration(output_complex=True)
    # TODO: Add subaperture.
    op.TerrainCorrection(map_projection='AUTO:42001', pixel_spacing_in_meter=5.0)
    return op.prod_path


def pipeline_biomass(
    product_path: Path,
    output_dir: Path,
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
        **gpt_kwargs,
    )
    op.Write()
    # TODO: Calculate SubApertures with BIOMASS Data.
    return op.prod_path


def pipeline_nisar(product_path: Path, output_dir: Path):
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

    global GPT_PATH, GRID_PATH, DB_DIR
    if args.gpt_path:
        GPT_PATH = args.gpt_path
    if args.grid_path:
        GRID_PATH = args.grid_path
    if args.db_dir:
        DB_DIR = args.db_dir

    product_path = Path(args.product_path)
    output_dir = Path(args.output_dir)
    product_wkt = args.product_wkt
    cuts_outdir = Path(args.cuts_outdir)
    grid_geoj_path = Path(GRID_PATH) if GRID_PATH else None
    product_mode = args.prod_mode
    gpt_memory = args.gpt_memory
    gpt_parallelism = args.gpt_parallelism

    # STEP1:
    if prepro:
        intermediate_product = ROUTER_PIPE[product_mode](
            product_path,
            output_dir,
            gpt_memory=gpt_memory,
            gpt_parallelism=gpt_parallelism,
        )
        print(f"Intermediate processed product located at: {intermediate_product}")
        assert Path(intermediate_product).exists(), f"Intermediate product {intermediate_product} does not exist."

    # STEP2:
    if tiling:
        # ------ Cutting according to the tile griding system: UTM / WGS84 Auto ------
        print(f'Checking points within polygon: {product_wkt}')
        assert grid_geoj_path is not None and grid_geoj_path.exists(), 'grid_10km.geojson does not exist.'
        # step 1: check the contained grid points in the prod
        contained = check_points_in_polygon(product_wkt, geojson_path=grid_geoj_path)
        if not contained:
            print('No grid points contained within the provided WKT.')
            raise ValueError('No grid points contained; check WKT and grid CRS alignment.')
        # step 2: Build the rectangles for cutting
        rectangles = rectanglify(contained)
        if not rectangles:
            print('No rectangles could be formed from contained points.')
            raise ValueError('No rectangles formed; check WKT coverage and grid alignment.')
        product_path = Path(intermediate_product)
        name = extract_product_id(product_path.as_posix()) if product_mode != 'NISAR' else product_path.stem
        if name is None:
            raise ValueError(f"Could not extract product id from: {product_path}")
        
        for rect in rectangles: # CUT!
            geo_region = rectangle_to_wkt(rect)
            if product_mode != 'NISAR':
                final_product = subset(
                    product_path,
                    cuts_outdir / name,
                    output_name=rect['BL']['properties']['name'],
                    geo_region=geo_region,
                    gpt_memory=gpt_memory,
                    gpt_parallelism=gpt_parallelism,
                )
                print(f"Final processed product located at: {final_product}")
            else:
                reader = NISARReader(product_path.as_posix())
                cutter = NISARCutter(reader)
                subset_data = cutter.cut_by_wkt(geo_region, "HH", apply_mask=False)
                nisar_tile_path = cuts_outdir / name / f"{rect['BL']['properties']['name']}.tiff"
                cutter.save_subset(subset_data, nisar_tile_path)
                # TODO: write write method to save to h5.
                print(f"Final processed NISAR tile saved at: {nisar_tile_path}")
                
                
                
        total_tiles = len(rectangles)
        num_cuts = list(Path(cuts_outdir / name).rglob('*.h5'))
        assert total_tiles == len(num_cuts), f"Expected {total_tiles} tiles, but found {len(num_cuts)}."
        
            
    # STEP3:
    # Database indexing
    if db_indexing:
        cuts_folder = cuts_outdir / name
        db = create_tile_database(cuts_folder.as_posix(), DB_DIR) # type: ignore
        assert not db.empty, "Database creation failed, resulting DataFrame is empty."
        print("Database created successfully.")
        
    sys.exit(0)
# ========================================================================================================================================  





if __name__ == "__main__":
    main()
        
