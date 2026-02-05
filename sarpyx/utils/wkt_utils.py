"""This script provides utility functions for extracting Well-Known Text (WKT) representations of geographic footprints 
from satellite product metadata. It includes functions for extracting WKT polygons from Sentinel-1 products using 
the Copernicus Data Search API and from Terrasar-X product XML files.
"""

from pathlib import Path
from phidown.search import CopernicusDataSearcher
from shapely.geometry import shape
import xml.etree.ElementTree as ET





def sentinel1_wkt_extractor_cdse(product_name: str, display_results: bool = False, verbose: bool = False) -> str | None:
    """
    Extract WKT footprint from Sentinel-1 product using Copernicus Data Search.

    Args:
        product_name (str): Name of the Sentinel-1 product to search for.
        display_results (bool): Whether to display search results. Defaults to False.

    Returns:
        str | None: WKT representation of the product footprint polygon, or None if not found.
    """
    searcher = CopernicusDataSearcher()
    if verbose:
        print(f"Searching for product with exact name: {product_name}\n")
    df_exact = searcher.query_by_name(product_name=product_name)

    if not df_exact.empty:
        if display_results:
            searcher.display_results(top_n=1)
            if verbose:
                print(df_exact)
                print("\nProduct found successfully.")
        
        geofootprint = df_exact['GeoFootprint'].values[0]
        if verbose:
            print(f"Product with GeoFootprint: '{geofootprint}' found.")
        
        polygon = shape(geofootprint)
        wkt_polygon = polygon.wkt
        
        if verbose:
            print(f"\nWKT Polygon:\n{wkt_polygon}")
        return wkt_polygon
    else:
        if verbose:
            print("No product found with the specified name.")
        return None


def terrasar_wkt_extractor(product_path: Path) -> str:
    """
    Extract WKT footprint from Terrasar-X product XML file.

    Args:
        product_path (Path): Path to the input Terrasar-X product XML file.

    Returns:
        str: WKT representation of the product footprint polygon.
    """
    if isinstance(product_path, str):
        product_path = Path(product_path)
    assert product_path.exists(), f"Product path {product_path} does not exist."
    assert product_path.suffix.lower() == '.xml', f"Product path {product_path} is not an XML file."
    
    tree = ET.parse(product_path)
    root = tree.getroot()

    # Find all sceneCornerCoord elements and extract coordinates
    corners = [(float(corner.find('lon').text), float(corner.find('lat').text)) 
               for corner in root.findall('.//sceneCornerCoord')]

    # Close the polygon by repeating the first point at the end
    if corners:
        corners.append(corners[0])

    # Create WKT polygon string
    return f"POLYGON(({', '.join(f'{lon} {lat}' for lon, lat in corners)}))"




if __name__ == "__main__":
    MODE = 'S1TOPS'  # Example mode, can be 'S1TOPS', 'S1STRIP', 'BM', 'TSX', etc.
    if MODE == 'S1TOPS' or MODE == 'S1STRIP':
        # Example usage for Sentinel-1 product
        sentinel1_product_name = "S1A_IW_GRDH_1SDV_20141031T161924_20141031T161949_003076_003856_634E.SAFE"
        wkt_sentinel1 = sentinel1_wkt_extractor_cdse(sentinel1_product_name, display_results=False)
    
    elif MODE == 'TSX':
        # Example usage for Terrasar-X product
        terrasar_product_path = Path("/path/to/terrasar_product.xml")
        wkt_terrasar = terrasar_wkt_extractor(terrasar_product_path)
        print(f"\nTerrasar-X WKT Polygon:\n{wkt_terrasar}")