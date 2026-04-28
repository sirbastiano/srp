"""This script provides utility functions for extracting Well-Known Text (WKT) representations of geographic footprints
from satellite product metadata. It includes functions for extracting WKT polygons from Sentinel-1 products using
the Copernicus Data Search API and from Terrasar-X product XML files.
"""

import re
from pathlib import Path
import xml.etree.ElementTree as ET

def _parse_gml_coordinates(text: str, axis_order: str) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for token in text.strip().split():
        if not token:
            continue
        parts = token.split(',')
        if len(parts) < 2:
            continue
        first = float(parts[0])
        second = float(parts[1])
        if axis_order == 'latlon':
            lat, lon = first, second
        else:
            lon, lat = first, second
        points.append((lon, lat))
    return points


def _parse_gml_pos_list(text: str, axis_order: str) -> list[tuple[float, float]]:
    values = [float(value) for value in text.replace(',', ' ').split()]
    points: list[tuple[float, float]] = []
    for i in range(0, len(values) - 1, 2):
        first = values[i]
        second = values[i + 1]
        if axis_order == 'latlon':
            lat, lon = first, second
        else:
            lon, lat = first, second
        points.append((lon, lat))
    return points


def sentinel1_wkt_extractor_manifest(
    product_path: str | Path,
    display_results: bool = False,
    verbose: bool = False,
    axis_order: str = 'latlon',
) -> str | None:
    """
    Extract WKT footprint from Sentinel-1 SAFE manifest (offline).

    Args:
        product_path (str | Path): Path to the Sentinel-1 SAFE folder or manifest.safe file.
        display_results (bool): Whether to display extracted coordinates. Defaults to False.
        verbose (bool): Whether to print debug output. Defaults to False.
        axis_order (str): Coordinate order in the manifest ('latlon' or 'lonlat'). Defaults to 'latlon'.

    Returns:
        str | None: WKT representation of the footprint polygon, or None if not found.
    """
    if axis_order not in {'latlon', 'lonlat'}:
        raise ValueError("axis_order must be 'latlon' or 'lonlat'.")

    if isinstance(product_path, str):
        product_path = Path(product_path)

    if product_path.is_dir():
        manifest_path = product_path / 'manifest.safe'
    elif product_path.name.lower() == 'manifest.safe':
        manifest_path = product_path
    elif product_path.parent.suffix.upper() == '.SAFE':
        manifest_path = product_path.parent / 'manifest.safe'
    else:
        manifest_path = product_path

    if not manifest_path.exists():
        if verbose:
            print(f"Manifest not found: {manifest_path}")
        return None

    if verbose:
        print(f"Reading manifest: {manifest_path}")

    tree = ET.parse(manifest_path)
    root = tree.getroot()
    namespaces = {
        'safe': 'http://www.esa.int/safe/sentinel-1.0',
        'gml': 'http://www.opengis.net/gml',
    }

    from shapely.geometry import MultiPolygon, Polygon

    polygons = []
    for footprint in root.findall('.//safe:footPrint', namespaces):
        coords_el = footprint.find('.//gml:coordinates', namespaces)
        if coords_el is not None and coords_el.text:
            coords = _parse_gml_coordinates(coords_el.text, axis_order=axis_order)
        else:
            poslist_el = footprint.find('.//gml:posList', namespaces)
            if poslist_el is None or not poslist_el.text:
                continue
            coords = _parse_gml_pos_list(poslist_el.text, axis_order=axis_order)

        if len(coords) < 3:
            continue
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        polygons.append(Polygon(coords))

    if not polygons:
        if verbose:
            print("No footprint coordinates found in manifest.")
        return None

    if len(polygons) == 1:
        polygon = polygons[0]
    else:
        polygon = MultiPolygon(polygons).convex_hull

    wkt_polygon = polygon.wkt
    if display_results:
        print(wkt_polygon)
    if verbose:
        print(f"WKT Polygon: {wkt_polygon}")
    return wkt_polygon





def sentinel1_wkt_extractor_cdse(product_name: str, display_results: bool = False, verbose: bool = False) -> str | None:
    """
    Extract WKT footprint from Sentinel-1 product using Copernicus Data Search.

    Args:
        product_name (str): Name of the Sentinel-1 product to search for.
        display_results (bool): Whether to display search results. Defaults to False.

    Returns:
        str | None: WKT representation of the product footprint polygon, or None if not found.
    """
    from shapely.geometry import MultiPolygon, shape
    from phidown.search import CopernicusDataSearcher

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
        if isinstance(polygon, MultiPolygon):
            # Take the convex hull to merge into a single polygon
            polygon = polygon.convex_hull
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
    if not product_path.exists():
        raise FileNotFoundError(f"Product path {product_path} does not exist.")
    if product_path.suffix.lower() != '.xml':
        raise ValueError(f"Product path {product_path} is not an XML file.")
    
    tree = ET.parse(product_path)
    root = tree.getroot()

    corners = []
    for corner in root.findall('.//sceneCornerCoord'):
        lon_text = corner.findtext('lon')
        lat_text = corner.findtext('lat')
        if lon_text is None or lat_text is None:
            continue
        corners.append((float(lon_text), float(lat_text)))

    if len(corners) < 3:
        raise ValueError(f"Could not find at least three sceneCornerCoord lon/lat pairs in {product_path}.")

    # Close the polygon by repeating the first point at the end
    if corners:
        corners.append(corners[0])

    # Create WKT polygon string
    return f"POLYGON(({', '.join(f'{lon} {lat}' for lon, lat in corners)}))"


def _safe_dir_from_product_path(product_path: str | Path) -> Path:
    if isinstance(product_path, str):
        product_path = Path(product_path)

    if product_path.is_dir() and product_path.suffix.upper() == '.SAFE':
        return product_path
    if product_path.name.lower() == 'manifest.safe':
        return product_path.parent
    if product_path.parent.suffix.upper() == '.SAFE':
        return product_path.parent
    return product_path


def sentinel1_swath_wkt_extractor_safe(
    product_path: str | Path,
    swath: str,
    display_results: bool = False,
    verbose: bool = False,
) -> str | None:
    """
    Extract a swath-specific WKT footprint from Sentinel-1 SAFE annotation XMLs.

    The function collects geolocation grid points from annotation XML files that
    belong to the requested swath (for example ``IW1``) and returns the convex
    hull of those geographic points as a lon/lat WKT polygon.
    """
    safe_dir = _safe_dir_from_product_path(product_path)
    annotation_dir = safe_dir / 'annotation'
    if not annotation_dir.exists():
        if verbose:
            print(f'Annotation directory not found: {annotation_dir}')
        return None

    swath_norm = swath.upper()
    annotation_paths = sorted(
        path for path in annotation_dir.glob('*.xml')
        if swath_norm.lower() in path.name.lower()
    )
    if not annotation_paths:
        if verbose:
            print(f'No annotation XMLs found for swath {swath_norm} in {annotation_dir}')
        return None

    points: list[tuple[float, float]] = []
    for xml_path in annotation_paths:
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for point_el in root.findall('.//geolocationGridPoint'):
                lat_text = point_el.findtext('latitude')
                lon_text = point_el.findtext('longitude')
                if lat_text is None or lon_text is None:
                    continue
                points.append((float(lon_text), float(lat_text)))
        except Exception as exc:
            if verbose:
                print(f'Skipping {xml_path}: {type(exc).__name__}: {exc}')

    if len(points) < 3:
        if verbose:
            print(f'Not enough geolocation points for swath {swath_norm}')
        return None

    from shapely.geometry import MultiPoint

    polygon = MultiPoint(points).convex_hull
    if polygon.is_empty:
        return None

    wkt_polygon = polygon.wkt
    if display_results:
        print(wkt_polygon)
    if verbose:
        print(f'Swath WKT Polygon ({swath_norm}): {wkt_polygon}')
    return wkt_polygon




def nisar_wkt_extractor(product_path: Path) -> str:
    """Extract a 2D bounding polygon WKT from a NISAR GSLC .h5 file.

    The NISAR GSLC product stores an OGR-compatible 3D WKT polygon
    (lon, lat, height) at ``science/LSAR/identification/boundingPolygon``.
    This function reads it, strips the Z (height) component, and returns
    a standard 2D WKT POLYGON string suitable for downstream tiling.

    Args:
        product_path: Path to the NISAR GSLC HDF5 file.

    Returns:
        2D WKT POLYGON string in EPSG:4326.
    """
    import h5py

    with h5py.File(str(product_path), 'r') as f:
        raw = f['science/LSAR/identification/boundingPolygon'][()]
        if isinstance(raw, bytes):
            raw = raw.decode('utf-8')

    def _strip_z(wkt_3d: str) -> str:
        def _replace_coords(m):
            pairs = []
            for pt in m.group(1).split(','):
                parts = pt.strip().split()
                if len(parts) >= 2:
                    pairs.append(f'{parts[0]} {parts[1]}')
            return '(' + ', '.join(pairs) + ')'
        return re.sub(r'\(([^()]+)\)', _replace_coords, wkt_3d)

    wkt_2d = _strip_z(raw)
    print(f'Extracted NISAR bounding polygon (2D, EPSG:4326): {wkt_2d[:120]}...')
    return wkt_2d


def _safe_dir_from_product_path(product_path: str | Path) -> Path:
    if isinstance(product_path, str):
        product_path = Path(product_path)

    if product_path.is_dir() and product_path.suffix.upper() == '.SAFE':
        return product_path
    if product_path.name.lower() == 'manifest.safe':
        return product_path.parent
    if product_path.parent.suffix.upper() == '.SAFE':
        return product_path.parent
    return product_path


def sentinel1_swath_wkt_extractor_safe(
    product_path: str | Path,
    swath: str,
    display_results: bool = False,
    verbose: bool = False,
) -> str | None:
    """
    Extract a swath-specific WKT footprint from Sentinel-1 SAFE annotation XMLs.

    The function collects geolocation grid points from annotation XML files that
    belong to the requested swath (for example ``IW1``) and returns the convex
    hull of those geographic points as a lon/lat WKT polygon.
    """
    safe_dir = _safe_dir_from_product_path(product_path)
    annotation_dir = safe_dir / 'annotation'
    if not annotation_dir.exists():
        if verbose:
            print(f'Annotation directory not found: {annotation_dir}')
        return None

    swath_norm = swath.upper()
    annotation_paths = sorted(
        path for path in annotation_dir.glob('*.xml')
        if swath_norm.lower() in path.name.lower()
    )
    if not annotation_paths:
        if verbose:
            print(f'No annotation XMLs found for swath {swath_norm} in {annotation_dir}')
        return None

    points: list[tuple[float, float]] = []
    for xml_path in annotation_paths:
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for point_el in root.findall('.//geolocationGridPoint'):
                lat_text = point_el.findtext('latitude')
                lon_text = point_el.findtext('longitude')
                if lat_text is None or lon_text is None:
                    continue
                points.append((float(lon_text), float(lat_text)))
        except Exception as exc:
            if verbose:
                print(f'Skipping {xml_path}: {type(exc).__name__}: {exc}')

    if len(points) < 3:
        if verbose:
            print(f'Not enough geolocation points for swath {swath_norm}')
        return None

    from shapely.geometry import MultiPoint

    polygon = MultiPoint(points).convex_hull
    if polygon.is_empty:
        return None

    wkt_polygon = polygon.wkt
    if display_results:
        print(wkt_polygon)
    if verbose:
        print(f'Swath WKT Polygon ({swath_norm}): {wkt_polygon}')
    return wkt_polygon


if __name__ == "__main__":
    import argparse 
    
    parser = argparse.ArgumentParser(description="Extract WKT footprint from satellite product metadata.")
    parser.add_argument('--mode', type=str, required=True, help="Mode of operation: 'S1TOPS', 'S1STRIP', 'BM', 'TSX', etc.")
    parser.add_argument('--product-path', type=str, required=True, help="Name of the satellite product.")
    args = parser.parse_args()
    
    
    MODE = args.mode  # Example mode, can be 'S1TOPS', 'S1STRIP', 'BM', 'TSX', etc.
    PRODUCT_PATH = Path(args.product_path)
    PRODUCT_NAME = PRODUCT_PATH.name
    
    if MODE == 'S1TOPS' or MODE == 'S1STRIP':
        # Example usage for Sentinel-1 product
        sentinel1_product_name = PRODUCT_NAME
        wkt_sentinel1 = sentinel1_wkt_extractor_cdse(sentinel1_product_name, display_results=False)
        print(f"\nSentinel-1 WKT Polygon:\n{wkt_sentinel1}")
    
    elif MODE == 'TSX':
        # Example usage for Terrasar-X product
        terrasar_product_path = PRODUCT_PATH
        wkt_terrasar = terrasar_wkt_extractor(terrasar_product_path)
        print(f"\nTerrasar-X WKT Polygon:\n{wkt_terrasar}")
