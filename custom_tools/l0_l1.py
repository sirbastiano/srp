import random
import time
from typing import Optional, Sequence
from phidown.search import CopernicusDataSearcher
import pandas as pd



def geojson_to_polygon_wkt(geometry, *, on_multipolygon="first"):
    """
    Convert a GeoJSON geometry to a POLYGON WKT string.

    This function guarantees a POLYGON WKT output. Behavior by geometry type:
    - Polygon: serialized directly to POLYGON (2D or Z).
    - MultiPolygon: by default takes the first polygon and returns it as POLYGON.
      You can control this with `on_multipolygon`.
    - Any other geometry type: raises ValueError.

    Parameters
    ----------
    geometry : dict
        A GeoJSON-like geometry dictionary with keys:
        - 'type' (str): One of {'Polygon','MultiPolygon'} is supported; others raise.
        - 'coordinates' (list): For 'Polygon', a list of linear rings; for 'MultiPolygon',
          a list of polygons (each a list of rings). Rings are expected in GeoJSON order
          [x, y] or [x, y, z]. Rings should be closed (first == last); this function
          does not enforce closure.
    on_multipolygon : {'first', 'error'}, optional
        Behavior when input is a MultiPolygon:
        - 'first' (default): Use the first polygon within the MultiPolygon.
        - 'error': Raise ValueError if there is more than one polygon.
        Note: True topological merging (dissolve/union) is not performed.

    Returns
    -------
    str
        A POLYGON (or 'POLYGON Z') WKT string.

    Raises
    ------
    ValueError
        If geometry type is unsupported, structure is invalid, or MultiPolygon handling
        is set to 'error' with multiple polygons.

    Notes
    -----
    - This function does not perform geometric operations (e.g., union/merge).
    - Presence of any Z value in the chosen polygon promotes output to 'POLYGON Z'.
    """

    def _is_3d_coords(obj):
        """Return True if any coordinate has 3 elements."""
        found = False

        def _walk(o):
            nonlocal found
            if found:
                return
            if isinstance(o, (list, tuple)):
                if o and all(isinstance(v, (int, float)) for v in o):
                    if len(o) >= 3:
                        found = True
                else:
                    for item in o:
                        _walk(item)

        _walk(obj)
        return found

    def _fmt_num(n):
        """Format numbers compactly, removing trailing zeros and unnecessary decimals."""
        if isinstance(n, int):
            return str(n)
        return f"{float(n):.15g}"

    def _fmt_coord(coord):
        return " ".join(_fmt_num(c) for c in coord[:3])  # x y or x y z

    def _fmt_ring(ring):
        return f"({_fmt_coord_list(ring)})"

    def _fmt_coord_list(coords):
        return ", ".join(_fmt_coord(c) for c in coords)

    gtype = geometry.get("type")
    if not gtype:
        raise ValueError("Geometry missing 'type'")

    if gtype == "Polygon":
        rings = geometry.get("coordinates")
        if not isinstance(rings, list):
            raise ValueError("Polygon 'coordinates' must be a list of rings")
        dim = " Z" if _is_3d_coords(rings) else ""
        return f"POLYGON{dim} ({', '.join(_fmt_ring(r) for r in rings)})"

    if gtype == "MultiPolygon":
        polys = geometry.get("coordinates")
        if not isinstance(polys, list) or not polys:
            raise ValueError("MultiPolygon 'coordinates' must be a non-empty list of polygons")
        if on_multipolygon == "error" and len(polys) != 1:
            raise ValueError("MultiPolygon has multiple polygons; set on_multipolygon='first' to pick the first")
        chosen = polys[0]
        if not isinstance(chosen, list):
            raise ValueError("Invalid MultiPolygon structure: expected list of polygons (list of rings)")
        dim = " Z" if _is_3d_coords(chosen) else ""
        return f"POLYGON{dim} ({', '.join(_fmt_ring(r) for r in chosen)})"

    raise ValueError(f"Unsupported geometry type for polygon output: {gtype}")



def get_corresponding_level_0(
    product,
    *,
    searcher_cls=CopernicusDataSearcher,
    persist_path: Optional[str] = None,
    existing_pairs: Optional[Sequence[dict]] = None,
    sleep_range=(5, 10),
    display_top: int = 1,
):
    """
    Given a SENTINEL-1 SLC product, locate the corresponding LEVEL0 product.

    Parameters
    ----------
    product : object
        Object with a `name` attribute for the SLC product.
    searcher_cls : type, optional
        Factory for CopernicusDataSearcher instances (allows dependency injection/testability).
    persist_path : str, optional
        If provided, append the resulting pair to `existing_pairs` (or a new list) and
        persist with `pd.to_pickle`.
    existing_pairs : sequence of dict, optional
        Collection that will be appended with the new mapping when `persist_path` is set.
    sleep_range : tuple(int, int), optional
        Inclusive range (seconds) for random sleep to throttle API calls.
    display_top : int, optional
        Number of rows to show via `display_results` for the name search.

    Returns
    -------
    dict
        Mapping of `{L1_product_name: L0_product_name}`.

    Raises
    ------
    ValueError
        If the SLC product has no name or no matching LEVEL0 product is found.
    """
    product_name = getattr(product, "name", None)
    if not product_name:
        raise ValueError("Product is missing a `name` attribute.")

    # Query by exact name to retrieve metadata
    searcher_by_name = searcher_cls()
    df_exact = searcher_by_name.query_by_name(product_name=product_name)
    if df_exact.empty:
        raise ValueError(f"No metadata found for product '{product_name}'.")

    if display_top:
        searcher_by_name.display_results(top_n=display_top)

    geo_footprint = df_exact.iloc[0]["GeoFootprint"]
    content_date = df_exact.iloc[0]["ContentDate"]
    start = content_date["Start"]
    end = content_date["End"]

    # Query for LEVEL0 products matching the footprint/time window
    searcher = searcher_cls()
    searcher.query_by_filter(
        collection_name="SENTINEL-1",
        product_type=None,
        orbit_direction=None,
        cloud_cover_threshold=None,
        aoi_wkt=geojson_to_polygon_wkt(geo_footprint),
        start_date=start,
        end_date=end,
        top=1000,
        attributes={"processingLevel": "LEVEL0"},
    )

    df = searcher.execute_query()
    if df.empty:
        raise ValueError(f"No LEVEL0 candidate found for '{product_name}'.")

    l0_product_name = df.sample(n=1, random_state=random.randint(0, 1_000_000))["Name"].iat[0]
    mapping = {product_name: l0_product_name}

    if persist_path:
        pairs = list(existing_pairs) if existing_pairs is not None else []
        pairs.append(mapping)
        pd.to_pickle(pairs, persist_path)

    if sleep_range:
        time.sleep(random.uniform(*sleep_range))

    return mapping
