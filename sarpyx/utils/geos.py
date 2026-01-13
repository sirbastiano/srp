"""Point containment checker for grid points within WKT polygons."""

import json
from pathlib import Path
from typing import List, Dict, Any

from shapely import wkt
from shapely.geometry import Point, Polygon


def check_points_in_polygon(
    wkt_polygon: str,
    geojson_path: str = '/Data_large/SARGFM/grid_10km.geojson'
) -> List[Dict[str, Any]]:
    """
    Check which points from the GeoJSON file are contained within the given WKT polygon.
    
    Args:
        wkt_polygon: A WKT (Well-Known Text) string representing a polygon
        geojson_path: Path to the GeoJSON file containing point features
    
    Returns:
        List of feature dictionaries containing points that fall within the polygon.
        Each feature includes properties (name, row, col, row_idx, col_idx, utm_zone, epsg)
        and geometry (coordinates).
    
    Example:
        >>> wkt_poly = "POLYGON((10 10, 10 20, 20 20, 20 10, 10 10))"
        >>> contained_points = check_points_in_polygon(wkt_poly)
        >>> print(f"Found {len(contained_points)} points")
    """
    # Parse the WKT polygon
    polygon = wkt.loads(wkt_polygon)
    
    # Load the GeoJSON file
    with open(geojson_path, 'r') as f:
        geojson_data = json.load(f)
    
    # Use list comprehension for faster filtering of contained points
    contained_features = [
        feature for feature in geojson_data['features']
        if polygon.contains(Point(*feature['geometry']['coordinates']))
    ]
    
    return contained_features


def get_point_names(
    wkt_polygon: str,
    geojson_path: str = '/Data_large/SARGFM/grid_10km.geojson'
) -> List[str]:
    """
    Get the names of points contained within the given WKT polygon.
    
    Args:
        wkt_polygon: A WKT (Well-Known Text) string representing a polygon
        geojson_path: Path to the GeoJSON file containing point features
    
    Returns:
        List of point names (e.g., ["946D_176L", "946D_175L", ...])
    """
    contained_features = check_points_in_polygon(wkt_polygon, geojson_path)
    return [feature['properties']['name'] for feature in contained_features]


def get_point_coordinates(
    wkt_polygon: str,
    geojson_path: str = '/Data_large/SARGFM/grid_10km.geojson'
) -> List[tuple]:
    """
    Get the coordinates of points contained within the given WKT polygon.
    
    Args:
        wkt_polygon: A WKT (Well-Known Text) string representing a polygon
        geojson_path: Path to the GeoJSON file containing point features
    
    Returns:
        List of coordinate tuples [(lon, lat), ...]
    """
    contained_features = check_points_in_polygon(wkt_polygon, geojson_path)
    return [
        tuple(feature['geometry']['coordinates'])
        for feature in contained_features
    ]


class GridNavigator:
    """
    Grid navigation utilities for moving between adjacent grid points.
    This module provides GridNavigator which navigates using only point names
    without requiring the Grid object.

    The grid naming convention:
    - Rows: XD (south of equator), 0U (equator), XU (north of equator)
    - Columns: XL (west of prime meridian), 0R (prime meridian), XR (east of prime meridian)

    Navigation:
    - North (up): Row number decreases in D hemisphere, increases in U hemisphere
    - South (down): Row number increases in D hemisphere, decreases in U hemisphere
    - East (right): Column number increases in R hemisphere, decreases in L hemisphere
    - West (left): Column number decreases in R hemisphere, increases in L hemisphere
    """
    
    def __init__(self):
        """Initialize navigator."""
        pass
    
    @staticmethod
    def _parse_row(row: str) -> tuple[int, str]:
        """Parse row name into number and hemisphere.
        
        Args:
            row: Row name like '298D' or '5U'
            
        Returns:
            Tuple of (number, hemisphere) e.g., (298, 'D')
        """
        if row.endswith('U'):
            return int(row[:-1]), 'U'
        elif row.endswith('D'):
            return int(row[:-1]), 'D'
        else:
            raise ValueError(f'Invalid row format: {row}')
    
    @staticmethod
    def _parse_col(col: str) -> tuple[int, str]:
        """Parse column name into number and hemisphere.
        
        Args:
            col: Column name like '323R' or '10L'
            
        Returns:
            Tuple of (number, hemisphere) e.g., (323, 'R')
        """
        if col.endswith('R'):
            return int(col[:-1]), 'R'
        elif col.endswith('L'):
            return int(col[:-1]), 'L'
        else:
            raise ValueError(f'Invalid column format: {col}')
    
    @staticmethod
    def _make_row(num: int, hemisphere: str) -> str:
        """Create row name from number and hemisphere.
        
        Args:
            num: Row number
            hemisphere: 'U' or 'D'
            
        Returns:
            Row name like '298D' or '5U'
        """
        return f'{num}{hemisphere}'
    
    @staticmethod
    def _make_col(num: int, hemisphere: str) -> str:
        """Create column name from number and hemisphere.
        
        Args:
            num: Column number
            hemisphere: 'R' or 'L'
            
        Returns:
            Column name like '323R' or '10L'
        """
        return f'{num}{hemisphere}'
    
    def move_up(self, row: str, col: str) -> str:
        """Get the name of the point immediately above (northward).
        
        Moving north:
        - In D hemisphere: 298D → 297D → ... → 1D → 0U
        - In U hemisphere: 0U → 1U → 2U → ...
        
        Args:
            row: Current row (e.g., '298D', '5U')
            col: Current column (e.g., '323R', '10L')
        
        Returns:
            Name of the upper point
        """
        num, hemisphere = self._parse_row(row)
        
        if hemisphere == 'D':
            # Moving north in southern hemisphere
            if num == 1:
                # Cross to northern hemisphere
                new_row = '0U'
            else:
                # Stay in southern hemisphere
                new_row = self._make_row(num - 1, 'D')
        else:  # hemisphere == 'U'
            # Moving north in northern hemisphere
            new_row = self._make_row(num + 1, 'U')
        
        return f'{new_row}_{col}'

    def move_down(self, row: str, col: str) -> str:
        """Get the name of the point immediately below (southward).
        
        Moving south:
        - In U hemisphere: 298U → 297U → ... → 1U → 0U
        - In D hemisphere: 0U → 1D → 2D → ...
        
        Args:
            row: Current row (e.g., '298D', '5U')
            col: Current column (e.g., '323R', '10L')
        
        Returns:
            Name of the lower point
        """
        num, hemisphere = self._parse_row(row)
        
        if hemisphere == 'U':
            # Moving south in northern hemisphere
            if num == 0:
                # Cross to southern hemisphere
                new_row = '1D'
            elif num == 1:
                # Move to equator
                new_row = '0U'
            else:
                # Stay in northern hemisphere
                new_row = self._make_row(num - 1, 'U')
        else:  # hemisphere == 'D'
            # Moving south in southern hemisphere
            new_row = self._make_row(num + 1, 'D')
        
        return f'{new_row}_{col}'

    def move_right(self, row: str, col: str) -> str:
        """Get the name of the point immediately to the right (eastward).
        
        Moving east:
        - In L hemisphere: 10L → 9L → ... → 1L → 0R
        - In R hemisphere: 0R → 1R → 2R → ...
        
        Args:
            row: Current row (e.g., '298D', '5U')
            col: Current column (e.g., '323R', '10L')
        
        Returns:
            Name of the right point
        """
        num, hemisphere = self._parse_col(col)
        
        if hemisphere == 'L':
            # Moving east in western hemisphere
            if num == 1:
                # Cross to eastern hemisphere
                new_col = '0R'
            else:
                # Stay in western hemisphere
                new_col = self._make_col(num - 1, 'L')
        else:  # hemisphere == 'R'
            # Moving east in eastern hemisphere
            new_col = self._make_col(num + 1, 'R')
        
        return f'{row}_{new_col}'

    def move_left(self, row: str, col: str) -> str:
        """Get the name of the point immediately to the left (westward).
        
        Moving west:
        - In R hemisphere: 10R → 9R → ... → 1R → 0R
        - In L hemisphere: 0R → 1L → 2L → ...
        
        Args:
            row: Current row (e.g., '298D', '5U')
            col: Current column (e.g., '323R', '10L')
        
        Returns:
            Name of the left point
        """
        num, hemisphere = self._parse_col(col)
        
        if hemisphere == 'R':
            # Moving west in eastern hemisphere
            if num == 0:
                # Cross to western hemisphere
                new_col = '1L'
            else:
                # Stay in eastern hemisphere
                new_col = self._make_col(num - 1, 'R')
        else:  # hemisphere == 'L'
            # Moving west in western hemisphere
            new_col = self._make_col(num + 1, 'L')
        
        return f'{row}_{new_col}'

    def move_up_right(self, row: str, col: str) -> str:
        """Get the name of the point diagonally upper-right (north-east).
        
        Args:
            row: Current row (e.g., '298D', '5U')
            col: Current column (e.g., '323R', '10L')
        
        Returns:
            Name of the upper-right point
        """
        # Move up
        upper_name = self.move_up(row, col)
        upper_row, upper_col = upper_name.split('_')
        
        # Then move right
        return self.move_right(upper_row, upper_col)

    def move_up_left(self, row: str, col: str) -> str:
        """Get the name of the point diagonally upper-left (north-west).
        
        Args:
            row: Current row (e.g., '298D', '5U')
            col: Current column (e.g., '323R', '10L')
        
        Returns:
            Name of the upper-left point
        """
        upper_name = self.move_up(row, col)
        upper_row, upper_col = upper_name.split('_')
        return self.move_left(upper_row, upper_col)

    def move_down_right(self, row: str, col: str) -> str:
        """Get the name of the point diagonally lower-right (south-east).
        
        Args:
            row: Current row (e.g., '298D', '5U')
            col: Current column (e.g., '323R', '10L')
        
        Returns:
            Name of the lower-right point
        """
        lower_name = self.move_down(row, col)
        lower_row, lower_col = lower_name.split('_')
        return self.move_right(lower_row, lower_col)

    def move_down_left(self, row: str, col: str) -> str:
        """Get the name of the point diagonally lower-left (south-west).
        
        Args:
            row: Current row (e.g., '298D', '5U')
            col: Current column (e.g., '323R', '10L')
        
        Returns:
            Name of the lower-left point
        """
        lower_name = self.move_down(row, col)
        lower_row, lower_col = lower_name.split('_')
        return self.move_left(lower_row, lower_col)


def is_point_in_contained(name, contained):
    for feature in contained:
        props = feature['properties']
        if props['name'] == name:
            return True
    return False


def rectangle_to_wkt(rectangle: dict) -> str:
    """Convert a rectangle dictionary to WKT POLYGON format.
    
    Takes a dictionary containing four corner points (TL, TR, BR, BL) with their
    coordinates and creates a closed polygon in Well-Known Text format. The polygon
    is closed by repeating the first point at the end.
    
    Args:
        rectangle: Dictionary with keys 'TL', 'TR', 'BR', 'BL', each containing
                  a GeoJSON Feature with geometry.coordinates [lon, lat].
                  
    Returns:
        WKT POLYGON string in format 'POLYGON ((lon lat, lon lat, ...))'
        
    Example:
        >>> rect = {
        ...     'TL': {'geometry': {'coordinates': [32.37, -26.68]}},
        ...     'TR': {'geometry': {'coordinates': [32.47, -26.68]}},
        ...     'BR': {'geometry': {'coordinates': [32.49, -26.77]}},
        ...     'BL': {'geometry': {'coordinates': [32.39, -26.77]}}
        ... }
        >>> rectangle_to_wkt(rect)
        'POLYGON ((32.37 -26.68, 32.47 -26.68, 32.49 -26.77, 32.39 -26.77, 32.37 -26.68))'
    """
    # Extract coordinates in clockwise order: TL -> TR -> BR -> BL -> TL (close polygon)
    corners = ['TL', 'TR', 'BR', 'BL']
    coords = [rectangle[corner]['geometry']['coordinates'] for corner in corners]
    
    # Close the polygon by adding the first point at the end
    coords.append(coords[0])
    
    # Format as WKT: 'lon lat' pairs separated by commas
    coord_strings = [f'{lon} {lat}' for lon, lat in coords]
    wkt = f"POLYGON (({', '.join(coord_strings)}))"
    
    return wkt
























# if __name__ == '__main__':
#     print('='*80)
#     print('GRID NAVIGATION TESTS (NAME-BASED)')
#     print('='*80)
#     print('\nNavigator uses only point names - no Grid object required!\n')
    
#     # Create navigator (no Grid needed!)
#     navigator = GridNavigator()
    
#     # Test 1: Basic movements in Southern Hemisphere
#     print('='*80)
#     print('TEST 1: Basic movements in Southern Hemisphere')
#     print('='*80)
#     test_row, test_col = '100D', '50R'
#     print(f'Starting point: {test_row}_{test_col}')
    
#     up = navigator.move_up(test_row, test_col)
#     print(f'  ↑ Up (north):    {up}')
    
#     down = navigator.move_down(test_row, test_col)
#     print(f'  ↓ Down (south):  {down}')
    
#     right = navigator.move_right(test_row, test_col)
#     print(f'  → Right (east):  {right}')
    
#     left = navigator.move_left(test_row, test_col)
#     print(f'  ← Left (west):   {left}')
    
#     up_right = navigator.move_up_right(test_row, test_col)
#     print(f'  ↗ Up-right (NE): {up_right}')
    
#     # Test 2: Basic movements in Northern Hemisphere
#     print('\n' + '='*80)
#     print('TEST 2: Basic movements in Northern Hemisphere')
#     print('='*80)
#     test_row, test_col = '50U', '100R'
#     print(f'Starting point: {test_row}_{test_col}')
    
#     up = navigator.move_up(test_row, test_col)
#     print(f'  ↑ Up (north):    {up}')
    
#     down = navigator.move_down(test_row, test_col)
#     print(f'  ↓ Down (south):  {down}')
    
#     right = navigator.move_right(test_row, test_col)
#     print(f'  → Right (east):  {right}')
    
#     left = navigator.move_left(test_row, test_col)
#     print(f'  ← Left (west):   {left}')
    
#     # Test 3: Crossing equator from south to north
#     print('\n' + '='*80)
#     print('TEST 3: Crossing the equator (South → North)')
#     print('='*80)
#     test_row, test_col = '1D', '100R'
#     print(f'Starting point: {test_row}_{test_col}')
    
#     up = navigator.move_up(test_row, test_col)
#     print(f'  ↑ Up:   {test_row}_{test_col} → {up}')
#     assert up == '0U_100R', f'Expected 0U_100R, got {up}'
#     print(f'  ✓ Successfully crossed from D to U!')
    
#     # Test 4: Crossing equator from north to south
#     print('\n' + '='*80)
#     print('TEST 4: Crossing the equator (North → South)')
#     print('='*80)
#     test_row, test_col = '0U', '100R'
#     print(f'Starting point: {test_row}_{test_col}')
    
#     down = navigator.move_down(test_row, test_col)
#     print(f'  ↓ Down: {test_row}_{test_col} → {down}')
#     assert down == '1D_100R', f'Expected 1D_100R, got {down}'
#     print(f'  ✓ Successfully crossed from U to D!')
    
#     # Test 5: Moving through equator
#     print('\n' + '='*80)
#     print('TEST 5: Sequential movement through equator')
#     print('='*80)
#     test_row, test_col = '2D', '50R'
#     print(f'Starting: {test_row}_{test_col}')
    
#     current_row, current_col = test_row, test_col
#     for i in range(5):
#         next_point = navigator.move_up(current_row, current_col)
#         print(f'  Step {i+1}: {current_row}_{current_col} → {next_point}')
#         current_row, current_col = next_point.split('_')
#     print(f'  ✓ Traversed: 2D → 1D → 0U → 1U → 2U → 3U')
    
#     # Test 6: Crossing prime meridian from west to east
#     print('\n' + '='*80)
#     print('TEST 6: Crossing prime meridian (West → East)')
#     print('='*80)
#     test_row, test_col = '50D', '1L'
#     print(f'Starting point: {test_row}_{test_col}')
    
#     right = navigator.move_right(test_row, test_col)
#     print(f'  → Right: {test_row}_{test_col} → {right}')
#     assert right == '50D_0R', f'Expected 50D_0R, got {right}'
#     print(f'  ✓ Successfully crossed from L to R!')
    
#     # Test 7: Crossing prime meridian from east to west
#     print('\n' + '='*80)
#     print('TEST 7: Crossing prime meridian (East → West)')
#     print('='*80)
#     test_row, test_col = '50D', '0R'
#     print(f'Starting point: {test_row}_{test_col}')
    
#     left = navigator.move_left(test_row, test_col)
#     print(f'  ← Left:  {test_row}_{test_col} → {left}')
#     assert left == '50D_1L', f'Expected 50D_1L, got {left}'
#     print(f'  ✓ Successfully crossed from R to L!')
    
#     # Test 8: Sequential movement through prime meridian
#     print('\n' + '='*80)
#     print('TEST 8: Sequential movement through prime meridian')
#     print('='*80)
#     test_row, test_col = '100U', '2L'
#     print(f'Starting: {test_row}_{test_col}')
    
#     current_row, current_col = test_row, test_col
#     for i in range(5):
#         next_point = navigator.move_right(current_row, current_col)
#         print(f'  Step {i+1}: {current_row}_{current_col} → {next_point}')
#         current_row, current_col = next_point.split('_')
#     print(f'  ✓ Traversed: 2L → 1L → 0R → 1R → 2R → 3R')
    
#     # Test 9: Consistency - round trip
#     print('\n' + '='*80)
#     print('TEST 9: Round-trip consistency')
#     print('='*80)
#     test_row, test_col = '200D', '150R'
#     print(f'Starting point: {test_row}_{test_col}')
    
#     # Up and down
#     up = navigator.move_up(test_row, test_col)
#     up_row, up_col = up.split('_')
#     back_down = navigator.move_down(up_row, up_col)
#     print(f'  Up then down:    {test_row}_{test_col} → {up} → {back_down}')
#     assert back_down == f'{test_row}_{test_col}', 'Should return to original'
#     print(f'  ✓ Passed!')
    
#     # Right and left
#     right = navigator.move_right(test_row, test_col)
#     right_row, right_col = right.split('_')
#     back_left = navigator.move_left(right_row, right_col)
#     print(f'  Right then left: {test_row}_{test_col} → {right} → {back_left}')
#     assert back_left == f'{test_row}_{test_col}', 'Should return to original'
#     print(f'  ✓ Passed!')
    
#     # Test 10: Diagonal movements
#     print('\n' + '='*80)
#     print('TEST 10: Diagonal movements')
#     print('='*80)
#     test_row, test_col = '50D', '50R'
#     print(f'Starting point: {test_row}_{test_col}')
    
#     ne = navigator.move_up_right(test_row, test_col)
#     print(f'  ↗ NE: {ne}')
    
#     nw = navigator.move_up_left(test_row, test_col)
#     print(f'  ↖ NW: {nw}')
    
#     se = navigator.move_down_right(test_row, test_col)
#     print(f'  ↘ SE: {se}')
    
#     sw = navigator.move_down_left(test_row, test_col)
#     print(f'  ↙ SW: {sw}')
    
#     print('\n' + '='*80)
#     print('ALL TESTS COMPLETED')
#     print('='*80)
#     print('\nSUMMARY:')
#     print('- GridNavigator works purely by naming conventions')
#     print('- No Grid object required for navigation')
#     print('- Correctly handles hemisphere transitions (D↔U, L↔R)')
#     print('- Round-trip navigation returns to origin')
#     print('- Diagonal movements combine cardinal directions')
#     print('='*80)
