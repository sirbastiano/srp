from pathlib import Path
from datetime import datetime
from typing import Dict, List, Union, Optional
import xml.etree.ElementTree as ET
import numpy as np

def write__xml(
    scene_id: str,
    sar_product: str,
    swath_id: int,
    corner_coords_sar: Dict[str, List[float]],
    corner_coords_scene: Dict[str, List[float]],
    lat_coords: List[float],
    lon_coords: List[float],
    rfi_detection: int,
    output_path: Union[str, Path],
    date: Optional[int] = None,
    version: int = 1,
    case_study: str = 'RFIDetection'
) -> bool:
    """
    Write RFI detection XML metadata file following the MATLAB structure.
    
    Args:
        scene_id: Unique scene identifier (e.g., 'DB_OPENSAR_RFI_1')
        sar_product: Full SAR product name
        swath_id: SLC swath identifier (1, 2, or 3)
        corner_coords_sar: Dictionary with 'sample' and 'line' keys containing 4 coordinates each
        corner_coords_scene: Dictionary with 'sample' and 'line' keys containing 4 coordinates each
        lat_coords: List of 4 latitude coordinates for corners
        lon_coords: List of 4 longitude coordinates for corners
        rfi_detection: RFI detection flag (0 or 1)
        output_path: Path where to save the XML file
        date: Date in YYYYMMDD format (defaults to current date)
        version: XML version number
        case_study: Case study identifier
        
    Returns:
        True if successfully written, False otherwise
    """
    try:
        # Set default date to current date if not provided
        if date is None:
            date = int(datetime.now().strftime('%Y%m%d'))
        
        # Extract SAR mission (first 3 characters)
        sar_mission = sar_product[:3]
        
        # Extract time intervals from product name
        # Format: S1A_IW_SLC__1SDV_20201122T025208_20201122T025235_035355_042179_3723
        time_start = f"{sar_product[17:25]}_{sar_product[26:32]}"
        time_stop = f"{sar_product[33:41]}_{sar_product[42:48]}"
        
        # Create root element
        root = ET.Element('outxml')
        
        # Scene Info section
        scene_info = ET.SubElement(root, 'Scene_Info')
        ET.SubElement(scene_info, 'Scene_ID').text = scene_id
        ET.SubElement(scene_info, 'Date').text = str(date)
        ET.SubElement(scene_info, 'Version').text = str(version)
        ET.SubElement(scene_info, 'CaseStudy').text = case_study
        
        # SAR Data section
        sar_data = ET.SubElement(root, 'SARData')
        ET.SubElement(sar_data, 'SAR_Mission').text = sar_mission
        ET.SubElement(sar_data, 'SARProduct').text = sar_product
        ET.SubElement(sar_data, 'SLCSwath').text = str(swath_id)
        
        # Time interval
        time_interval = ET.SubElement(sar_data, 'Time_Interval')
        ET.SubElement(time_interval, 'Start').text = time_start
        ET.SubElement(time_interval, 'Stop').text = time_stop
        
        # Processing Data section
        processing_data = ET.SubElement(root, 'ProcessingData')
        
        # Corner coordinates
        corner_coord = ET.SubElement(processing_data, 'Corner_Coord')
        
        # SAR Data coordinates
        ET.SubElement(corner_coord, 'SARData_Sample').text = ' '.join(map(str, corner_coords_sar['sample']))
        ET.SubElement(corner_coord, 'SARData_Line').text = ' '.join(map(str, corner_coords_sar['line']))
        
        # Scene coordinates
        ET.SubElement(corner_coord, 'Scene_Sample').text = ' '.join(map(str, corner_coords_scene['sample']))
        ET.SubElement(corner_coord, 'Scene_Line').text = ' '.join(map(str, corner_coords_scene['line']))
        
        # Geographic coordinates
        ET.SubElement(corner_coord, 'Latitude').text = ' '.join(f'{lat:.6f}' for lat in lat_coords)
        ET.SubElement(corner_coord, 'Longitude').text = ' '.join(f'{lon:.6f}' for lon in lon_coords)
        
        # Statistics Report
        stats_report = ET.SubElement(processing_data, 'StatisticsReport')
        ET.SubElement(stats_report, 'RFIDetection').text = str(rfi_detection)
        
        # Create XML tree and write to file
        tree = ET.ElementTree(root)
        ET.indent(tree, space='  ', level=0)  # Pretty print
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        tree.write(output_path, encoding='utf-8', xml_declaration=True)
        
        return True
        
    except Exception as e:
        print(f'Error writing XML file: {e}')
        return False

def create_corner_coordinates(
    col_start: int,
    row_start: int,
    tile_size: int
) -> tuple[Dict[str, List[int]], Dict[str, List[int]]]:
    """
    Create corner coordinate dictionaries for SAR and scene data.
    
    Args:
        col_start: Starting column in SAR data
        row_start: Starting row in SAR data
        tile_size: Size of the extracted tile
        
    Returns:
        Tuple of (sar_coords, scene_coords) dictionaries
    """
    # SAR data coordinates (in original SAR space)
    sar_coords = {
        'sample': [col_start, col_start, col_start + tile_size - 1, col_start + tile_size - 1],
        'line': [row_start, row_start + tile_size - 1, row_start + tile_size - 1, row_start]
    }
    
    # Scene coordinates (in extracted tile space)
    scene_coords = {
        'sample': [1, 1, tile_size, tile_size],
        'line': [1, tile_size, tile_size, 1]
    }
    
    return sar_coords, scene_coords

# Example usage function
def example_usage():
    """Example of how to use the XML writer function."""
    
    # Example parameters
    scene_id = 'DB_OPENSAR_RFI_1'
    sar_product = 'S1A_IW_SLC__1SDV_20201122T025208_20201122T025235_035355_042179_3723'
    swath_id = 1
    
    # Create corner coordinates for a 512x512 tile starting at (1000, 2000)
    sar_coords, scene_coords = create_corner_coordinates(
        col_start=1000,
        row_start=2000,
        tile_size=512
    )
    
    # Example geographic coordinates (4 corners)
    lat_coords = [45.123456, 45.124567, 45.125678, 45.122345]
    lon_coords = [-123.456789, -123.455678, -123.454567, -123.457890]
    
    # Write XML file
    success = write_rfi_xml(
        scene_id=scene_id,
        sar_product=sar_product,
        swath_id=swath_id,
        corner_coords_sar=sar_coords,
        corner_coords_scene=scene_coords,
        lat_coords=lat_coords,
        lon_coords=lon_coords,
        rfi_detection=1,  # 1 for RFI present, 0 for no RFI
        output_path='output/xml_labelling/DB_OPENSAR_RFI_1.xml'
    )
    
    print(f'XML written successfully: {success}')