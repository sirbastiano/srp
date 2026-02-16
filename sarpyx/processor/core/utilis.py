from pathlib import Path
import os
import shutil
from zipfile import ZipFile
import subprocess
import random
import numpy as np
import rasterio
import time


def delete(path_to_delete):
    """
    Deletes the specified file or directory.
    If the provided path corresponds to a directory, the directory and all its
    contents are removed. If the path corresponds to a file, the file is deleted.
    Args:
        path_to_delete (str or pathlib.Path): The path to the file or directory to delete.
            Can be provided as a string or a pathlib.Path object.
    Raises:
        FileNotFoundError: If the specified path does not exist.
        PermissionError: If the operation lacks the necessary permissions.
        OSError: For other errors related to file or directory removal.
    """
    if isinstance(path_to_delete, str):
        path_to_delete = Path(path_to_delete)
    elif isinstance(path_to_delete, Path):
        pass
    
    if path_to_delete.is_dir():
        shutil.rmtree(path_to_delete.as_posix())
    else:
        os.remove(path_to_delete.as_posix())


def unzip(path_to_zip_file):
    """
    Extracts the contents of a ZIP file to the directory containing the ZIP file.

    Args:
        path_to_zip_file (str): The file path to the ZIP file to be extracted.

    Returns:
        None
    """
    zip_Path = Path(path_to_zip_file)
    with ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(zip_Path.parent.as_posix())


def delProd(prodToDelete):
    """
    Deletes a product file and its associated data folder, while ensuring the last 
    processed product is retained.

    Parameters:
        prodToDelete (Path or str): The product file to delete. Can be a Path object 
                                    or a string representing the file path.

    Behavior:
        - If `prodToDelete` is a Path object, it is converted to a POSIX-style string.
        - Deletes the specified product file.
        - Deletes the associated `.data` folder by replacing the `.dim` extension 
          in the file name with `.data`.
    """
    if isinstance(prodToDelete, Path):
        prodToDelete = prodToDelete.as_posix()
    delete(prodToDelete)
    delete(prodToDelete.replace('.dim','.data'))


def command_line(cmd:str):
    """
    Executes a shell command provided as a string and prints its output.

    Args:
        cmd (str): The shell command to execute. The command should be provided
                   as a single string, with arguments separated by spaces.

    Raises:
        FileNotFoundError: If the specified command is not found.
        OSError: If there is an error executing the command.

    Note:
        This function uses `subprocess.Popen` to execute the command and capture
        its output. It splits the command string into a list of arguments using
        spaces as delimiters. Ensure the command string is properly formatted.
    """
    print(subprocess.Popen(cmd.split(' '), stdout=subprocess.PIPE, universal_newlines=True).communicate()[0])


def iterNodes(root, Val_Dict={}):
    """
    Recursively iterates through the children of an XML element tree and populates a dictionary 
    with the tags and text values of the leaf nodes.
    Args:
        root (xml.etree.ElementTree.Element): The root element of the XML tree to iterate through.
        Val_Dict (dict): A dictionary to store the tags and text values of the leaf nodes.
    Returns:
        dict: The updated dictionary containing tags as keys and their corresponding text values 
        as values from the leaf nodes of the XML tree.
    """
    # Check if it has children:
    def hasChildren(elem):
        if len(elem):
            return True
        else:
            return False

    # Iterator:
    for child in root:
        if hasChildren(child):
            iterNodes(child, Val_Dict)
        else:
            # print(child.tag, child.text)
            Val_Dict[child.tag]=child.text
    
    return Val_Dict


def get_annotations(filepath):
    """
    Retrieves the list of annotation files from the .SAFE directory.
    """
    try:
   
        # Find all XML files in annotation folders
        annotations = []
        for root, dirs, files in os.walk(filepath):
            if "annotation" in root and "calibration" not in root and "rfi" not in root:
                for file in files:
                    if file.endswith(".xml"):
                        annotations.append(os.path.join(root, file))
        
        if not annotations:
            raise ValueError("No valid annotation files found in the .SAFE directory.")
        
        return annotations
    except Exception as e:
        raise RuntimeError(f"Failed to get annotations: {e}")


def read_annotation(annotations):
    """
    Reads the content of the first annotation file.
    """
    try:
        # Use the first annotation file found
        annotation_file = annotations[0]
        
        with open(annotation_file, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        raise RuntimeError(f"Failed to read annotation file: {e}")


def create_complex_image_from_file(i_path, q_path):
    """
    Create a complex image array from the in-phase and quadrature components files in a beam-dimap file.
    """
    # Read the in-phase and quadrature components using rasterio.
    with rasterio.open(i_path) as src_i:
        img_i = src_i.read(1)
    with rasterio.open(q_path) as src_q:
        img_q = src_q.read(1)

    # Create a complex image array
    img_complex = img_i + 1j * img_q

    return img_complex


def randomize_subset_origins(file_path, subset_count = 500, sub_dims = 1024):
    """
    Get set of random origins for subsetting to, adjusted to image shape and desired subset dimension (square subset)
    """
    # Get image dimensions using rasterio
    with rasterio.open(file_path) as src:
        img_shape = src.shape
            
    rand_coords = np.zeros([subset_count,2])
    for i in range(subset_count):
        rand_coords[i,0] = round(random.randint(0, img_shape[1]-sub_dims))
        rand_coords[i,1] = round(random.randint(0, img_shape[0]-sub_dims))
    return rand_coords

def find_measurement_image(asset_path):
    '''Get path to a tiff image for an asset.
    
    Args:
        assetPath (str): The path to the asset directory.
        
    Returns:
        Path to the first matching tiff image in the measurement folder.
    '''
            
    tiff_files = list(Path(asset_path).glob("measurement/*.tiff"))
    if not tiff_files:
        raise FileNotFoundError(f"No .tiff files found in {asset_path}/measurement/")
    return tiff_files[0]

def create_complex_image_from_array(i_image, q_image):
    """
    Create a complex image array from the in-phase and quadrature components already loaded into a numpy array.
    
    """
    img_complex = np.zeros((i_image.shape[0], i_image.shape[1]), dtype=np.complex128)
    # Create a complex image array
    img_complex = i_image + 1j * q_image
    
    return img_complex


def check_file(file_path, max_retries=5, retry_delay=0.5):
    """
    Check for existence and completeness of input file with attempts and time delay. Returns file path.
    """
    
    for attempt in range(max_retries):
        try:
            # Check if file exists and is accessible
            if os.path.exists(file_path) and os.access(file_path, os.R_OK):
                # Additional check: ensure file isn't empty (still being written)
                if os.path.getsize(file_path) > 0:
                    # source = open(file_path, "rb")
                    return file_path
                    
            if attempt < max_retries - 1:  # Don't sleep on last attempt
                time.sleep(retry_delay)
                
        except (FileNotFoundError, PermissionError, OSError) as e:
            if attempt == max_retries - 1:  # Last attempt
                raise e
    
    raise FileNotFoundError(f"Could not access file after {max_retries} attempts: {file_path}")


