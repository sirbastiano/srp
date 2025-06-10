import shutil
import subprocess
from pathlib import Path
from typing import Any, Union
from zipfile import ZipFile
from scipy import io
import gc
from typing import Optional, Tuple, Union, Dict, Any, List, Callable

# ------- Functions for memory efficiency -------

def gc_collect(func: Callable) -> Callable:
    """
    Decorator to perform garbage collection after function execution.

    Args:
        func (Callable): The function to decorate.

    Returns:
        Callable: The wrapped function with garbage collection.
    """
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        gc.collect()
        return result
    return wrapper


# ------- Functions for file operations -------
def save_matlab_mat(data_object: Any, filename: str, filepath: Union[str, Path]) -> bool:
    """Saves a Python object to a MATLAB .mat file.

    Args:
        data_object: The Python object to save in the MATLAB file.
        filename: The name for the MATLAB file (without .mat extension).
        filepath: Path to the directory where the file will be saved.

    Returns:
        True if save was successful, False otherwise.

    Examples:
        >>> save_matlab_mat(my_array, "data", "path/to/dir")
        Saved MATLAB file to: path/to/dir/data.mat
        True
    """
    try:
        filepath = Path(filepath)
        savename = filepath / f"{filename}.mat"
        io.savemat(savename, {filename: data_object})
        print(f"Saved MATLAB file to: {savename}")
        return True
    except Exception as e:
        print(f"Could not save MATLAB file to {savename}: {e}")
        return False

def delete(path_to_delete: Union[str, Path]):
    """Deletes a file or directory.

    Args:
        path_to_delete: The path to the file or directory to delete.
    """
    if isinstance(path_to_delete, str):
        path_to_delete = Path(path_to_delete)

    if path_to_delete.exists():
        if path_to_delete.is_dir():
            shutil.rmtree(path_to_delete)
        else:
            path_to_delete.unlink()  # Use unlink for files

def unzip(path_to_zip_file: Union[str, Path]):
    """Unzips a file to its parent directory.

    Args:
        path_to_zip_file: The path to the zip file to extract.
    """
    zip_path = Path(path_to_zip_file)
    output_dir = zip_path.parent
    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)

def delProd(prodToDelete: Union[str, Path]):
    """Deletes a SNAP product (.dim file and associated .data directory).

    Args:
        prodToDelete: The path to the .dim file of the product to delete.
    """
    if isinstance(prodToDelete, str):
        prodToDelete = Path(prodToDelete)

    dim_file = prodToDelete
    data_dir = prodToDelete.with_suffix('.data')

    delete(dim_file)
    delete(data_dir)

def command_line(cmd: str):
    """Executes a command line process and prints its output.

    Args:
        cmd: The command string to execute in the shell.
    """
    try:
        # Use shell=True carefully, consider security implications if cmd comes from untrusted input
        # Using list of args is generally safer if possible
        process = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True, universal_newlines=True)
        print(process.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {cmd}")
        print(f"Return code: {e.returncode}")
        print(f"Stderr: {e.stderr}")
    except FileNotFoundError:
        print(f"Error: Command not found - ensure the executable is in the system's PATH or provide the full path.")

def iterNodes(root, val_dict: dict) -> dict:
    """Recursively iterates through XML nodes and extracts tag/text pairs.

    Args:
        root: The root element of the XML tree or subtree to iterate.
        val_dict: A dictionary to store the extracted tag/text pairs.

    Returns:
        The dictionary containing the extracted tag/text pairs.
    """
    # Check if it has children:
    def hasChildren(elem):
        return len(elem) > 0  # Simplified boolean check

    # Iterator:
    for child in root:
        if hasChildren(child):
            iterNodes(child, val_dict)  # Pass the same dict down
        elif child.text is not None:  # Ensure text exists
            # print(child.tag, child.text)
            val_dict[child.tag] = child.text.strip()  # Add strip() to remove potential whitespace

    return val_dict

def find_dat_file(folder: Path, pol: str) -> Path:
    """
    Find the .dat file in a SAFE folder for a specific polarization.
    
    Args:
        folder (Path): Path to the SAFE folder to search in
        pol (str): Polarization string to match (e.g., 'vh', 'vv')
        
    Returns:
        Path: Path to the matching .dat file
        
    Raises:
        AssertionError: If folder doesn't exist or is not a directory
        FileNotFoundError: If no valid .dat file is found matching criteria
    """
    assert folder.exists() and folder.is_dir(), f'Invalid folder: {folder}'
    
    for file in folder.iterdir():
        if (file.suffix == '.dat' and 
            'annot' not in file.name and 
            'index' not in file.name and 
            pol in file.name):
            return file
    
    raise FileNotFoundError(f'No valid .dat file found in {folder}')