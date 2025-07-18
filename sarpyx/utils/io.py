import shutil
import subprocess
from pathlib import Path
from typing import Any, Union
from zipfile import ZipFile
from scipy import io
import gc
from typing import Optional, Tuple, Union, Dict, Any, List, Callable
import numpy as np
import matplotlib.pyplot as plt


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


# ------- Classes for file operations -------

class ArraySlicer:
    """Class for slicing arrays into overlapping portions with specific drop rules.
    
    This class implements a sophisticated slicing strategy where:
    - First slice drops last 30% of rows
    - Subsequent slices start at 50% overlap and drop 30% from both ends
    - Final slice preserves the end portion
    - All slices are merged without gaps
    - Optional processing function can be applied to each slice before dropping
    
    Args:
        array (np.ndarray): Input array to be sliced
        slice_height (int): Height of each slice portion
        processing_func (Optional[callable]): Function to apply to each slice before dropping
    """
    
    def __init__(self, array: np.ndarray, slice_height: int, processing_func: Optional[callable] = None) -> None:
        """Initialize the ArraySlicer.
        
        Args:
            array (np.ndarray): Input array to slice
            slice_height (int): Height of each slice portion
            processing_func (Optional[callable]): Function to apply to each slice before dropping.
                Should accept (slice_data: np.ndarray, slice_info: dict) and return np.ndarray
            
        Raises:
            ValueError: If slice_height is larger than array height
            AssertionError: If input parameters are invalid
        """
        assert isinstance(array, np.ndarray), 'Input must be a numpy array'
        assert len(array.shape) >= 2, 'Array must be at least 2D'
        assert slice_height > 0, 'Slice height must be positive'
        
        if slice_height > array.shape[0]:
            raise ValueError(f'Slice height {slice_height} cannot be larger than array height {array.shape[0]}')
            
        self.array = array
        self.slice_height = slice_height
        self.processing_func = processing_func
        self.array_height, self.array_width = array.shape[:2]
        self.slices: List[np.ndarray] = []
        self.slice_info: List[dict] = []
        
    def slice_array(self) -> List[np.ndarray]:
        """Perform the complete slicing operation using calculate_slice_indices.
        
        Returns:
            List[np.ndarray]: List of sliced array portions
        """
        self.slices = []
        
        # Calculate all slice indices using the standalone function
        self.slice_info = calculate_slice_indices(self.array_height, self.slice_height)
        
        for slice_info in self.slice_info:
            # Extract the slice
            slice_data = self.array[slice_info['actual_start']:slice_info['actual_end']]
            
            # Apply processing function if provided
            if self.processing_func is not None:
                try:
                    slice_data = self.processing_func(slice_data, slice_info)
                    assert isinstance(slice_data, np.ndarray), 'Processing function must return numpy array'
                except Exception as e:
                    print(f'Warning: Processing function failed for slice {slice_info["slice_index"]}: {e}')
                    # Continue with original slice if processing fails
                    slice_data = self.array[slice_info['actual_start']:slice_info['actual_end']]
            
            self.slices.append(slice_data)
            
            # Update shape in slice_info after processing
            slice_info['shape'] = slice_data.shape
            
            print(f'Slice {slice_info["slice_index"]}: rows {slice_info["actual_start"]}-{slice_info["actual_end"]} '
                  f'(original: {slice_info["original_start"]}-{slice_info["original_end"]}) '
                  f'shape: {slice_data.shape}')
            
        return self.slices
    
    def merge_slices(self) -> np.ndarray:
        """Merge all slices back into a single array without gaps.
        
        Returns:
            np.ndarray: Merged array
        """
        if not self.slices:
            raise ValueError('No slices available. Call slice_array() first.')
            
        return np.concatenate(self.slices, axis=0)
    
    def get_slice_info(self) -> List[dict]:
        """Get detailed information about each slice.
        
        Returns:
            List[dict]: Information about each slice
        """
        return self.slice_info
    
    def visualize_slicing(self, figsize: Tuple[int, int] = (12, 8)) -> None:
        """Visualize the slicing process.
        
        Args:
            figsize (Tuple[int, int]): Figure size for the plot
        """
        if not self.slice_info:
            raise ValueError('No slice information available. Call slice_array() first.')
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot original array coverage
        ax1.set_title('Original Array Slicing Coverage')
        ax1.set_xlabel('Columns')
        ax1.set_ylabel('Rows')
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.slice_info)))
        
        for i, (info, color) in enumerate(zip(self.slice_info, colors)):
            # Show original slice boundaries
            rect_orig = plt.Rectangle(
                (0, info['original_start']), 
                self.array_width, 
                info['original_end'] - info['original_start'],
                fill=False, edgecolor=color, linewidth=2, linestyle='--',
                label=f'Slice {i} original'
            )
            ax1.add_patch(rect_orig)
            
            # Show actual slice after drops
            rect_actual = plt.Rectangle(
                (0, info['actual_start']), 
                self.array_width, 
                info['actual_end'] - info['actual_start'],
                fill=True, facecolor=color, alpha=0.3, edgecolor=color, linewidth=2,
                label=f'Slice {i} actual'
            )
            ax1.add_patch(rect_actual)
            
        ax1.set_xlim(0, self.array_width)
        ax1.set_ylim(0, self.array_height)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot merged result
        if self.slices:
            merged = self.merge_slices()
            ax2.imshow(merged, aspect='auto', cmap='viridis')
            ax2.set_title(f'Merged Result\nShape: {merged.shape}')
            ax2.set_xlabel('Columns')
            ax2.set_ylabel('Rows (Merged)')
        
        plt.tight_layout()
        plt.show()


# ------- Functions for file operations -------






def calculate_slice_indices(array_height: int, slice_height: int) -> List[dict]:
    """Calculate slice indices and drop information for array slicing.
    
    Args:
        array_height (int): Total height of the array to slice
        slice_height (int): Height of each slice portion
        
    Returns:
        List[dict]: List of dictionaries containing slice information with keys:
            - slice_index: Index of the slice
            - original_start: Original starting row
            - original_end: Original ending row
            - actual_start: Actual starting row after drops
            - actual_end: Actual ending row after drops
            - is_first: Whether this is the first slice
            - is_last: Whether this is the last slice
            - drop_start: Number of rows dropped from start
            - drop_end: Number of rows dropped from end
            - original_height: Height of original slice
            - actual_height: Height after drops
    """
    assert array_height > 0, 'Array height must be positive'
    assert slice_height > 0, 'Slice height must be positive'
    
    if slice_height > array_height:
        raise ValueError(f'Slice height {slice_height} cannot be larger than array height {array_height}')
    
    slice_info_list = []
    current_start = 0
    slice_index = 0
    last_actual_end = 0
    
    def _get_next_start_position(current_start: int, overlap: float = 0.5) -> int:
        """Calculate the start position for the next slice (50% overlap)."""
        return current_start + int(overlap * slice_height)

    def _calculate_drop_amounts(original_height: int, is_first: bool, is_last: bool, drop_ratio: float = 0.15) -> Tuple[int, int]:
        """Calculate drop amounts for start and end.
        
        Returns:
            Tuple[int, int]: (drop_start, drop_end)
        """
        if is_first:
            # First slice: drop last
            drop_start = 0
            drop_end = int(drop_ratio * original_height)
        elif is_last:
            # Last slice: drop from start, preserve end
            drop_start = int(drop_ratio * original_height)
            drop_end = 0
        else:
            # Middle slices: drop from both ends
            drop_start = int(drop_ratio * original_height)
            drop_end = int(drop_ratio * original_height)
            
        return drop_start, drop_end
    
    while current_start < array_height:
        # Determine if this is first or last slice
        is_first = slice_index == 0
        next_start = _get_next_start_position(current_start)
        
        # Check if this should be the last slice
        # Last slice if: next start would be beyond array OR remaining data is less than minimum slice
        remaining_data = array_height - current_start
        is_last = (next_start >= array_height) or (remaining_data <= slice_height * 1.2)
        
        # Calculate original slice boundaries
        if is_last:
            # For last slice, use all remaining data
            original_end = array_height
        else:
            original_end = min(current_start + slice_height, array_height)
        
        original_height = original_end - current_start
        
        # Skip if original slice would be too small
        if original_height < 10:  # Minimum viable slice height
            break
        
        # Calculate drop amounts ONCE
        drop_start, drop_end = _calculate_drop_amounts(original_height, is_first, is_last)
        
        # Calculate actual boundaries after drops
        actual_start = current_start + drop_start
        actual_end = original_end - drop_end
        
        # For non-first slices, ensure no gaps by starting where previous ended
        if not is_first and last_actual_end > actual_start:
            # Adjust start to connect with previous slice, but ensure we don't go backwards
            gap_adjusted_start = max(last_actual_end, current_start)
            
            # Only adjust if it doesn't create unreasonable overlap (more than 75% of slice)
            max_allowable_start = current_start + int(0.75 * original_height)
            if gap_adjusted_start <= max_allowable_start:
                actual_start = gap_adjusted_start
        
        # Ensure we don't go beyond array bounds and maintain proper ordering
        actual_start = max(current_start, min(actual_start, array_height - 1))
        actual_end = min(array_height, max(actual_start + 1, actual_end))
        
        # For last slice, ensure we reach the end of the array
        if is_last:
            actual_end = array_height
        
        # Final validation: ensure positive slice height
        if actual_start >= actual_end:
            # Try to salvage by using remaining data
            if last_actual_end < array_height:
                actual_start = last_actual_end
                actual_end = array_height
            else:
                break
        
        actual_height = actual_end - actual_start
        
        # Calculate final drop values - ensure they are non-negative
        final_drop_start = max(0, actual_start - current_start)
        final_drop_end = max(0, original_end - actual_end)
        
        # Skip if slice would be too small after all adjustments
        if actual_height < 5:
            break
        
        # Create slice information dictionary
        slice_info = {
            'slice_index': slice_index,
            'original_start': current_start,
            'original_end': original_end,
            'actual_start': actual_start,
            'actual_end': actual_end,
            'is_first': is_first,
            'is_last': is_last,
            'drop_start': final_drop_start,
            'drop_end': final_drop_end,
            'original_height': original_height,
            'actual_height': actual_height
        }
        
        slice_info_list.append(slice_info)
        
        # Update last_actual_end for next iteration
        last_actual_end = actual_end
        
        # Break if this was the last slice or we've reached the end
        if is_last or actual_end >= array_height:
            break
        
        # Move to next slice position
        current_start = next_start
        slice_index += 1
    
    return slice_info_list








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
    Find the .dat file in a SAFE folder for a specific polarization using recursive search.

    Args:
        folder (Path): Path to the SAFE folder to search in.
        pol (str): Polarization string to match (e.g., 'vh', 'vv').

    Returns:
        Path: Path to the matching .dat file.

    Raises:
        AssertionError: If folder doesn't exist or is not a directory.
        FileNotFoundError: If no valid .dat file is found matching criteria.
    """
    assert folder.exists() and folder.is_dir(), f'Invalid folder: {folder}'

    for file in folder.rglob('*.dat'):
        if 'annot' not in file.name and 'index' not in file.name and pol in file.name:
            return file

    raise FileNotFoundError(f'No valid .dat file found in {folder} for polarization {pol}')
