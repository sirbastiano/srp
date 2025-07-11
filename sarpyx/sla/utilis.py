from pathlib import Path
import os
import shutil
from zipfile import ZipFile
import subprocess
import xml.etree.ElementTree as ET

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


def iterNodes(root, Val_Dict: dict):
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