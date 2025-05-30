"""
Processor utilities for SAR data handling.

This submodule contains utility functions for I/O operations,
downloading, and other helper functions specific to SAR processing.
"""

import configparser
import os

from .io import *
from .printsummary import *
from .unzip import *

# Import downloader submodule
from . import downloader

def load_config(config_file_paths):
    """
    Load configuration variables from a list of potential config file paths.

    Parameters:
    config_file_paths (list of str): List of file paths to search for the configuration file.

    Returns:
    configparser.ConfigParser: The loaded configuration object.

    Raises:
    FileNotFoundError: If the configuration file is not found in any of the provided paths.
    KeyError: If the required configuration key is not found in the configuration file.
    """
    config = configparser.ConfigParser()
    
    for config_file in config_file_paths:
        if os.path.exists(config_file):
            config.read(config_file)
            try:
                sarlens_dir = config["DIRECTORIES"]["SARLENS_DIR"]
                return config  # Return the loaded configuration
            except KeyError as e:
                raise KeyError(f"Missing key in configuration file: {e}")
    
    raise FileNotFoundError("Configuration file not found in any of the specified paths.")

def set_environment_variables(config):
    """
    Set environment variables from the loaded configuration.

    Parameters:
    config (configparser.ConfigParser): The loaded configuration object.
    """
    os.environ["SARLENS_DIR"] = config["DIRECTORIES"]["SARLENS_DIR"]

if __name__ == "__main__":
    config_paths = ["config.ini", "../config.ini","../../config.ini"]
    try:
        config = load_config(config_paths)
        set_environment_variables(config)
        print(f"SARLENS_DIR is set to: {os.environ['SARLENS_DIR']}")
    except (FileNotFoundError, KeyError) as e:
        print(f"Error: {e}")