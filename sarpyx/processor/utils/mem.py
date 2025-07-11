import gc
import sys
from typing import Dict, Any


def cleanup_memory(caller_globals: Dict[str, Any] = None) -> int:
    """Clean up memory by deleting large variables and running garbage collection.
    
    Args:
        caller_globals: Dictionary of caller's global variables. If None, gets caller's globals.
            
    Returns:
        Number of variables deleted.
    """
    # Get caller's globals if not provided
    if caller_globals is None:
        caller_globals = sys._getframe(1).f_globals
    
    # Variables that are safe to delete
    deletable_vars = ['echo', 'metadata', 'ephemeris', 'raw_data']
    variables_to_keep = ['focused_radar_data', 'focuser']
    
    # Delete variables
    deleted_count = 0
    for var_name in deletable_vars:
        if var_name in caller_globals and var_name not in variables_to_keep:
            try:
                del caller_globals[var_name]
                deleted_count += 1
            except KeyError:
                pass
    
    # Run garbage collection
    gc.collect()
    
    return deleted_count