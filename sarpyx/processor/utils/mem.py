

def cleanup_memory(variables_to_keep: Optional[list] = None) -> None:
    """Clean up memory by deleting large variables and running garbage collection.
    
    Args:
        variables_to_keep (Optional[list]): List of variable names to preserve.
    """
    import gc
    
    variables_to_keep = variables_to_keep or ['focused_radar_data', 'focuser']
    
    # Get current globals
    current_globals = list(globals().keys())
    
    # Variables that are safe to delete (large data arrays)
    deletable_vars = ['echo', 'metadata', 'ephemeris', 'raw_data']
    
    deleted_count = 0
    for var_name in deletable_vars:
        if var_name in current_globals and var_name not in variables_to_keep:
            try:
                del globals()[var_name]
                deleted_count += 1
                print(f'  ✅ Deleted variable: {var_name}')
            except KeyError:
                pass
    
    # Run garbage collection
    collected = gc.collect()
    print(f'  🗑️  Garbage collector freed {collected} objects')
    print(f'  📝 Deleted {deleted_count} large variables')