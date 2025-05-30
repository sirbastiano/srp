import numpy as np

def summarize_2d_array(arr):
    # Ensure the input is a NumPy array
    arr = np.array(arr)
    
    # Basic summary statistics
    shape = arr.shape
    mean = np.mean(arr)
    median = np.median(arr)
    min_value = np.min(arr)
    max_value = np.max(arr)
    std_dev = np.std(arr)
    
    # Print the summary
    print(f"Shape: {shape}")
    print(f"Mean: {mean}")
    print(f"Median: {median}")
    print(f"Min: {min_value}")
    print(f"Max: {max_value}")
    print(f"Standard Deviation: {std_dev}")