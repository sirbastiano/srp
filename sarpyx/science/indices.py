import numpy as np


def calculate_rvi(sigma_vv, sigma_vh):
    """
    Calculates the Radar Vegetation Index (RVI) from Sentinel-1 dual-pol data.

    RVI is computed using the simplified formula:
        RVI = (4 * sigma_vh) / (sigma_vv + sigma_vh)

    Args:
        sigma_vv (np.ndarray): Backscatter coefficient (linear scale) for VV polarization.
        sigma_vh (np.ndarray): Backscatter coefficient (linear scale) for VH polarization.

    Returns:
        np.ndarray: The Radar Vegetation Index (RVI) array.
                    Returns NaN where denominator is zero or negative.
                    
    References:
        - https://www.mdpi.com/2076-3417/9/4/655
        - https://forum.step.esa.int/t/creating-radar-vegetation-index/12444/18
    """
    sigma_vv = np.asarray(sigma_vv)
    sigma_vh = np.asarray(sigma_vh)

    numerator = 4 * sigma_vh
    denominator = sigma_vv + sigma_vh

    rvi = np.full_like(denominator, np.nan)
    valid_mask = denominator > 0
    rvi[valid_mask] = numerator[valid_mask] / denominator[valid_mask]

    return rvi




# Example Usage (can be removed or kept for testing)
if __name__ == '__main__':
    # Example dummy data (replace with actual data loading)
    # Assuming backscatter values are in linear scale
    sigma_hh_example = np.array([[0.1, 0.2], [0.3, 0.4]])
    sigma_vv_example = np.array([[0.2, 0.3], [0.4, 0.5]])
    sigma_hv_example = np.array([[0.01, 0.02], [0.03, 0.04]])

    rvi_values = calculate_rvi(sigma_hh_example, sigma_vv_example, sigma_hv_example)
    print("Calculated RVI values:")
    print(rvi_values)

    # Example with zero denominator
    sigma_hh_zero = np.array([[0.0, 0.2], [0.3, 0.4]])
    sigma_vv_zero = np.array([[0.0, 0.3], [0.4, 0.5]])
    sigma_hv_zero = np.array([[0.0, 0.02], [0.03, 0.04]])
    rvi_zero_den = calculate_rvi(sigma_hh_zero, sigma_vv_zero, sigma_hv_zero)
    print("\\nCalculated RVI with potential zero denominator:")
    print(rvi_zero_den)

    # Example with non-positive input
    sigma_hh_neg = np.array([[-0.1, 0.2], [0.3, 0.4]])
    rvi_neg_input = calculate_rvi(sigma_hh_neg, sigma_vv_example, sigma_hv_example)
    print("\\nCalculated RVI with non-positive input:")
    print(rvi_neg_input)
