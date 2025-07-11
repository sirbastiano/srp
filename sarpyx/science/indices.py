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


def calculate_dpdd(sigma_vv, sigma_vh):
    """
    Calculates the Dual-Pol Diagonal Distance (DPDD) from Sentinel-1 dual-pol data.
    
    DPDD is computed using the formula:
        DPDD = (VV + VH) / (2^0.5)
    
    Args:
        sigma_vv (np.ndarray): Backscatter coefficient (linear scale) for VV polarization.
        sigma_vh (np.ndarray): Backscatter coefficient (linear scale) for VH polarization.
        
    Returns:
        np.ndarray: The Dual-Pol Diagonal Distance (DPDD) array.
    
    References:
        - See documentation: https://www.sciencedirect.com/science/article/pii/S0034425718304140?via%3Dihub
    """
    sigma_vv = np.asarray(sigma_vv)
    sigma_vh = np.asarray(sigma_vh)
    
    numerator = sigma_vv + sigma_vh
    denominator = np.sqrt(2.0)
    
    dpdd = numerator / denominator
    
    return dpdd


def calculate_dprvi_hh(sigma_hh, sigma_hv):
    """
    Calculates the Dual-Polarized Radar Vegetation Index HH (DpRVIHH) from radar data.
    
    DpRVIHH is computed using the formula:
        DpRVIHH = (4.0 * HV) / (HH + HV)
    
    Args:
        sigma_hh (np.ndarray): Backscatter coefficient (linear scale) for HH polarization.
        sigma_hv (np.ndarray): Backscatter coefficient (linear scale) for HV polarization.
        
    Returns:
        np.ndarray: The Dual-Polarized Radar Vegetation Index HH array.
                    Returns NaN where denominator is zero or negative.
                    
    References:
        - See documentation for full reference details: https://www.tandfonline.com/doi/abs/10.5589/m12-043
    """
    sigma_hh = np.asarray(sigma_hh)
    sigma_hv = np.asarray(sigma_hv)
    
    numerator = 4.0 * sigma_hv
    denominator = sigma_hh + sigma_hv
    
    dprvi_hh = np.full_like(denominator, np.nan)
    valid_mask = denominator > 0
    dprvi_hh[valid_mask] = numerator[valid_mask] / denominator[valid_mask]
    
    return dprvi_hh


def calculate_dprvi_vv(sigma_vv, sigma_vh):
    """
    Calculates the Dual-Polarized Radar Vegetation Index VV (DpRVIVV) from radar data.
    
    DpRVIVV is computed using the formula:
        DpRVIVV = (4.0 * VH) / (VV + VH)
    
    Args:
        sigma_vv (np.ndarray): Backscatter coefficient (linear scale) for VV polarization.
        sigma_vh (np.ndarray): Backscatter coefficient (linear scale) for VH polarization.
        
    Returns:
        np.ndarray: The Dual-Polarized Radar Vegetation Index VV array.
                    Returns NaN where denominator is zero or negative.
                    
    References:
        - See ref: https://www.mdpi.com/2076-3417/9/4/655
    """
    sigma_vv = np.asarray(sigma_vv)
    sigma_vh = np.asarray(sigma_vh)
    
    numerator = 4.0 * sigma_vh
    denominator = sigma_vv + sigma_vh
    
    dprvivv = np.full_like(denominator, np.nan)
    valid_mask = denominator > 0
    dprvivv[valid_mask] = numerator[valid_mask] / denominator[valid_mask]

    return dprvivv


def calculate_ndpoll(sigma_vv, sigma_vh):
    """
    Calculates the Normalized Difference Polarization Index (NDPoll).

    NDPoll = (VV - VH) / (VV + VH)

    Args:
        sigma_vv (np.ndarray): Backscatter coefficient (linear scale) for VV polarization.
        sigma_vh (np.ndarray): Backscatter coefficient (linear scale) for VH polarization.

    Returns:
        np.ndarray: The NDPoll array. Returns NaN where the denominator is zero.
    """
    sigma_vv = np.asarray(sigma_vv)
    sigma_vh = np.asarray(sigma_vh)

    numerator = sigma_vv - sigma_vh
    denominator = sigma_vv + sigma_vh

    ndpoll = np.full_like(denominator, np.nan, dtype=np.float64)
    valid_mask = denominator != 0
    ndpoll[valid_mask] = numerator[valid_mask] / denominator[valid_mask]

    return ndpoll


def calculate_qprvi(sigma_hh, sigma_vv, sigma_hv):
    """
    Calculates the Quad-Polarized Radar Vegetation Index (QpRVI).

    QpRVI = (8.0 * HV) / (HH + VV + 2.0 * HV)

    Args:
        sigma_hh (np.ndarray): Backscatter coefficient (linear scale) for HH polarization.
        sigma_vv (np.ndarray): Backscatter coefficient (linear scale) for VV polarization.
        sigma_hv (np.ndarray): Backscatter coefficient (linear scale) for HV polarization.

    Returns:
        np.ndarray: The QpRVI array. Returns NaN where the denominator is zero.
    """
    sigma_hh = np.asarray(sigma_hh)
    sigma_vv = np.asarray(sigma_vv)
    sigma_hv = np.asarray(sigma_hv)

    numerator = 8.0 * sigma_hv
    denominator = sigma_hh + sigma_vv + 2.0 * sigma_hv

    qprvi = np.full_like(denominator, np.nan, dtype=np.float64)
    valid_mask = denominator != 0
    qprvi[valid_mask] = numerator[valid_mask] / denominator[valid_mask]

    return qprvi


def calculate_rfdi(sigma_hh, sigma_hv):
    """
    Calculates the Radar Forest Degradation Index (RFDI).

    RFDI = (HH - HV) / (HH + HV)

    Args:
        sigma_hh (np.ndarray): Backscatter coefficient (linear scale) for HH polarization.
        sigma_hv (np.ndarray): Backscatter coefficient (linear scale) for HV polarization.

    Returns:
        np.ndarray: The RFDI array. Returns NaN where the denominator is zero.
    """
    sigma_hh = np.asarray(sigma_hh)
    sigma_hv = np.asarray(sigma_hv)

    numerator = sigma_hh - sigma_hv
    denominator = sigma_hh + sigma_hv

    rfdi = np.full_like(denominator, np.nan, dtype=np.float64)
    valid_mask = denominator != 0
    rfdi[valid_mask] = numerator[valid_mask] / denominator[valid_mask]

    return rfdi


def calculate_vddpi(sigma_vv, sigma_vh):
    """
    Calculates the Vertical Dual De-Polarization Index (VDDPI).

    VDDPI = (VV + VH) / VV

    Args:
        sigma_vv (np.ndarray): Backscatter coefficient (linear scale) for VV polarization.
        sigma_vh (np.ndarray): Backscatter coefficient (linear scale) for VH polarization.

    Returns:
        np.ndarray: The VDDPI array. Returns NaN where VV is zero.
    """
    sigma_vv = np.asarray(sigma_vv)
    sigma_vh = np.asarray(sigma_vh)

    numerator = sigma_vv + sigma_vh
    denominator = sigma_vv

    vddpi = np.full_like(denominator, np.nan, dtype=np.float64)
    valid_mask = denominator != 0
    vddpi[valid_mask] = numerator[valid_mask] / denominator[valid_mask]

    return vddpi


def calculate_vhvvd(sigma_vh, sigma_vv):
    """
    Calculates the VH-VV Difference.

    VHVVD = VH - VV

    Args:
        sigma_vh (np.ndarray): Backscatter coefficient (linear scale) for VH polarization.
        sigma_vv (np.ndarray): Backscatter coefficient (linear scale) for VV polarization.

    Returns:
        np.ndarray: The VH-VV Difference array.
    """
    sigma_vh = np.asarray(sigma_vh)
    sigma_vv = np.asarray(sigma_vv)
    return sigma_vh - sigma_vv


def calculate_vhvvp(sigma_vh, sigma_vv):
    """
    Calculates the VH-VV Product.

    VHVVP = VH * VV

    Args:
        sigma_vh (np.ndarray): Backscatter coefficient (linear scale) for VH polarization.
        sigma_vv (np.ndarray): Backscatter coefficient (linear scale) for VV polarization.

    Returns:
        np.ndarray: The VH-VV Product array.
    """
    sigma_vh = np.asarray(sigma_vh)
    sigma_vv = np.asarray(sigma_vv)
    return sigma_vh * sigma_vv


def calculate_vhvvr(sigma_vh, sigma_vv):
    """
    Calculates the VH-VV Ratio.

    VHVVR = VH / VV

    Args:
        sigma_vh (np.ndarray): Backscatter coefficient (linear scale) for VH polarization.
        sigma_vv (np.ndarray): Backscatter coefficient (linear scale) for VV polarization.

    Returns:
        np.ndarray: The VH-VV Ratio array. Returns NaN where VV is zero.
    """
    sigma_vh = np.asarray(sigma_vh)
    sigma_vv = np.asarray(sigma_vv)

    vhvvr = np.full_like(sigma_vv, np.nan, dtype=np.float64)
    valid_mask = sigma_vv != 0
    vhvvr[valid_mask] = sigma_vh[valid_mask] / sigma_vv[valid_mask]

    return vhvvr


def calculate_vvvhd(sigma_vv, sigma_vh):
    """
    Calculates the VV-VH Difference.

    VVVHD = VV - VH

    Args:
        sigma_vv (np.ndarray): Backscatter coefficient (linear scale) for VV polarization.
        sigma_vh (np.ndarray): Backscatter coefficient (linear scale) for VH polarization.

    Returns:
        np.ndarray: The VV-VH Difference array.
    """
    sigma_vv = np.asarray(sigma_vv)
    sigma_vh = np.asarray(sigma_vh)
    return sigma_vv - sigma_vh



def calculate_vvvhr(sigma_vv, sigma_vh):
    """
    Calculates the VV-VH Ratio.

    VVVHR = VV / VH

    Args:
        sigma_vv (np.ndarray): Backscatter coefficient (linear scale) for VV polarization.
        sigma_vh (np.ndarray): Backscatter coefficient (linear scale) for VH polarization.

    Returns:
        np.ndarray: The VV-VH Ratio array. Returns NaN where VH is zero.
    """
    sigma_vv = np.asarray(sigma_vv)
    sigma_vh = np.asarray(sigma_vh)

    vvvhr = np.full_like(sigma_vh, np.nan, dtype=np.float64)
    valid_mask = sigma_vh != 0
    vvvhr[valid_mask] = sigma_vv[valid_mask] / sigma_vh[valid_mask]

    return vvvhr


def calculate_vvvhs(sigma_vv, sigma_vh):
    """
    Calculates the VV-VH Sum.

    VVVHS = VV + VH

    Args:
        sigma_vv (np.ndarray): Backscatter coefficient (linear scale) for VV polarization.
        sigma_vh (np.ndarray): Backscatter coefficient (linear scale) for VH polarization.

    Returns:
        np.ndarray: The VV-VH Sum array.
    """
    sigma_vv = np.asarray(sigma_vv)
    sigma_vh = np.asarray(sigma_vh)
    return sigma_vv + sigma_vh