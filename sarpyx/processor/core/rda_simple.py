import numpy as np
from scipy.fft import fft, ifft
from scipy.interpolate import interp1d

def SAR_focus(raw, Vr, fc, PRF, fs, swst, ch_R, ch_T):
    """
    Focuses raw SAR data using a basic range-Doppler algorithm to produce a single-look-complex (SLC) image.
    
    Args:
    raw : ndarray
        Baseband RAW data (complex pixels, range along rows).
    Vr : float
        Sensor velocity [m/s].
    fc : float
        SAR central frequency [Hz].
    PRF : float
        Pulse repetition frequency [Hz].
    fs : float
        Range sampling frequency [Hz].
    swst : float
        Sampling window start time [s] (fast-time to first sample).
    ch_R : float
        Range chirp rate [Hz/s] (positive for up-chirp).
    ch_T : float
        Range chirp duration [s].

    Returns:
    slc : ndarray
        Single-look-complex focused image.
    """
    c0 = 299792458  # Speed of light [m/s]
    lambda_ = c0 / fc  # SAR wavelength [m]
    Nl, Ns = raw.shape  # Number of lines and samples
    
    # Slant range [m] for each range pixel
    SR = (swst + np.arange(Ns) / fs) * c0 / 2
    
    # Doppler frequency vector after FFT (assumes DC=0)
    fdopp = np.arange(Nl) / Nl * PRF
    idx_wrap = fdopp >= PRF / 2
    fdopp[idx_wrap] -= PRF
    
    # Range compression
    print('Range compression')
    t_rg = np.arange(0, ch_T, 1/fs)
    t_rg -= np.mean(t_rg)
    chr_rg = np.exp(1j * np.pi * ch_R * t_rg**2)
    rgc = ifft(fft(raw, axis=1) * np.conj(fft(chr_rg, Ns)), axis=1)
    
    # Range cell migration correction
    print('RCMC')
    rgc_dopp = fft(rgc, axis=0)  # Range/Doppler domain
    for k in range(Nl):
        SR2 = SR + SR * (lambda_ * fdopp[k])**2 / (8 * Vr**2)
        interp_func = interp1d(SR, rgc_dopp[k, :], kind='cubic', fill_value=0, bounds_error=False)
        rgc_dopp[k, :] = interp_func(SR2)
    
    # Azimuth compression
    print('Azimuth compression')
    dopp_R = -2 * Vr**2 / lambda_ / SR  # Doppler rate
    slc = ifft(rgc_dopp * np.exp(1j * np.pi * fdopp[:, None]**2 / dopp_R), axis=0)
    
    print('END :)')
    return slc