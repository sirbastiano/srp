import numpy as np
from scipy.constants import c

def freq2wavelen(f_hz):
    return c / f_hz

def fspl(r_m, wavelen_m):
    return (4 * np.pi * r_m / wavelen_m) ** 2

def GenerateRFISignal(sqd, RadPar=None, verbose=False):
    """Generate Radio Frequency Interference (RFI) signal for SAR data.
    
    Args:
        sqd (np.ndarray): Input SAR single look complex data with shape (rows, cols).
        RadPar (dict): Radar parameters containing:
            - 'bw': Bandwidth in Hz
            - 'fo': Center frequency in Hz
            - 'fs': Sampling frequency in Hz
        
    Returns:
        tuple: A tuple containing:
            - sInterf (np.ndarray): Generated interference signal with same shape as sqd
            - IR (dict): Dictionary containing RFI parameters used in generation
    """
    
    
    if sqd is None:
        raise ValueError('The parameter "sqd" must be a valid numpy ndarray.')
    
    
    RadPar = {
            'fo': 5405000454.33435,
            'fs': 64345238.12571428,
            'bw': 56504455.48389234,
            'ts': 1.554116558005821e-08
    } if RadPar is None else RadPar
    
    
    IR = {}
    
    # === RFI Physical Parameters ===
    IR['freqShift'] = (np.random.rand() - 0.5) * RadPar['bw']  # shift within radar BW
    IR['fc'] = RadPar['fo'] + IR['freqShift']
    IR['SIR_dB'] = np.random.uniform(-25, -5)  # control strength
    IR['bw'] = np.random.uniform(0.005, 0.05) * RadPar['bw']  # narrower than radar BW
    IR['PRF'] = np.random.randint(1000, 2000)
    IR['duty'] = np.random.uniform(0.1, 0.5)  # sparse duty
    IR['Gain'] = 1.0
    IR['Lambda'] = freq2wavelen(IR['fc'])
    IR['T'] = IR['duty'] / IR['PRF']
    IR['K'] = IR['bw'] / IR['T']
    fs = RadPar['fs']
    IR['t'] = np.arange(-IR['T']/2, IR['T']/2, 1/fs)

    # === Chirp Pulse Generation ===
    chirp = np.exp(1j * np.pi * IR['K'] * IR['t']**2)
    chirp *= np.exp(1j * 2 * np.pi * IR['freqShift'] * IR['t'])  # center-shift
    chirp *= np.hanning(len(chirp))  # envelope
    chirp /= np.max(np.abs(chirp))

    # === Pulse Train (1D) ===
    rows, cols = sqd.shape
    burst_duration = rows / IR['PRF']
    IRstream_len = int(burst_duration * fs)
    pulse_spacing = int(fs / IR['PRF'])
    pulse_len = len(chirp)
    IRstream = np.zeros(IRstream_len, dtype=np.complex64)

    pulse_indices = np.arange(0, IRstream_len - pulse_len, pulse_spacing)
    N_total = len(pulse_indices)
    N_active = int(IR['duty'] * N_total)
    active_idx = np.random.choice(pulse_indices, N_active, replace=False)
    for idx in active_idx:
        IRstream[idx:idx+pulse_len] += chirp

    # === Map to Burst Grid (Spatial Submask) ===
    IRmap = np.zeros((rows, cols), dtype=np.complex64)
    # Random RFI spatial extent (5-95% of image dimensions)
    az_fraction = np.random.uniform(0.05, 0.95)
    rg_fraction = np.random.uniform(0.05, 0.95)
    sub_rows = int(rows * az_fraction)
    sub_cols = int(cols * rg_fraction)
    
    # Random position ensuring no border overflow
    az_start = np.random.randint(0, max(1, rows - sub_rows + 1))
    az_stop = az_start + sub_rows
    rg_start = np.random.randint(0, max(1, cols - sub_cols + 1))
    rg_stop = rg_start + sub_cols

    # reshape with tiling
    IRstream_crop = IRstream[:sub_rows * sub_cols]
    IRmap[az_start:az_stop, rg_start:rg_stop] = IRstream_crop.reshape((sub_rows, sub_cols))

    # === Power Calibration ===
    signal_power_lin = np.mean(np.abs(sqd)**2)
    signal_power_dB = 10 * np.log10(signal_power_lin)
    PrRFI_dB = signal_power_dB - IR['SIR_dB']
    PrRFI = 10 ** (PrRFI_dB / 10)
    sInterf = IRmap * np.sqrt(PrRFI)

    # === Metadata & Logging ===
    IR['AffectedAz'] = (az_start, az_stop)
    IR['AffectedRg'] = (rg_start, rg_stop)
    if verbose:
        print(f"[INFO] Injected RFI: freq offset = {IR['freqShift']/1e6:.2f} MHz, bw = {IR['bw']/1e6:.2f} MHz")
        print(f"[INFO] SIR = {IR['SIR_dB']} dB, Affected area: Az[{az_start}:{az_stop}], Rg[{rg_start}:{rg_stop}]")

    return sInterf, IR
