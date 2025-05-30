import torch
import sentinel1decoder

def load_constants_from_meta(meta, patch_dim = (4096,4096)):
    constants = {}
    constants['wavelength'] = sentinel1decoder.constants.TX_WAVELENGTH_M
    constants['c'] = sentinel1decoder.constants.SPEED_OF_LIGHT_MPS  # Speed of light
    
    constants['PRI'] = meta["PRI"].unique()[0]
    constants['len_az_line'] = patch_dim[0]
    constants['len_range_line'] = patch_dim[1]
     
    constants['az_sample_freq'] = 1 / constants['PRI']
    RGDEC = meta["Range Decimation"].unique()[0]
    constants['range_sample_freq'] = sentinel1decoder.utilities.range_dec_to_sample_rate(RGDEC)
    constants['range_sample_period'] = 1 / constants['range_sample_freq']
    constants['pi'] = 3.141592653589793
    constants['rank'] = meta["Rank"].unique()[0]
    suppressed_data_time = 320/(8*sentinel1decoder.constants.F_REF)
    constants['range_start_time'] = meta["SWST"].unique()[0] + suppressed_data_time
    constants['sample_num_along_range_line'] = torch.arange(start=0, end=constants['len_range_line'], step=1)
    fast_time_vec = constants['range_start_time'] + (constants['range_sample_period'] * constants['sample_num_along_range_line'])
    constants['fast_time_vec'] = fast_time_vec
    TXPSF = meta["Tx Pulse Start Frequency"].unique()[0]
    TXPRR = meta["Tx Ramp Rate"].unique()[0]
    TXPL = meta["Tx Pulse Length"].unique()[0]
    num_tx_vals = int(TXPL*constants['range_sample_freq'])
    tx_replica_time_vals = torch.linspace(-TXPL/2, TXPL/2, steps=num_tx_vals)
    phi1 = TXPSF + TXPRR*TXPL/2
    phi2 = TXPRR/2
    tx_replica = torch.exp(2j * constants['pi'] * (phi1*tx_replica_time_vals + phi2*tx_replica_time_vals**2))
    # print('Len tx replica:', len(tx_replica))
    # # Create range filter from replica pulse
    # range_filter = torch.zeros(constants['len_range_line'], dtype=torch.complex64)
    # index_start = torch.ceil(torch.tensor((constants['len_range_line']-num_tx_vals)/2))-1
    # index_end = num_tx_vals + torch.ceil(torch.tensor((constants['len_range_line']-num_tx_vals)/2))-2
    # range_filter[int(index_start):int(index_end+1)] = tx_replica
    # constants['tx_replica'] = torch.conj(torch.fft.fft(range_filter))
    constants['tx_replica'] = tx_replica
    return constants


def load_constants():
    constants = {}
    constants['wavelength'] = torch.tensor(0.055465764662349676) 
    constants['c'] = torch.tensor(299792458.0)  # Speed of light
    
    constants['PRI'] = torch.tensor(0.0005345716926237736)
    constants['len_az_line'] = 18710
    constants['len_range_line'] = 25780
    constants['az_sample_freq'] = 1 / constants['PRI']
    start = -constants['az_sample_freq'] / 2
    end = constants['az_sample_freq'] / 2
    step = 1 / (constants['PRI'] * constants['len_az_line'])
    constants['f_eta'] = torch.arange(start=start, end=end, step=step, dtype=torch.float32)
    constants['range_sample_freq'] = 100092592.64 # ok
    constants['range_sample_period'] = 1 / constants['range_sample_freq']
    constants['pi'] = 3.141592653589793
    constants['rank'] = 9
    constants['range_start_time'] = 0.00011360148005720262
    constants['sample_num_along_range_line'] = torch.arange(start=0, end=constants['len_range_line'], step=1)
    fast_time_vec = constants['range_start_time'] + (constants['range_sample_period'] * constants['sample_num_along_range_line'])
    constants['fast_time_vec'] = fast_time_vec

    return constants
