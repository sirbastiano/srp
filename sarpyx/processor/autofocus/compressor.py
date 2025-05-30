import torch 
from torch import nn 

import matplotlib.pyplot as plt
import matplotlib.colors as colors

from constants import load_constants, load_constants_from_meta
import pandas as pd 
import numpy as np

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class estimate_V_eff(nn.Module):
    def __init__(self, V_aux = 7900, patch_dim=(4096, 4096)):
        super().__init__()
        Vx, Vy, Vz = V_aux
        self.patch_dim = patch_dim
        self.V0 = torch.sqrt(torch.tensor(Vx*Vx + Vy*Vy + Vz*Vz)).to(device)

        self.a0 = nn.Parameter(torch.tensor(1.0))  # a0 parameter
        self.ar = nn.Parameter(torch.tensor(0.0))  # ar parameter (riga)
        self.an = nn.Parameter(torch.tensor(0.0))  # an parameter (colonna)

    def forward(self):
        x = torch.ones(self.patch_dim, device=device) * self.V0
        # A0 addition
        x += self.a0
        # Create a range tensor for rows and columns
        r = torch.arange(self.patch_dim[1], device=x.device).view(-1, 1) * self.ar
        c = torch.arange(self.patch_dim[0], device=x.device).view(1, -1) * self.an
        # Add to x using broadcasting
        x += r
        x += c
        return x
            
class estimate_D(nn.Module):
    def __init__(self, meta):
        super().__init__()
        # self.constants = load_constants()
        self.constants = load_constants_from_meta(meta)
        self.wavelength = self.constants['wavelength'] 
        PRI = self.constants['PRI'] 
        len_az_line = self.constants['len_az_line'] 
        az_sample_freq = 1 / PRI
        start = -az_sample_freq / 2
        end = az_sample_freq / 2
        self.f_eta = torch.linspace(start=start, end=end, steps=4096, dtype=torch.float32).to(device)
        
    def forward(self, V):
        A = self.wavelength**2 * self.f_eta**2
        B = torch.tensor(4.0, device=device) * V*V
        C = A/B
        E = 1 - C
        F = torch.sqrt(E)
        return F.T
   
class Focalizer(nn.Module):
    def __init__(self, metadata: dict):
        super().__init__()
        self.device = device
        self.patch_dim = (4096, 4096)
        
        self.meta = metadata['aux']
        self.constants = load_constants_from_meta(self.meta)
        self.V_metadata = self.extract_velocity(metadata) 
        # slant range vector:
        self.R0 = ((self.constants['rank'] * self.constants['PRI']) + self.constants['fast_time_vec']) * self.constants['c']/2
        # layers initialization:
        self.V = estimate_V_eff(self.V_metadata, self.patch_dim)
       
       
    @staticmethod
    def fft2D(x):
        """
        Perform FFT on radar data along range and azimuth lines.

        Parameters:
            x (torch.Tensor): Input tensor of shape [batch, channels, height, width].

        Returns:
            torch.Tensor: Transformed tensor after FFT operations.
        """
        # FFT each range line (along width=range, which is dim=-1)
        x_fft_w = torch.fft.fft(x, dim=-1)
        # FFT each azimuth line (along height=azimuth, which is dim=-2) and shift the zero frequency component to the center
        x_fft_hw = torch.fft.fft(x_fft_w, dim=-2)
        x_fftshift_hw = torch.fft.fftshift(x_fft_hw, dim=-2)
        return x_fftshift_hw
    
    def _plot_rg_filter(self):
        rg = self.constants['range_filter'].cpu().numpy().real
        
        plt.figure(figsize=(13,6))
        plt.subplot(2, 1, 1)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power')
        
        plt.subplot(2, 1, 2)
        plt.plot(torch.fft.ifft(rg).cpu().numpy().real)
        plt.xlabel('Time (s)')
        plt.ylabel('Power')
        plt.show()
    
    @staticmethod
    def _plot_tensor(x, save=False, name=None):
        if type(x) == torch.Tensor:
            input_tensor = x.cpu().numpy()
        else:
            input_tensor = x
        
        input_data_mean = np.mean(np.abs(input_tensor))
        input_data_std = np.std(np.abs(input_tensor))
        
        plt.figure(figsize=(15,15))
        norm_input = colors.LogNorm(vmin=input_data_mean - input_data_std * 0.5 + 1e-10, vmax=input_data_mean + input_data_std * 2)
        plt.imshow(np.abs(input_tensor), cmap='viridis', norm=norm_input)
        plt.axis('off')
        if save:
            if name is not None:
                plt.savefig(name)
            else:
                plt.savefig('output.png')
        plt.close()
    
        
    @staticmethod
    def extract_velocity(metadata):
        ephemeris = metadata['ephemeris']
        Vx = ephemeris["X-axis velocity ECEF"][0].item()
        Vy = ephemeris["Y-axis velocity ECEF"][0].item()
        Vz = ephemeris["Z-axis velocity ECEF"][0].item()
        return (Vx,Vy,Vz) 

    def _compute_filter_rcmc(self):
        """ Compute the RCMC shift filter. """
        self.D = estimate_D(meta=self.meta)(self.V())
        self.R0 = self.R0.to(device)
        
        # self.RO is slant range vec
        rcmc_shift = self.R0 * ( (1/self.D) - 1)
        range_freq_vals = torch.linspace(-self.constants['range_sample_freq']/2, self.constants['range_sample_freq']/2, steps=self.constants['len_range_line'])
        self.rcmc_filter = torch.exp(4j * self.constants['pi'] * range_freq_vals.to(self.device) * rcmc_shift / self.constants['c'])
        return self.rcmc_filter

    def _compute_azimuth_filter(self):
        """ Compute the Azimuth filter. """
        self.azimuth_filter = torch.exp(4j * self.constants['pi'] * self.R0 * self.D / self.constants['wavelength']).to(self.device)
        return self.azimuth_filter

    def _compute_range_filter(self):
        tx_replica = torch.tensor(self.constants['tx_replica'], device=device)
        h, w = self.patch_dim # zero-pad or trim the filter to the patch dimension
        return torch.conj(torch.fft.fft(tx_replica, w))
        
    def forward(self, x):
        with torch.no_grad():
            x = self.fft2D(x)
            B,C,H,W = x.shape
            rg_filter = self._compute_range_filter().reshape(1, 1, 1, W)
            x = x * rg_filter # Range Compression
        x = x * self._compute_filter_rcmc().unsqueeze(0).unsqueeze(0)
        x = torch.fft.ifftshift(torch.fft.ifft(x, dim=-1), dim=-1) # Convert to Range-Doppler
        x = x * self._compute_azimuth_filter().unsqueeze(0).unsqueeze(0)
        x = torch.fft.ifft(x, dim=-2)
        return x   


if __name__ == '__main__':

        
    def test_model():
        x = torch.load('/home/roberto/PythonProjects/SSFocus/Data/4096_test_fft2D.pt').to(device)
        aux = pd.read_pickle('/home/roberto/PythonProjects/SSFocus/Data/RAW/SM/numpy/s1a-s1-raw-s-vv-20200509t183227-20200509t183238-032492-03c34a_pkt_8_metadata.pkl')
        eph = pd.read_pickle('/home/roberto/PythonProjects/SSFocus/Data/RAW/SM/numpy/s1a-s1-raw-s-vv-20200509t183227-20200509t183238-032492-03c34a_ephemeris.pkl')
        model = Focalizer(metadata={'aux':aux, 'ephemeris':eph})
        print('Model parameters:', list(model.parameters()))
    
    test_model()