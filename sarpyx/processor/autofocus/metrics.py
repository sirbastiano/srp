import torch
import torch.nn.functional as F
from torch.nn.functional import conv2d
import numpy as np

# Constants for SSIM
K1 = 0.01
K2 = 0.03

def luminance(img1, img2):
    mu_x = np.mean(img1)
    mu_y = np.mean(img2)
    L = max(np.max(img1), np.max(img2)) - min(np.min(img1), np.min(img2))
    C1 = (K1 * L) ** 2
    return (2 * mu_x * mu_y + C1) / (mu_x ** 2 + mu_y ** 2 + C1)

def contrast(img1, img2):
    sigma_x = np.std(img1)
    sigma_y = np.std(img2)
    L = max(np.max(img1), np.max(img2)) - min(np.min(img1), np.min(img2))
    C2 = (K2 * L) ** 2
    return (2 * sigma_x * sigma_y + C2) / (sigma_x ** 2 + sigma_y ** 2 + C2)

def structure(img1, img2):
    sigma_x = np.std(img1)
    sigma_y = np.std(img2)
    sigma_xy = np.mean((img1 - np.mean(img1)) * (img2 - np.mean(img2)))
    L = max(np.max(img1), np.max(img2)) - min(np.min(img1), np.min(img2))
    C2 = (K2 * L) ** 2
    C3 = C2 / 2
    return (sigma_xy + C3) / (sigma_x * sigma_y + C3)


def create_window(window_size, channel):
    _1D_window = torch.hann_window(window_size, periodic=False).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Convert numpy arrays to tensors if needed
    if isinstance(img1, np.ndarray):
        img1 = torch.from_numpy(img1).float()
    if isinstance(img2, np.ndarray):
        img2 = torch.from_numpy(img2).float()

    # Ensure the images are at least 4D tensors
    if img1.dim() == 2:  # If 2D, make it 4D
        img1 = img1.unsqueeze(0).unsqueeze(0)
    elif img1.dim() == 3:  # If 3D, make it 4D
        img1 = img1.unsqueeze(0)

    if img2.dim() == 2:  # If 2D, make it 4D
        img2 = img2.unsqueeze(0).unsqueeze(0)
    elif img2.dim() == 3:  # If 3D, make it 4D
        img2 = img2.unsqueeze(0)

    if val_range is None:
        max_val = 1
        min_val = 0
    else:
        max_val = val_range[1]
        min_val = val_range[0]
    L = max_val - min_val

    padd = 0
    if window is None:
        real_size = min(window_size, img1.shape[-1], img1.shape[-2])
        window = create_window(real_size, img1.shape[1]).to(img1.device)
        padd = window_size // 2

    mu1 = conv2d(img1, window, padding=padd, groups=img1.shape[1])
    mu2 = conv2d(img2, window, padding=padd, groups=img2.shape[1])

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = conv2d(img1 * img1, window, padding=padd, groups=img1.shape[1]) - mu1_sq
    sigma2_sq = conv2d(img2 * img2, window, padding=padd, groups=img2.shape[1]) - mu2_sq
    sigma12 = conv2d(img1 * img2, window, padding=padd, groups=img1.shape[1]) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def psnr(img1, img2, max_val=1.0):
    # Convert np arrays to torch tensors if necessary
    if isinstance(img1, np.ndarray):
        img1 = torch.tensor(img1, dtype=torch.float32)
    if isinstance(img2, np.ndarray):
        img2 = torch.tensor(img2, dtype=torch.float32)
    
    if not isinstance(img1, torch.Tensor) or not isinstance(img2, torch.Tensor):
        raise TypeError('Inputs should be torch tensors or numpy arrays')
    
    if img1.size() != img2.size():
        raise ValueError(f"Input shapes should match. Got {img1.size()} and {img2.size()}")

    mse = F.mse_loss(img1, img2)
    psnr_value = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr_value