import matplotlib.pyplot as plt  
from matplotlib.colors import LogNorm
import numpy as np
import os 
from matplotlib import colors
import torch
import pickle
import joblib

try:
    import zarr 
except:
    print('Zarr packet not installed. Zarr functionalities will not work.')

def plot_with_logscale(res, k=1):
    plt.figure(dpi=300, figsize=(12,12))
    if type(res) == torch.Tensor:
        res = np.abs(res.detach().cpu().numpy())
    else:
        res = np.abs(res)
    plt.imshow(res, norm=LogNorm(vmin=res.mean()-k*res.std(), vmax=res.mean()+k*res.std()))  # vmin should be > 0 for LogNorm
    plt.colorbar
    plt.show()

def plot_with_cdf(img, figsize=(22, 22), savepath=None):
    plt.figure(figsize=figsize)
    plt.title("Sentinel-1 Processed SAR Image")
    
    img_abs = abs(img)
    vmin, vmax = np.percentile(img_abs, [25, 99])
    
    plt.imshow(img_abs, origin='lower', norm=colors.LogNorm(vmin=vmin, vmax=vmax))
    plt.xlabel("Down Range (samples)")
    plt.ylabel("Cross Range (samples)")
    plt.colorbar()
    if savepath is not None:
        plt.savefig(savepath)
    plt.show()


def plot2_with_cdf(img1, img2):
    fig, axes = plt.subplots(1, 2, figsize=(24, 12))

    # Plot first image
    axes[0].set_title("Image 1")
    img1_abs = abs(img1)
    vmin1, vmax1 = np.percentile(img1_abs, [25, 99])
    axes[0].imshow(img1_abs, origin='lower', norm=colors.LogNorm(vmin=vmin1, vmax=vmax1))
    axes[0].set_xlabel("Down Range (samples)")
    axes[0].set_ylabel("Cross Range (samples)")

    # Plot second image
    axes[1].set_title("Image 2")
    img2_abs = abs(img2)
    vmin2, vmax2 = np.percentile(img2_abs, [25, 99])
    axes[1].imshow(img2_abs, origin='lower', norm=colors.LogNorm(vmin=vmin2, vmax=vmax2))
    axes[1].set_xlabel("Down Range (samples)")
    axes[1].set_ylabel("Cross Range (samples)")

    plt.show()
    return fig




def get_lognorm(output_data):
    output_data_mean = np.mean(np.abs(output_data))
    output_data_std = np.std(np.abs(output_data))
    norm_output = colors.LogNorm(vmin=output_data_mean - output_data_std * 0.5 + 1e-10, 
                          vmax=output_data_mean + output_data_std * 2)
    return norm_output

def find_checkpoint(folder):
    subdir = os.path.join(folder, 'checkpoints')
    checkpoint_filepath = os.listdir(subdir)[0]
    return os.path.join(subdir, checkpoint_filepath)    

def plot_histogram(img,xlim=[0,500], savepath=None):
    img = img[np.isfinite(img)]

    plt.figure(figsize=(10, 6))
    plt.hist(img.ravel(), bins=4096*4, fc='k', ec='b')
    plt.title("Histogram of Image Values")
    plt.xlim(xlim)
    plt.xlabel("Pixel Values")
    plt.ylabel("Frequency")
    if savepath is not None:
        plt.savefig(savepath)
    plt.show()


def dump(obj, filename):
    """
    Save an object to a pickle file using joblib.

    Parameters:
    obj (object): The object to be saved.
    filename (str): The name of the file where the object will be saved.
    """
    joblib.dump(obj, filename)

def load(filename):
    """
    Load an object from a pickle file.

    Parameters:
    filename (str): The name of the file from which the object will be loaded.

    Returns:
    object: The object loaded from the pickle file.
    """
    with open(filename, 'rb') as file:
        return pickle.load(file)

def main():
    print("Starting main..")
    pass


if __name__ == '__main__':
    main()