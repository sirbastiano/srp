import numpy as np
from typing import Optional, Union
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize


def show_image(image: np.ndarray, title: Optional[str] = None, cmap: Optional[str] = None,
               vmin: Optional[float] = None, vmax: Optional[float] = None,
               colorbar: bool = True, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Display an image using matplotlib with optional color normalization and colorbar.
    
    Args:
        image: Input image to display as numpy array
        title: Optional title to display above the image
        cmap: Optional colormap to use for displaying the image
        vmin: Optional minimum data value for colormap scaling
        vmax: Optional maximum data value for colormap scaling
        colorbar: Whether to add a colorbar (default True)
        ax: Optional matplotlib Axes to plot on, if None a new figure will be created
        
    Returns:
        The matplotlib Axes containing the image
    """
    if ax is None:
        fig, ax = plt.subplots()
    
    norm = None
    if vmin is not None or vmax is not None:
        norm = Normalize(vmin=vmin, vmax=vmax)
    
    im = ax.imshow(image, cmap=cmap, norm=norm)
    
    if title:
        ax.set_title(title)
    
    if colorbar:
        plt.colorbar(im, ax=ax)
    
    return ax
    
    
def image_histogram_equalization(image: np.ndarray, number_bins: int = 8) -> tuple:
    """Perform histogram equalization on an input image.
    Args:
        image: Input image as a numpy array.
        number_bins: Number of bins for the histogram. Defaults to 8.
    Returns:
            - Equalized image with the same shape as input.
            - Cumulative distribution function used for equalization.
    References:
    ----------
    Adapted from http://www.janeriksolem.net/histogram-equalization-with-python-and.html
    """
    # Calculate the image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, density=True)
    
    # Compute the cumulative distribution function
    cdf = image_histogram.cumsum()
    
    # Normalize the CDF to the range [0, 255]
    cdf_normalized = 255 * cdf / cdf[-1]
    
    # Apply linear interpolation to map original pixel values to equalized values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf_normalized)
    
    # Reshape the equalized values back to the original image shape
    return image_equalized.reshape(image.shape), cdf_normalized


def show_histogram(image: np.ndarray, title: Optional[str] = None, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Display a histogram of the pixel values in an image.
    
    Args:
        image: Input image as a numpy array
        title: Optional title for the histogram
        ax: Optional matplotlib Axes to plot on, if None a new figure will be created
        
    Returns:
        The matplotlib Axes containing the histogram
    """
    if ax is None:
        fig, ax = plt.subplots()
    
    ax.hist(image.flatten(), bins=256, color='gray', alpha=0.7)
    
    if title:
        ax.set_title(title)
    
    return ax


def show_histogram_equalization(image: np.ndarray, number_bins: int = 8,
                                title: Optional[str] = None, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Display the histogram equalization of an image.
    
    Args:
        image: Input image as a numpy array
        number_bins: Number of bins for the histogram
        title: Optional title for the histogram
        ax: Optional matplotlib Axes to plot on, if None a new figure will be created
        
    Returns:
        The matplotlib Axes containing the histogram
    """
    if ax is None:
        fig, ax = plt.subplots()
    
    equalized_image, cdf = image_histogram_equalization(image, number_bins)
    
    ax.hist(equalized_image.flatten(), bins=256, color='gray', alpha=0.7)
    
    if title:
        ax.set_title(title)
    
    return ax