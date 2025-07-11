import numpy as np
from typing import Tuple, Optional
from scipy import ndimage

class AdaptiveThresholding:
    """Adaptive thresholding class for vessel detection using boxcar approach."""
    
    def __init__(
        self, 
        target_size: Tuple[int, int] = (3, 3),
        guard_size: Tuple[int, int] = (5, 5),
        background_size: Tuple[int, int] = (21, 21),
        pfa: float = 1e-6
    ):
        """Initialize adaptive thresholding parameters.
        
        Args:
            target_size: Size of target window (height, width).
            guard_size: Size of guard window (height, width).
            background_size: Size of background window (height, width).
            pfa: Probability of false alarm.
            
        Raises:
            AssertionError: If window sizes are invalid or pfa is out of range.
        """
        assert len(target_size) == 2 and all(s > 0 for s in target_size), 'Target size must be positive tuple of length 2'
        assert len(guard_size) == 2 and all(s > 0 for s in guard_size), 'Guard size must be positive tuple of length 2'
        assert len(background_size) == 2 and all(s > 0 for s in background_size), 'Background size must be positive tuple of length 2'
        assert 0 < pfa < 1, 'Probability of false alarm must be between 0 and 1'
        assert all(g >= t for g, t in zip(guard_size, target_size)), 'Guard window must be larger than or equal to target window'
        assert all(b >= g for b, g in zip(background_size, guard_size)), 'Background window must be larger than or equal to guard window'
        
        self.target_size = target_size
        self.guard_size = guard_size
        self.background_size = background_size
        self.pfa = pfa
        
    def _create_windows(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create target, guard, and background windows.
        
        Returns:
            Tuple of target, guard, and background window masks.
        """
        # Create background window (ones)
        background_window = np.ones(self.background_size, dtype=np.float64)
        
        # Create guard window (zeros in the middle)
        guard_window = np.ones(self.background_size, dtype=np.float64)
        h_bg, w_bg = self.background_size
        h_g, w_g = self.guard_size
        start_h = (h_bg - h_g) // 2
        start_w = (w_bg - w_g) // 2
        guard_window[start_h:start_h + h_g, start_w:start_w + w_g] = 0
        
        # Create target window (ones only in the center)
        target_window = np.zeros(self.background_size, dtype=np.float64)
        h_t, w_t = self.target_size
        start_h_t = (h_bg - h_t) // 2
        start_w_t = (w_bg - w_t) // 2
        target_window[start_h_t:start_h_t + h_t, start_w_t:start_w_t + w_t] = 1
        
        return target_window, guard_window, background_window
        
    def _calculate_threshold_multiplier(self, n_background: int) -> float:
        """Calculate threshold multiplier based on PFA and number of background samples.
        
        Args:
            n_background: Number of background samples.
            
        Returns:
            Threshold multiplier factor.
        """
        assert n_background > 0, 'Number of background samples must be positive'
        
        # For exponential distribution (common for SAR intensity data)
        # Threshold = mean * (-ln(PFA))
        return -np.log(self.pfa)
    
    def detect_vessels(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detect vessels using adaptive thresholding.
        
        Args:
            image: Input SAR image as 2D numpy array.
            
        Returns:
            Tuple of (detection_map, threshold_map) where detection_map is binary
            and threshold_map contains the adaptive threshold values.
            
        Raises:
            AssertionError: If input image is invalid.
        """
        assert isinstance(image, np.ndarray), 'Image must be a numpy array'
        assert image.ndim == 2, 'Image must be 2D'
        assert image.size > 0, 'Image must not be empty'
        
        target_window, guard_window, background_window = self._create_windows()
        
        # Calculate number of background samples
        n_background = int(np.sum(guard_window))
        assert n_background > 0, 'Background window must contain at least one sample'
        
        # Calculate threshold multiplier
        threshold_multiplier = self._calculate_threshold_multiplier(n_background)
        
        # Convolve image with windows to get local statistics
        target_sum = ndimage.convolve(image, target_window, mode='constant', cval=0.0)
        background_sum = ndimage.convolve(image, guard_window, mode='constant', cval=0.0)
        
        # Calculate target and background means
        n_target = int(np.sum(target_window))
        target_mean = target_sum / n_target
        background_mean = background_sum / n_background
        
        # Calculate adaptive threshold
        threshold_map = background_mean * threshold_multiplier
        
        # Create detection map
        detection_map = (target_mean > threshold_map).astype(np.uint8)
        
        return detection_map, threshold_map
    
    def set_pfa(self, pfa: float) -> None:
        """Set new probability of false alarm.
        
        Args:
            pfa: New probability of false alarm (e.g., 1e-6).
            
        Raises:
            AssertionError: If pfa is out of valid range.
        """
        assert isinstance(pfa, (int, float)), 'PFA must be a number'
        assert 0 < pfa < 1, 'Probability of false alarm must be between 0 and 1'
        self.pfa = pfa