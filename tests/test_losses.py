"""
Test module for SAR loss functions.

This module provides comprehensive testing for all loss functions in the sarpyx.utils.losses module,
including standard loss functions and specialized SAR loss functions for complex-valued data.
"""

import torch
import pytest
import numpy as np
from typing import Type

from sarpyx.utils.losses import (
    BaseLoss,
    MSELoss,
    MAELoss,
    HuberLoss,
    FocalLoss,
    ComplexMSELoss,
    PhaseLoss,
    CombinedComplexLoss,
    get_loss_function
)


class TestBaseLoss:
    """Test cases for BaseLoss abstract base class."""
    
    def test_invalid_reduction(self):
        """Test that invalid reduction raises AssertionError."""
        with pytest.raises(AssertionError):
            class DummyLoss(BaseLoss):
                def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
                    return torch.tensor(0.0)
            DummyLoss(reduction='invalid')
    
    def test_valid_reductions(self):
        """Test that valid reductions are accepted."""
        class DummyLoss(BaseLoss):
            def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
                return torch.tensor([1.0, 2.0, 3.0])
        
        for reduction in ['none', 'mean', 'sum']:
            loss_fn = DummyLoss(reduction=reduction)
            assert loss_fn.reduction == reduction
    
    def test_reduce_functionality(self):
        """Test the _reduce method works correctly."""
        class DummyLoss(BaseLoss):
            def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
                return torch.tensor([1.0, 2.0, 3.0])
        
        loss_tensor = torch.tensor([1.0, 2.0, 3.0])
        
        # Test mean reduction
        loss_fn = DummyLoss(reduction='mean')
        assert torch.isclose(loss_fn._reduce(loss_tensor), torch.tensor(2.0))
        
        # Test sum reduction
        loss_fn = DummyLoss(reduction='sum')
        assert torch.isclose(loss_fn._reduce(loss_tensor), torch.tensor(6.0))
        
        # Test none reduction
        loss_fn = DummyLoss(reduction='none')
        assert torch.allclose(loss_fn._reduce(loss_tensor), loss_tensor)


class TestMSELoss:
    """Test cases for MSE Loss function."""
    
    @pytest.fixture
    def mse_loss(self):
        """Create MSE loss instance."""
        return MSELoss()
    
    def test_identical_tensors(self, mse_loss):
        """Test MSE loss with identical prediction and target."""
        prediction = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.0, 2.0, 3.0])
        loss = mse_loss(prediction, target)
        assert torch.isclose(loss, torch.tensor(0.0))
    
    def test_different_tensors(self, mse_loss):
        """Test MSE loss with different prediction and target."""
        prediction = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([2.0, 3.0, 4.0])
        expected_loss = torch.mean((prediction - target) ** 2)
        loss = mse_loss(prediction, target)
        assert torch.isclose(loss, expected_loss)
    
    def test_shape_mismatch(self, mse_loss):
        """Test that shape mismatch raises AssertionError."""
        prediction = torch.tensor([1.0, 2.0])
        target = torch.tensor([1.0, 2.0, 3.0])
        with pytest.raises(AssertionError):
            mse_loss(prediction, target)
    
    def test_multidimensional_input(self, mse_loss):
        """Test MSE loss with multidimensional tensors."""
        prediction = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        target = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
        expected_loss = torch.mean((prediction - target) ** 2)
        loss = mse_loss(prediction, target)
        assert torch.isclose(loss, expected_loss)
    
    def test_reduction_modes(self):
        """Test different reduction modes for MSE loss."""
        prediction = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([2.0, 3.0, 4.0])
        squared_diff = (prediction - target) ** 2
        
        # Test mean reduction
        mse_mean = MSELoss(reduction='mean')
        assert torch.isclose(mse_mean(prediction, target), squared_diff.mean())
        
        # Test sum reduction
        mse_sum = MSELoss(reduction='sum')
        assert torch.isclose(mse_sum(prediction, target), squared_diff.sum())
        
        # Test no reduction
        mse_none = MSELoss(reduction='none')
        assert torch.allclose(mse_none(prediction, target), squared_diff)


class TestMAELoss:
    """Test cases for MAE Loss function."""
    
    @pytest.fixture
    def mae_loss(self):
        """Create MAE loss instance."""
        return MAELoss()
    
    def test_identical_tensors(self, mae_loss):
        """Test MAE loss with identical prediction and target."""
        prediction = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.0, 2.0, 3.0])
        loss = mae_loss(prediction, target)
        assert torch.isclose(loss, torch.tensor(0.0))
    
    def test_different_tensors(self, mae_loss):
        """Test MAE loss with different prediction and target."""
        prediction = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([2.0, 3.0, 4.0])
        expected_loss = torch.mean(torch.abs(prediction - target))
        loss = mae_loss(prediction, target)
        assert torch.isclose(loss, expected_loss)
    
    def test_negative_values(self, mae_loss):
        """Test MAE loss with negative values."""
        prediction = torch.tensor([-1.0, -2.0, -3.0])
        target = torch.tensor([1.0, 2.0, 3.0])
        expected_loss = torch.mean(torch.abs(prediction - target))
        loss = mae_loss(prediction, target)
        assert torch.isclose(loss, expected_loss)
    
    def test_shape_mismatch(self, mae_loss):
        """Test that shape mismatch raises AssertionError."""
        prediction = torch.tensor([1.0, 2.0])
        target = torch.tensor([1.0, 2.0, 3.0])
        with pytest.raises(AssertionError):
            mae_loss(prediction, target)


class TestHuberLoss:
    """Test cases for Huber Loss function."""
    
    def test_delta_validation(self):
        """Test that non-positive delta raises AssertionError."""
        with pytest.raises(AssertionError):
            HuberLoss(delta=0.0)
        with pytest.raises(AssertionError):
            HuberLoss(delta=-1.0)
    
    def test_quadratic_region(self):
        """Test Huber loss in quadratic region (|residual| < delta)."""
        huber_loss = HuberLoss(delta=2.0)
        prediction = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.5, 2.5, 3.5])  # residuals = [0.5, 0.5, 0.5] < 2.0
        
        # Should be quadratic: 0.5 * residual^2
        expected_loss = torch.mean(0.5 * torch.tensor([0.5, 0.5, 0.5]) ** 2)
        loss = huber_loss(prediction, target)
        assert torch.isclose(loss, expected_loss)
    
    def test_linear_region(self):
        """Test Huber loss in linear region (|residual| >= delta)."""
        huber_loss = HuberLoss(delta=1.0)
        prediction = torch.tensor([0.0, 0.0, 0.0])
        target = torch.tensor([3.0, 3.0, 3.0])  # residuals = [3.0, 3.0, 3.0] >= 1.0
        
        # Should be linear: delta * residual - 0.5 * delta^2
        expected_loss = torch.mean(1.0 * 3.0 - 0.5 * 1.0 ** 2)
        loss = huber_loss(prediction, target)
        assert torch.isclose(loss, expected_loss)
    
    def test_transition_point(self):
        """Test Huber loss at the transition point."""
        huber_loss = HuberLoss(delta=1.0)
        prediction = torch.tensor([0.0])
        target = torch.tensor([1.0])  # residual = 1.0 = delta
        
        # Both formulas should give the same result at transition
        quadratic_result = 0.5 * 1.0 ** 2
        linear_result = 1.0 * 1.0 - 0.5 * 1.0 ** 2
        assert torch.isclose(torch.tensor(quadratic_result), torch.tensor(linear_result))
        
        loss = huber_loss(prediction, target)
        assert torch.isclose(loss, torch.tensor(quadratic_result))


class TestFocalLoss:
    """Test cases for Focal Loss function."""
    
    def test_parameter_validation(self):
        """Test parameter validation for Focal Loss."""
        # Valid parameters
        FocalLoss(alpha=1.0, gamma=2.0)
        
        # Invalid gamma
        with pytest.raises(AssertionError):
            FocalLoss(gamma=-1.0)
        
        # Invalid alpha
        with pytest.raises(AssertionError):
            FocalLoss(alpha=0.0)
        with pytest.raises(AssertionError):
            FocalLoss(alpha=-1.0)
    
    def test_input_shape_validation(self):
        """Test input shape validation for Focal Loss."""
        focal_loss = FocalLoss()
        
        # Valid inputs
        prediction = torch.randn(10, 5)  # (batch_size, num_classes)
        target = torch.randint(0, 5, (10,))  # (batch_size,)
        focal_loss(prediction, target)
        
        # Invalid prediction shape
        with pytest.raises(AssertionError):
            prediction_1d = torch.randn(10)
            focal_loss(prediction_1d, target)
        
        # Invalid target shape
        with pytest.raises(AssertionError):
            target_2d = torch.randint(0, 5, (10, 1))
            focal_loss(prediction, target_2d)
        
        # Batch size mismatch
        with pytest.raises(AssertionError):
            target_wrong_size = torch.randint(0, 5, (5,))
            focal_loss(prediction, target_wrong_size)
    
    def test_focal_loss_computation(self):
        """Test Focal Loss computation with known values."""
        focal_loss = FocalLoss(alpha=1.0, gamma=2.0)
        
        # Simple case: 2 classes, perfect prediction
        prediction = torch.tensor([[10.0, -10.0], [-10.0, 10.0]])  # Strong predictions
        target = torch.tensor([0, 1])
        
        loss = focal_loss(prediction, target)
        # With perfect predictions, focal loss should be very small
        assert loss < 0.1
    
    def test_gamma_effect(self):
        """Test that gamma parameter affects focusing."""
        prediction = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        target = torch.tensor([0, 1])
        
        # Higher gamma should give lower loss for well-classified examples
        focal_loss_low_gamma = FocalLoss(gamma=0.0)
        focal_loss_high_gamma = FocalLoss(gamma=2.0)
        
        loss_low = focal_loss_low_gamma(prediction, target)
        loss_high = focal_loss_high_gamma(prediction, target)
        
        # With gamma=0, focal loss reduces to cross-entropy
        # With gamma>0, well-classified examples get down-weighted
        assert loss_high <= loss_low


class TestComplexMSELoss:
    """Test cases for Complex MSE Loss function."""
    
    @pytest.fixture
    def complex_mse_loss(self):
        """Create Complex MSE loss instance."""
        return ComplexMSELoss()
    
    def test_identical_complex_tensors(self, complex_mse_loss):
        """Test Complex MSE loss with identical prediction and target."""
        prediction = torch.complex(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))
        target = torch.complex(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))
        loss = complex_mse_loss(prediction, target)
        assert torch.isclose(loss, torch.tensor(0.0))
    
    def test_different_complex_tensors(self, complex_mse_loss):
        """Test Complex MSE loss with different prediction and target."""
        prediction = torch.complex(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))
        target = torch.complex(torch.tensor([2.0, 3.0]), torch.tensor([4.0, 5.0]))
        
        diff = prediction - target
        expected_loss = torch.mean(torch.abs(diff) ** 2)
        loss = complex_mse_loss(prediction, target)
        assert torch.isclose(loss, expected_loss)
    
    def test_real_tensor_rejection(self, complex_mse_loss):
        """Test that real tensors are rejected."""
        prediction = torch.tensor([1.0, 2.0])
        target = torch.tensor([2.0, 3.0])
        
        with pytest.raises(AssertionError):
            complex_mse_loss(prediction, target)
    
    def test_mixed_tensor_types(self, complex_mse_loss):
        """Test that mixed real/complex tensors are rejected."""
        prediction = torch.complex(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))
        target = torch.tensor([2.0, 3.0])
        
        with pytest.raises(AssertionError):
            complex_mse_loss(prediction, target)


class TestPhaseLoss:
    """Test cases for Phase Loss function."""
    
    @pytest.fixture
    def phase_loss(self):
        """Create Phase loss instance."""
        return PhaseLoss()
    
    def test_identical_phases(self, phase_loss):
        """Test Phase loss with identical phases."""
        # Complex numbers with same phase
        magnitude = torch.tensor([1.0, 2.0, 3.0])
        phase = torch.tensor([0.5, 1.0, 1.5])
        
        prediction = magnitude * torch.exp(1j * phase)
        target = magnitude * torch.exp(1j * phase)
        
        loss = phase_loss(prediction, target)
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)
    
    def test_opposite_phases(self, phase_loss):
        """Test Phase loss with opposite phases."""
        # Create complex numbers with opposite phases
        prediction = torch.complex(torch.tensor([1.0]), torch.tensor([0.0]))  # phase = 0
        target = torch.complex(torch.tensor([-1.0]), torch.tensor([0.0]))  # phase = π
        
        loss = phase_loss(prediction, target)
        expected_loss = torch.pi  # Maximum phase difference
        assert torch.isclose(loss, expected_loss, atol=1e-5)
    
    def test_circular_phase_distance(self, phase_loss):
        """Test that phase distance is computed circularly."""
        # Phases close to 0 and 2π should have small distance
        prediction = torch.complex(torch.tensor([1.0]), torch.tensor([0.1]))  # Small positive phase
        target = torch.complex(torch.tensor([1.0]), torch.tensor([-0.1]))  # Small negative phase
        
        loss = phase_loss(prediction, target)
        expected_loss = 0.2  # Direct difference
        assert torch.isclose(loss, expected_loss, atol=1e-5)
    
    def test_real_tensor_rejection(self, phase_loss):
        """Test that real tensors are rejected."""
        prediction = torch.tensor([1.0, 2.0])
        target = torch.tensor([2.0, 3.0])
        
        with pytest.raises(AssertionError):
            phase_loss(prediction, target)


class TestCombinedComplexLoss:
    """Test cases for Combined Complex Loss function."""
    
    def test_parameter_validation(self):
        """Test parameter validation for Combined Complex Loss."""
        # Valid parameters
        CombinedComplexLoss(magnitude_weight=1.0, phase_weight=1.0)
        
        # Invalid weights
        with pytest.raises(AssertionError):
            CombinedComplexLoss(magnitude_weight=-1.0)
        with pytest.raises(AssertionError):
            CombinedComplexLoss(phase_weight=-1.0)
    
    def test_weight_effects(self):
        """Test that weights affect the combined loss appropriately."""
        prediction = torch.complex(torch.tensor([1.0]), torch.tensor([0.0]))
        target = torch.complex(torch.tensor([2.0]), torch.tensor([1.0]))
        
        # Test magnitude-only loss
        mag_only_loss = CombinedComplexLoss(magnitude_weight=1.0, phase_weight=0.0)
        mag_loss = mag_only_loss(prediction, target)
        
        # Test phase-only loss
        phase_only_loss = CombinedComplexLoss(magnitude_weight=0.0, phase_weight=1.0)
        phase_loss = phase_only_loss(prediction, target)
        
        # Test equal weights
        equal_loss = CombinedComplexLoss(magnitude_weight=1.0, phase_weight=1.0)
        combined_loss = equal_loss(prediction, target)
        
        # Combined loss should be roughly the sum of individual losses
        assert torch.isclose(combined_loss, mag_loss + phase_loss, atol=1e-5)
    
    def test_zero_weights(self):
        """Test combined loss with zero weights."""
        prediction = torch.complex(torch.tensor([1.0]), torch.tensor([0.0]))
        target = torch.complex(torch.tensor([2.0]), torch.tensor([1.0]))
        
        # Zero weights should give zero loss
        zero_loss = CombinedComplexLoss(magnitude_weight=0.0, phase_weight=0.0)
        loss = zero_loss(prediction, target)
        assert torch.isclose(loss, torch.tensor(0.0))


class TestLossFactory:
    """Test cases for loss function factory."""
    
    def test_get_valid_loss_functions(self):
        """Test getting all valid loss functions."""
        loss_names = ['mse', 'mae', 'huber', 'focal', 'complex_mse', 'phase', 'combined_complex']
        
        for loss_name in loss_names:
            loss_fn = get_loss_function(loss_name)
            assert isinstance(loss_fn, BaseLoss)
    
    def test_case_insensitive(self):
        """Test that loss function names are case insensitive."""
        loss_fn_lower = get_loss_function('mse')
        loss_fn_upper = get_loss_function('MSE')
        loss_fn_mixed = get_loss_function('MsE')
        
        assert type(loss_fn_lower) == type(loss_fn_upper) == type(loss_fn_mixed)
    
    def test_invalid_loss_name(self):
        """Test that invalid loss names raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            get_loss_function('invalid_loss')
        
        assert 'Unknown loss function' in str(exc_info.value)
        assert 'invalid_loss' in str(exc_info.value)
    
    def test_loss_with_parameters(self):
        """Test creating loss functions with parameters."""
        # Test Huber loss with custom delta
        huber_loss = get_loss_function('huber', delta=2.0)
        assert isinstance(huber_loss, HuberLoss)
        assert huber_loss.delta == 2.0
        
        # Test Focal loss with custom parameters
        focal_loss = get_loss_function('focal', alpha=0.5, gamma=3.0)
        assert isinstance(focal_loss, FocalLoss)
        assert focal_loss.alpha == 0.5
        assert focal_loss.gamma == 3.0
        
        # Test Combined Complex loss with custom weights
        combined_loss = get_loss_function(
            'combined_complex', 
            magnitude_weight=2.0, 
            phase_weight=0.5
        )
        assert isinstance(combined_loss, CombinedComplexLoss)
        assert combined_loss.magnitude_weight == 2.0
        assert combined_loss.phase_weight == 0.5


class TestLossIntegration:
    """Integration tests for loss functions."""
    
    def test_loss_functions_with_gradients(self):
        """Test that loss functions work with gradient computation."""
        # Test with real-valued tensors
        prediction = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        target = torch.tensor([1.5, 2.5, 3.5])
        
        mse_loss = MSELoss()
        loss = mse_loss(prediction, target)
        loss.backward()
        
        assert prediction.grad is not None
        assert prediction.grad.shape == prediction.shape
    
    def test_complex_loss_with_gradients(self):
        """Test that complex loss functions work with gradient computation."""
        # Create complex tensors with gradients
        real_part = torch.tensor([1.0, 2.0], requires_grad=True)
        imag_part = torch.tensor([3.0, 4.0], requires_grad=True)
        prediction = torch.complex(real_part, imag_part)
        
        target = torch.complex(torch.tensor([1.5, 2.5]), torch.tensor([3.5, 4.5]))
        
        complex_mse = ComplexMSELoss()
        loss = complex_mse(prediction, target)
        loss.backward()
        
        assert real_part.grad is not None
        assert imag_part.grad is not None
    
    def test_loss_numerical_stability(self):
        """Test numerical stability of loss functions."""
        # Test with very small values
        prediction = torch.tensor([1e-8, 1e-8])
        target = torch.tensor([2e-8, 2e-8])
        
        mse_loss = MSELoss()
        loss = mse_loss(prediction, target)
        assert torch.isfinite(loss)
        
        # Test with very large values
        prediction = torch.tensor([1e8, 1e8])
        target = torch.tensor([2e8, 2e8])
        
        loss = mse_loss(prediction, target)
        assert torch.isfinite(loss)
    
    def test_batch_processing(self):
        """Test loss functions with batch processing."""
        batch_size = 32
        feature_dim = 64
        
        prediction = torch.randn(batch_size, feature_dim)
        target = torch.randn(batch_size, feature_dim)
        
        mse_loss = MSELoss()
        loss = mse_loss(prediction, target)
        
        assert loss.shape == torch.Size([])  # Scalar loss
        assert torch.isfinite(loss)


if __name__ == '__main__':
    pytest.main([__file__])