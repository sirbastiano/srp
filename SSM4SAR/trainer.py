'''
Azimuth model trainer - Updated to work with DataModule

Key improvements for SAR image processing:
1. SSIM calculation now works on H×W dimensions (H=seq_len=10000, W=batch_size)
2. All losses (MAE, MSE, Huber, SSIM, Edge) now use magnitude of complex SAR data
3. Positional encoding channels are excluded from loss calculations
4. Edge loss operates on 2D magnitude structure for proper edge detection
5. Consistent magnitude extraction across all loss functions
'''

# import utilities
from sarSSM import sarSSM
from train_utils import EdgeLoss
from argparse import ArgumentParser
from skimage.metrics import structural_similarity as ssim
from lightning.pytorch.loggers import TensorBoardLogger
from torchmetrics import MeanMetric

import torch
import torch.nn as nn
import torch.optim as optim
import pykeops
import os
import sys
import numpy as np
import math
import lightning as pl
import kornia.losses as losses
import kornia as K


class azimuthModelTrainer(pl.LightningModule):
    def __init__(self, model, ssim_proportion, lr, weight_decay):
        super(azimuthModelTrainer, self).__init__()
        self.model = model
        self.delta = ssim_proportion
        self.lr = lr
        self.wd = weight_decay

        # define the loss functions
        self.mae_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.huber_loss = nn.HuberLoss(delta=1.0)
        self.ssim_loss = losses.SSIMLoss(window_size=7, reduction='mean')
        self.edge_loss = EdgeLoss()
        
        # initialize metrics to accumulate losses
        self.train_loss_metric = MeanMetric()
        self.val_loss_metric = MeanMetric()
        self.val_ssim_metric = MeanMetric()
        
    
    def forward(self, x):
        return self.model(x)
    
    def extract_magnitude_for_ssim(self, data):
        """
        Extract magnitude from complex SAR data for loss calculations.
        
        Args:
            data: Tensor of shape [batch_size, seq_len, channels] where channels = [real, imag, pos_enc_y, pos_enc_x]
                  Can be complex or real tensor
        
        Returns:
            magnitude: Tensor of shape [seq_len, batch_size] containing magnitude values for H×W structure
        """
        # Handle complex tensors by converting to real first
        if torch.is_complex(data):
            # If data is complex, extract real and imaginary parts
            real_part = data.real[:, :, 0]  # [batch_size, seq_len]
            imag_part = data.imag[:, :, 0] if data.shape[2] > 0 else torch.zeros_like(real_part)  # [batch_size, seq_len]
        else:
            # If data is real, extract real and imaginary parts from channels
            real_part = data[:, :, 0]  # [batch_size, seq_len]
            imag_part = data[:, :, 1] if data.shape[2] > 1 else torch.zeros_like(real_part)  # [batch_size, seq_len]
        
        # Ensure tensors are float
        real_part = real_part.float()
        imag_part = imag_part.float()
        
        # Calculate magnitude: sqrt(real^2 + imag^2)
        magnitude = torch.sqrt(real_part**2 + imag_part**2)  # [batch_size, seq_len]
        
        # Transpose to get [seq_len, batch_size] for H×W where H=seq_len, W=batch_size
        magnitude = magnitude.transpose(0, 1)  # [seq_len, batch_size]
        
        return magnitude

    def compute_loss(self, outputs_orig, targets_orig):
        """
        Compute all loss functions using magnitude data from complex SAR images.
        
        Args:
            outputs_orig: Original outputs with shape [batch_size, seq_len, channels] 
            targets_orig: Original targets with shape [batch_size, seq_len, channels]
                         channels = [real, imag, pos_enc_y, pos_enc_x]
        
        Returns:
            Tuple of all computed losses
        """
        # Handle complex tensors properly
        if torch.is_complex(outputs_orig):
            # Convert complex to real representation [batch, seq, channels]
            outputs_real = torch.stack([outputs_orig.real, outputs_orig.imag], dim=-1)
            if outputs_orig.dim() == 2:  # [batch, seq] -> [batch, seq, 2]
                outputs_orig = outputs_real
            else:  # [batch, seq, complex_channels] -> [batch, seq, 2*complex_channels]
                outputs_orig = outputs_real.view(*outputs_orig.shape[:-1], -1)
        
        if torch.is_complex(targets_orig):
            # Convert complex to real representation [batch, seq, channels]
            targets_real = torch.stack([targets_orig.real, targets_orig.imag], dim=-1)
            if targets_orig.dim() == 2:  # [batch, seq] -> [batch, seq, 2]
                targets_orig = targets_real
            else:  # [batch, seq, complex_channels] -> [batch, seq, 2*complex_channels]
                targets_orig = targets_real.view(*targets_orig.shape[:-1], -1)
        
        # Ensure data is float
        outputs_orig = outputs_orig.float()
        targets_orig = targets_orig.float()
        
        # Extract magnitude for all loss calculations (excludes positional encoding)
        outputs_magnitude = self.extract_magnitude_for_ssim(outputs_orig)  # [seq_len, batch_size]
        targets_magnitude = self.extract_magnitude_for_ssim(targets_orig)  # [seq_len, batch_size]
        
        # Ensure magnitude tensors are float
        outputs_magnitude = outputs_magnitude.float()
        targets_magnitude = targets_magnitude.float()
        
        # For pixel-wise losses (MAE, MSE, Huber), we can use the magnitude directly
        loss_mae = self.mae_loss(outputs_magnitude, targets_magnitude)
        loss_mse = self.mse_loss(outputs_magnitude, targets_magnitude)
        loss_huber = self.huber_loss(outputs_magnitude, targets_magnitude)
        
        # For SSIM loss, use kornia with proper reshaping
        # Reshape to [batch=1, channel=1, H, W] for kornia
        outputs_ssim = outputs_magnitude.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, batch_size]
        targets_ssim = targets_magnitude.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, batch_size]
        loss_ssim = self.ssim_loss(outputs_ssim, targets_ssim)
        
        # For Edge loss, we need 4D tensors [batch, channel, H, W]
        # The EdgeLoss expects [N, C, H, W] format
        outputs_edge = outputs_magnitude.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, batch_size]
        targets_edge = targets_magnitude.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, batch_size]
        loss_edge = self.edge_loss(outputs_edge, targets_edge)
        
        # Ensure all losses are real scalars and handle any remaining complex values
        loss_mae = loss_mae.real if torch.is_complex(loss_mae) else loss_mae
        loss_mse = loss_mse.real if torch.is_complex(loss_mse) else loss_mse
        loss_huber = loss_huber.real if torch.is_complex(loss_huber) else loss_huber
        loss_ssim = loss_ssim.real if torch.is_complex(loss_ssim) else loss_ssim
        loss_edge = loss_edge.real if torch.is_complex(loss_edge) else loss_edge
        
        # Replace any NaN or infinite losses with zeros
        loss_mae = torch.where(torch.isfinite(loss_mae), loss_mae, torch.tensor(0.0, device=loss_mae.device))
        loss_mse = torch.where(torch.isfinite(loss_mse), loss_mse, torch.tensor(0.0, device=loss_mse.device))
        loss_huber = torch.where(torch.isfinite(loss_huber), loss_huber, torch.tensor(0.0, device=loss_huber.device))
        loss_ssim = torch.where(torch.isfinite(loss_ssim), loss_ssim, torch.tensor(0.0, device=loss_ssim.device))
        loss_edge = torch.where(torch.isfinite(loss_edge), loss_edge, torch.tensor(0.0, device=loss_edge.device))
        
        return loss_mae, loss_mse, loss_huber, loss_ssim, loss_edge
        
    def configure_optimizers(self):
        '''
        S4 requires a specific optimizer setup.

        The S4 layer (A, B, C, dt) parameters typically require
        a smaller learning rate with no weight decay.

        The rest of the model can be trained with a higher learning rate (e.g. 0.004, 0.01)
        and weight decay (if desired)
        '''
        # All parameters in the model
        all_parameters = list(self.model.parameters())
        
        # General parameters don't contain the special _optim key
        params = [p for p in all_parameters if not hasattr(p, "_optim")]
        
        # Create an optimizer with general paramters
        optimizer = optim.AdamW(params, lr=self.lr, weight_decay=self.wd)
        
        # Add parameters with special hyperparameters
        hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
        hps = [
            dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
        ] # unique dicts
        for hp in hps:
            params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
            optimizer.add_param_group(
                {"params": params, **hp}
            )  
            
        # Print optimizer info
        keys = sorted(set([k for hp in hps for k in hp.keys()]))
        for i, g in enumerate(optimizer.param_groups):
            group_hps = {k: g.get(k, None) for k in keys}
            print(' | '.join([
                f"Optimizer group {i}", 
                f"{len(g['params'])} tensors",
            ] + [f"{k} {v}" for k, v in group_hps.items()]))
            
        print(f"\n\n stepping_batches = {self.trainer.estimated_stepping_batches}", flush=True)
                
        return optimizer
    
    def training_step(self, batch, batch_idx):
        raw, gt = batch
        if raw.shape[0] < 10:
            raise ValueError("batch size must be above 10 for your loss functions to work")
        
        # Squeeze the width dimension: [batch, seq, 1, channels] -> [batch, seq, channels]
        raw = raw.squeeze(2)
        gt = gt.squeeze(2)
        
        # Store original data for SSIM calculation
        gt_orig = gt.clone()
        
        # Convert complex to real if needed
        if torch.is_complex(raw):
            raw = torch.abs(raw).float()
        if torch.is_complex(gt):
            gt = torch.abs(gt).float()
            
        outputs = self.model(raw).squeeze()
        
        # Store original outputs for SSIM calculation (ensure it has same structure as gt_orig)
        if outputs.dim() == 2:
            # If outputs is [batch, seq], we need to add channels dimension
            # Assume outputs represents the complex magnitude (single channel)
            outputs_orig = outputs.unsqueeze(-1).repeat(1, 1, 2)  # [batch, seq, 2] for real/imag
            # Set imaginary part to zero since we only have magnitude
            outputs_orig[:, :, 1] = 0
        elif outputs.dim() == 3:
            outputs_orig = outputs.clone()
        else:
            raise ValueError(f"Unexpected outputs dimension: {outputs.dim()}")
        
        # compute combined loss with original data for all losses (now using magnitude)
        loss_mae, loss_mse, loss_huber, loss_ssim, loss_edge = self.compute_loss(outputs_orig, gt_orig)
        
        # get ssim of output and gt using corrected magnitude calculation
        outputs_magnitude = self.extract_magnitude_for_ssim(outputs_orig)
        targets_magnitude = self.extract_magnitude_for_ssim(gt_orig)
        train_ssim = self.calculate_ssim(outputs_magnitude, targets_magnitude)
        
        # update training loss metric
        self.train_loss_metric.update(loss_mae)
        
        # Log losses - show in progress bar and log every step
        self.log("train_loss", loss_mae, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train_ssim", train_ssim, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train_mae", loss_mae, on_step=True, on_epoch=False)
        self.log("train_mse", loss_mse, on_step=True, on_epoch=False)
        self.log("train_huber", loss_huber, on_step=True, on_epoch=False)
        self.log("train_ssim_loss", loss_ssim, on_step=True, on_epoch=False)
        self.log("train_edge", loss_edge, on_step=True, on_epoch=False)
        
        return loss_mae*(1-self.delta) + loss_ssim*(self.delta)
    
    def validation_step(self, batch, idx):
        raw, gt = batch
        if raw.shape[0] < 10:
            raise ValueError("batch size must be above 10 for your loss functions to work")
        
        # Squeeze the width dimension: [batch, seq, 1, channels] -> [batch, seq, channels]
        raw = raw.squeeze(2)
        gt = gt.squeeze(2)
        
        # Store original data for SSIM calculation
        gt_orig = gt.clone()
        
        # Convert complex to real if needed
        if torch.is_complex(raw):
            raw = torch.abs(raw).float()
        if torch.is_complex(gt):
            gt = torch.abs(gt).float()
        
        outputs = self.model(raw).squeeze()
        
        # Store original outputs for SSIM calculation (ensure it has same structure as gt_orig)
        if outputs.dim() == 2:
            # If outputs is [batch, seq], we need to add channels dimension
            # Assume outputs represents the complex magnitude (single channel)
            outputs_orig = outputs.unsqueeze(-1).repeat(1, 1, 2)  # [batch, seq, 2] for real/imag
            # Set imaginary part to zero since we only have magnitude
            outputs_orig[:, :, 1] = 0
        elif outputs.dim() == 3:
            outputs_orig = outputs.clone()
        else:
            raise ValueError(f"Unexpected outputs dimension: {outputs.dim()}")
        
        # compute combined loss with original data for all losses (now using magnitude)
        loss_mae, loss_mse, loss_huber, loss_ssim, loss_edge = self.compute_loss(outputs_orig, gt_orig)
        
        # update validation loss metric
        self.val_loss_metric.update(loss_mae)
        
        # compute SSIM for logging using corrected magnitude calculation
        outputs_magnitude = self.extract_magnitude_for_ssim(outputs_orig)
        targets_magnitude = self.extract_magnitude_for_ssim(gt_orig)
        ssim_value = self.calculate_ssim(outputs_magnitude, targets_magnitude)
        self.val_ssim_metric.update(ssim_value)
        
        # Log metrics
        self.log("val_loss", loss_mae, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_ssim", ssim_value, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_mae", loss_mae, on_step=False, on_epoch=True)
        self.log("val_mse", loss_mse, on_step=False, on_epoch=True)
        self.log("val_huber", loss_huber, on_step=False, on_epoch=True)
        self.log("val_ssim_loss", loss_ssim, on_step=False, on_epoch=True)
        self.log("val_edge", loss_edge, on_step=False, on_epoch=True)
        
        return loss_mae
       
    def on_training_epoch_end(self):
        # this is where you can perform actions at the end of the test epoch
        self.train_loss_metric.compute()
        self.log("epoch_train_loss", self.train_loss_metric, on_step=False, on_epoch=True, prog_bar=False)
        self.train_loss_metric.reset()
        pass
         
    def on_validation_epoch_end(self):
        # Compute and log average validation loss
        avg_val_loss = self.val_loss_metric.compute()
        avg_val_ssim = self.val_ssim_metric.compute()
        
        # Log summary metrics with clear names
        self.log('val_loss_avg', avg_val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_ssim_avg', avg_val_ssim, on_step=False, on_epoch=True, prog_bar=True)
        
        # Print summary for visibility
        print(f"\n[Validation] Epoch End - Avg Loss: {avg_val_loss:.4f}, Avg SSIM: {avg_val_ssim:.4f}")
        
        self.val_loss_metric.reset()
        self.val_ssim_metric.reset()
        
    def calculate_psnr(self, outputs, targets):
        """Calculates PSNR between outputs and targets"""
        mse = torch.mean((outputs - targets) ** 2)
        if mse == 0:
            return torch.tensor(float("inf")) # Avoid division by zero
        max_pixel = 1.0 # assuming normalized [0, 1] images
        psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
        return psnr
    
    def calculate_ssim(self, outputs_magnitude, targets_magnitude):
        """
        Calculate SSIM between magnitude outputs and targets with proper H×W dimensions.
        
        Args:
            outputs_magnitude: Tensor of shape [seq_len, batch_size] (H×W where H=seq_len, W=batch_size)
            targets_magnitude: Tensor of shape [seq_len, batch_size] (H×W where H=seq_len, W=batch_size)
        
        Returns:
            ssim_value: Scalar tensor containing SSIM value
        """
        # Convert to numpy for skimage SSIM calculation
        outputs = outputs_magnitude.detach().cpu().numpy()
        targets = targets_magnitude.detach().cpu().numpy()
        
        # outputs and targets are now [seq_len, batch_size] which is [H, W]
        height, width = outputs.shape
        
        # Dynamically set win_size based on smallest dimension
        win_size = min(7, height, width)
        if win_size % 2 == 0:
            win_size -= 1
        
        # Make sure win_size is at least 3
        win_size = max(3, win_size)
        
        # Calculate SSIM on the 2D image where H=seq_len and W=batch_size
        ssim_value = ssim(
            targets, outputs,  # Note: targets first for consistency
            data_range=targets.max() - targets.min(),
            win_size=win_size
        )
        
        return torch.tensor(ssim_value, dtype=torch.float32)