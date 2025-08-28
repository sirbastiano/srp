'''
Azimuth model trainer - Updated to work with DataModule
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

    def compute_loss(self, outputs, targets):
        # compute individual losses
        loss_mae = self.mae_loss(outputs, targets)
        loss_mse = self.mse_loss(outputs, targets)
        loss_huber = self.huber_loss(outputs, targets)
        loss_ssim = self.ssim_loss(outputs.unsqueeze(0).unsqueeze(0), targets.unsqueeze(0).unsqueeze(0))
        loss_edge = self.edge_loss(outputs.unsqueeze(0).unsqueeze(0), targets.unsqueeze(0).unsqueeze(0))
        
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
        
        # Convert complex to real if needed
        if torch.is_complex(raw):
            raw = torch.abs(raw).float()
        if torch.is_complex(gt):
            gt = torch.abs(gt).float()
            
        outputs = self.model(raw).squeeze()
        
        # you need to flatten gt and outputs so that you can calculate loss on them
        # Dynamically handle shapes
        batch_size, seq_len, channels = gt.shape
        gt_flat = gt.permute(1, 0, 2)  # [seq_len, batch_size, channels]
        gt_flat = gt_flat.reshape(seq_len, batch_size * channels)
        
        outputs_flat = outputs.permute(1, 0, 2) if outputs.dim() == 3 else outputs
        if outputs_flat.shape != gt_flat.shape:
            # Handle shape mismatch - outputs might have different channels
            outputs_flat = outputs_flat.reshape(seq_len, -1)
            if outputs_flat.shape[1] != gt_flat.shape[1]:
                # Pad or truncate to match gt shape
                min_channels = min(outputs_flat.shape[1], gt_flat.shape[1])
                outputs_flat = outputs_flat[:, :min_channels]
                gt_flat = gt_flat[:, :min_channels]
        
        # compute combined loss
        loss_mae, loss_mse, loss_huber, loss_ssim, loss_edge = self.compute_loss(outputs_flat, gt_flat)
        
        # get ssim of output and gt
        train_ssim = self.calculate_ssim(outputs_flat.unsqueeze(0).unsqueeze(0), gt_flat.unsqueeze(0).unsqueeze(0))
        
        # update training loss metric
        self.train_loss_metric.update(loss_mae)
        
        # Log losses - show in progress bar and log every step
        self.log("train_loss", loss_mae, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_ssim", train_ssim, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_mae", loss_mae, on_step=True, on_epoch=True)
        self.log("train_mse", loss_mse, on_step=True, on_epoch=True)
        self.log("train_huber", loss_huber, on_step=False, on_epoch=True)
        self.log("train_ssim_loss", loss_ssim, on_step=False, on_epoch=True)
        self.log("train_edge", loss_edge, on_step=False, on_epoch=True)
        
        return loss_mae*(1-self.delta) + loss_ssim*(self.delta)
    
    def validation_step(self, batch, idx):
        raw, gt = batch
        if raw.shape[0] < 10:
            raise ValueError("batch size must be above 10 for your loss functions to work")
        
        # Squeeze the width dimension: [batch, seq, 1, channels] -> [batch, seq, channels]
        raw = raw.squeeze(2)
        gt = gt.squeeze(2)
        
        # Convert complex to real if needed
        if torch.is_complex(raw):
            raw = torch.abs(raw).float()
        if torch.is_complex(gt):
            gt = torch.abs(gt).float()
        
        outputs = self.model(raw).squeeze()
        
        # you need to flatten gt and outputs so that you can calculate loss on them
        # Dynamically handle shapes
        batch_size, seq_len, channels = gt.shape
        gt_flat = gt.permute(1, 0, 2)  # [seq_len, batch_size, channels]
        gt_flat = gt_flat.reshape(seq_len, batch_size * channels)
        
        outputs_flat = outputs.permute(1, 0, 2) if outputs.dim() == 3 else outputs
        if outputs_flat.shape != gt_flat.shape:
            # Handle shape mismatch - outputs might have different channels
            outputs_flat = outputs_flat.reshape(seq_len, -1)
            if outputs_flat.shape[1] != gt_flat.shape[1]:
                # Pad or truncate to match gt shape
                min_channels = min(outputs_flat.shape[1], gt_flat.shape[1])
                outputs_flat = outputs_flat[:, :min_channels]
                gt_flat = gt_flat[:, :min_channels]
        
        # compute combined loss
        loss_mae, loss_mse, loss_huber, loss_ssim, loss_edge = self.compute_loss(outputs_flat, gt_flat)
        
        # update validation loss metric
        self.val_loss_metric.update(loss_mae)
        
        # compute PSNR and SSIM for logging 
        # psnr_value = self.calculate_psnr(outputs_flat, gt_flat)
        ssim_value = self.calculate_ssim(outputs_flat.unsqueeze(0).unsqueeze(0), gt_flat.unsqueeze(0).unsqueeze(0))
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
    
    def calculate_ssim(self, outputs, targets):
        """Calculates SSIMbetween outputs and targets."""
        outputs = outputs.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()
        
        # Squeeze the channel dimension if it's 1
        if outputs.shape[1] == 1:
            outputs = outputs.squeeze(1)
            targets = targets.squeeze(1)
            channel_axis = None
        else:
            outputs = np.moveaxis(outputs, 1, -1)
            targets = np.moveaxis(targets, 1, -1)
            channel_axis = -1
            
        ssim_values = []
        for i in range(outputs.shape[0]): # iterate over the batch
            output = outputs[i]
            target = targets[i]
            
            # Dynamically set win_size based on smallest dimension
            height, width = output.shape[:2]
            win_size = min(7, height, width)
            if win_size % 2 == 0:
                win_size -=1
                
            ssim_value = ssim(
                output, target,
                data_range=output.max() - output.min(),
                win_size=win_size,
                channel_axis=channel_axis # None for grayscale, -1 for multichannel
            )
            ssim_values.append(ssim_value)
            
        return torch.tensor(np.mean(ssim_values))