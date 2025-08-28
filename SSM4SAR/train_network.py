"""
Argumentized training file to train azimuth compression SSM model
"""

# -- File info -- #
__author__ = 'Sebastian Fieldhouse'
__contact__ = 'sebastianfieldhouse.2@gmail.com'
__date__ = '2024-12-5'

# -- Import Utitlities -- #
from s4d2 import S4D, S4DKernel #, DropoutNd
from torch.utils.data import Dataset, DataLoader
from argparse import ArgumentParser
from skimage.metrics import structural_similarity as ssim
from pytorch_lightning.loggers import TensorBoardLogger
from config import valid_list, train_list # this is importing dataset address tuples
from torchmetrics import MeanMetric

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pykeops
import pickle
import os
import sys
import pandas as pd
import numpy as np
import math
import lightning as pl
import zarr
import multiprocessing
import kornia.losses as losses
import kornia as K

# -- Loss Function Definitions -- #
class EdgeLoss(nn.Module): # Currently not using this
    def __init__(self):
        super(EdgeLoss, self).__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, pred, target):
        # Compute gradients (edges) for predictions
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]

        # Compute gradients (edges) for targets
        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]

        # Compute L1 loss between gradients
        loss_x = self.l1_loss(pred_dx, target_dx)
        loss_y = self.l1_loss(pred_dy, target_dy)

        return loss_x + loss_y

class complexLoss(nn.Module): # Currently not using this
    """
    Loss function class for calculating loss between two azimuth columns in the complex domain
    """
    def __init__(self):
        super(complexLoss, self).__init__()
    
    def forward(self,
                pred: torch.Tensor, 
                target: torch.Tensor
                ) -> float:
        """
        Calculate loss between two azimuth columns in the complex domain
        
        Parameters 
        ----------
        pred : torch.Tensor
            tensor of shape (batchsize, columns, rows) or (columns, rows) whose loss must be compared to "target"
        target : torch.Tensor
            tensor of shape (batchsize, columns, rows) or (columns, rows) whose loss is calculated in comparison to "pred"
            
        Returns
        ----------
        total_loss : float
            complex loss between the two tensors
        """
        
        if pred.dim() != target.dim(): 
            raise ValueError(f"Expected pred and target tensor dimensions to be equal but instead \
                                received pred dim = {pred.dim()} and target dim = {target.dim()}")
        elif pred.dim() == 3:
            # flatten the batch dimension - create a tensor of shape (x, 2)
            pred = pred.permute(1, 0, 2)
            x = pred.numel() // 2
            pred = pred.reshape(x, 2)
            
            target = target.permute(1, 0, 2)
            x = target.numel() // 2
            target = target.reshape(x, 2)
            
        elif pred.dim() not in [2, 3]:
            raise ValueError(f"Expected pred and target tensor dimensions to be equal to 2 or 3 \
                                but got tensor dimension = {pred.dim()}")
            
        a = torch.sub(target, output)
        a_sq = torch.pow(a, 2)
        ab_sq = torch.add(a_sq[:, 0], a_sq[:, 1])
        c = torch.pow(ab_sq, 0.5)
        
        std = torch.std(c)
        mean = torch.mean(c)
        
        lower_bound = mean - 2 * std
        upper_bound = mean + 2 * std
        
        filtered_c = c[(c >= lower_bound) & (c <= upper_bound)]
        
        
        total_loss = torch.sum(filtered_c)
        return total_loss

# TODO: entropyLoss is not doing anything at the moment           
class entropyLoss(nn.Module): # Currently not using this
    def __init__(self):
        super(entropyLoss, self).__init__()
        
    def forward(self,
                I: torch.Tensor
                ) -> float:
        """
        Calculate the shannon entropy of a azimuth column tensor
        
        Parameters 
        ----------
        I : torch.Tensor
            tensor of shape (batchsize, columns, rows) or (columns, rows) whose loss must be compared to "target"
            
        Returns
        ----------
        entropy : float
            shannon entropy loss of the tensor
        """
              
# -- Dataset Definition -- #

class azimuthColumnDataset2(Dataset):
    def __init__(self, samples: list[tuple[str, str]], normalization: str):
        '''
        samples:
            list of tuples where the first string is the address of the rcmc sample and the second string is the address of the corresponding ac sample
        '''  
        # data min and max values - I got these from manually testing the data (for real/im values the max and min are around the same)
        # TODO: add a mode so that I get the gt out as the normal, non-thresholded image so that I can run experiments on the efficacy of the
        # normalization method
        
        # set normalization method
        self.normalization = normalization
        
        if self.normalization == 'broad':
            self.rc_min = -10000
            self.rc_max = 10000
            
            self.gt_min = -40000
            self.gt_max = 40000
        elif self.normalization == 'medium':
            self.rc_min = -6000
            self.rc_max = 6000
            
            self.gt_min = -24000
            self.gt_max = 24000
        elif self.normalization == 'tight':
            self.rc_min = -4000
            self.rc_max = 4000
            
            self.gt_min = -16000
            self.gt_max = 16000
        elif self.normalization == 'extra_tight':
            self.rc_min = -3000
            self.rc_max = 3000
            
            self.gt_min = -12000
            self.gt_max = 12000 
        else:
            self.rc_min = -2000
            self.rc_max = 2000
            
            self.gt_min = -9000
            self.gt_max = 9000
                 
        
        # calculate length of dataset - putting it here because multiple uses
        self.length = 0
        self.samples = []
        
        print(f"Opening zarr files...", flush=True)
        for (rcmc_address, ac_address) in samples:       
            zarr_array = zarr.open(rcmc_address, mode='r')
            shape = zarr_array.shape
            chunks = zarr_array.chunks
            num_chunks = np.prod([int(np.ceil(s / c)) for s, c in zip(shape, chunks)])
            
            # increment the total num_chunks counter
            self.length += num_chunks
            
            # create a new tuple (rc_addr, ac_addr, num_of_columns)
            self.samples.append((rcmc_address, ac_address, num_chunks))
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):       
        # search for which array the sample is in
        cumulative_length = 0
        for i, (rcmc_address, ac_address, num_chunks) in enumerate(self.samples):
            #print(f" for i = {i} cumulative_length = {cumulative_length} & num_chunks = {num_chunks}")
            if idx < cumulative_length + num_chunks:
                # the index is in this list
                list_index = i # TODO: this variable is not being used
                item_index = idx - cumulative_length
                break
            cumulative_length += num_chunks
            
        rcmc_array = zarr.open(rcmc_address, mode='r')
        rcmc_chunk = rcmc_array[:, item_index]
        
        ac_array = zarr.open(ac_address, mode='r')
        ac_chunk = ac_array[:, item_index]
        
        inp = self._prepare_sample(rcmc_chunk, self.rc_min, self.rc_max)
        target = self._prepare_sample(ac_chunk, self.gt_min, self.gt_max)
        
        # add position embedding to the input
        position_embedding = torch.full((10000, 1), (item_index + 1) / 20000)
        # position_embedding = torch.full((10000, 1), (idx + 1) / 10000) # TODO: this looks like its WRONG
        inp = torch.cat((inp, position_embedding), dim=1)
        
        # send everything to float32 so it matches the datatype that the outputs will be in 
        return inp.to(torch.float32), target.to(torch.float32)
        
    def _prepare_sample(self, array, array_min, array_max):
        array = self._iq_split(array, array.shape)
        array = torch.tensor(array)
        array = self._normalize(array, array_min, array_max)
        return array

        
    def _normalize(self, tensor, tensor_min, tensor_max):
        # normalize the data to the range between 0 and 1
        # note that because data is normalized to this range the center of the data ends up at 0.5
        normalized_tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
        normalized_tensor = torch.clamp(normalized_tensor, min=tensor_min, max=tensor_max)         
            
        return normalized_tensor
        
        
    def _iq_split(self, array, array_shape):      
        # if the shape of the array is (10000,) then this tuple needs to be expanded for the code further down to work
        if len(array_shape) == 1:
            array_shape = array_shape + (1,)

        # calculate double the shape of the input
        combined_array_shape = (array_shape[0], array_shape[1]*2)        
        combined_array = np.empty(combined_array_shape)
        combined_array[:, 0] = array.real[:]
        combined_array[:, 1] = array.imag[:]
        return combined_array             

# -- Model Definition -- #
class sarSSM(nn.Module):
    def __init__(self,
                num_layers: int,
                d_state: int = 16,
                activation_function: str = 'gelu',
                batch_normalization: bool = False
                ):
        super(sarSSM, self).__init__()
        self.num_layers = num_layers
        self.ssm = nn.ModuleList()
        

        # position embedding mixing
        self.fc1 = nn.Linear(3, 2)
        
        # ssm layers
        for _ in range(num_layers):
            self.ssm.append(
                S4D(d_model=2, d_state=d_state, transposed=False, activation=activation_function),                
            )
        
        if batch_normalization == True:    
            self.bn = nn.ModuleList()
            for _ in range(num_layers):
                self.bn.append(nn.BatchNorm1d(10000))
            
        
        self.fc2 = nn.Linear(2, 2)
            
    def forward(self, x):    
  
        # position embedding
        x = self.fc1(x)
        
        # s4 layers
        if batch_normalization == True:
            for i, ssm in enumerate(self.ssm):
                x, _ = ssm(x)
                x = self.bn[i](x)
        else:  
            for ssm in self.ssm:
                x, _ = ssm(x)
        
        x = self.fc2(x)
        
        return x
    
class azimuthModel(pl.LightningModule):
    def __init__(self, model, train_list, valid_list, train_batch, valid_batch, normalization, ssim_proportion, lr, weight_decay):
        super(azimuthModel, self).__init__()
        self.model = model
        self.train_list = train_list
        self.valid_list = valid_list
        self.train_batch = train_batch
        self.valid_batch = valid_batch
        self.normalization = normalization
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
        
    def train_dataloader(self):
        samples = self.train_list
        trainset = azimuthColumnDataset2(samples=samples, normalization=self.normalization)
        trainloader = DataLoader(dataset=trainset, num_workers=16, batch_size=self.train_batch, shuffle=True, drop_last=True)   
        return trainloader
        
    def val_dataloader(self):
        samples = self.valid_list
        validset = azimuthColumnDataset2(samples=samples, normalization=self.normalization)
        validloader = DataLoader(dataset=validset, num_workers=16, batch_size=self.valid_batch, shuffle=False, drop_last=True)
        return validloader
        
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
            
        outputs = model(raw).squeeze()
        
        # you need to flatten gt and outputs so that you can calculate loss on them
        gt_flat = gt.permute(1, 0, 2)
        gt_flat = gt_flat.reshape(10000, gt.shape[0]*gt.shape[2])
        
        outputs_flat = outputs.permute(1, 0, 2)
        outputs_flat = outputs_flat.reshape(10000, gt.shape[0]*gt.shape[2])
        
        # compute combined loss
        loss_mae, loss_mse, loss_huber, loss_ssim, loss_edge = self.compute_loss(outputs_flat, gt_flat)
        
        # get ssim of output and gt
        train_ssim = self.calculate_ssim(outputs_flat.unsqueeze(0).unsqueeze(0), gt_flat.unsqueeze(0).unsqueeze(0))
        
        # update training loss metric
        self.train_loss_metric.update(loss_mae)
        
        # Log losses
        self.log("train_loss", loss_mae, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train_ssim", train_ssim, on_step=True, on_epoch=False, prog_bar=True)
        '''
        self.log("train_loss_huber", loss_huber, on_step=True, on_epoch=False)
        self.log("train_loss_ssim", loss_ssim, on_step=True, on_epoch=False)
        self.log("train_loss_edge", loss_edge, on_step=True, on_epoch=False)
        '''
        
        return loss_mae*(1-self.delta) + loss_ssim*(self.delta)
    
    def validation_step(self, batch, idx):
        raw, gt = batch
        if raw.shape[0] < 10:
            raise ValueError("batch size must be above 10 for your loss functions to work")
        
        outputs = model(raw).squeeze()
        
        # you need to flatten gt and outputs so that you can calculate loss on them
        gt_flat = gt.permute(1, 0, 2)
        gt_flat = gt_flat.reshape(10000, gt.shape[0]*gt.shape[2])
        
        outputs_flat = outputs.permute(1, 0, 2)
        outputs_flat = outputs_flat.reshape(10000, gt.shape[0]*gt.shape[2])
        
        # compute combined loss
        loss_mae, loss_mse, loss_huber, loss_ssim, loss_edge = self.compute_loss(outputs_flat, gt_flat)
        
        # update validation loss metric
        self.val_loss_metric.update(loss_mae)
        
        # compute PSNR and SSIM for logging 
        # psnr_value = self.calculate_psnr(outputs_flat, gt_flat)
        ssim_value = self.calculate_ssim(outputs_flat.unsqueeze(0).unsqueeze(0), gt_flat.unsqueeze(0).unsqueeze(0))
        self.val_ssim_metric.update(ssim_value)
        
        
        # Log metrics
        self.log("val_loss", loss_mae, on_step=True, on_epoch=False, prog_bar=True)
        '''
        self.log("val_loss", val_loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log("val_psnr", psnr_value, on_step=True, on_epoch=False, prog_bar=True)
        self.log("val_ssim", ssim_value, on_step=True, on_epoch=False, prog_bar=True)
        self.log("val_loss_huber", loss_huber, on_step=True, on_epoch=False)
        self.log("val_loss_ssim", loss_ssim, on_step=True, on_epoch=False)
        self.log("val_loss_edge", loss_edge, on_step=True, on_epoch=False)
        '''
        
        return loss_mae
    
    def test_step(self, batch, idx):
        raw, gt = batch
        if raw.shape[0] < 10:
            raise ValueError("batch size must be above 10 for your loss functions to work")
        
        outputs = model(raw).squeeze()
        
        # you need to flatten gt and outputs so that you can calculate loss on them
        gt_flat = gt.permute(1, 0, 2)
        gt_flat = gt_flat.reshape(10000, gt.shape[0]*gt.shape[2])
        
        outputs_flat = outputs.permute(1, 0, 2)
        outputs_flat = outputs_flat.reshape(10000, gt.shape[0]*gt.shape[2])
        
        # compute combined loss
        val_loss, loss_mae, loss_mse, loss_huber, loss_ssim, loss_edge = self.compute_loss(outputs_flat, gt_flat)
        
        # update validation loss metric
        self.val_loss_metric.update(val_loss)
        
        # compute PSNR and SSIM for logging 
        psnr_value = self.calculate_psnr(outputs_flat, gt_flat)
        ssim_value = self.calculate_ssim(outputs_flat.unsqueeze(0).unsqueeze(0), gt_flat.unsqueeze(0).unsqueeze(0))
        
        # Log metrics
        self.log("val_loss", val_loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log("val_psnr", psnr_value, on_step=True, on_epoch=False, prog_bar=True)
        self.log("val_ssim", ssim_value, on_step=True, on_epoch=False, prog_bar=True)
        self.log("val_loss_huber", loss_huber, on_step=True, on_epoch=False)
        self.log("val_loss_ssim", loss_ssim, on_step=True, on_epoch=False)
        self.log("val_loss_edge", loss_edge, on_step=True, on_epoch=False)
        
        return val_loss
       
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
        #self.val_losses_metric.append(avg_val_loss.item())
        self.log('val_loss_metric', avg_val_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_ssim', avg_val_ssim, on_step=False, on_epoch=True, prog_bar=False)
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
    
    def on_fit_end(self):
        '''
        samples = self.valid_list
        validset = azimuthColumnDataset2(samples=samples)
        validloader = DataLoader(dataset=validset, num_workers=32, batch_size=10000, shuffle=False, drop_last=True)
        
        
        raw, gt = next(iter(validloader))
        if raw.shape[0] < 10:
            raise ValueError("batch size must be above 10 for your loss functions to work")
        self.eval()
        with torch.no_grad():
            outputs = model(raw).squeeze()
        
        # you need to flatten gt and outputs so that you can calculate loss on them
        gt_flat = gt.permute(1, 0, 2)
        gt_flat = gt_flat.reshape(10000, gt.shape[0]*gt.shape[2])
        
        outputs_flat = outputs.permute(1, 0, 2)
        outputs_flat = outputs_flat.reshape(10000, gt.shape[0]*gt.shape[2])
        
        # Now plot both of these
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))

        axs[0].imshow(abs_05(gt_flat.detach().numpy()), vmin=0.5, vmax=vmax(gt_flat.detach()))
        axs[0].set_title('gt')

        axs[1].imshow(abs_05(outputs_flat.detach().numpy()), vmin=0.5, vmax=vmax(outputs_flat.detach()))
        axs[1].set_title('prediction')

        plt.show()
        '''
        
        
        '''
        plt.figure()
        epochs = range(1, len(self.train_losses) + 1)
        plt.plot(epochs, self.train_losses, label="Training Loss")
        plt.plot(epochs, self.val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training and Validation Loss")
        plt.savefig("train_val_loss_plot.png")
        plt.close()
        print("Training and validation plot saved as train_val_loss_plot.png")
        '''
        
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        print(f"batch: {batch}")
        print(f" shape of batch = {len(batch)}")
        print(f"batch_idx: {batch_idx}")
        print(f"dataloader_idx: {dataloader_idx}")
        return self.forward(batch)    
        
         
    
def parse_arguments():
    """
    Parses arguments
    
    Parameters
    ----------
    arguments List
        Arguments for parsing
        
    Returns
    -------
    arguments : dict
        Dictionary of parsed arguments
    
    """
    
    parser = ArgumentParser(description='sarSSM experiment script')
    
    # - Parameters
    
    # -- Experiment Directory Name
    parser.add_argument('-dn',
                        '--directory',
                        type=str,
                        default='experiment_1',
                        help='Experiment directory name, can be any valid string'
    )
    
    # -- Model Name
    parser.add_argument('-mn',
                        '--model_name',
                        type=str,
                        default='model_1',
                        help='Experiment model name, can be any valid? string'
    )
    
    # -- GPU Number
    parser.add_argument('-gpu',
                        '--gpu_no',
                        type=str,
                        default='1',
                        help='GPU number to run on'
    )
    
    
    # -- Model Number of Layers
    parser.add_argument('-nl',
                         '--num_layers',
                         type=int,
                         default=4,
                         help='Number of ssm layers after the embedding layer'
    )
    
    # -- Model hidden state size
    parser.add_argument('-hs',
                        '--hidden_state_size',
                        type=int,
                        default=16,
                        help='Size of the hidden state of the SSM layer'
    )
    
    # -- Number of training epochs
    parser.add_argument('-ep',
                        '--epochs',
                        type=int,
                        default=3,
                        help='Number of training epochs for the experiment'
    )
    
    # -- Training Batch Size
    parser.add_argument('-bs',
                        '--batch_size',
                        type=int,
                        default=10,
                        help='Batch size for the training loop'
    )
    
    # -- Validation Batch Size
    parser.add_argument('-vb',
                        '--valid_batch_size',
                        type=int,
                        default=10,
                        help='Batch size for the validation set'
    )
    # -- Learning Rate
    parser.add_argument('-lr',
                        '--learning_rate',
                        type=float,
                        default=0.005,
                        help='Learning rate for the non-SSM layers'
    )
    
    # -- Weight Decay
    parser.add_argument('-wd',
                        '--weight_decay',
                        type=float,
                        default=0.01,
                        help='Weight decay for the training optimizer'
    )
    
    # -- Normalization Mode
    parser.add_argument('-nm',
                        '--norm_mode',
                        type=str,
                        choices=['broad', 'medium', 'tight', 'extra_tight', 'extra_extra_tight'],
                        default='broad',
                        help='Normalization Mode for the dataset'
    )
    
    # -- ssim_proportion
    parser.add_argument('-sp',
                        '--ssim',
                        type=float,
                        default=0,
                        help='ssim proportion'
    )

    # -- activation function
    parser.add_argument('-af',
                        '--act_fun',
                        type=str,
                        default='default',
                        choices=['default', 'relu', 'hardtanh', 'hardsigmoid', 'hardshrink', 'gelu', 'leakyrelu', 'hardswish', 'prelu'],
                        help='ssim proportion'
    )
    
    # -- batch normalization
    parser.add_argument('-bn',
                        '--batch_normalization',
                        type=bool,
                        default=False,
                        help='whether or not to include batch normalization'
    )

    # what else do I want to test - the next thing to test is different loss functions

    arguments = vars(parser.parse_args())
    
    return arguments


# function to get vmax for my images when they are tensors
def vmax(array):
    # if is tensor convert it to numpy array
    #TODO: this function was breaking when I try to plot my array (something to do with dividing by an invalid value during scalar divide)
    if torch.is_tensor(array):
        array = array.numpy()
        
    image = array[1100:1900, 1100:1900] # this is wrong - when I submit an array with three columns this breaks 
    mean = np.mean(abs(image))
    std  = np.std(abs(image))
    vmax = mean + 3*std
    return vmax
    
   
def extract_dataset_chunk(i, dataset):
    array = torch.empty(10000, 200)
    for j in range(100):
        _ , array[:10000, 2*j:(2*j)+2] = dataset[(100*i)+j]
    return array

def abs_05(data):
    reflected = 0.5 + np.abs(data - 0.5)
    return reflected
         
   
def save_script():
    # open the current script
    with open(__file__, 'r') as script_file:
        contents = script_file.read()
        
    # Write the contents to a new file
    with open(os.path.join(exp_dir, model_name, 'original_script.py'), 'w') as new_file:
        new_file.write(contents)     
             
          
if __name__ == '__main__': 
    arguments = parse_arguments()
    
    # directory parameters
    exp_dir = arguments['directory']
    model_name = arguments['model_name']
    
    # training parameters
    num_epochs = arguments['epochs']
    train_batch = arguments['batch_size']
    valid_batch = arguments['valid_batch_size']
    lr = arguments['learning_rate'] # TODO: not using this at the moment
    weight_decay = arguments['weight_decay']
    ssim_proportion = arguments['ssim']
    
    # device parameters
    gpu_no = arguments['gpu_no']
    
    # model parameters
    d_state = arguments['hidden_state_size']
    num_layers = arguments['num_layers']
    activation_function = arguments['act_fun']
    batch_normalization = arguments['batch_normalization']
    
    # dataset parameters
    normalization = arguments['norm_mode']
    
    
    # now that the dataset is verified, you need to run a training cycle
    model = sarSSM(num_layers=num_layers, d_state=d_state, activation_function=activation_function, batch_normalization=batch_normalization)
    # train_list and valid_list coming from an external document at the moment
    lightning_model = azimuthModel(model, 
                                   train_list=train_list,
                                   valid_list=valid_list, 
                                   train_batch=train_batch,
                                   valid_batch=valid_batch,
                                   normalization=normalization,
                                   ssim_proportion=ssim_proportion,
                                   lr=lr,
                                   weight_decay=weight_decay
                                    )

    # I think this "logs" argument tells it where to put the logs data that tensorboard creates
    logger = TensorBoardLogger(exp_dir, name=model_name)

    trainer = pl.Trainer(max_epochs=num_epochs,
                         logger=logger,
                         devices=gpu_no,
                         fast_dev_run=False,
                         log_every_n_steps=1,
                         )
    
    print(f"\n\nEstimated number of stepping batches : {trainer.estimated_stepping_batches} \n\n")
    
    '''
    print(f"Size of the train set is: {len(trainer.train_dataloader.dataset)}", flush=True)
    print(f"Size of the val set is: {len(trainer.val_dataloader.dataset)}", flush=True)
    '''
    
    trainer.fit(lightning_model)
    
    print(trainer.__dict__)
    
    
    # save the script to the same directory as the tensorboard logging
    save_script()

    
    
    '''
    # once we have finished training I want to get the model prediction
    validset = azimuthColumnDataset2(samples=valid_list)
    validloader = DataLoader(dataset=validset, num_workers=32, batch_size=10000, shuffle=False)
    
    rcmc, gt = next(iter(validloader))
    

    outputs = trainer.predict(lightning_model, validloader)
    
    
    # you need to flatten gt and outputs so that you can calculate loss on them 
    gt_flat = gt.permute(1, 0, 2)
    gt_flat = gt_flat.reshape(10000, gt.shape[0]*gt.shape[2])
    
    outputs_flat = outputs.permute(1, 0, 2)
    outputs_flat = outputs_flat.reshape(10000, outputs.shape[0]*outputs.shape[2])
    
    # Now plot both of these
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))

    axs[0].imshow(abs_05(gt_flat.detach().numpy()), vmin=0.5, vmax=vmax(gt_flat.detach()))
    axs[0].set_title('gt')

    axs[1].imshow(abs_05(outputs_flat.detach().numpy()), vmin=0.5, vmax=vmax(outputs_flat.detach()))
    axs[1].set_title('prediction')

    plt.show()
    '''
    
    
    
    

    

    





    
            
    
        
        