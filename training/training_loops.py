import json
import logging
import torch
from torch import device, nn, optim
from torch.utils.data import DataLoader
from typing import Tuple, Union
from tqdm import tqdm
import os
from typing import Optional, Callable, Union
from dataloader.dataloader import SARZarrDataset, get_sar_dataloader, SARDataloader
from model.transformers.rv_transformer import RealValuedTransformer  
from model.transformers.cv_transformer import ComplexTransformer
from training.visualize import compute_metrics, save_metrics
from sarpyx.utils.losses import get_loss_function
from training.visualize import save_results_and_metrics, get_full_image_and_prediction, compute_metrics, display_inference_results
import pytorch_lightning as pl

import numpy as np
class TrainerBase(pl.LightningModule):
    def __init__(
            self,      
            base_save_dir:str,
            model , 
            train_loader: SARDataloader,
            val_loader: SARDataloader,
            test_loader: SARDataloader,
            mode: str = "parallel",
            criterion: Callable = nn.MSELoss, 
            scheduler_type: str = 'cosine', 
            inference_loader: Optional[SARDataloader] = None,
            metrics_file_name: str = "test_metrics.json", 
            lr: int = 1e-4
        ):
        super().__init__()
        self.base_save_dir = base_save_dir
        self.model = model 
        self.mode = mode
        self.criterion_fn = criterion
        self.scheduler_type = scheduler_type
        self.inference_loader = inference_loader
        self.lr = lr
        
        self._train_loader = train_loader
        self._val_loader = val_loader
        self._test_loader = test_loader

        self.test_metrics = []
        self.metrics_file_name = metrics_file_name

    def train_dataloader(self):
        return self._train_loader

    def val_dataloader(self):
        return self._val_loader

    def test_dataloader(self):
        return self._test_loader

    def compute_loss(self, output: torch.Tensor, target: torch.Tensor):
        return self.criterion_fn(output, target)
    def preprocess_sample(self, x: Union[np.ndarray, torch.Tensor], device: Union[str, torch.device]):                
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        return x.to(device)
    def forward(self, x: Union[np.ndarray, torch.Tensor], y: Optional[Union[np.ndarray, torch.Tensor]]=None, device: Union[str, torch.device]="cuda") -> torch.Tensor:
        """
        Forward pass through the model.
        Args:
            x: Input tensor
            y: Target tensor
        Returns:
            Model output tensor
        """
        x_preprocessed = self.preprocess_sample(x, device)
        if y is None:
            y_preprocessed = None
        else:
            y_preprocessed = self.preprocess_sample(y, device)
        return self.model(x_preprocessed, y_preprocessed)

    def show_example(self, loader: SARDataloader, window: Tuple[Tuple[int, int], Tuple[int, int]] = ((1000, 1000), (5000, 5000)), vminmax=(4000, 4200), figsize=(20, 6), metrics_save_path: str = "metrics.json", img_save_path: str = "test.png"):
        try:
            gt, pred, input = get_full_image_and_prediction(
                dataloader=loader,
                show_window=window,
                zfile=0,
                inference_fn= self.forward_pass,
                return_input=True, 
                device="cuda", 
                vminmax=vminmax
            )

            display_inference_results(
                input_data=input,
                gt_data=gt,
                pred_data=pred,
                figsize=figsize,
                vminmax=vminmax,  
                show=True, 
                save=False,
                save_path=os.path.join(self.base_save_dir, img_save_path)
            )
            metrics = compute_metrics(gt, pred)
            with open(metrics_save_path, 'w') as f:
                json.dump(metrics, f)
                
        except Exception as e:
            print(f"Visualization failed with error: {str(e)}")
            raise

    def save_checkpoint(self, epoch: int, optimizer, scheduler=None, is_best: bool = False, **kwargs):
        """Save training checkpoint with all necessary information."""
        from model.model_utils import save_checkpoint
        
        # Determine filename based on checkpoint type
        if is_best:
            filename = "checkpoint_best.pth"
        else:
            filename = f"checkpoint_epoch_{epoch}.pth"
            
        checkpoint_path = os.path.join(self.base_save_dir, filename)
        
        # Prepare metrics
        metrics = {
            'epoch': epoch,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }
        metrics.update(kwargs)
        
        # Save comprehensive checkpoint
        save_checkpoint(
            model=self.model,
            save_path=checkpoint_path,
            epoch=epoch,
            optimizer=optimizer,
            scheduler=scheduler,
            metrics=metrics,
            model_config=getattr(self.model, '_model_config', None)
        )
        
        # Also save legacy format for compatibility
        if is_best:
            torch.save(self.model.state_dict(), f"{self.base_save_dir}/sar_transformer_best.pth")
        torch.save(self.model.state_dict(), f"{self.base_save_dir}/sar_transformer_last.pth")

    def resume_from_checkpoint(self, checkpoint_path: str, optimizer, scheduler=None):
        """Resume training from a checkpoint."""
        from model.model_utils import load_pretrained_weights
        
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            return False
            
        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Load model weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer state
            if optimizer and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
            # Load scheduler state
            if scheduler and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                
            # Load training state
            self.start_epoch = checkpoint.get('epoch', 0) + 1
            if 'metrics' in checkpoint:
                metrics = checkpoint['metrics']
                self.val_losses = metrics.get('val_losses', [])
                self.train_losses = metrics.get('train_losses', [])
                self.best_val_loss = metrics.get('best_val_loss', float('inf'))
                self.last_improve_epochs = metrics.get('last_improve_epochs', 0)

            print(f"✅ Resumed training from epoch {self.start_epoch}")
            print(f"📊 Best validation loss so far: {self.best_val_loss:.6f}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to resume from checkpoint: {str(e)}")
            return False
    def on_train_start(self):
        self.resume_from_checkpoint(self.resume_from, self.trainer.optimizers[0], None)
    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x, y)  
        loss = self.compute_loss(output, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        y_np = y.cpu().numpy() #.squeeze()
        output_np = output.cpu().numpy() #.squeeze()

        m = compute_metrics(y_np, output_np)
        self.log_dict('train_metrics', m, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x, y)
        loss = self.compute_loss(output, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        y_np = y.cpu().numpy() #.squeeze()
        output_np = output.cpu().numpy() #.squeeze()

        m = compute_metrics(y_np, output_np)
        self.log_dict('val_metrics', m, on_step=False, on_epoch=True, prog_bar=False)
        return loss
    def on_validation_epoch_end(self):
        if self.inference_loader is not None:
            self.show_example(self.inference_loader, window=((1000, 1000), (5000, 5000)), vminmax=(4000, 4200), figsize=(20, 6), metrics_save_path=f"metrics_{self.current_epochs}.json", img_save_path=f"val_{self.current_epoch}.png")
        self.save_checkpoint(epoch=self.current_epoch, optimizer=self.trainer.optimizers[0], scheduler=None, is_best=(self.trainer.callback_metrics['val_loss'] < self.best_val_loss), val_loss=self.trainer.callback_metrics['val_loss'])
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x, y)
                
        y_np = y.cpu().numpy() #.squeeze()
        output_np = output.cpu().numpy() #.squeeze()

        m = compute_metrics(y_np, output_np)
        self.log_dict('test_metrics', m, on_step=False, on_epoch=True, prog_bar=False)

        self.test_metrics.append(m)

        loss = self.criterion(output, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    def on_test_end(self):
        avg_psnr = sum(m['psnr_raw_vs_focused'] or 0 for m in self.test_metrics) / len(self.test_metrics)
        avg_pslr = sum(m['pslr_focused'] or 0 for m in self.test_metrics) / len(self.test_metrics)
        summary = {
            'avg_psnr_raw_vs_focused': avg_psnr,
            'avg_pslr_focused': avg_pslr,
            'num_samples': len(self.test_metrics),
            'val_losses': self.val_losses
        }
        metrics_path = os.path.join(self.base_save_dir, self.metrics_file_name)
        save_metrics(summary, metrics_path)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        if self.scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
            return [optimizer], [scheduler]
        else:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss'
                }
            }

class TrainRVTransformer(TrainerBase):
    def __init__(self, 
                base_save_dir:str,
                model , 
                train_loader: SARDataloader,
                val_loader: SARDataloader,
                test_loader: SARDataloader,
                mode: str = "parallel",
                criterion: Callable = nn.MSELoss,
            
        ):
        super().__init__(base_save_dir, model, train_loader, val_loader, test_loader, mode, criterion=criterion)
        assert mode == "parallel" or "autoregressive", "training mode must be either 'parallel' or 'autoregressive'"

        self.base_save_dir = base_save_dir
        self.model = model 
        self.mode = mode
        self.criterion_fn = criterion
        if not os.path.exists(self.base_save_dir):
            os.makedirs(self.base_save_dir)
    def compute_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute loss for complex-valued output."""
        # For complex output, we can use MSE on both real and imaginary parts
        if output.shape[-1] > 2:
            output = output[..., :-2]
        if target.shape[-1] > 2:
            target = target[..., :-2]

        loss = self.criterion_fn(output, target)
        return loss
    def preprocess_sample(self, x: torch.Tensor, device: Union[str, torch.device]):    
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        return x.float().to(device)
        
        

class TrainCVTransformer(TrainRVTransformer):
    """
    Enhanced trainer for SAR Complex Transformer with optional patch preprocessing.
    Extends the existing TrainCVTransformer to handle patch preprocessing.
    """
    
    def __init__(
            self, 
            base_save_dir: str, 
            model, 
            train_loader: SARDataloader,
            val_loader: SARDataloader,
            test_loader: SARDataloader,
            mode: str = "parallel",  
            criterion: Callable = nn.MSELoss
        ):
        super().__init__(base_save_dir, model, train_loader, val_loader, test_loader, mode, criterion=criterion)
        
        # # Count parameters and log model info
        # if hasattr(model, 'parameters'):
        #     total_params = sum(p.numel() for p in model.parameters())
        #     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        #     logging.info(f"Model created with {total_params:,} total parameters")
        #     logging.info(f"Trainable parameters: {trainable_params:,}")

    def preprocess_sample(self, x: Union[torch.Tensor, np.ndarray], device: Union[str, torch.device]):   
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)              
        if torch.is_complex(x):
            x = x.to(device).to(torch.complex64)
        else:
            x = x.to(device).float()
        return x
    def compute_loss(self, output: torch.Tensor, target: torch.Tensor, device: Union[str, torch.device]) -> torch.Tensor:
        """
        Compute loss for complex-valued output.
        Converts complex tensors to real-valued with last dimension 2 (real, imag).
        """
        # Remove positional embedding if present
        if output.shape[-1] > 2:
            output = output[..., :-2]
        if target.shape[-1] > 2:
            target = target[..., :-2]

        # print(f"Output shape: {output.shape}, Target shape: {target.shape}")
        # print(f"Output patch: {output}")
        # print(f"Target patch: {target}")
        loss = self.criterion_fn(output.to(device), target.to(device))
        return loss

class TrainSSM(TrainerBase):
    """
    Training class for State Space Models (SSM) for SAR focusing.
    Handles column-wise processing where each column is treated as a sequence.
    """
    
    def __init__(
            self, 
            base_save_dir: str, 
            model, 
            train_loader: SARDataloader,
            val_loader: SARDataloader,
            test_loader: SARDataloader,
            mode: str = "parallel", 
            criterion: Callable = nn.MSELoss, 
            scheduler_type: str = 'cosine'
        ):
        super().__init__(base_save_dir, model, train_loader, val_loader, test_loader, mode, criterion=criterion, scheduler_type=scheduler_type)
        # self.logger = logging.getLogger(__name__)
        
        # # Count parameters and log model info
        # if hasattr(model, 'parameters'):
        #     total_params = sum(p.numel() for p in model.parameters())
        #     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        #     self.logger.info(f"Model created with {total_params:,} total parameters")
        #     self.logger.info(f"Trainable parameters: {trainable_params:,}")
    def forward(self, x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor], device: Union[str, torch.device]="cuda") -> torch.Tensor:
        """
        Forward pass through the model.
        Args:
            x: Input tensor
            y: Target tensor
        Returns:
            Model output tensor
        """
        x_preprocessed = self.preprocess_sample(x, device=device)
        return self.model(x_preprocessed)
    def preprocess_sample(self, x: Union[torch.Tensor, np.ndarray], device: Union[str, torch.device]="cuda"):   
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)              
        if torch.is_complex(x):
            x = x.to(torch.complex64)
        else:
            x = x.float()
        return x.to(device)
    def compute_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute loss for complex-valued output.
        Converts complex tensors to real-valued with last dimension 2 (real, imag).
        """
        # Remove positional embedding if present
        if output.shape[-1] > 2:
            output = output[..., :-2]
        elif output.shape[-1] == 2:
            output = output[..., :-1]
        if target.shape[-1] > 2:
            target = target[..., :-2]
        elif target.shape[-1] == 2:
            target = target[..., :-1]

        # print(f"Output shape: {output.shape}, Target shape: {target.shape}")
        # print(f"Output patch: {output}")
        # print(f"Target patch: {target}")
        loss = self.criterion_fn(output, target)
        return loss


class TrainDUN(TrainerBase):
    pass

def get_training_loop_by_model_name(
        model_name: str, 
        model: nn.Module,  
        train_loader: SARDataloader,
        val_loader: SARDataloader,
        test_loader: SARDataloader,
        save_dir: Union[str, os.PathLike] = './results', 
        loss_fn_name: str = "mse", 
        mode: str = 'parallel', 
        scheduler_type: str = 'cosine', 
        num_epochs: int = 250, 
        logger: Optional[pl.loggers.Logger] = None, 
        device_no: int= 0
    ) -> Tuple[TrainerBase, pl.Trainer]:

    if model_name == 'cv_transformer':
        lightning_model = TrainCVTransformer(
            base_save_dir=str(save_dir),
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            mode=mode,
            criterion=get_loss_function(loss_fn_name),
            scheduler_type=scheduler_type
        )
    elif 'transformer' in model_name.lower():
        lightning_model = TrainRVTransformer(
            base_save_dir=str(save_dir),
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            criterion=get_loss_function(loss_fn_name),
            mode=mode,
            scheduler_type=scheduler_type
        )
    elif 'ssm' in model_name.lower():
        lightning_model = TrainSSM(
            base_save_dir=str(save_dir),
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            criterion=get_loss_function(loss_fn_name),
            mode=mode,
            scheduler_type=scheduler_type
        )
    else:
        raise ValueError(f"Unsupported model type: {model_name}")
    trainer = pl.Trainer(max_epochs=num_epochs,
                        logger=logger,
                        devices=[device_no],
                        fast_dev_run=False,
                        log_every_n_steps=1,
                        accelerator="gpu" if torch.cuda.is_available() else "cpu",
                        )
    return lightning_model, trainer