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
from model.SSMs.SSM import OverlapSaveWrapper
from model.transformers.rv_transformer import RealValuedTransformer  
from model.transformers.cv_transformer import ComplexTransformer
from training.visualize import compute_metrics, save_metrics, average_metrics
from sarpyx.utils.losses import get_loss_function
from training.visualize import get_full_image_and_prediction, compute_metrics, display_inference_results
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
            inference_loader: Optional[SARDataloader] = None,
            mode: str = "parallel",
            criterion: Callable = nn.MSELoss, 
            scheduler_type: str = 'cosine', 
            metrics_file_name: str = "test_metrics.json", 
            lr: int = 1e-4, 
            resume_from: Optional[str] = None
        ):
        super().__init__()
        self.base_save_dir = base_save_dir
        print(f"BASE SAVE DIR: {self.base_save_dir}")
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
        self.best_val_loss = float('inf')
        self.val_losses = []
        self.train_losses = []
        self.start_epoch = 0
        self.last_improve_epochs = 0
        self.resume_from = resume_from
        self.validation_metrics = []
        self.train_metrics = []

    def train_dataloader(self):
        return self._train_loader

    def val_dataloader(self):
        return self._val_loader

    def test_dataloader(self):
        return self._test_loader

    def compute_loss(self, target: torch.Tensor, output: torch.Tensor):
        target, output = self.preprocess_output_and_prediction_before_comparison(target, output)
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
        if device is None:
            device = self.device
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
                inference_fn= self.forward,
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
                save=True,
                save_path=os.path.join(self.base_save_dir, img_save_path)
            )
            gt, pred = self.preprocess_output_and_prediction_before_comparison(gt, pred)
            metrics = compute_metrics(gt, pred)
            with open(os.path.join(self.base_save_dir, metrics_save_path), 'w') as f:
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
            torch.save(self.model.state_dict(), f"{self.base_save_dir}/model_best.pth")
        torch.save(self.model.state_dict(), f"{self.base_save_dir}/model_last.pth")

    def resume_from_checkpoint(self, checkpoint_path: Optional[Union[str, os.PathLike]], optimizer, scheduler=None):
        """Resume training from a checkpoint."""
        from model.model_utils import load_pretrained_weights
        if not self.resume_from:
            print("No resume checkpoint specified.")
            return False
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
    def log_metrics(self, metrics:dict, prefix: str, on_step: bool=False, on_epoch: bool=True, prog_bar: bool=False):
        for key, value in metrics.items():
            self.log(f"{prefix}_{key}", value, on_step=on_step, on_epoch=on_epoch, prog_bar=prog_bar)
    def preprocess_output_and_prediction_before_comparison(self, target: torch.Tensor, output: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        if output.shape[-1] > 2:
            output = output[..., :-2]
        elif output.shape[-1] == 2 and np.iscomplex(output.dtype):
            output = output[..., :-1]
        if target.shape[-1] > 2:
            target = target[..., :-2]
        elif target.shape[-1] == 2 and np.iscomplex(target.dtype):
            target = target[..., :-1]
        return target, output
    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x, y)  
        #print(f"Output from model: {output}")
        #print(f"Ground truth: {y}")
        loss = self.compute_loss(output, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        y_np = y.detach().cpu().numpy() #.squeeze()
        output_np = output.detach().cpu().numpy() #.squeeze()
        y_np, output_np = self.preprocess_output_and_prediction_before_comparison(y_np, output_np)
        for i in range(y_np.shape[0]):
            gt_patch = self.train_dataloader().dataset.get_patch_visualization(y_np[i], self.train_dataloader().dataset.level_to, vminmax=(4000, 4200), restore_complex=True, remove_positional_encoding=False)
            #print(f"Ground truth patch with index {idx} has shape: {gt_patch.shape}, while reconstructed ground truth patch has dimension {gt_patch.shape}")
            pred_patch = self.train_dataloader().dataset.get_patch_visualization(output_np[i], self.train_dataloader().dataset.level_to, vminmax=(4000, 4200), restore_complex=True, remove_positional_encoding=False)
            gt_patch, pred_patch = self.preprocess_output_and_prediction_before_comparison(gt_patch, pred_patch)
            m = compute_metrics(gt_patch, pred_patch)
            # self.log_metrics(m, 'train_metrics', on_step=False, on_epoch=True, prog_bar=False)
            self.train_metrics.append(m)
        return loss
    def on_train_epoch_end(self):
        super().on_train_epoch_end()
        avg_metrics = average_metrics(self.train_metrics)
        self.train_metrics = []
        self.log_metrics(avg_metrics, 'train_metrics', on_step=False, on_epoch=True, prog_bar=False)
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x, y)
        loss = self.compute_loss(output, y)
        self.val_losses.append(loss.item())
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        y_np = y.cpu().numpy() #.squeeze()
        output_np = output.cpu().numpy() #.squeeze()
        
        y_np, output_np = self.preprocess_output_and_prediction_before_comparison(y_np, output_np)
        for i in range(y_np.shape[0]):
            gt_patch = self.val_dataloader().dataset.get_patch_visualization(y_np[i], self.train_dataloader().dataset.level_to, vminmax=(4000, 4200), restore_complex=True, remove_positional_encoding=False)
            #print(f"Ground truth patch with index {idx} has shape: {gt_patch.shape}, while reconstructed ground truth patch has dimension {gt_patch.shape}")
            pred_patch = self.val_dataloader().dataset.get_patch_visualization(output_np[i], self.train_dataloader().dataset.level_to, vminmax=(4000, 4200), restore_complex=True, remove_positional_encoding=False)
            gt_patch, pred_patch = self.preprocess_output_and_prediction_before_comparison(gt_patch, pred_patch)
            m = compute_metrics(gt_patch, pred_patch)
            self.validation_metrics.append(m)

        return loss
    def on_validation_epoch_end(self):
        avg_metrics = average_metrics(self.validation_metrics)
        self.validation_metrics = []
        self.log_metrics(avg_metrics, 'val_metrics', on_step=False, on_epoch=True, prog_bar=False)

        if self.inference_loader is not None:
            self.show_example(self.inference_loader, window=((1000, 1000), (2000, 2000)), vminmax=(4000, 4200), figsize=(20, 6), metrics_save_path=f"metrics_{self.current_epoch}.json", img_save_path=f"val_{self.current_epoch}.png")
        self.save_checkpoint(epoch=self.current_epoch, optimizer=self.trainer.optimizers[0], scheduler=None, is_best=(self.trainer.callback_metrics['val_loss'] < self.best_val_loss), val_loss=self.trainer.callback_metrics['val_loss'])
        if self.trainer.callback_metrics['val_loss'] < self.best_val_loss:
            self.best_val_loss = self.trainer.callback_metrics['val_loss']
            self.last_improve_epochs = 0
        else:
            self.last_improve_epochs += 1
        super().on_validation_epoch_end()
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        print(f"Test batch {batch_idx}: ")
        print(f"x={x}, ")
        print(f"y={y}")
        output = self.forward(x, y)
                
        y_np = y.cpu().numpy() #.squeeze()
        output_np = output.cpu().numpy() #.squeeze()
        y_np, output_np = self.preprocess_output_and_prediction_before_comparison(y_np, output_np)
        for i in range(y_np.shape[0]):
            gt_patch = self.test_dataloader().dataset.get_patch_visualization(y_np[i], self.train_dataloader().dataset.level_to, vminmax=(4000, 4200), restore_complex=True, remove_positional_encoding=True)
            #print(f"Ground truth patch with index {idx} has shape: {gt_patch.shape}, while reconstructed ground truth patch has dimension {gt_patch.shape}")
            pred_patch = self.test_dataloader().dataset.get_patch_visualization(output_np[i], self.train_dataloader().dataset.level_to, vminmax=(4000, 4200), restore_complex=True, remove_positional_encoding=True)
            gt_patch, pred_patch = self.preprocess_output_and_prediction_before_comparison(gt_patch, pred_patch)
            m = compute_metrics(gt_patch, pred_patch)
            self.test_metrics.append(m)

        loss = self.criterion(output, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    def on_test_end(self):
        avg_metrics = average_metrics(self.test_metrics)
        self.test_metrics = []
        self.log_metrics(avg_metrics, 'test_metrics', on_step=False, on_epoch=True, prog_bar=False)

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
                inference_loader: Optional[SARDataloader] = None,
                mode: str = "parallel",
                criterion: Callable = nn.MSELoss,
                scheduler_type: str = 'cosine'
            
        ):
        super().__init__(base_save_dir, model, train_loader, val_loader, test_loader, inference_loader, mode, criterion=criterion, scheduler_type=scheduler_type)
        assert mode == "parallel" or "autoregressive", "training mode must be either 'parallel' or 'autoregressive'"

    def preprocess_output_and_prediction_before_comparison(self, target: torch.Tensor, output: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        if output.shape[-1] > 2:
            output = output[..., :-2]
        if target.shape[-1] > 2:
            target = target[..., :-2]
        return target, output

    def preprocess_sample(self, x: torch.Tensor, device: Union[str, torch.device]):    
        if device is None:
            device = self.device
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
            inference_loader: Optional[SARDataloader] = None,
            mode: str = "parallel",  
            criterion: Callable = nn.MSELoss, 
            scheduler_type: str = 'cosine'
        ):
        super().__init__(base_save_dir, model, train_loader, val_loader, test_loader, inference_loader, mode, criterion=criterion, scheduler_type=scheduler_type)

        # # Count parameters and log model info
        # if hasattr(model, 'parameters'):
        #     total_params = sum(p.numel() for p in model.parameters())
        #     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        #     logging.info(f"Model created with {total_params:,} total parameters")
        #     logging.info(f"Trainable parameters: {trainable_params:,}")

    def preprocess_sample(self, x: Union[torch.Tensor, np.ndarray], device: Union[str, torch.device]):   
        if device is None:
            device = self.device
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)              
        if torch.is_complex(x):
            x = x.to(device).to(torch.complex64)
        else:
            x = x.to(device).float()
        return x
    def preprocess_output_and_prediction_before_comparison(self, target, output):
        if output.shape[-1] > 2:
            output = output[..., :-2]
        if target.shape[-1] > 2:
            target = target[..., :-2]
        return target, output

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
            inference_loader: Optional[SARDataloader] = None,
            criterion: Callable = nn.MSELoss, 
            mode: str = "parallel",
            scheduler_type: str = 'cosine', 
            wrapper: bool = True
        ):
        super().__init__(
            base_save_dir=base_save_dir, 
            model=model, 
            train_loader=train_loader, 
            val_loader=val_loader, 
            test_loader=test_loader, 
            inference_loader=inference_loader, 
            criterion=criterion, 
            mode=mode,
            scheduler_type=scheduler_type
        )
        if wrapper: 
            self.wrapper = OverlapSaveWrapper(model=model)
        else:
            self.wrapper = None
        # self.logger = logging.getLogger(__name__)
        
        # # Count parameters and log model info
        # if hasattr(model, 'parameters'):
        #     total_params = sum(p.numel() for p in model.parameters())
        #     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        #     self.logger.info(f"Model created with {total_params:,} total parameters")
        #     self.logger.info(f"Trainable parameters: {trainable_params:,}")
    def forward(self, x: Union[np.ndarray, torch.Tensor], y: Optional[Union[np.ndarray, torch.Tensor]] = None, device: Union[str, torch.device]="cuda") -> torch.Tensor:
        """
        Forward pass through the model.
        Args:
            x: Input tensor
            y: Target tensor
        Returns:
            Model output tensor
        """
        if device is None:
            device = self.device
        x_preprocessed = self.preprocess_sample(x, device=device)
        if y is not None and self.mode == 'autoregressive':
            y_preprocessed = self.preprocess_sample(y, device=device)
            if self.wrapper is not None:
                out = self.wrapper(x_preprocessed, y_preprocessed)
            else:
                out = self.model(x_preprocessed, y_preprocessed)
        else:
            if self.wrapper is not None:
                out = self.wrapper(x_preprocessed)
            else:
                out = self.model(x_preprocessed)

        return out
    def preprocess_output_and_prediction_before_comparison(self, target: torch.Tensor, output: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        # print(f"Output shape: {output.shape}, dtype={output.dtype} iscomplex={torch.is_complex(output)}")
        # print(f"Target shape: {target.shape}, dtype={target.dtype} iscomplex={torch.is_complex(target)}")

        if output.shape[-1] > 2:
            output = output[..., :2]
        elif output.shape[-1] == 2:
            if (isinstance(output, torch.Tensor) and torch.is_complex(output)) or (isinstance(output, np.ndarray) and np.iscomplexobj(output)):
                output = output[..., :1]
        if target.shape[-1] > 2:
            target = target[..., :2]
        elif target.shape[-1] == 2:
            if (isinstance(target, torch.Tensor) and torch.is_complex(target)) or (isinstance(target, np.ndarray) and np.iscomplexobj(target)):
                target = target[..., :1]

        if len(output.shape) > 3:
            output = output.squeeze(-1)
        if len(target.shape) > 3:
            target = target.squeeze(-1)

        return target, output
    def preprocess_sample(self, x: Union[torch.Tensor, np.ndarray], device: Union[str, torch.device]=None):   
        if device is None:
            device = self.device
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)              
        if torch.is_complex(x):
            x = x.to(torch.complex64)
        else:
            x = x.float()
        return x.to(device)


class TrainDUN(TrainerBase):
    pass

def get_training_loop_by_model_name(
        model_name: str, 
        model: nn.Module,  
        train_loader: SARDataloader,
        val_loader: SARDataloader,
        test_loader: SARDataloader,
        inference_loader: Optional[SARDataloader] = None,
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
            inference_loader=inference_loader,
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
            inference_loader=inference_loader,
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
            inference_loader=inference_loader,
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
                        enable_progress_bar=True
                        )
    return lightning_model, trainer