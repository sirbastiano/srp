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
from model.SSMs.SSM import OverlapSaveWrapper, WindowedOverlapWrapper
from model.transformers.rv_transformer import RealValuedTransformer  
from model.transformers.cv_transformer import ComplexTransformer
from training.visualize import compute_metrics, save_metrics, average_metrics
from sarpyx.utils.losses import get_loss_function
from training.visualize import get_full_image_and_prediction, compute_metrics, display_inference_results, log_inference_to_wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

# Import wandb with fallback for gradient tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

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
            lr: float = 1e-4, 
            warmup_epochs: int = 3,
            warmup_start_lr: float = 1e-6,
            weight_decay: float = 1e-6,
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
        self.last_improve_steps = 0
        
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        self.weight_decay = weight_decay
        # Initialize gradient tracking if wandb is available
        self.gradient_tracker = None
        self.global_step_count = 0
        if WANDB_AVAILABLE:
            self.gradient_tracker = self._create_gradient_tracker()
        
    def get_current_step_or_epoch(self):
        """Get current step or epoch depending on training mode."""
        if hasattr(self.trainer, 'max_steps') and self.trainer.max_steps > 0:
            return self.global_step, 'step'
        else:
            return self.current_epoch, 'epoch'

    def _create_gradient_tracker(self):
        """Create gradient tracker utility"""
        class SimpleGradientTracker:
            def __init__(self, log_frequency: int = 100):
                self.log_frequency = log_frequency
            
            def log_gradients(self, model: nn.Module, model_name: str, global_step: int):
                if global_step % self.log_frequency != 0:
                    return
                    
                gradient_stats = {}
                total_norm = 0.0
                param_count = 0
                
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad = param.grad.data
                        grad_norm = torch.norm(grad).item()
                        total_norm += grad_norm ** 2
                        param_count += 1
                        
                        # Log key layer gradients
                        if any(key in name for key in ['input_proj', 'output_proj', 'layers.0', 'layers.-1']):
                            clean_name = name.replace('.', '/')
                            gradient_stats[f"{model_name}/gradients/{clean_name}/norm"] = grad_norm
                            gradient_stats[f"{model_name}/gradients/{clean_name}/mean"] = torch.mean(grad).item()
                            gradient_stats[f"{model_name}/gradients/{clean_name}/std"] = torch.std(grad).item()
                
                if param_count > 0:
                    total_norm = (total_norm ** 0.5)
                    gradient_stats[f"{model_name}/gradients/total_norm"] = total_norm
                    gradient_stats[f"{model_name}/gradients/param_count"] = param_count
                
                try:
                    if WANDB_AVAILABLE:
                        wandb.log(gradient_stats, step=global_step)
                except:
                    pass  # Fail silently if wandb is not available
        
        return SimpleGradientTracker()
    
    def on_after_backward(self):
        """Hook called after backward pass to log gradients"""
        if self.gradient_tracker is not None:
            self.global_step_count += 1
            self.gradient_tracker.log_gradients(
                self.model,
                "model",
                self.global_step_count
            )
        self.last_improve_steps = 0
        
        
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

    def show_example(self, loader: SARDataloader, window: Tuple[Tuple[int, int], Tuple[int, int]] = ((1000, 1000), (5000, 5000)), vminmax=(2000, 6000), figsize=(20, 6), metrics_save_path: str = "metrics.json", img_save_path: str = "test.png"):
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
            metrics = compute_metrics(gt, pred)
            with open(os.path.join(self.base_save_dir, metrics_save_path), 'w') as f:
                json.dump(metrics, f)
                

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
            # gt, pred = self.preprocess_output_and_prediction_before_comparison(gt, pred)

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
            # gt_patch, pred_patch = self.preprocess_output_and_prediction_before_comparison(gt_patch, pred_patch)
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
            # gt_patch, pred_patch = self.preprocess_output_and_prediction_before_comparison(gt_patch, pred_patch)
            m = compute_metrics(gt_patch, pred_patch)
            self.validation_metrics.append(m)

        return loss
    def on_validation_epoch_end(self):
        avg_metrics = average_metrics(self.validation_metrics)
        self.validation_metrics = []
        self.log_metrics(avg_metrics, 'val_metrics', on_step=False, on_epoch=True, prog_bar=False)

        if self.inference_loader is not None:
            self.show_example(self.inference_loader, window=((1000, 1000), (2000, 2000)), vminmax='auto', figsize=(20, 6), metrics_save_path=f"metrics_{self.current_epoch}.json", img_save_path=f"val_{self.current_epoch}.png")
        if 'val_loss' in self.trainer.callback_metrics:
            self.save_checkpoint(epoch=self.current_epoch, optimizer=self.trainer.optimizers[0], scheduler=None)#, is_best=(self.trainer.callback_metrics['val_loss'] < self.best_val_loss), val_loss=self.trainer.callback_metrics['val_loss'])
            if self.trainer.callback_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = self.trainer.callback_metrics['val_loss']
                self.last_improve_epochs = 0
            else:
                self.last_improve_epochs += 1
        super().on_validation_epoch_end()
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        # print(f"Test batch {batch_idx}: ")
        # print(f"x={x}, ")
        # print(f"y={y}")
        output = self.forward(x, y)
                
        y_np = y.cpu().numpy() #.squeeze()
        output_np = output.cpu().numpy() #.squeeze()
        y_np, output_np = self.preprocess_output_and_prediction_before_comparison(y_np, output_np)
        for i in range(y_np.shape[0]):
            gt_patch = self.test_dataloader().dataset.get_patch_visualization(y_np[i], self.train_dataloader().dataset.level_to, vminmax=(4000, 4200), restore_complex=True, remove_positional_encoding=False)
            #print(f"Ground truth patch with index {idx} has shape: {gt_patch.shape}, while reconstructed ground truth patch has dimension {gt_patch.shape}")
            pred_patch = self.test_dataloader().dataset.get_patch_visualization(output_np[i], self.train_dataloader().dataset.level_to, vminmax=(4000, 4200), restore_complex=True, remove_positional_encoding=False)
            # gt_patch, pred_patch = self.preprocess_output_and_prediction_before_comparison(gt_patch, pred_patch)
            m = compute_metrics(gt_patch, pred_patch)
            self.test_metrics.append(m)

        loss = self.criterion(output, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    def on_test_end(self):
        avg_metrics = average_metrics(self.test_metrics)
        self.log_metrics(avg_metrics, 'test_metrics', on_step=False, on_epoch=True, prog_bar=False)
        self.show_example(self.test_dataloader, window=((1000, 1000), (2000, 2000)), vminmax='auto', figsize=(20, 6), metrics_save_path=f"metrics_{self.current_epoch}.json", img_save_path=f"test.png")
        avg_metrics['num_samples'] = len(self.test_metrics)
        avg_metrics['val_losses'] = self.val_losses
        metrics_path = os.path.join(self.base_save_dir, self.metrics_file_name)
        save_metrics(avg_metrics, metrics_path)

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
            input_dim: int = 3,
            mode: str = "parallel",
            scheduler_type: str = 'cosine',
            lr: float = 1e-4, 
            wrapper: bool = False, 
            real: bool = False, 
            step_mode: bool = False
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
            scheduler_type=scheduler_type, 
            lr=lr
        )
        if wrapper: 
            self.wrapper = OverlapSaveWrapper(model=model, step_mode=step_mode)
        else:
            self.wrapper = None
        self.real = real
        self.step_mode = step_mode
        self.input_dim = input_dim
        # self.logger = logging.getLogger(__name__)
        
        # # Count parameters and log model info
        # if hasattr(model, 'parameters'):
        #     total_params = sum(p.numel() for p in model.parameters())
        #     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        #     self.logger.info(f"Model created with {total_params:,} total parameters")
        #     self.logger.info(f"Trainable parameters: {trainable_params:,}")
    def _convert_complex_to_real(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert complex SAR input to real format for student model.
        Student expects real tensors with channels: (real_part, imag_part, horizontal_pos_embedding)
        
        Args:
            x: Complex input tensor (B, L, 2) where last dim is [real+1j*imag, horizontal_pos_embedding]
            
        Returns:
            Real tensor (B, L, 3) with [real_part, imag_part, horizontal_pos_embedding]
        """
        if torch.is_complex(x):
            # x is complex tensor (B, L, 2) - [complex_value, horizontal_pos_embedding]
            if x.shape[-1] >= 2:
                complex_part = x[..., 0]  # (B, L) - complex values
                horizontal_pos = x[..., 1].imag  # (B, L) - horizontal position (take imag part)
                vertical_pos = x[..., 1].real  # (B, L) - vertical position (take real part)
                # Convert complex to (real, imag, horizontal_pos)
                real_part = complex_part.real  # (B, L)
                imag_part = complex_part.imag  # (B, L)
                
                if self.input_dim == 2:
                    # Stack to create (B, L, 2)
                    student_input = torch.stack([real_part, imag_part], dim=-1)
                elif self.input_dim == 3: 
                    # Stack to create (B, L, 3)
                    student_input = torch.stack([real_part, imag_part, horizontal_pos], dim=-1)
                else:
                    # Stack to create (B, L, 4)
                    student_input = torch.stack([real_part, imag_part, vertical_pos, horizontal_pos], dim=-1)

            else:
                # If only complex values, add zero horizontal position
                complex_part = x[..., 0] if x.shape[-1] > 0 else x.squeeze(-1)
                real_part = complex_part.real
                imag_part = complex_part.imag
                horizontal_pos = torch.zeros_like(real_part)
                
                student_input = torch.stack([real_part, imag_part, horizontal_pos], dim=-1)
        else:
            # x is already real - check if it needs conversion
            if x.shape[-1] == 2:
                # Assume it's [real, imag] format, add zero horizontal position
                real_part = x[..., 0]
                imag_part = x[..., 1]
                horizontal_pos = torch.zeros_like(real_part)
                student_input = torch.stack([real_part, imag_part, horizontal_pos], dim=-1)
            elif x.shape[-1] == 3:
                # Already in correct format
                student_input = x
            else:
                # Add padding dimensions if needed
                if x.shape[-1] == 1:
                    # Single channel - assume real part, add imag and pos
                    real_part = x[..., 0]
                    imag_part = torch.zeros_like(real_part)
                    horizontal_pos = torch.zeros_like(real_part)
                    student_input = torch.stack([real_part, imag_part, horizontal_pos], dim=-1)
                else:
                    student_input = x
        self.squeeze_dim = None
        if student_input.shape[1] == 1:
            self.squeeze_dim = 1
        elif student_input.shape[2] == 1:
            self.squeeze_dim = 2
        if self.squeeze_dim is not None:
            student_input = student_input.squeeze(self.squeeze_dim)
        return student_input.float()
    
    def _convert_real_to_complex(self, student_real_output: torch.Tensor) -> torch.Tensor:
        """
        Convert student's real output back to complex format for loss computation.
        
        Args:
            student_real_output: Real output from student (B, L, 2) with [real_part, imag_part]
            
        Returns:
            Complex tensor (B, L, 1) for loss computation
        """
        if student_real_output.shape[-1] >= 2:
            real_part = student_real_output[..., 0]
            imag_part = student_real_output[..., 1]
            complex_output = torch.complex(real_part, imag_part)
            # Add dimension to match expected format (B, L, 1)
            complex_output = complex_output.unsqueeze(-1)
        else:
            # If only one channel, treat as real part
            real_part = student_real_output[..., 0] if student_real_output.shape[-1] > 0 else student_real_output.squeeze(-1)
            complex_output = torch.complex(real_part, torch.zeros_like(real_part))
            complex_output = complex_output.unsqueeze(-1)
        if self.squeeze_dim is not None:
            complex_output = complex_output.unsqueeze(self.squeeze_dim)
        return complex_output
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
        if self.real:
            x_preprocessed = self._convert_complex_to_real(x_preprocessed)
        # if y is not None and self.mode == 'autoregressive':
        #     y_preprocessed = self.preprocess_sample(y, device=device)
        #     if self.wrapper is not None:
        #         out = self.wrapper(x_preprocessed, y_preprocessed)
        #     else:
        #         if self.step_mode:
        #             out = self.model.step(x_preprocessed, y_preprocessed)
        #        else:
        #             out = self.model(x_preprocessed, y_preprocessed)
        # else:
        

        if self.wrapper is not None:
            out = self.wrapper(x_preprocessed)
        else:
            if self.step_mode:
                state = self.model.setup_step()
                for t in range(x_preprocessed.shape[1]):
                    xt = x_preprocessed[:, t:t+1, :]  # (B, 1, C)
                    print(f"Input to step: {xt.shape}")
                    out_step, state = self.model(xt, state)
                    if t == 0:
                        out = out_step
                    else:
                        out = torch.cat((out, out_step), dim=1)  # Concatenate along sequence dimension
            else:
                out = self.model(x_preprocessed)
        # print(f"Output from model before conversion: shape={out.shape}, dtype={out.dtype}, iscomplex={torch.is_complex(out)}")
        if self.real:
            out = self._convert_real_to_complex(out)
        return out
    def preprocess_output_and_prediction_before_comparison(self, target: torch.Tensor, output: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        # print(f"Output shape: {output.shape}, dtype={output.dtype} iscomplex={torch.is_complex(output) if isinstance(output, torch.Tensor) else np.iscomplexobj(output)}")
        # print(f"Target shape: {target.shape}, dtype={target.dtype} iscomplex={torch.is_complex(target) if isinstance(target, torch.Tensor) else np.iscomplexobj(target)}")

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
        if not ((isinstance(target, torch.Tensor) and torch.is_complex(target)) or (isinstance(target, np.ndarray) and np.iscomplexobj(target))):
            target = target[..., 0] + 1j * target[..., 1]
        if not ((isinstance(output, torch.Tensor) and torch.is_complex(output)) or (isinstance(output, np.ndarray) and np.iscomplexobj(output))):
            output = output[..., 0] + 1j * output[..., 1]
        # print(f"After processing - Output shape: {output.shape}, dtype={output.dtype} iscomplex={torch.is_complex(output) if isinstance(output, torch.Tensor) else np.iscomplexobj(output)}")
        # print(f"After processing - Target shape: {target.shape}, dtype={target.dtype} iscomplex={torch.is_complex(target) if isinstance(target, torch.Tensor) else np.iscomplexobj(target)}")
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
        device_no: int= 0, 
        input_dim: int = 3, 
        patience: int = 50,
        **kwargs  # Accept additional parameters to prevent TypeError
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
            scheduler_type=scheduler_type,
            real=('final' in model_name.lower()), 
            input_dim=input_dim
        )
    else:
        raise ValueError(f"Unsupported model type: {model_name}")
    
    # Create early stopping callback
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        verbose=True,
        mode='min'
    )
    
    trainer = pl.Trainer(max_epochs=num_epochs,
                        logger=logger,
                        devices=[device_no],
                        fast_dev_run=False,
                        log_every_n_steps=1,
                        accelerator="gpu" if torch.cuda.is_available() else "cpu",
                        enable_progress_bar=True,
                        callbacks=[early_stop_callback]
                        )
    return lightning_model, trainer