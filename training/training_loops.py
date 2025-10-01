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
from training.distillation_models import DistillationSSM, create_teacher_student_pair
from training.distillation_losses import FeatureDistillationLoss, CombinedDistillationLoss, AttentionTransferLoss
from training.visualize import get_full_image_and_prediction, compute_metrics, display_inference_results, log_inference_to_wandb
import pytorch_lightning as pl

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
        
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        self.weight_decay = weight_decay
        
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
            step_or_epoch = self.current_epoch if self.current_epoch is not None else self.current
            log_inference_to_wandb(
                input_data=input,
                gt_data=gt,
                pred_data=pred,
                logger=self.logger,
                vminmax=vminmax,
                step_or_epoch=step_or_epoch,
                save_path=os.path.join(self.base_save_dir, img_save_path)
            )
    
            # display_inference_results(
            #     input_data=input,
            #     gt_data=gt,
            #     pred_data=pred,
            #     figsize=figsize,
            #     vminmax=vminmax,  
            #     show=True, 
            #     save=True,
            #     save_path=os.path.join(self.base_save_dir, img_save_path)
            # )
            gt, pred = self.preprocess_output_and_prediction_before_comparison(gt, pred)
            metrics = compute_metrics(gt, pred)
            with open(os.path.join(self.base_save_dir, metrics_save_path), 'w') as f:
                json.dump(metrics, f)
                
        except Exception as e:
            print(f"Visualization failed with error: {str(e)}")
            raise

    def save_checkpoint(self, optimizer, scheduler=None, is_best: bool = False, **kwargs):
        """Save training checkpoint with all necessary information."""
        from model.model_utils import save_checkpoint
        
        # Get current progress indicator
        current_progress, progress_type = self.get_current_step_or_epoch()
        
        # Always save the latest checkpoint
        latest_checkpoint_path = os.path.join(self.base_save_dir, "checkpoint_latest.pth")
        latest_model_path = os.path.join(self.base_save_dir, "model_last.pth")
        
        # Prepare metrics
        metrics = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'val_losses': self.val_losses,
            'train_losses': self.train_losses,
            'best_val_loss': self.best_val_loss,
            'last_improve_epochs': self.last_improve_epochs,
            # 'last_improve_steps': self.last_improve_steps,
            'progress_type': progress_type
        }
        metrics.update(kwargs)
        
        # Save comprehensive latest checkpoint
        save_checkpoint(
            model=self.model,
            save_path=latest_checkpoint_path,
            epoch=self.current_epoch,
            optimizer=optimizer,
            scheduler=scheduler,
            metrics=metrics,
            model_config=getattr(self.model, '_model_config', None)
        )
        
        # Save latest model weights
        torch.save(self.model.state_dict(), latest_model_path)
        print(f"✅ Saved latest checkpoint: {progress_type} {current_progress}")
        
        # Save best checkpoint if this is the best model
        if is_best:
            best_checkpoint_path = os.path.join(self.base_save_dir, "checkpoint_best.pth")
            best_model_path = os.path.join(self.base_save_dir, "model_best.pth")
            
            # Copy latest checkpoint to best checkpoint
            save_checkpoint(
                model=self.model,
                save_path=best_checkpoint_path,
                epoch=self.current_epoch,
                optimizer=optimizer,
                scheduler=scheduler,
                metrics=metrics,
                model_config=getattr(self.model, '_model_config', None)
            )
            
            # Save best model weights
            torch.save(self.model.state_dict(), best_model_path)
            val_loss = kwargs.get('val_loss', 'N/A')
            print(f"🎉 New best model saved! {progress_type.capitalize()} {current_progress}, Validation loss: {val_loss:.6f}")

    def resume_from_checkpoint(self, checkpoint_path: Optional[Union[str, os.PathLike]], optimizer, scheduler=None):
        """Resume training from a checkpoint."""
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            return False
            
        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load model weights
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # Fallback for simple state dict
                self.model.load_state_dict(checkpoint)
            
            # Load optimizer state
            if optimizer and 'optimizer_state_dict' in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                except Exception as e:
                    print(f"⚠️  Could not load optimizer state: {e}")
                
            # Load scheduler state
            if scheduler and 'scheduler_state_dict' in checkpoint:
                try:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                except Exception as e:
                    print(f"⚠️  Could not load scheduler state: {e}")
                
            # Load training state
            if 'metrics' in checkpoint:
                metrics = checkpoint['metrics']
                self.val_losses = metrics.get('val_losses', [])
                self.train_losses = metrics.get('train_losses', [])
                self.best_val_loss = metrics.get('best_val_loss', float('inf'))
                self.last_improve_epochs = metrics.get('last_improve_epochs', 0)
                self.last_improve_steps = metrics.get('last_improve_steps', 0)
                self.start_epoch = checkpoint.get('epoch', 0) + 1

            progress_type = checkpoint.get('metrics', {}).get('progress_type', 'epoch')
            current_progress = checkpoint.get('global_step', 0) if progress_type == 'step' else checkpoint.get('epoch', 0)
            
            print(f"✅ Resumed training from {progress_type} {current_progress}")
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
        elif output.shape[-1] == 2 and torch.is_complex(output):
            output = output[..., :-1]
        if target.shape[-1] > 2:
            target = target[..., :-2]
        elif target.shape[-1] == 2 and torch.is_complex(target):
            target = target[..., :-1]
        return target, output
    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x, y)  
        #print(f"Output from model: {output}")
        #print(f"Ground truth: {y}")
        loss = self.compute_loss(output, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Log gradient norms for monitoring
        if self.global_step % 100 == 0:  # Log every 100 steps to avoid spam
            total_grad_norm = 0.0
            num_params = 0
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    param_grad_norm = param.grad.data.norm(2).item()
                    total_grad_norm += param_grad_norm ** 2
                    num_params += 1
                    # Log individual layer gradient norms for complex models
                    if 'layers' in name or 'kernel' in name:
                        self.log(f'grad_norm/{name}', param_grad_norm, on_step=True, on_epoch=False)
            
            if num_params > 0:
                total_grad_norm = total_grad_norm ** 0.5
                self.log('grad_norm/total', total_grad_norm, on_step=True, on_epoch=False)
        
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
        
        # Log learning rate
        optimizer = self.trainer.optimizers[0] if self.trainer.optimizers else None
        if optimizer:
            current_lr = optimizer.param_groups[0]['lr']
            self.log('learning_rate', current_lr, on_step=False, on_epoch=True, prog_bar=False)
        
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
        if hasattr(self.trainer, 'max_steps') and self.trainer.max_steps > 0:    
            if self.validation_metrics:
                avg_metrics = average_metrics(self.validation_metrics)
                self.validation_metrics = []
                self.log_metrics(avg_metrics, 'val_metrics', on_step=False, on_epoch=True, prog_bar=False)

            # Get current validation loss
            # current_val_loss = self.trainer.callback_metrics.get('val_loss', float('inf'))
            
            # Check if this is the best model so far
            is_best = loss < self.best_val_loss
            optimizer = self.trainer.optimizers[0] if self.trainer.optimizers else None
            scheduler = self.trainer.lr_scheduler_configs[0].scheduler if self.trainer.lr_scheduler_configs else None
            self.save_checkpoint(
                epoch=self.current_epoch, 
                optimizer=optimizer, 
                scheduler=scheduler, 
                is_best=is_best,
                val_loss=float(loss),
                train_loss=self.trainer.callback_metrics.get('train_loss_epoch', float('inf'))
            )
        return loss
    def on_validation_epoch_end(self):
        # Calculate average validation metrics
        if self.validation_metrics:
            avg_metrics = average_metrics(self.validation_metrics)
            self.validation_metrics = []
            self.log_metrics(avg_metrics, 'val_metrics', on_step=False, on_epoch=True, prog_bar=False)

        # Get current validation loss
        current_val_loss = self.trainer.callback_metrics.get('val_loss', float('inf'))
        
        # Check if this is the best model so far
        is_best = current_val_loss < self.best_val_loss
        
        if is_best:
            print(f"🎉 New best validation loss: {current_val_loss:.6f} (previous: {self.best_val_loss:.6f})")
            self.best_val_loss = float(current_val_loss)
            self.last_improve_epochs = 0
        else:
            self.last_improve_epochs += 1
            print(f"📊 Validation loss: {current_val_loss:.6f} (best: {self.best_val_loss:.6f}) - No improvement for {self.last_improve_epochs} epochs")

        # Save checkpoints
        optimizer = self.trainer.optimizers[0] if self.trainer.optimizers else None
        scheduler = self.trainer.lr_scheduler_configs[0].scheduler if self.trainer.lr_scheduler_configs else None
        
        self.save_checkpoint(
            epoch=self.current_epoch, 
            optimizer=optimizer, 
            scheduler=scheduler, 
            is_best=is_best,
            val_loss=float(current_val_loss),
            train_loss=self.trainer.callback_metrics.get('train_loss_epoch', float('inf'))
        )

        # Show example inference if inference loader is available
        if self.inference_loader is not None:
            try:
                self.show_example(
                    self.inference_loader, 
                    window=((1000, 1000), (2000, 2000)), 
                    vminmax=(2000, 6000), 
                    figsize=(20, 6), 
                    metrics_save_path=f"metrics_{self.current_epoch}.json", 
                    img_save_path=f"val_{self.current_epoch}.png"
                )
            except Exception as e:
                print(f"⚠️  Inference visualization failed: {str(e)}")

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
        """Configure optimizers with warmup support."""
        import math
        
        # Create optimizer with weight decay
        optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        
        # Calculate total training steps
        if hasattr(self.trainer, 'estimated_stepping_batches') and self.trainer.estimated_stepping_batches:
            total_steps = self.trainer.estimated_stepping_batches
        else:
            # Fallback calculation
            steps_per_epoch = len(self.train_dataloader())
            if hasattr(self.trainer, 'max_epochs') and self.trainer.max_epochs:
                total_steps = steps_per_epoch * self.trainer.max_epochs
            elif hasattr(self.trainer, 'max_steps') and self.trainer.max_steps:
                total_steps = self.trainer.max_steps
            else:
                total_steps = steps_per_epoch * 50  # Default fallback
        
        # Calculate warmup steps
        if hasattr(self.trainer, 'max_steps') and self.trainer.max_steps > 0:
            # Step-based training
            warmup_steps = min(self.warmup_epochs * 100, total_steps // 10)  # 10% of total or warmup_epochs * 100
        else:
            # Epoch-based training
            steps_per_epoch = len(self.train_dataloader())
            warmup_steps = self.warmup_epochs * steps_per_epoch
        
        print(f"🔥 Warmup configuration:")
        print(f"   Warmup epochs: {self.warmup_epochs}")
        print(f"   Warmup steps: {warmup_steps}")
        print(f"   Total steps: {total_steps}")
        print(f"   Start LR: {self.warmup_start_lr:.2e}")
        print(f"   Target LR: {self.lr:.2e}")
        
        # Create learning rate scheduler with warmup
        def lr_lambda(step):
            if step < warmup_steps:
                # Linear warmup: from warmup_start_lr to lr
                warmup_factor = self.warmup_start_lr / self.lr
                return warmup_factor + (1.0 - warmup_factor) * step / warmup_steps
            else:
                # Post-warmup scheduling
                if self.scheduler_type == 'cosine':
                    # Cosine annealing after warmup
                    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                    return 0.5 * (1 + math.cos(math.pi * progress))
                elif self.scheduler_type == 'linear':
                    # Linear decay after warmup
                    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                    return 1.0 - progress
                elif self.scheduler_type == 'constant':
                    # Constant LR after warmup
                    return 1.0
                else:
                    # Default: cosine annealing
                    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                    return 0.5 * (1 + math.cos(math.pi * progress))
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # Update every step for smooth warmup
                "frequency": 1,
                "name": "lr_with_warmup"
            },
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
                scheduler_type: str = 'cosine',
                weight_decay: float = 1e-6,
                warmup_epochs: int = 3,
                warmup_start_lr: float = 1e-6,
                lr: float = 1e-4
            ):
        super().__init__(base_save_dir, model, train_loader, val_loader, test_loader, inference_loader, mode, criterion=criterion, scheduler_type=scheduler_type,  weight_decay=weight_decay, warmup_epochs=warmup_epochs, warmup_start_lr=warmup_start_lr, lr=lr)
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
            scheduler_type: str = 'cosine', 
            weight_decay: float = 1e-6,
            warmup_epochs: int = 3,
            warmup_start_lr: float = 1e-6,
            lr: float = 1e-4
        ):
        super().__init__(base_save_dir, model, train_loader, val_loader, test_loader, inference_loader, mode, criterion=criterion, scheduler_type=scheduler_type, weight_decay=weight_decay, warmup_epochs=warmup_epochs, warmup_start_lr=warmup_start_lr, lr=lr)

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
            wrapper: bool = False, 
            weight_decay: float = 1e-6,
            warmup_epochs: int = 3,
            warmup_start_lr: float = 1e-6, 
            lr: float = 1e-4
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
            weight_decay=weight_decay,
            warmup_epochs=warmup_epochs,
            warmup_start_lr=warmup_start_lr,
            lr=lr
        )
        print(f"Model is using wrapper? {wrapper}")
        if wrapper: 
            self.wrapper = WindowedOverlapWrapper(model=model) #OverlapSaveWrapper(model=model)
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
        device_no: int= 0, 
        weight_decay: float = 1e-6,
        warmup_epochs: int = 3,
        warmup_start_lr: float = 1e-6,
        lr: float = 1e-4, 
        wrapper: bool = True
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
            scheduler_type=scheduler_type,
            weight_decay=weight_decay,
            warmup_epochs=warmup_epochs,
            warmup_start_lr=warmup_start_lr,
            lr=lr
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
            scheduler_type=scheduler_type,
            weight_decay=weight_decay,
            warmup_epochs=warmup_epochs,
            warmup_start_lr=warmup_start_lr,
            lr=lr
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
            weight_decay=weight_decay,
            warmup_epochs=warmup_epochs,
            warmup_start_lr=warmup_start_lr,
            lr=lr, 
            wrapper=wrapper
        )
    else:
        raise ValueError(f"Unsupported model type: {model_name}")
    trainer = pl.Trainer(max_epochs=num_epochs,
                        logger=logger,
                        devices=[device_no],
                        fast_dev_run=False,
                        log_every_n_steps=1,
                        accelerator="gpu" if torch.cuda.is_available() else "cpu",
                        enable_progress_bar=True,
                        )
    print(f"Created training loop for model: {model_name}, with parameters: weight_decay={weight_decay}, warmup_epochs={warmup_epochs}, warmup_start_lr={warmup_start_lr}, lr={lr}, scheduler_type={scheduler_type}, num_epochs={num_epochs},  mode={mode}, loss_fn={loss_fn_name}, device_no={device_no}, save_dir={save_dir}, trainer={trainer}, lightning_model={lightning_model}")
    return lightning_model, trainer


class DistillationTrainer(TrainerBase):
    """
    Trainer class for knowledge distillation with teacher-student models.
    
    Supports training a student model to mimic both the outputs and 
    intermediate features of a pretrained teacher model.
    """
    
    def __init__(
        self,
        teacher_model: DistillationSSM,
        student_model: DistillationSSM,
        train_loader: SARDataloader,
        val_loader: SARDataloader,
        test_loader: SARDataloader,
        base_save_dir: str = "/tmp",
        inference_loader: Optional[SARDataloader] = None,
        mode: str = "parallel",
        criterion: Callable = nn.MSELoss, 
        scheduler_type: str = 'cosine', 
        metrics_file_name: str = "test_metrics.json", 
        lr: float = 1e-4, 
        warmup_epochs: int = 3,
        warmup_start_lr: float = 1e-6,
        weight_decay: float = 1e-6,
        resume_from: Optional[str] = None,
        **kwargs
    ):
        # Initialize with student model as the main model
        super().__init__(
            model=student_model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            base_save_dir=base_save_dir,
            inference_loader=inference_loader,
            mode=mode,
            criterion=criterion,
            scheduler_type=scheduler_type,
            metrics_file_name=metrics_file_name,
            lr=lr,
            warmup_epochs=warmup_epochs,
            warmup_start_lr=warmup_start_lr,
            weight_decay=weight_decay,
            resume_from=resume_from,
            **kwargs
        )
        
        self.teacher_model = teacher_model
        self.student_model = student_model
        
        # Freeze teacher model
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
            
        # Setup distillation losses with default config
        distillation_config = kwargs.get('distillation_config', {})
        self.setup_distillation_losses(distillation_config)
        
        print(f"🎓 Distillation trainer initialized")
        print(f"   Teacher: {self.teacher_model.__class__.__name__}")
        print(f"   Student: {self.student_model.__class__.__name__}")
        print(f"   Task loss weight: {self.task_loss_weight}")
        print(f"   Feature loss weight: {self.feature_loss_weight}")
        print(f"   Temperature: {self.temperature}")
    
    def setup_distillation_losses(self, config: dict):
        """Setup the distillation loss components."""
        distill_config = config.get('distillation', {})
        
        # Loss weights
        self.task_loss_weight = distill_config.get('task_loss_weight', 1.0)
        self.feature_loss_weight = distill_config.get('feature_loss_weight', 1.0)
        self.attention_loss_weight = distill_config.get('attention_loss_weight', 0.0)
        
        # Temperature for output distillation
        self.temperature = distill_config.get('temperature', 4.0)
        
        # Feature distillation loss
        self.feature_distill_loss = FeatureDistillationLoss(
            temperature=self.temperature,
            alpha=self.feature_loss_weight
        )
        
        # Attention transfer loss (optional)
        if self.attention_loss_weight > 0:
            self.attention_loss = AttentionTransferLoss()
        else:
            self.attention_loss = None
            
        # Combined distillation loss
        self.distillation_criterion = CombinedDistillationLoss(
            task_loss=self.criterion,
            feature_loss=self.feature_distill_loss,
            attention_loss=self.attention_loss,
            task_weight=self.task_loss_weight,
            feature_weight=self.feature_loss_weight,
            attention_weight=self.attention_loss_weight
        )
    
    def forward(self, x: Union[np.ndarray, torch.Tensor], y: Optional[Union[np.ndarray, torch.Tensor]]=None, device: Union[str, torch.device]="cuda") -> torch.Tensor:
        """Forward pass through student model only."""
        if device is None:
            device = self.device
        x_preprocessed = self.preprocess_sample(x, device)
        if y is None:
            y_preprocessed = None
        else:
            y_preprocessed = self.preprocess_sample(y, device)
        return self.student_model(x_preprocessed, y_preprocessed)
    
    def distillation_forward(self, x: torch.Tensor, y: torch.Tensor) -> dict:
        """
        Forward pass through both teacher and student models.
        Returns predictions and intermediate features.
        """
        # Teacher forward (no gradients)
        with torch.no_grad():
            teacher_output, teacher_features = self.teacher_model(x, y, return_features=True)
        
        # Student forward (with gradients)
        student_output, student_features = self.student_model(x, y, return_features=True)
        
        return {
            'teacher_output': teacher_output,
            'teacher_features': teacher_features,
            'student_output': student_output,
            'student_features': student_features,
            'target': y
        }
    
    def compute_distillation_loss(self, batch_results: dict) -> torch.Tensor:
        """Compute the combined distillation loss."""
        return self.distillation_criterion(
            student_output=batch_results['student_output'],
            teacher_output=batch_results['teacher_output'],
            student_features=batch_results['student_features'],
            teacher_features=batch_results['teacher_features'],
            target=batch_results['target']
        )
    
    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Single training step with distillation."""
        # Forward pass through both models
        batch_results = self.distillation_forward(x, y)
        
        # Compute distillation loss
        loss = self.compute_distillation_loss(batch_results)
        
        return loss
    
    def val_step(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Single validation step with distillation."""
        # Forward pass through both models
        batch_results = self.distillation_forward(x, y)
        
        # Compute distillation loss
        loss = self.compute_distillation_loss(batch_results)
        
        return loss
    
    def load_teacher_from_checkpoint(self, teacher_checkpoint_path: str):
        """Load pretrained teacher model from checkpoint."""
        if not os.path.exists(teacher_checkpoint_path):
            raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_checkpoint_path}")
        
        checkpoint = torch.load(teacher_checkpoint_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            self.teacher_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.teacher_model.load_state_dict(checkpoint)
        
        print(f"✅ Teacher model loaded from: {teacher_checkpoint_path}")
    
    def save_student_checkpoint(self, optimizer, scheduler=None, is_best: bool = False, **kwargs):
        """Save checkpoint for student model only."""
        # Use parent's save_checkpoint but ensure we're saving student model
        original_model = self.model
        self.model = self.student_model
        try:
            self.save_checkpoint(optimizer, scheduler, is_best, **kwargs)
        finally:
            self.model = original_model


def create_distillation_trainer(
    teacher_config: dict,
    student_config: dict, 
    distillation_config: dict,
    teacher_checkpoint_path: str,
    train_loader: SARDataloader,
    val_loader: SARDataloader,
    test_loader: SARDataloader,
    base_save_dir: str = "/tmp",
    device: str = "cuda",
    **kwargs
) -> DistillationTrainer:
    """
    Factory function to create a distillation trainer.
    
    Args:
        teacher_config: Configuration for teacher model
        student_config: Configuration for student model
        distillation_config: Distillation-specific configuration
        teacher_checkpoint_path: Path to pretrained teacher checkpoint
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        base_save_dir: Directory for saving training artifacts
        device: Device for training
    
    Returns:
        Configured DistillationTrainer instance
    """
    # Create teacher and student models
    teacher_model, student_model = create_teacher_student_pair(
        teacher_config=teacher_config,
        student_config=student_config,
        device=device
    )
    
    # Create trainer
    trainer = DistillationTrainer(
        teacher_model=teacher_model,
        student_model=student_model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        base_save_dir=base_save_dir,
        distillation_config=distillation_config,
        **kwargs
    )
    
    # Load pretrained teacher
    trainer.load_teacher_from_checkpoint(teacher_checkpoint_path)
    
    return trainer
