import json
import logging
import torch
from torch import nn, optim
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

import numpy as np

class TrainerBase():
    def __init__(
        self, 
        base_save_dir:str,
        model , 
        mode: str = "parallel",
        criterion: Callable = nn.MSELoss
    ):
        self.base_save_dir = base_save_dir
        self.model = model 
        self.mode = mode
        self.criterion_fn = criterion
        
    def compute_loss(self, output: torch.Tensor, target: torch.Tensor):
        pass
    def preprocess_sample(self, x: Union[np.ndarray, torch.Tensor], device: Union[str, torch.device]):                
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        return x.to(device)
    def forward_pass(self, x: Union[np.ndarray, torch.Tensor], y: Optional[Union[np.ndarray, torch.Tensor]]=None, device: Union[str, torch.device]="cuda") -> torch.Tensor:
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
        return self.model(src=x_preprocessed, tgt=y_preprocessed)
    def train(
            self,
            train_loader,
            val_loader,
            device:Union[str, torch.device], 
            epochs:int=50, 
            patience:int=10, 
            delta:float=0.001, 
            lr: float=1e-5, 
            resume_from: Optional[str|os.PathLike] = None):
        pass

class TrainRVTransformer(TrainerBase):
    def __init__(self, 
                base_save_dir:str,
                model , 
                mode: str = "parallel",
                criterion: Callable = nn.MSELoss,
            
        ):
        assert mode == "parallel" or "autoregressive", "training mode must be either 'parallel' or 'autoregressive'"

        self.base_save_dir = base_save_dir
        self.model = model 
        self.mode = mode
        self.val_losses = []
        self.train_losses = []
        self.criterion_fn = criterion
        if not os.path.exists(self.base_save_dir):
            os.makedirs(self.base_save_dir)
    def compute_loss(self, output: torch.Tensor, target: torch.Tensor, device: Union[str, torch.device]) -> torch.Tensor:
        """Compute loss for complex-valued output."""
        # For complex output, we can use MSE on both real and imaginary parts
        if output.shape[-1] > 2:
            output = output[..., :-2]
        if target.shape[-1] > 2:
            target = target[..., :-2]

        loss = self.criterion_fn(output.to(device), target.to(device))
        return loss
    def preprocess_sample(self, x: torch.Tensor, device: Union[str, torch.device]):    
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        return x.float().to(device)
        
    def train(
            self, 
            train_loader, 
            val_loader, 
            device:Union[str, torch.device], 
            epochs:int=50, 
            patience:int=10, 
            delta:float=0.001, 
            lr: float=1e-5, 
            resume_from: Optional[str|os.PathLike] = None
        ):
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        if resume_from:
            if self.resume_from_checkpoint(str(resume_from), optimizer, None):
                print(f"Resuming training from epoch {self.start_epoch}")
            else:
                print("Failed to resume, starting from scratch")
                self.start_epoch = 0
        last_improve_epochs = 0
        min_val_loss = float('inf')

        # Training loop
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            train_bar = tqdm(train_loader, unit="batch", desc=f"Training epoch {epoch}/{epochs}", leave=True)
            for x, y in train_bar:

                optimizer.zero_grad()
                output = self.forward_pass(x, y, device)  
                loss = self.compute_loss(output, y, device)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                train_bar.set_postfix(train_loss=loss.item())
                
            avg_loss = epoch_loss / len(train_loader)
            self.train_losses.append(avg_loss)
            print(f"Training epoch {epoch+1}/{epochs} — avg_loss: {avg_loss:.5f}")
            
            val_bar = tqdm(val_loader, unit="batch", desc=f"Validation epoch {epoch}/{epochs}", leave=True)
            val_loss = 0
            for x, y in val_bar:
                if torch.is_complex(x):
                    x = x.to(device).to(torch.complex64)
                else:
                    x = x.to(device).float()

                if torch.is_complex(y):
                    y = y.to(device).to(torch.complex64)
                else:
                    y = y.to(device).float()
                #gt_inp, tgt_real = make_seq_batch(y)

                output = self.model(src=x, tgt=y)  # [B, T-1, 1]
                loss = self.compute_loss(output, y, device)
                val_loss += loss.item()
                val_bar.set_postfix(train_loss=loss.item())
            
            avg_val_loss = val_loss / len(val_loader)
            self.val_losses.append(avg_val_loss)
            print(f"Validation epoch {epoch+1}/{epochs} — avg_val_loss: {avg_val_loss:.5f}")
            if epoch % 2 == 0 :
                print(f"Saving metrics and inference results respectively to metrics_{epoch}.json and val_{epoch}.png...")
                self.show_example(val_loader, window=((1000, 1000), (5000, 5000)), vminmax=(4000, 4200), figsize=(20, 6), metrics_save_path=f"metrics_{epoch}.json", img_save_path=f"val_{epoch}.png")
            if val_loss < min_val_loss + delta:
                min_val_loss = val_loss
                torch.save(self.model.state_dict(), f"{self.base_save_dir}/sar_transformer_best.pth")
                last_improve_epochs = 0
            else: 
                last_improve_epochs += 1
                if last_improve_epochs >= patience:
                    print(f"Early stopping triggered: validation loss did not improve for the last {last_improve_epochs} epochs")
                    break

        torch.save(self.model.state_dict(), f"{self.base_save_dir}/sar_transformer_last.pth")
        metrics_path = os.path.join(self.base_save_dir, "train_metrics.json")
        with open(metrics_path, 'w') as f:
            f.write(str(self.val_losses))
        print("Training complete and model saved.")  
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
                self.best_val_loss = metrics.get('best_val_loss', float('inf'))
                
            print(f"✅ Resumed training from epoch {self.start_epoch}")
            print(f"📊 Best validation loss so far: {self.best_val_loss:.6f}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to resume from checkpoint: {str(e)}")
            return False
    def test(
        self,
        test_loader: torch.utils.data.DataLoader,
        device: Union[str, torch.device] = "cpu",
        num_visuals: int = 5,
        mainlobe_size: int = 5
    ):
        """
        Run the model in eval mode on test_loader, save a few example visualizations,
        and compute+save overall PSNR/PSLR metrics.
        """
        os.makedirs(self.base_save_dir, exist_ok=True)
        self.model.to(device)
        self.model.eval()

        all_metrics = []
        with torch.no_grad():
            for batch_idx, (raw, target) in enumerate(tqdm(test_loader, desc="Testing")):
                raw = raw.to(device).float()       # [B, T, 1] or [B, H, W]
                target = target.to(device).float()

                # for parallel mode: model(src=raw, tgt=target_inp)
                # here assume parallel and full-target pass
                output = self.model(src=raw, tgt=target)  
                
                raw_np = raw.cpu().numpy() #.squeeze()
                out_np = output.cpu().numpy() #.squeeze()
                tgt_np = target.cpu().numpy() #.squeeze()

                # Compute metrics for this sample
                m = compute_metrics(raw_np, out_np)
                all_metrics.append(m)

                # Save a few visual examples
                # if batch_idx < num_visuals:
                #     fig_path = os.path.join(output_dir, f"sample_{batch_idx}.png")
                #     visualize_pair(raw_np, out_np, fig_path)

            # Aggregate metrics (e.g., mean PSNR, PSLR)
            avg_psnr = sum(m['psnr_raw_vs_focused'] or 0 for m in all_metrics) / len(all_metrics)
            avg_pslr = sum(m['pslr_focused'] or 0 for m in all_metrics) / len(all_metrics)
            summary = {
                'avg_psnr_raw_vs_focused': avg_psnr,
                'avg_pslr_focused': avg_pslr,
                'num_samples': len(all_metrics),
                'val_losses': self.val_losses
            }
            # Save metrics JSON
            metrics_path = os.path.join(self.base_save_dir, "test_metrics.json")
            save_metrics(summary, metrics_path)

        print(f"Test visuals saved to: {self.base_save_dir}")
        print(f"Test metrics saved to: {metrics_path}")
            
    def train_loop(
            self, 
            train_loader, 
            val_loader, 
            device:Union[str,torch.device], 
            epochs:int=50, 
            patience:int=10, 
            delta:float=0.001, 
        ):
        
        self.model.to(device)
        self.train(
            train_loader=train_loader, 
            val_loader=val_loader, 
            device=device, 
            epochs=epochs, 
            patience=patience, 
            delta=delta, 
        )

        

class TrainCVTransformer(TrainRVTransformer):
    """
    Enhanced trainer for SAR Complex Transformer with optional patch preprocessing.
    Extends the existing TrainCVTransformer to handle patch preprocessing.
    """
    
    def __init__(self, base_save_dir: str, model, mode: str = "parallel",  criterion: Callable = nn.MSELoss):
        super().__init__(base_save_dir, model, mode, criterion=criterion)
        self.logger = logging.getLogger(__name__)
        
        # Count parameters and log model info
        if hasattr(model, 'parameters'):
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.logger.info(f"Model created with {total_params:,} total parameters")
            self.logger.info(f"Trainable parameters: {trainable_params:,}")
        
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
    
    def __init__(self, base_save_dir: str, model, mode: str = "parallel", criterion: Callable = nn.MSELoss):
        super().__init__(base_save_dir, model, mode, criterion=criterion)
        self.logger = logging.getLogger(__name__)
        
        # Count parameters and log model info
        if hasattr(model, 'parameters'):
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.logger.info(f"Model created with {total_params:,} total parameters")
            self.logger.info(f"Trainable parameters: {trainable_params:,}")
        
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


class TrainDUN(TrainerBase):
    pass

def get_training_loop_by_model_name(model_name: str, model: nn.Module, save_dir: Union[str, os.PathLike] = './results', loss_fn_name: str = "mse", mode: str = 'parallel') -> TrainerBase:
    if model_name == 'cv_transformer':
        # Use enhanced trainer for complex transformers
        trainer = TrainCVTransformer(
            base_save_dir=str(save_dir),
            model=model,
            mode=mode,
            criterion = get_loss_function("complex_mse")
        )
        
        #logger.info("Created Complex transformer")
    elif 'transformer' in model_name.lower():
        trainer = TrainRVTransformer(
            base_save_dir=str(save_dir),
            model=model,
            criterion = get_loss_function("mse"),
            mode=mode
        )
        #logger.info("Created TrainRVTransformer")
    # elif 'ssm' in model_name.lower():
    #     trainer = TrainSSM(
    #         base_save_dir=str(save_dir),
    #         model=model,
    #         criterion = get_loss_function("mse"),
    #         mode=mode
    #     )
        #logger.info("Created TrainSSM")
    else:
        raise ValueError(f"Unsupported model type: {model_name}")
    return trainer