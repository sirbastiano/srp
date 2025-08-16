import json
import logging
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from typing import Union
from tqdm import tqdm
import os
from typing import Optional, Callable, Union
from dataloader.dataloader import SARZarrDataset, get_sar_dataloader
from model.transformers.rv_transformer import RealValuedTransformer  
from model.transformers.cv_transformer import ComplexTransformer
from training.visualize import compute_metrics, visualize_pair, save_metrics
                
class TrainRVTransformer():
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
        self.criterion_fn = criterion
        if not os.path.exists(self.base_save_dir):
            os.makedirs(self.base_save_dir)
    def compute_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute loss for complex-valued output."""
        # For complex output, we can use MSE on both real and imaginary parts

        loss = self.criterion_fn(output[..., :-2], target[..., :-2])
        return loss
    def train(
            self, 
            train_loader, 
            val_loader, 
            device:Union[str, torch.device], 
            epochs:int=50, 
            patience:int=10, 
            delta:float=0.001, 
            lr: float=1e-5
        ):
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        last_improve_epochs = 0
        min_val_loss = float('inf')

        # Training loop
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            train_bar = tqdm(train_loader, unit="batch", desc=f"Training epoch {epoch}/{epochs}", leave=True)
            for x, y in train_bar:
                #print(f"x={x}, y={y}")
                #print(f"Input shape: {x.shape}, Target shape: {y.shape}")
                if torch.is_complex(x):
                    x = x.to(device).to(torch.complex64)
                else:
                    x = x.to(device).float()

                if torch.is_complex(y):
                    y = y.to(device).to(torch.complex64)
                else:
                    y = y.to(device).float()
                #tgt_inp, tgt_real = make_seq_batch(y)

                optimizer.zero_grad()
                output = self.model(src=x, tgt=y)  
                loss = self.compute_loss(output, y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                train_bar.set_postfix(train_loss=loss.item())
                
            avg_loss = epoch_loss / len(train_loader)
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
                loss = self.compute_loss(output, y)
                val_loss += loss.item()
                val_bar.set_postfix(train_loss=loss.item())
            
            avg_val_loss = val_loss / len(val_loader)
            self.val_losses.append(avg_val_loss)
            print(f"Validation epoch {epoch+1}/{epochs} — avg_val_loss: {avg_val_loss:.5f}")
            
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
                
                # Move to CPU numpy arrays (squeeze channel dim if needed)
                raw_np = raw.cpu().numpy().squeeze()
                out_np = output.cpu().numpy().squeeze()
                tgt_np = target.cpu().numpy().squeeze()

                # Compute metrics for this sample
                m = compute_metrics(raw_np, out_np, mainlobe_size=mainlobe_size)
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
            device:Union[str|torch.device], 
            epochs:int=50, 
            patience:int=10, 
            delta:int=0.001, 
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
    
    def __init__(self, base_save_dir: str, model, mode: str = "parallel", 
                 patch_processor=None, use_preprocessing=False, criterion: Callable = nn.MSELoss,
):
        super().__init__(base_save_dir, model, mode, criterion=criterion)
        self.patch_processor = patch_processor
        self.use_preprocessing = use_preprocessing
        self.logger = logging.getLogger(__name__)
        
        # Count parameters and log model info
        if hasattr(model, 'parameters'):
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.logger.info(f"Model created with {total_params:,} total parameters")
            self.logger.info(f"Trainable parameters: {trainable_params:,}")
        
        if self.use_preprocessing:
            self.logger.info("Using patch preprocessing pipeline")
        else:
            self.logger.info("Using standard complex transformer")

    def compute_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
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
        loss = self.criterion_fn(output, target)
        return loss

class TrainSSM:
    """
    Training class for State Space Models (SSM) for SAR focusing.
    Handles column-wise processing where each column is treated as a sequence.
    """
    
    def __init__(
        self, 
        base_save_dir: str,
        model: nn.Module,
        mode: str = "column_wise"
    ):
        """
        Args:
            base_save_dir: Directory to save models and metrics
            model: SSM model instance
            mode: Training mode (currently supports "column_wise")
        """
        self.base_save_dir = base_save_dir
        self.model = model
        self.mode = mode
        self.val_losses = []
        
        if not os.path.exists(self.base_save_dir):
            os.makedirs(self.base_save_dir)
    
    def train(
        self,
        train_loader,
        val_loader,
        device: Union[str, torch.device],
        epochs: int = 50,
        patience: int = 10,
        delta: float = 0.001,
        lr: float = 1e-4,
        weight_decay: float = 1e-5
    ):
        """
        Train the SSM model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader  
            device: Training device
            epochs: Number of training epochs
            patience: Early stopping patience
            delta: Minimum improvement threshold
            criterion: Loss function
            lr: Learning rate
            weight_decay: Weight decay for optimizer
        """
        
        self.model.to(device)
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience//2, factor=0.5)
        
        last_improve_epochs = 0
        min_val_loss = float('inf')

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            epoch_loss = 0.0
            train_bar = tqdm(train_loader, unit="batch", desc=f"Training epoch {epoch+1}/{epochs}", leave=True)
            
            for batch_idx, (x, y) in enumerate(train_bar):
                x = x.to(device)  # [B, H, W, C] where H=column_height, W=num_columns
                y = y.to(device)
                
                # Handle different input formats
                x_processed, y_processed = self._preprocess_batch(x, y)
                
                optimizer.zero_grad()
                output = self.model(x_processed)
                
                # Compute loss only on data channels (exclude positional encoding)
                loss = self._compute_loss(output, y_processed)
                
                loss.backward()
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                train_bar.set_postfix(train_loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
                
            avg_loss = epoch_loss / len(train_loader)
            print(f"Training epoch {epoch+1}/{epochs} — avg_loss: {avg_loss:.6f}")
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_bar = tqdm(val_loader, unit="batch", desc=f"Validation epoch {epoch+1}/{epochs}", leave=True)
            
            with torch.no_grad():
                for x, y in val_bar:
                    x = x.to(device)
                    y = y.to(device)
                    
                    x_processed, y_processed = self._preprocess_batch(x, y)
                    output = self.model(x_processed)
                    
                    loss = self._compute_loss(output, y_processed)
                    val_loss += loss.item()
                    val_bar.set_postfix(val_loss=loss.item())
            
            avg_val_loss = val_loss / len(val_loader)
            self.val_losses.append(avg_val_loss)
            scheduler.step(avg_val_loss)
            
            print(f"Validation epoch {epoch+1}/{epochs} — avg_val_loss: {avg_val_loss:.6f}")
            
            # Early stopping check
            if avg_val_loss < min_val_loss - delta:
                min_val_loss = avg_val_loss
                torch.save(self.model.state_dict(), f"{self.base_save_dir}/ssm_model_best.pth")
                last_improve_epochs = 0
                print(f"New best model saved with validation loss: {min_val_loss:.6f}")
            else:
                last_improve_epochs += 1
                if last_improve_epochs >= patience:
                    print(f"Early stopping triggered: validation loss did not improve for {last_improve_epochs} epochs")
                    break

        # Save final model and training history
        torch.save(self.model.state_dict(), f"{self.base_save_dir}/ssm_model_last.pth")
        
        training_history = {
            'val_losses': self.val_losses,
            'final_val_loss': self.val_losses[-1] if self.val_losses else None,
            'best_val_loss': min_val_loss,
            'epochs_trained': len(self.val_losses)
        }
        
        with open(os.path.join(self.base_save_dir, "training_history.json"), 'w') as f:
            json.dump(training_history, f, indent=2)
            
        print("Training complete and model saved.")
    
    def _preprocess_batch(self, x: torch.Tensor, y: torch.Tensor):
        """
        Preprocess input batch for SSM training.
        
        Args:
            x: Input tensor - can be various formats from dataloader
            y: Target tensor
            
        Returns:
            Tuple of processed tensors in format [B, H, W, C]
        """
        # Handle different input shapes from dataloader
        if len(x.shape) == 4:  # [B, H, W, C]
            x_processed = x
            y_processed = y
        elif len(x.shape) == 3:  # [B, H, C] - single column
            x_processed = x.unsqueeze(2)  # [B, H, 1, C]  
            y_processed = y.unsqueeze(2)  # [B, H, 1, C]
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")
        
        # Ensure float32 type
        x_processed = x_processed.float()
        y_processed = y_processed.float()
        
        return x_processed, y_processed
    
    def _compute_loss(self, output: torch.Tensor, target: torch.Tensor):
        """
        Compute loss excluding positional encoding channels.
        
        Args:
            output: Model output [B, H, W, output_dim]
            target: Target tensor [B, H, W, C] 
                        
        Returns:
            Computed loss
        """
        # Extract only the data channels from target (exclude positional encoding)
        if target.shape[-1] > output.shape[-1]:
            # Target has positional encoding, extract only data channels
            target_data = target[..., :output.shape[-1]]
        else:
            target_data = target
            
        return self.criterion_fn(output, target_data)
    
    def test(
        self,
        test_loader,
        device: Union[str, torch.device] = "cpu", 
        mainlobe_size: int = 5
    ):
        """
        Test the trained SSM model and compute metrics.
        
        Args:
            test_loader: Test data loader
            device: Device for testing
            mainlobe_size: Size parameter for PSLR computation
        """
        self.model.to(device)
        self.model.eval()
        
        all_metrics = []
        
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(tqdm(test_loader, desc="Testing")):
                x = x.to(device)
                y = y.to(device)
                
                x_processed, y_processed = self._preprocess_batch(x, y)
                output = self.model(x_processed)
                
                # Convert to numpy for metrics computation
                x_np = x_processed.cpu().numpy()
                output_np = output.cpu().numpy()
                
                # Compute metrics for each sample in batch
                for i in range(x_np.shape[0]):
                    # Extract real and imaginary parts and combine into complex
                    if x_np.shape[-1] >= 2:
                        raw_complex = x_np[i, :, :, 0] + 1j * x_np[i, :, :, 1]
                    else:
                        raw_complex = x_np[i, :, :, 0]
                        
                    if output_np.shape[-1] >= 2:
                        focused_complex = output_np[i, :, :, 0] + 1j * output_np[i, :, :, 1]
                    else:
                        focused_complex = output_np[i, :, :, 0]
                    
                    # Compute metrics on magnitude
                    m = compute_metrics(
                        np.abs(raw_complex), 
                        np.abs(focused_complex), 
                        mainlobe_size=mainlobe_size
                    )
                    all_metrics.append(m)

        # Aggregate metrics
        valid_psnr = [m['psnr_raw_vs_focused'] for m in all_metrics if m['psnr_raw_vs_focused'] is not None]
        valid_pslr = [m['pslr_focused'] for m in all_metrics if m['pslr_focused'] is not None]
        
        summary = {
            'avg_psnr_raw_vs_focused': float(np.mean(valid_psnr)) if valid_psnr else None,
            'avg_pslr_focused': float(np.mean(valid_pslr)) if valid_pslr else None,
            'num_samples': len(all_metrics),
            'val_losses': self.val_losses
        }
        
        # Save test metrics
        metrics_path = os.path.join(self.base_save_dir, "test_metrics.json")
        save_metrics(summary, metrics_path)
        
        print(f"Test metrics saved to: {metrics_path}")
        print(f"Average PSNR: {summary['avg_psnr_raw_vs_focused']:.4f}")
        print(f"Average PSLR: {summary['avg_pslr_focused']:.4f}")
        
        return summary
    
    def train_loop(
        self,
        train_loader,
        val_loader, 
        device: Union[str, torch.device],
        epochs: int = 50,
        patience: int = 10,
        delta: float = 0.001,
        lr: float = 1e-4
    ):
        """
        Convenience method that wraps the training process.
        """
        self.model.to(device)
        self.train(
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=epochs,
            patience=patience,
            delta=delta,
            lr=lr
        )

class TrainDUN():
    pass