import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
from pathlib import Path
from cv_transformer import ComplexTransformer
from dataloader.dataloader import get_sar_dataloader
from typing import Optional

def train(
    data_dir: str,
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 1e-4,
    dim: int = 64,
    depth: int = 6,
    heads: int = 8,
    dim_head: int = 32,
    ff_mult: int = 4,
    device: str = 'cuda'
):
    # Dataset & DataLoader
    loader = get_sar_dataloader(
        data_dir=data_dir,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

    # Model
    model = ComplexTransformer(
        dim=dim,
        depth=depth,
        num_tokens=None,
        causal=False,
        dim_head=dim_head,
        heads=heads,
        ff_mult=ff_mult,
        relu_squared=True,
        complete_complex=False,
        rotary_emb=True,
        flash_attn=True
    ).to(device)

    # Loss & Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for batch_idx, (rcmc, az_gt) in enumerate(loader):
            rcmc = rcmc.to(device)  # shape: (B,1,L)
            az_gt = az_gt.to(device)

            # forward
            pred = model(rcmc)  # expects input shape (B, seq_len, dim) but here seq_len=L, dim=1
            # If needed, project input to model dim via a linear layer
            # Compute loss
            loss = criterion(pred.squeeze(-1), az_gt.squeeze(1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f'Epoch {epoch:02d}/{epochs} — Loss: {avg_loss:.6f}')

    # Save checkpoint
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/complex_transformer_azimuth.pt')
    print('Training complete. Model saved.')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train Complex Transformer for SAR Azimuth Compression')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--mode', choices=['parallel','autoregressive'], default='parallel',
                        help='Choose training mode: parallel (many-to-many) or autoregressive')
    args = parser.parse_args()
    if args.mode == 'parallel':
        train(
            data_dir=args.data_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr
        )
    else:
        train_ar(
            data_dir=args.data_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr
        )

# Autoregressive training
def train_ar(
    data_dir: str,
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 1e-4,
    dim: int = 64,
    depth: int = 6,
    heads: int = 8,
    dim_head: int = 32,
    ff_mult: int = 4,
    device: str = 'cuda'
):
    dataset = SARAzimuthDataset(data_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Autoregressive (causal) ComplexTransformer
    model = ComplexTransformer(
        dim=dim,
        depth=depth,
        num_tokens=None,
        causal=True,
        dim_head=dim_head,
        heads=heads,
        ff_mult=ff_mult,
        relu_squared=True,
        complete_complex=False,
        rotary_emb=True,
        flash_attn=True
    ).to(device)

