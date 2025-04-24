#!/usr/bin/env python3
"""
train_sae.py  –  Tinygrad script that trains a small convolutional
auto‑encoder on Stable‑Diffusion latents (shape 1×4×64×64).

Architecture (all ASCII):

    input  (B, 4, 64, 64)
      |  flatten     -> (B, 16384)
      |  linear      -> (B, 16384)  # same size, adds capacity
      |  reshape     -> (B, 64, 16, 16)
      |  upsample2x  -> (B, 64, 32, 32)
      |  conv 3x3 64->64 + ReLU
      |  upsample2x  -> (B, 64, 64, 64)
      |  conv 3x3 64->32 + ReLU
      |  conv 3x3 32->4          
      v  output      -> (B, 4, 64, 64)

Loss: mean absolute error (L1) between output and input.

Changes in this version:
  * **NaN‑safe data loader** – drops any latent that contains NaNs (these occur
    occasionally when SD produces Inf / NaN during sampling).
  * **Gradient clipping** at 1.0‑norm to prevent blow‑ups that also cause NaNs.
  * Uses `dtypes.float32` explicitly to avoid dtype inference issues.

Example run:

    python train_sae.py \
        --csv data/latents.csv \
        --steps 10000 --bs 256 --lr 3e-4 \
        --target_loss 0.15
"""

from __future__ import annotations

import argparse
import os
from typing import List, Callable

import numpy as np
import pandas as pd
from tqdm import trange

from tinygrad import Tensor, TinyJit, nn, GlobalCounters, dtypes
from tinygrad.nn.state import safe_load, get_parameters
from tinygrad.helpers import colored

# ----------------------------- dataset loader -----------------------------

def load_latents(csv_path: str) -> Tensor:
    """Load all latents from csv → single float32 Tensor.
    Drops any sample that contains NaNs."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    if "latent_path" not in df.columns:
        raise ValueError("CSV must contain a 'latent_path' column")

    arrays: list[np.ndarray] = []
    dropped = 0
    for p in df["latent_path"]:
        t = safe_load(p)["data"].cast(dtypes.float32).numpy()  # (1,4,64,64)
        if np.isnan(t).any():
            dropped += 1
            continue
        arrays.append(t)
    if not arrays:
        raise RuntimeError("No clean latents found – all contained NaNs")

    if dropped:
        print(f"[INFO] dropped {dropped} latent(s) with NaNs → using {len(arrays)}")

    data = np.concatenate(arrays, axis=0)  # (N,4,64,64)
    return Tensor(data)

# ------------------------------ upsample op -------------------------------

def upsample2x(x: Tensor) -> Tensor:
    """Nearest‑neighbor 2× upsample implemented with reshape/expand."""
    b, c, h, w = x.shape
    x = x.reshape(b, c, h, 1, w, 1)
    x = x.expand(b, c, h, 2, w, 2)
    return x.reshape(b, c, h * 2, w * 2)

# ------------------------------ model class -------------------------------

class Model:
    def __init__(self):
        self.flatten = lambda z: z.reshape(z.shape[0], -1)
        self.fc      = nn.Linear(16384, 16384)
        self.conv1   = nn.Conv2d(64, 64, 3, padding=1)
        self.conv2   = nn.Conv2d(64, 32, 3, padding=1)
        self.conv3   = nn.Conv2d(32, 4,  3, padding=1)

    def __call__(self, x: Tensor) -> Tensor:
        z = self.flatten(x)
        z = self.fc(z).relu()
        z = z.reshape(x.shape[0], 64, 16, 16)
        z = upsample2x(z)
        z = self.conv1(z).relu()
        z = upsample2x(z)
        z = self.conv2(z).relu()
        out = self.conv3(z)
        return out

# ------------------------------ training loop -----------------------------

def main():
    parser = argparse.ArgumentParser("train_sae")
    parser.add_argument("--csv", default="data/latents.csv")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--bs", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--target_loss", type=float, default=0.0,
                        help="abort if final val loss > this")
    args = parser.parse_args()

    latents = load_latents(args.csv)  # (N,4,64,64)
    n = latents.shape[0]

    model = Model()
    opt = nn.optim.Adam(get_parameters(model), lr=args.lr)

    def _clip_grads(max_norm: float = 1.0):
        total_norm = 0.0
        for p in get_parameters(model):
            if p.grad is None:  # should not happen but be safe
                continue
            param_norm = float((p.grad * p.grad).sum().sqrt().item())
            total_norm += param_norm * param_norm
        total_norm = total_norm ** 0.5
        if total_norm > max_norm:
            scale = max_norm / (total_norm + 1e-6)
            for p in get_parameters(model):
                if p.grad is not None:
                    p.grad *= scale

    @TinyJit
    @Tensor.train()
    def train_step() -> Tensor:
        opt.zero_grad()
        idx = Tensor.randint(args.bs, high=n)
        batch = latents[idx]
        recon = model(batch)
        loss = (recon - batch).abs().mean()
        loss.backward()
        _clip_grads(1.0)
        opt.step()
        return loss

    @TinyJit
    @Tensor.test()
    def val_step() -> Tensor:
        idx = Tensor.randint(args.bs, high=n)
        batch = latents[idx]
        recon = model(batch)
        return (recon - batch).abs().mean()

    val_loss = float('nan')
    for i in (bar := trange(args.steps)):
        GlobalCounters.reset()
        loss = train_step()
        if (i + 1) % 500 == 0:
            val_loss = val_step().item()
        bar.set_description(f"L_train {loss.item():.4f}  L_val {val_loss:.4f}")

    if args.target_loss and (val_loss > args.target_loss):
        raise ValueError(colored(f"val_loss {val_loss:.4f} > target {args.target_loss}", "red"))

if __name__ == "__main__":
    main()
