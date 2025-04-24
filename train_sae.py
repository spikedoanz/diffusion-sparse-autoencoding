"""
train_sae.py  –  Tinygrad script that learns a very small convolutional
auto‑encoder on Stable‑Diffusion latents (shape 1×4×64×64).

Rough architecture requested by the user (ASCII only):

    input : (B, 4, 64, 64)
       |   flatten -> (B, 16384)
       |   linear  -> (B, 16384)                # keeps same size, adds capacity
       |   reshape -> (B, 64, 16, 16)           # 64 feature maps, 16×16
       |   upsample2x (nearest) -> 32×32        # no new params
       |   conv 3×3, 64 -> 64, ReLU            
       |   upsample2x (nearest) -> 64×64        
       |   conv 3×3, 64 -> 32, ReLU            
       |   conv 3×3, 32 -> 4                   
       v   output : (B, 4, 64, 64)

Loss: mean absolute error (L1) between output and input.

This code mirrors the style of tinygrad/examples/mnist.py: one Model class,
JIT‑compiled train_step / val_step, progress bar with trange, fail‑fast gate.

Usage example:

    python train_sae.py --csv data/latents.csv --steps 10000 --bs 256 --lr 3e-4
"""

from __future__ import annotations

import argparse
import os
from typing import List, Callable

import numpy as np
from tqdm import trange

from tinygrad import Tensor, TinyJit, nn, GlobalCounters, dtypes
from tinygrad.nn.state import safe_load, get_parameters
from tinygrad.helpers import colored
import pandas as pd

# ----------------------------- dataset loader -----------------------------

def load_latents(csv_path: str) -> Tensor:
    """Load every latent from csv into one big float32 Tensor on host memory."""
    df = pd.read_csv(csv_path)
    arr: list[np.ndarray] = []
    for p in df["latent_path"]:
        t = safe_load(p)["data"].cast(dtypes.float).numpy()
        arr.append(t)
    return Tensor(np.concatenate(arr, axis=0))  # (N, 4, 64, 64)

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
        # convs
        self.conv1   = nn.Conv2d(64, 64, 3, padding=1)
        self.conv2   = nn.Conv2d(64, 32, 3, padding=1)
        self.conv3   = nn.Conv2d(32, 4,  3, padding=1)
        self.layers: List[Callable[[Tensor], Tensor]] = []  # unused (sequential written by hand)

    def __call__(self, x: Tensor) -> Tensor:
        # flatten -> linear -> reshape to 64x16x16
        z = self.flatten(x)
        z = self.fc(z).relu()
        z = z.reshape(x.shape[0], 64, 16, 16)
        # upsample conv -> relu
        z = upsample2x(z)
        z = self.conv1(z).relu()
        # second stage
        z = upsample2x(z)
        z = self.conv2(z).relu()
        # output projection (no activation)
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

    @TinyJit
    @Tensor.train()
    def train_step() -> Tensor:
        opt.zero_grad()
        idx = Tensor.randint(args.bs, high=n)
        batch = latents[idx]
        recon = model(batch)
        loss = (recon - batch).abs().mean()
        loss.backward()
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
    bar = trange(args.steps)
    for i in bar:
        GlobalCounters.reset()
        loss = train_step()
        if (i + 1) % 500 == 0:
            val_loss = val_step().item()
        bar.set_description(f"L_train {loss.item():.4f}  L_val {val_loss:.4f}")

    if args.target_loss and (val_loss > args.target_loss):
        raise ValueError(colored(f"val_loss {val_loss:.4f} > target {args.target_loss}", "red"))

if __name__ == "__main__":
    main()
