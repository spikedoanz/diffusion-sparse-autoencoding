# vis_latents.py – visualise Stable‑Diffusion latents
"""Mini‑pipeline
  • read data/latents.csv (produced by obtain_latents.py)
  • load each .safetensors with tinygrad.nn.state.safe_load
  • flatten to 16 384‑D, z‑score, PCA→50
  • 2‑D projections: PCA‑2, UMAP‑2 (if installed), t‑SNE‑2 (if sklearn ≥1.2)

Colour **reflects the iteration count** embedded in every filename
(`s{iter}_…`) so you can visually compare latents generated at
different diffusion steps.

All trustworthiness / silhouette metrics have been removed – these
plots are now purely for qualitative inspection.
"""

from __future__ import annotations

import os, re, argparse, pathlib, json
from typing import Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# optional deps --------------------------------------------------------------
try:
    import umap  # type: ignore
except ModuleNotFoundError:
    umap = None  # type: ignore
try:
    from sklearn.manifold import TSNE  # type: ignore
except ModuleNotFoundError:
    TSNE = None  # type: ignore

from tinygrad.nn.state import safe_load  # tinygrad‑native safetensor loader

# ---------------------------------------------------------------------------

def extract_iter(path: str) -> int:
    """Pull the integer immediately after the leading 's' in a file name.
    Example:  's8_g7.5_abcdef.safetensors'  ->  8
    """
    m = re.match(r"s(\d+)_", os.path.basename(path))
    if not m:
        raise ValueError(f"Cannot parse iteration from file name: {path}")
    return int(m.group(1))


def load_vectors(paths: Sequence[str]) -> np.ndarray:
    """Load each safetensor file → flattened float32 vector."""
    vecs: list[np.ndarray] = []
    for p in tqdm(paths, desc="latents"):
        t = safe_load(p)["data"]  # tinygrad Tensor (1,4,64,64)
        vecs.append(t.numpy().astype("float32").squeeze(0).reshape(-1))
    return np.vstack(vecs)

# colour helper --------------------------------------------------------------
import matplotlib as mpl
COLOR_MAP = mpl.cm.get_cmap("tab10")  # cycles every 10

def label_to_color(labels: np.ndarray) -> np.ndarray:
    """Map integer labels → RGBA colours (cycling every 10)."""
    return COLOR_MAP(labels % 10)


def scatter(ax, xy: np.ndarray, labels: np.ndarray, title: str):
    ax.scatter(xy[:, 0], xy[:, 1], c=label_to_color(labels), s=4, alpha=0.8)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", "box")


def save_projection(proj: np.ndarray, name: str, out_dir: pathlib.Path, labels: np.ndarray):
    out_dir.mkdir(exist_ok=True, parents=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    scatter(ax, proj, labels, name)
    fig.tight_layout()
    fig.savefig(out_dir / f"{name.lower().replace(' ', '_')}.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualise latents (colour by iter)")
    parser.add_argument("--csv", default="data/latents.csv")
    parser.add_argument("--max", type=int, default=5000, help="max samples to plot")
    parser.add_argument("--out", default="plots")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if args.max and len(df) > args.max:
        df = df.sample(args.max, random_state=0).reset_index(drop=True)

    # extract iteration labels
    df["iter"] = df["latent_path"].apply(extract_iter)

    X = load_vectors(df.latent_path)

    # --- NaN filtering ------------------------------------------------------
    nan_rows = np.isnan(X).any(axis=1)
    if nan_rows.any():
        n_bad = int(nan_rows.sum())
        print(f"[WARN] {n_bad} / {len(X)} samples contain NaNs – dropping")
        X = X[~nan_rows]
        df = df.loc[~nan_rows].reset_index(drop=True)

    # z‑score ---------------------------------------------------------------
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)

    # PCA -------------------------------------------------------------------
    X50 = PCA(n_components=50, random_state=0).fit_transform(X)

    out_path = pathlib.Path(args.out)

    labels = df.iter.to_numpy()
    save_projection(X50[:, :2], "PCA 2D", out_path, labels)

    if umap is not None:
        u2 = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=0).fit_transform(X50)
        save_projection(u2, "UMAP 2D", out_path, labels)
    else:
        print("[INFO] umap‑learn not installed – skipping UMAP plot")

    if TSNE is not None:
        perp = 30 if len(X50) <= 5000 else 50
        ts2 = TSNE(n_components=2, perplexity=perp, init="pca", random_state=0).fit_transform(X50)
        save_projection(ts2, "t‑SNE 2D", out_path, labels)
    else:
        print("[INFO] scikit‑learn too old – skipping t‑SNE plot")

    print("Done – plots saved to", out_path)
