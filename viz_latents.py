# vis_latents.py – visualise Stable‑Diffusion latents
"""Mini‑pipeline
  • read data/latents.csv (produced by obtain_latents.py)
  • load each .safetensors with tinygrad.nn.state.safe_load
  • flatten to 16 384‑D, z‑score, PCA→50
  • 2‑D projections: PCA‑2, UMAP‑2 (if installed), t‑SNE‑2 (if sklearn ≥1.2)
  • k‑means on 2‑D projection for colours
  • store PNGs to ./plots and one JSON‑line per projection with metrics

NaNs in the latent vectors (rare—but can occur when SD spits
out Inf/NaN during decoding) are *filtered out* automatically so
scikit‑learn doesn’t choke.
"""

from __future__ import annotations

import os, random, json, argparse, pathlib
from typing import Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# trustworthiness changed module path in sklearn >=1.4
try:
    from sklearn.metrics import trustworthiness  # type: ignore
except ImportError:
    from sklearn.manifold import trustworthiness  # type: ignore

# optional deps – script still runs without them
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

def load_vectors(paths: Sequence[str]) -> np.ndarray:
    """Load each safetensor file → flattened float32 vector."""
    vecs: list[np.ndarray] = []
    for p in tqdm(paths, desc="latents"):
        t = safe_load(p)["data"]  # tinygrad Tensor (1,4,64,64)
        vecs.append(t.numpy().astype("float32").squeeze(0).reshape(-1))
    return np.vstack(vecs)


def scatter(ax, xy: np.ndarray, labels: np.ndarray, title: str):
    ax.scatter(xy[:, 0], xy[:, 1], c=labels, s=4, alpha=0.8, cmap="tab10")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", "box")


def save_projection(proj: np.ndarray, name: str, out_dir: pathlib.Path, X50: np.ndarray):
    km = KMeans(10, n_init="auto", random_state=0).fit(proj)
    sil = float(silhouette_score(proj, km.labels_))     # py‑floats for JSON
    trust = float(trustworthiness(X50, proj, n_neighbors=5))

    out_dir.mkdir(exist_ok=True, parents=True)
    with (out_dir / "scores.jsonl").open("a") as fp:
        fp.write(json.dumps({"projection": name, "silhouette": sil, "trustworthiness": trust}) + "\n")

    fig, ax = plt.subplots(figsize=(5, 4))
    scatter(ax, proj, km.labels_, f"{name}\nSil={sil:.3f}  Trust={trust:.3f}")
    fig.tight_layout()
    fig.savefig(out_dir / f"{name.lower().replace(' ', '_')}.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualise latents")
    parser.add_argument("--csv", default="data/latents.csv")
    parser.add_argument("--max", type=int, default=5000, help="max samples")
    parser.add_argument("--out", default="plots")
    args = parser.parse_args()

    out_path = pathlib.Path(args.out)
    # clear previous metrics for clean reruns
    if (out_path / "scores.jsonl").exists():
        (out_path / "scores.jsonl").unlink()

    df = pd.read_csv(args.csv)
    if args.max and len(df) > args.max:
        df = df.sample(args.max, random_state=0).reset_index(drop=True)

    X = load_vectors(df.latent_path)

    # -------- NaN filtering (before z‑score!) ----------------------------
    nan_rows = np.isnan(X).any(axis=1)
    if nan_rows.any():
        n_bad = int(nan_rows.sum())
        print(f"[WARN] {n_bad} / {len(X)} samples contain NaNs → dropping them")
        X = X[~nan_rows]
        df = df.loc[~nan_rows].reset_index(drop=True)

    # z‑score per feature (add eps to avoid /0 but we already cleaned NaNs)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)

    # PCA ---------------------------------------------------------------
    pca = PCA(n_components=50, random_state=0).fit(X)
    X50 = pca.transform(X)
    evr = pca.explained_variance_ratio_.cumsum()[1]
    print(f"PCA‑2 explains {evr*100:.1f}% variance (after NaN filter)")

    # projections -------------------------------------------------------
    save_projection(X50[:, :2], "PCA 2D", out_path, X50)

    if umap is not None:
        u2 = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=0).fit_transform(X50)
        save_projection(u2, "UMAP 2D", out_path, X50)
    else:
        print("[INFO] umap‑learn not installed – skipping UMAP plot")

    if TSNE is not None:
        perp = 30 if len(X50) <= 5000 else 50
        ts2 = TSNE(n_components=2, perplexity=perp, init="pca", random_state=0).fit_transform(X50)
        save_projection(ts2, "t‑SNE 2D", out_path, X50)
    else:
        print("[INFO] scikit‑learn too old – skipping t‑SNE plot")

    print("Done – projections + metrics saved to", out_path)
