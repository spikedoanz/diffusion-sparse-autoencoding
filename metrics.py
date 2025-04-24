from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Optional

import math
import random

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm.auto import tqdm

# -----------------------------------------------------------------------------
# 1.  Monosemanticity score
# -----------------------------------------------------------------------------

def monosemanticity_score(
    sae: nn.Module,
    data_loader: DataLoader,
    device: torch.device | str = "cuda",
    max_batches: int | None = 250,
    eps: float = 1e-8,
) -> float:
    """Compute the global monosemanticity score for *sae*.

    Args
    ----
    sae          –   nn.Module with attributes `.encoder` and `.decoder`.
                     The encoder is expected to be a single Linear layer or
                     behave like one (output shape = (B, k)).
    data_loader  –   Yields latent tensors with shape (B, C, H, W) or flattened
                     (B, d).  They are auto‑flattened inside.
    max_batches  –   Caps the number of minibatches used for activations to
                     keep memory/time bounded.  ~250 × BS=512 is ~128k samples.
    eps          –   Small constant for safe normalisation.

    Returns
    -------
    float ∈ [0, 1] – higher implies more monosemantic features.
    """
    sae.eval()
    k = sae.encoder.out_features

    # First pass: gather √N‑normalised activation vectors per feature.
    # We accumulate mean and cov in float64 to keep numerical error small.
    sum_acts = torch.zeros(k, dtype=torch.float64, device=device)
    sq_sum_acts = torch.zeros_like(sum_acts)
    n_samples = 0

    with torch.no_grad():
        for b_idx, x in enumerate(tqdm(data_loader, desc="[mono] batches")):
            if max_batches is not None and b_idx >= max_batches:
                break
            x = x.to(device, non_blocking=True)
            if x.ndim == 4:  # (B, C, H, W) → flatten
                x = x.view(x.size(0), -1)
            z = sae.encoder(x)  # (B, k)
            z = F.relu(z)  # assume ReLU SAE; safe for GLU because b ≤ z ≤ a ⊙ σ(b)

            sum_acts.add_(z.sum(dim=0, dtype=torch.float64))
            sq_sum_acts.add_((z ** 2).sum(dim=0, dtype=torch.float64))
            n_samples += z.size(0)

    # μ_i and σ_i for each feature i
    mu = sum_acts / max(n_samples, 1)
    var = sq_sum_acts / max(n_samples, 1) - mu ** 2
    std = (var + eps).sqrt()

    # Second pass: dot〈(z_i – μ_i)/σ_i, dec_col_i〉 / (‖dec_col_i‖·√N)
    # Aggregate per‑feature cosine, then average.
    score_acc = torch.zeros(k, dtype=torch.float64, device=device)
    dec_cols = sae.decoder.weight.data.t()  # (k, d)
    dec_norm = dec_cols.norm(dim=1).clamp(min=eps)  # (k,)

    with torch.no_grad():
        for b_idx, x in enumerate(tqdm(data_loader, desc="[mono] pass‑2")):
            if max_batches is not None and b_idx >= max_batches:
                break
            x = x.to(device, non_blocking=True)
            if x.ndim == 4:
                x = x.view(x.size(0), -1)
            z = sae.encoder(x)
            z = F.relu(z)
            z = (z - mu) / std  # (B, k) whitened activations
            # contribution of batch to per‑feature dot products
            score_acc += (z.t() @ dec_cols).sum(dim=1, dtype=torch.float64)

    # final cosine for each feature
    feat_cos = score_acc / (dec_norm * n_samples + eps)
    mono_score = feat_cos.abs().mean().item()
    return mono_score


# -----------------------------------------------------------------------------
# 2.  Directional causal test ("feature -> concept" alignment)
# -----------------------------------------------------------------------------

class DiffusionPipelineWrapper:
    """Minimal adapter around your Stable‑Diffusion inference util.

    Only two methods are required:
      • encode_latents(prompt, device) -> latent Tensor float32 (1, C=4, 64, 64)
      • decode_latents(latent) -> PIL.Image  (or uint8 Tensor)

    Replace with calls into your existing Tinygrad pipeline.
    """

    def __init__(self, encode_fn, decode_fn):
        self.encode = encode_fn
        self.decode = decode_fn

    # For compatibility with diffusers‑style API
    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)


def causal_alignment(
    sae: nn.Module,
    pipe: DiffusionPipelineWrapper,
    clip_model: nn.Module,
    concept_prompts: List[str],
    feature_indices: List[int],
    intervention_delta: float = 2.0,
    device: str | torch.device = "cuda",
) -> Dict[str, float]:
    """Quantify how often toggling *feature_indices* makes *concept_prompts*
    more CLIP‑aligned.

    Returns dict mapping concept → alignment‑fraction ‑‑ i.e. proportion of
    test prompts where CLIP score improves after intervention.
    """
    clip_model.eval().to(device)
    sae.eval().to(device)

    alignment: Dict[str, float] = {}
    for concept in concept_prompts:
        improved = 0
        total = 0
        # generate a triplet of prompts: base, base+attr, negative("not attr")
        base_prompt = concept
        for _ in range(8):  # 8 runs for cheap estimate – tune as needed
            with torch.no_grad():
                latent = pipe.encode(base_prompt, device)
                latent = latent.to(device)
                x_flat = latent.view(1, -1)
                z = F.relu(sae.encoder(x_flat))
                z_int = z.clone()
                z_int[:, feature_indices] += intervention_delta
                x_mod = sae.decoder(z_int).view_as(latent)
                img_mod = pipe.decode(x_mod)  # PIL.Image
                img_base = pipe.decode(latent)

                # CLIP score (+ higher is better alignment)
                clip_inp = T.Compose([
                    T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    T.Normalize(
                        (0.48145466, 0.4578275, 0.40821073),
                        (0.26862954, 0.26130258, 0.27577711),
                    ),
                ])
                img_b_t = clip_inp(img_base).unsqueeze(0).to(device)
                img_m_t = clip_inp(img_mod).unsqueeze(0).to(device)

                text_tok = clip_model.tokenize([base_prompt]).to(device)
                img_b_feat = clip_model.encode_image(img_b_t)
                img_m_feat = clip_model.encode_image(img_m_t)
                txt_feat = clip_model.encode_text(text_tok)

                sim_base = F.cosine_similarity(img_b_feat, txt_feat)
                sim_mod = F.cosine_similarity(img_m_feat, txt_feat)

                if sim_mod > sim_base:
                    improved += 1
                total += 1
        alignment[concept] = improved / max(1, total)
    return alignment


# -----------------------------------------------------------------------------
# 3.  Temporal drift metric
# -----------------------------------------------------------------------------

def temporal_drift(
    sae_prev: nn.Module,
    sae_next: nn.Module,
    topk: int = 256,
    metric: str = "cosine",
) -> float:
    """Compute feature overlap / drift between two SAEs trained on different
    diffusion timesteps.

    We take *topk* decoder columns by L2 norm from each SAE, match them by
    nearest‑neighbour search, and average their similarity.

    Returns float ∈ [0,1] – higher = more stable features (less drift).
    """
    assert metric in {"cosine", "dot"}
    D1 = sae_prev.decoder.weight.data  # (d, k1)
    D2 = sae_next.decoder.weight.data  # (d, k2)

    # pick columns with largest norm
    norms1 = D1.norm(dim=0)
    norms2 = D2.norm(dim=0)
    idx1 = norms1.topk(topk).indices
    idx2 = norms2.topk(topk).indices

    cols1 = F.normalize(D1[:, idx1], dim=0)  # (d, topk)
    cols2 = F.normalize(D2[:, idx2], dim=0)  # (d, topk)

    # cosine sim matrix
    sim = cols1.t() @ cols2  # (topk, topk)
    max_sim, _ = sim.max(dim=1)
    drift_score = max_sim.mean().item() if metric == "cosine" else (max_sim * norms1[idx1]).mean().item()
    return drift_score


# -----------------------------------------------------------------------------
# CLI / quick test harness
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SAE metric computation")
    parser.add_argument("sae", type=Path, help="Path to SAE .pt file (torch save)")
    parser.add_argument("dataset", type=Path, help=".pt file containing latent tensor dataset or folder of .pt shards")
    parser.add_argument("--metric", choices=["mono", "drift"], default="mono")
    parser.add_argument("--sae2", type=Path, help="second SAE for drift metric")
    parser.add_argument("--batch", type=int, default=512)

    args = parser.parse_args()

    sae = torch.load(args.sae, map_location="cpu")

    if args.metric == "mono":
        dataset = torch.load(args.dataset, map_location="cpu")  # expect (N, 4, 64, 64)
        loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, pin_memory=True)
        score = monosemanticity_score(sae, loader)
        print(f"Monosemanticity score: {score:.4f}")
    else:
        if args.sae2 is None:
            parser.error("--sae2 required for drift metric")
        sae2 = torch.load(args.sae2, map_location="cpu")
        score = temporal_drift(sae, sae2)
        print(f"Temporal drift similarity: {score:.4f}")
