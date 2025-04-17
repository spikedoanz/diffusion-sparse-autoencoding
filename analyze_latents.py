#!/usr/bin/env python3
"""
Simple script to analyze latent activations from stable diffusion using tinygrad.
Uses random projections for dimensionality reduction and visualizes the latent space.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tinygrad.nn import state
from tinygrad.tensor import Tensor

# Directory containing the latent files
LATENTS_DIR = "data/latents"

def main():
    # Load latent files
    latent_files = glob.glob(os.path.join(LATENTS_DIR, "*.safetensors"))[:100]
    print(f"Loading {len(latent_files)} latent files...")
    
    # Load and flatten latents
    latents = []
    for file_path in latent_files:
        # Load using tinygrad's safe_load
        tensor_dict = state.safe_load(file_path)
        latent = tensor_dict["data"].numpy()  # Shape: [1, 4, 64, 64]
        latents.append(latent.reshape(-1))
    
    latents_array = np.vstack(latents)
    print(f"Loaded latents with shape {latents_array.shape}")
    
    # Create random projection matrix (2 x n_features)
    n_features = latents_array.shape[1]
    projection_matrix = np.random.normal(size=(2, n_features))
    projection_matrix /= np.sqrt(np.sum(projection_matrix**2, axis=1, keepdims=True))
    
    # Project to 2D using matrix multiplication
    latents_2d = latents_array @ projection_matrix.T
    print(f"Projected to 2D with shape {latents_2d.shape}")
    
    # Plot results
    plt.figure(figsize=(10, 8))
    plt.scatter(latents_2d[:, 0], latents_2d[:, 1], alpha=0.7)
    plt.title("Random Projection of Stable Diffusion Latents")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.tight_layout()
    
    # Save plot
    output_path = "latent_projection.png"
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()