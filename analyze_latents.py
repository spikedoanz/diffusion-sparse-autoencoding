#!/usr/bin/env python3
"""
Script to analyze latent activations from stable diffusion using tinygrad.
Implements multiple dimensionality reduction techniques to visualize the latent space.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tinygrad.nn import state
from tinygrad.tensor import Tensor
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import time
import argparse
from matplotlib.colors import ListedColormap

# Directory containing the latent files
LATENTS_DIR = "data/latents"

def plot_projection(data_2d, title, filename, metadata=None):
    """Plot and save a 2D projection with optional metadata coloring"""
    plt.figure(figsize=(12, 10))
    
    if metadata is not None:
        # Use categorical coloring if metadata is provided
        unique_vals = np.unique(metadata)
        cmap = plt.cm.get_cmap('tab20', len(unique_vals))
        
        # Map metadata values to indices
        color_indices = np.zeros(len(metadata))
        for i, val in enumerate(unique_vals):
            color_indices[metadata == val] = i
            
        scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], alpha=0.7, c=color_indices, cmap=cmap)
        
        # Add legend for up to 20 categories
        if len(unique_vals) <= 20:
            plt.colorbar(scatter, label="Category", ticks=range(len(unique_vals)))
    else:
        # Regular scatter plot
        plt.scatter(data_2d[:, 0], data_2d[:, 1], alpha=0.7)
    
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.tight_layout()
    
    # Save plot
    plt.savefig(filename, dpi=300)
    print(f"Plot saved to {filename}")

def main():
    parser = argparse.ArgumentParser(description="Analyze latent space of stable diffusion")
    parser.add_argument("--methods", type=str, default="random,pca,tsne,umap", 
                        help="Comma-separated list of dimensionality reduction methods to use")
    parser.add_argument("--sample", type=int, default=None, 
                        help="Number of samples to use (default: all)")
    parser.add_argument("--perplexity", type=int, default=30,
                        help="Perplexity parameter for t-SNE (default: 30)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--color-by", type=str, default="steps", 
                        choices=["steps", "guidance", "none"],
                        help="Metadata to use for coloring points")
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    
    # Parse methods
    methods = [m.strip().lower() for m in args.methods.split(",")]
    
    # Load latent files
    latent_files = glob.glob(os.path.join(LATENTS_DIR, "*.safetensors"))
    print(f"Found {len(latent_files)} latent files")
    
    # Sample if requested
    if args.sample and args.sample < len(latent_files):
        indices = np.random.choice(len(latent_files), args.sample, replace=False)
        latent_files = [latent_files[i] for i in indices]
        print(f"Sampled {len(latent_files)} files for analysis")
    
    print(f"Loading {len(latent_files)} latent files...")
    
    # Load and flatten latents
    latents = []
    steps_metadata = []  # Store steps info from filenames
    guidance_metadata = []  # Store guidance values from filenames
    
    for file_path in latent_files:
        # Load using tinygrad's safe_load
        tensor_dict = state.safe_load(file_path)
        latent = tensor_dict["data"].numpy()  # Shape: [1, 4, 64, 64]
        latents.append(latent.reshape(-1))
        
        # Extract steps and guidance from filename
        # Format: s{steps}_g{guidance}_{hash}.safetensors
        filename = os.path.basename(file_path)
        try:
            # Extract steps (s1)
            steps_str = filename.split('_')[0]
            steps = int(steps_str.replace('s', ''))
            steps_metadata.append(steps)
            
            # Extract guidance (g7.5)
            guidance_str = filename.split('_')[1]
            guidance = float(guidance_str.replace('g', ''))
            guidance_metadata.append(guidance)
        except (IndexError, ValueError) as e:
            print(f"Could not parse metadata from filename: {filename}, error: {e}")
            # Use default values if parsing fails
            steps_metadata.append(0)
            guidance_metadata.append(0.0)
    
    latents_array = np.vstack(latents)
    print(f"Loaded latents with shape {latents_array.shape}")
    
    # Convert metadata to numpy arrays
    steps_array = np.array(steps_metadata) if steps_metadata else None
    guidance_array = np.array(guidance_metadata) if guidance_metadata else None
    
    # Select which metadata to use for coloring
    metadata_array = None
    if args.color_by == "steps" and steps_array is not None:
        metadata_array = steps_array
        print(f"Using steps for coloring (values: {np.unique(steps_array)})")
    elif args.color_by == "guidance" and guidance_array is not None:
        metadata_array = guidance_array
        print(f"Using guidance for coloring (values: {np.unique(guidance_array)})")
    else:
        print("No metadata used for coloring")
    
    # Apply dimensionality reduction techniques
    for method in methods:
        start_time = time.time()
        
        if method == "random":
            # Random projection
            n_features = latents_array.shape[1]
            projection_matrix = np.random.normal(size=(2, n_features))
            projection_matrix /= np.sqrt(np.sum(projection_matrix**2, axis=1, keepdims=True))
            latents_2d = latents_array @ projection_matrix.T
            
            plot_projection(
                latents_2d, 
                "Random Projection of Stable Diffusion Latents", 
                "latent_projection_random.png",
                metadata_array
            )
            
        elif method == "pca":
            # PCA
            pca = PCA(n_components=2)
            latents_2d = pca.fit_transform(latents_array)
            variance_explained = pca.explained_variance_ratio_.sum() * 100
            
            plot_projection(
                latents_2d,
                f"PCA of Stable Diffusion Latents (Variance Explained: {variance_explained:.2f}%)",
                "latent_projection_pca.png",
                metadata_array
            )
            
            # Also save component information
            np.save("pca_components.npy", pca.components_)
            
        elif method == "tsne":
            # t-SNE
            tsne = TSNE(
                n_components=2, 
                perplexity=args.perplexity, 
                n_iter=1000, 
                random_state=args.seed
            )
            latents_2d = tsne.fit_transform(latents_array)
            
            plot_projection(
                latents_2d,
                f"t-SNE of Stable Diffusion Latents (Perplexity: {args.perplexity})",
                "latent_projection_tsne.png",
                metadata_array
            )
            
        elif method == "umap":
            # UMAP
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=15,
                min_dist=0.1,
                metric='euclidean',
                random_state=args.seed
            )
            latents_2d = reducer.fit_transform(latents_array)
            
            plot_projection(
                latents_2d,
                "UMAP of Stable Diffusion Latents",
                "latent_projection_umap.png",
                metadata_array
            )
        
        print(f"{method.upper()} completed in {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
