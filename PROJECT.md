# Diffusion Sparse Autoencoding Project

This project studies patterns in stable diffusion latent activations. The focus is on analyzing the latent space of stable diffusion models, potentially for finding patterns or developing more efficient representations.

## Project Structure

### Main Files

- `README.md`: Basic setup instructions and workflow overview
- `sample.py`: Prepares prompts for latent generation by deduplicating and filtering
- `obtain_latents.py`: Core file that generates latent activations from text prompts

### Utility Files

- `utils/clip.py`: Contains CLIP text encoding functionality for processing prompts
- `utils/unet.py`: Implements the UNet architecture used in the diffusion model

## Workflow

1. Download prompt dataset from Kaggle to `./data/`
2. Run `sample.py` to deduplicate prompts and filter for shorter-than-average ones
3. Execute `obtain_latents.py` to generate latent representations
4. The system saves results as safetensors files with metadata
5. The system tracks processed prompts to enable batch processing and avoid duplication

## File Functionality

- **sample.py**: Processes a dataset of diffusion prompts from Kaggle, filtering and deduplicating them for later use in latent generation.

- **obtain_latents.py**: The main processing file that:
  - Implements a Stable Diffusion pipeline
  - Loads prompt data from the preprocessed CSV
  - Generates latent representations for each prompt
  - Saves the resulting latents as safetensors files with metadata
  - Tracks processed prompts to avoid duplication

- **utils/clip.py**: Provides text encoding capabilities with both closed (OpenAI) and open implementations of CLIP text transformers for embedding text prompts.

- **utils/unet.py**: Implements the UNet architecture with components like ResBlocks, cross-attention mechanisms, SpatialTransformers, and upsampling/downsampling operations.