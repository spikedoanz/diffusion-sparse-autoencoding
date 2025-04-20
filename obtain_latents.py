# https://arxiv.org/pdf/2112.10752.pdf
# https://github.com/ekagra-ranjan/huggingface-blog/blob/main/stable_diffusion.md

import argparse
import hashlib
import math
import os
from collections import namedtuple
from typing import Dict, Any

import numpy as np
import pandas as pd

from tinygrad import Tensor, TinyJit, dtypes
from tinygrad.helpers import Context, fetch, tqdm
import tinygrad.nn as nn
from tinygrad.nn import Conv2d, GroupNorm
from tinygrad.nn.state import torch_load, load_state_dict, get_state_dict

from utils.clip import Tokenizer, Closed
from utils.unet import UNetModel


def get_alphas_cumprod(beta_start: float = 0.00085, beta_end: float = 0.0120, n_training_steps: int = 1000) -> Tensor:
    """
    Create a beta schedule and return cumulative product of alphas as a Tensor.
    """
    betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, n_training_steps, dtype=np.float32) ** 2
    alphas = 1.0 - betas
    return Tensor(np.cumprod(alphas, axis=0))


# --- autoencoder definitions ---
class AttnBlock:
    def __init__(self, in_channels):
        self.norm = GroupNorm(32, in_channels)
        self.q = Conv2d(in_channels, in_channels, 1)
        self.k = Conv2d(in_channels, in_channels, 1)
        self.v = Conv2d(in_channels, in_channels, 1)
        self.proj_out = Conv2d(in_channels, in_channels, 1)

    def __call__(self, x: Tensor) -> Tensor:
        h_ = self.norm(x)
        q, k, v = self.q(h_), self.k(h_), self.v(h_)
        b, c, h, w = q.shape
        q, k, v = [t.reshape(b, c, h*w).transpose(1,2) for t in (q,k,v)]
        h_ = Tensor.scaled_dot_product_attention(q, k, v).transpose(1,2).reshape(b, c, h, w)
        return x + self.proj_out(h_)


class ResnetBlock:
    def __init__(self, in_channels, out_channels=None):
        out_channels = out_channels or in_channels
        self.norm1 = GroupNorm(32, in_channels)
        self.conv1 = Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = GroupNorm(32, out_channels)
        self.conv2 = Conv2d(out_channels, out_channels, 3, padding=1)
        self.nin_shortcut = Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else (lambda x: x)

    def __call__(self, x: Tensor) -> Tensor:
        h = self.conv1(self.norm1(x).swish())
        h = self.conv2(self.norm2(h).swish())
        return self.nin_shortcut(x) + h


class Mid:
    def __init__(self, block_in):
        self.block_1 = ResnetBlock(block_in)
        self.attn_1 = AttnBlock(block_in)
        self.block_2 = ResnetBlock(block_in)

    def __call__(self, x: Tensor) -> Tensor:
        return x.sequential([self.block_1, self.attn_1, self.block_2])


class Decoder:
    def __init__(self):
        sz = [(128, 256), (256, 512), (512, 512), (512, 512)]
        self.conv_in = Conv2d(4, 512, 3, padding=1)
        self.mid = Mid(512)
        self.up = []
        for i, (in_c, out_c) in enumerate(sz):
            layers = [ResnetBlock(out_c, in_c), ResnetBlock(in_c), ResnetBlock(in_c)]
            block = {'block': layers}
            if i != 0:
                block['upsample'] = {'conv': Conv2d(in_c, in_c, 3, padding=1)}
            self.up.append(block)
        self.norm_out = GroupNorm(32, 128)
        self.conv_out = Conv2d(128, 3, 3, padding=1)

    def __call__(self, x: Tensor) -> Tensor:
        x = self.conv_in(x)
        x = self.mid(x)
        for layer in reversed(self.up):
            for b in layer['block']:
                x = b(x)
            if 'upsample' in layer:
                bs, c, h, w = x.shape
                x = x.reshape(bs, c, h, 1, w, 1).expand(bs, c, h, 2, w, 2).reshape(bs, c, h*2, w*2)
                x = layer['upsample']['conv'](x)
            x.realize()
        return self.conv_out(self.norm_out(x).swish())


class Encoder:
    def __init__(self):
        sz = [(128, 128), (128, 256), (256, 512), (512, 512)]
        self.conv_in = Conv2d(3, 128, 3, padding=1)
        self.down = []
        for i, (in_c, out_c) in enumerate(sz):
            layers = [ResnetBlock(in_c, out_c), ResnetBlock(out_c)]
            block = {'block': layers}
            if i != len(sz)-1:
                block['downsample'] = {'conv': Conv2d(out_c, out_c, 3, stride=2, padding=(0,1,0,1))}
            self.down.append(block)
        self.mid = Mid(512)
        self.norm_out = GroupNorm(32, 512)
        self.conv_out = Conv2d(512, 8, 3, padding=1)

    def __call__(self, x: Tensor) -> Tensor:
        x = self.conv_in(x)
        for layer in self.down:
            for b in layer['block']:
                x = b(x)
            if 'downsample' in layer:
                x = layer['downsample']['conv'](x)
        x = self.mid(x)
        return self.conv_out(self.norm_out(x).swish())


class AutoencoderKL:
    def __init__(self):
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.quant_conv = Conv2d(8, 8, 1)
        self.post_quant_conv = Conv2d(4, 4, 1)

    def __call__(self, x: Tensor) -> Tensor:
        latent = self.encoder(x)
        latent = self.quant_conv(latent)
        latent = latent[:, :4]
        latent = self.post_quant_conv(latent)
        return self.decoder(latent)


# UNet hyperparameters
unet_params: Dict[str, Any] = {
    "adm_in_ch": None,
    "in_ch": 4,
    "out_ch": 4,
    "model_ch": 320,
    "attention_resolutions": [4, 2, 1],
    "num_res_blocks": 2,
    "channel_mult": [1, 2, 4, 4],
    "n_heads": 8,
    "transformer_depth": [1, 1, 1, 1],
    "ctx_dim": 768,
    "use_linear": False,
}


class StableDiffusion:
    def __init__(self):
        self.alphas_cumprod = get_alphas_cumprod()
        self.model = namedtuple("DiffusionModel", ["diffusion_model"])(
            diffusion_model=UNetModel(**unet_params)
        )
        self.first_stage_model = AutoencoderKL()
        self.cond_stage_model = namedtuple("CondStageModel", ["transformer"])(
            transformer=namedtuple("Transformer", ["text_model"])(
                text_model=Closed.ClipTextTransformer()
            )
        )

    def get_x_prev_and_pred_x0(self, x: Tensor, e_t: Tensor, a_t: Tensor, a_prev: Tensor):
        sigma_t = 0
        sqrt_one_minus_at = (1 - a_t).sqrt()
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt
        return x_prev, pred_x0

    def get_model_output(self, uncond_ctx: Tensor, ctx: Tensor, latent: Tensor, timestep: Tensor, guidance: float) -> Tensor:
        prior = latent.expand(2, *latent.shape[1:])
        latents = self.model.diffusion_model(prior, timestep, uncond_ctx.cat(ctx, dim=0))
        uncond_latent, cond_latent = latents[0:1], latents[1:2]
        return uncond_latent + guidance * (cond_latent - uncond_latent)

    def __call__(self, uncond_ctx: Tensor, ctx: Tensor, latent: Tensor,
                 timestep: Tensor, a_t: Tensor, a_prev: Tensor, guidance: float) -> Tensor:
        e_t = self.get_model_output(uncond_ctx, ctx, latent, timestep, guidance)
        x_prev, _ = self.get_x_prev_and_pred_x0(latent, e_t, a_t, a_prev)
        return x_prev.realize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract latents using Stable Diffusion",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--fp16', action='store_true', help="Cast weights to float16")
    parser.add_argument('--timing', action='store_true', help="Print timing info")
    parser.add_argument('--guidance', type=float, default=7.5, help="Guidance scale")
    parser.add_argument('--bs', type=int, default=8, help="Batch size for CLIP encoding")
    args = parser.parse_args()

    BS = args.bs
    STEPS = 1
    SEED = 1337
    CSV_PATH = "./data/latents.csv"
    PROMPT_PATH = "./data/sampled_prompts.csv"

    # Load or initialize tracking CSV
    if os.path.exists(CSV_PATH):
        latents_df = pd.read_csv(CSV_PATH)
        done = set(latents_df['prompt'])
    else:
        latents_df = pd.DataFrame(columns=['prompt','latent_path'])
        done = set()

    # Load prompts
    try:
        with open(PROMPT_PATH) as f:
            lines = [l.strip() for l in f if l.strip()]
        if lines and lines[0]=='prompt': lines = lines[1:]
        all_prompts = lines
    except FileNotFoundError:
        all_prompts = pd.read_csv(PROMPT_PATH)['prompt'].tolist()

    prompts = [p for p in all_prompts if p not in done]
    if not prompts:
        print("All prompts processed. Exiting.")
        exit(0)

    # Seed & no grad
    Tensor.manual_seed(SEED)
    Tensor.no_grad = True

    # Instantiate model & load weights
    model = StableDiffusion()
    ckpt = torch_load(fetch(
        'https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt',
        'sd-v1-4.ckpt'
    ))['state_dict']
    load_state_dict(model, ckpt, strict=False)
    if args.fp16:
        for k,v in get_state_dict(model).items():
            if k.startswith('model'):
                v.replace(v.cast(dtypes.float16).realize())

    # JIT-compile text encoder now that model exists
    @TinyJit
    def text_forward(x: Tensor) -> Tensor:
        return model.cond_stage_model.transformer.text_model(x)

    # Prepare CLIP tokenizer & empty context
    tokenizer = Tokenizer.ClipTokenizer()
    empty_tokens = Tensor([tokenizer.encode("")])
    uncond_ctx = text_forward(empty_tokens).realize()

    # Diffusion schedule
    timesteps = list(range(1,1000,1000//STEPS))
    alphas = model.alphas_cumprod[Tensor(timesteps)]
    alphas_prev = Tensor([1.0]).cat(alphas[:-1])

    # Process in batches
    def batchify(lst, n):
        for i in range(0,len(lst),n): yield lst[i:i+n]

    bar = tqdm(batchify(prompts,BS), total=math.ceil(len(prompts)/BS))
    for batch in bar:
        bar.set_description(f"Batch size {len(batch)}")
        # Encode batch texts
        token_list = [Tensor([tokenizer.encode(p)]) for p in batch]  # shape (1, seq_len) each
        prompt_tensor = Tensor.cat(*token_list, dim=0)                # shape (BS, seq_len)
        contexts = text_forward(prompt_tensor).realize()
        prompt_tensor.realize()
        
        # Per-prompt diffusion
        for i,prompt in enumerate(batch):
            ctx = contexts[i:i+1]
            lat = Tensor.randn(1,4,64,64)
            for idx,t in reversed(list(enumerate(timesteps))):
                lat = model(uncond_ctx, ctx, lat,
                            Tensor([t]), alphas[idx:idx+1], alphas_prev[idx:idx+1], args.guidance)
            # Save
            h = hashlib.sha256(prompt.encode()).hexdigest()
            fn = f"s{STEPS}_g{args.guidance:.1f}_{h}.safetensors"
            out = os.path.join("data/latents",fn)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            nn.state.safe_save({'data':lat}, out,
                               metadata={'prompt':prompt,'steps':STEPS,'guidance':args.guidance})
            # Record
            latents_df = pd.concat([latents_df,
                                     pd.DataFrame([{'prompt':prompt,'latent_path':out}])],
                                     ignore_index=True)
        latents_df.to_csv(CSV_PATH,index=False)

    print("Done processing all prompts.")
