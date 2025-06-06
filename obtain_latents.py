# https://arxiv.org/pdf/2112.10752.pdf
# https://github.com/ekagra-ranjan/huggingface-blog/blob/main/stable_diffusion.md

import argparse
from collections import namedtuple
from typing import Dict, Any

import hashlib
import numpy as np

from tinygrad import Tensor, Device, TinyJit, dtypes, nn 
from tinygrad.helpers import Timing, Context, getenv, fetch, colored, tqdm, GlobalCounters
from tinygrad.nn import Conv2d, GroupNorm
from tinygrad.nn.state import torch_load, load_state_dict, get_state_dict

from utils.clip import Closed, Tokenizer
from utils.unet import UNetModel

class AttnBlock:
  def __init__(self, in_channels):
    self.norm = GroupNorm(32, in_channels)
    self.q = Conv2d(in_channels, in_channels, 1)
    self.k = Conv2d(in_channels, in_channels, 1)
    self.v = Conv2d(in_channels, in_channels, 1)
    self.proj_out = Conv2d(in_channels, in_channels, 1)

  # copied from AttnBlock in ldm repo
  def __call__(self, x):
    h_ = self.norm(x)
    q,k,v = self.q(h_), self.k(h_), self.v(h_)

    # compute attention
    b,c,h,w = q.shape
    q,k,v = [x.reshape(b,c,h*w).transpose(1,2) for x in (q,k,v)]
    h_ = Tensor.scaled_dot_product_attention(q,k,v).transpose(1,2).reshape(b,c,h,w)
    return x + self.proj_out(h_)

class ResnetBlock:
  def __init__(self, in_channels, out_channels=None):
    self.norm1 = GroupNorm(32, in_channels)
    self.conv1 = Conv2d(in_channels, out_channels, 3, padding=1)
    self.norm2 = GroupNorm(32, out_channels)
    self.conv2 = Conv2d(out_channels, out_channels, 3, padding=1)
    self.nin_shortcut = Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else lambda x: x

  def __call__(self, x):
    h = self.conv1(self.norm1(x).swish())
    h = self.conv2(self.norm2(h).swish())
    return self.nin_shortcut(x) + h

class Mid:
  def __init__(self, block_in):
    self.block_1 = ResnetBlock(block_in, block_in)
    self.attn_1 = AttnBlock(block_in)
    self.block_2 = ResnetBlock(block_in, block_in)

  def __call__(self, x):
    return x.sequential([self.block_1, self.attn_1, self.block_2])

class Decoder:
  def __init__(self):
    sz = [(128, 256), (256, 512), (512, 512), (512, 512)]
    self.conv_in = Conv2d(4,512,3, padding=1)
    self.mid = Mid(512)

    arr = []
    for i,s in enumerate(sz):
      arr.append({"block":
        [ResnetBlock(s[1], s[0]),
         ResnetBlock(s[0], s[0]),
         ResnetBlock(s[0], s[0])]})
      if i != 0: arr[-1]['upsample'] = {"conv": Conv2d(s[0], s[0], 3, padding=1)}
    self.up = arr

    self.norm_out = GroupNorm(32, 128)
    self.conv_out = Conv2d(128, 3, 3, padding=1)

  def __call__(self, x):
    x = self.conv_in(x)
    x = self.mid(x)

    for l in self.up[::-1]:
      for b in l['block']: x = b(x)
      if 'upsample' in l:
        # https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html ?
        bs,c,py,px = x.shape
        x = x.reshape(bs, c, py, 1, px, 1).expand(bs, c, py, 2, px, 2).reshape(bs, c, py*2, px*2)
        x = l['upsample']['conv'](x)
      x.realize()

    return self.conv_out(self.norm_out(x).swish())

class Encoder:
  def __init__(self):
    sz = [(128, 128), (128, 256), (256, 512), (512, 512)]
    self.conv_in = Conv2d(3,128,3, padding=1)

    arr = []
    for i,s in enumerate(sz):
      arr.append({"block":
        [ResnetBlock(s[0], s[1]),
         ResnetBlock(s[1], s[1])]})
      if i != 3: arr[-1]['downsample'] = {"conv": Conv2d(s[1], s[1], 3, stride=2, padding=(0,1,0,1))}
    self.down = arr

    self.mid = Mid(512)
    self.norm_out = GroupNorm(32, 512)
    self.conv_out = Conv2d(512, 8, 3, padding=1)

  def __call__(self, x):
    x = self.conv_in(x)

    for l in self.down:
      print("encode", x.shape)
      for b in l['block']: x = b(x)
      if 'downsample' in l: x = l['downsample']['conv'](x)

    x = self.mid(x)
    return self.conv_out(self.norm_out(x).swish())

class AutoencoderKL:
  def __init__(self):
    self.encoder = Encoder()
    self.decoder = Decoder()
    self.quant_conv = Conv2d(8, 8, 1)
    self.post_quant_conv = Conv2d(4, 4, 1)

  def __call__(self, x):
    latent = self.encoder(x)
    latent = self.quant_conv(latent)
    latent = latent[:, 0:4]  # only the means
    print("latent", latent.shape)
    latent = self.post_quant_conv(latent)
    return self.decoder(latent)

def get_alphas_cumprod(beta_start=0.00085, beta_end=0.0120, n_training_steps=1000):
  betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, n_training_steps, dtype=np.float32) ** 2
  alphas = 1.0 - betas
  alphas_cumprod = np.cumprod(alphas, axis=0)
  return Tensor(alphas_cumprod)

unet_params: Dict[str,Any] = {
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
    self.model = namedtuple("DiffusionModel", ["diffusion_model"])(diffusion_model = UNetModel(**unet_params))
    self.first_stage_model = AutoencoderKL()
    self.cond_stage_model = namedtuple("CondStageModel", ["transformer"])(transformer = namedtuple("Transformer", ["text_model"])(text_model = Closed.ClipTextTransformer()))

  def get_x_prev_and_pred_x0(self, x, e_t, a_t, a_prev):
    temperature = 1
    sigma_t = 0
    sqrt_one_minus_at = (1-a_t).sqrt()
    #print(a_t, a_prev, sigma_t, sqrt_one_minus_at)

    pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()

    # direction pointing to x_t
    dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t

    x_prev = a_prev.sqrt() * pred_x0 + dir_xt
    return x_prev, pred_x0

  def get_model_output(self, unconditional_context, context, latent, timestep, unconditional_guidance_scale):
    prior = latent.expand(2, *latent.shape[1:])
    latents = self.model.diffusion_model(prior, timestep, unconditional_context.cat(context, dim=0))
    unconditional_latent, latent = latents[0:1], latents[1:2]

    e_t = unconditional_latent + unconditional_guidance_scale * (latent - unconditional_latent)
    return e_t

  def decode(self, x):
    x = self.first_stage_model.post_quant_conv(1/0.18215 * x)
    x = self.first_stage_model.decoder(x)

    # make image correct size and scale
    x = (x + 1.0) / 2.0
    x = x.reshape(3,512,512).permute(1,2,0).clip(0,1)*255
    return x.cast(dtypes.uint8)

  def __call__(self, unconditional_context, context, latent, timestep, alphas, alphas_prev, guidance):
    e_t = self.get_model_output(unconditional_context, context, latent, timestep, guidance)
    x_prev, _ = self.get_x_prev_and_pred_x0(latent, e_t, alphas, alphas_prev)
    #e_t_next = get_model_output(x_prev)
    #e_t_prime = (e_t + e_t_next) / 2
    #x_prev, pred_x0 = get_x_prev_and_pred_x0(latent, e_t_prime, index)
    return x_prev.realize()

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Run Stable Diffusion', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--fp16', action='store_true', help="Cast the weights to float16")
  parser.add_argument('--timing', action='store_true', help="Print timing per step")
  parser.add_argument('--guidance', type=float, default=7.5, help="Prompt strength")
  args = parser.parse_args()
  BS            = 1 # TODO: Higher BS
  STEPS         = 5
  SEED          = 1337
  CSV_PATH      = "./data/latents.csv"
  PROMPT_PATH   = "./data/sampled_prompts.csv"
  
  # Load prompts from the sampled file
  import pandas as pd
  import os
  
  # Load or create the tracking CSV
  if os.path.exists(CSV_PATH):
    latents_df = pd.read_csv(CSV_PATH)
    processed_prompts = set(latents_df['prompt'].tolist())
  else:
    latents_df = pd.DataFrame(columns=['prompt', 'latent_path'])
    processed_prompts = set()
  
  # Load sampled prompts
  try:
    with open(PROMPT_PATH, 'r') as f:
      all_prompts = [line.strip() for line in f.readlines() if line.strip()]
    # Skip header if it exists
    if all_prompts and all_prompts[0] == 'prompt':
      all_prompts = all_prompts[1:]
  except:
    # Fallback to DataFrame approach if file format is different
    prompts_df = pd.read_csv(PROMPT_PATH)
    all_prompts = prompts_df['prompt'].tolist()
  
  # Filter out already processed prompts
  prompts = [p for p in all_prompts if p not in processed_prompts]
  
  if not prompts:
    print("All prompts have been processed!")
    exit(0)
    
  print(f"Processing {len(prompts)} remaining prompts out of {len(all_prompts)} total")
  
  Tensor.manual_seed(SEED)
  Tensor.no_grad = True
  # load in weights
  model = StableDiffusion()
  load_state_dict(model, torch_load(fetch('https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt', 'sd-v1-4.ckpt'))['state_dict'], strict=False)
  if args.fp16:
    for k,v in get_state_dict(model).items():
      if k.startswith("model"):
        v.replace(v.cast(dtypes.float16).realize())
  @TinyJit
  def run(model, *x): return model(*x).realize()
  # Invariants
  tokenizer = Tokenizer.ClipTokenizer()
  empty_prompt = Tensor([tokenizer.encode("")])
  timesteps = list(range(1, 1000, 1000//STEPS))
  alphas = model.alphas_cumprod[Tensor(timesteps)]
  alphas_prev = Tensor([1.0]).cat(alphas[:-1])
  print(f"running for {timesteps} timesteps")

  progress_bar = tqdm(prompts, desc="Processing prompts") 
  for prompt_str in progress_bar:
    display_prompt = prompt_str[:40] + ('...' if len(prompt_str) > 40 else '')
    progress_bar.set_description(f"Processing: {display_prompt}")
    # run through CLIP to get context
    tokenizer = Tokenizer.ClipTokenizer()
    prompt = Tensor([tokenizer.encode(prompt_str)])
    context = model.cond_stage_model.transformer.text_model(prompt).realize()
    unconditional_context = model.cond_stage_model.transformer.text_model(empty_prompt).realize()
    """
    # done with clip model
    del model.cond_stage_model
    """
    # start with random noise
    latent = Tensor.randn(1,4,64,64)
    # this is diffusion
    with Context(BEAM=getenv("LATEBEAM")):
      for index, timestep in reversed(list(enumerate(timesteps))):
        tid = Tensor([index])
        latent = run(model, unconditional_context, context, latent, Tensor([timestep]), alphas[tid], alphas_prev[tid], Tensor([args.guidance]))
    # export latent
    latent = latent.realize()
    hash_value = hashlib.sha256(prompt_str.encode()).hexdigest()
    # Format filename with steps and guidance
    filename = f"s{STEPS}_g{args.guidance:.1f}_{hash_value}"
    latent_path = f"data/latents/{filename}.safetensors"
    # TODO: shove the hyperparams in here
    metadata = {
            'prompt'    : prompt_str,
            'steps'     : STEPS,
            'guidance'  : args.guidance,
            }
    
    # Ensure directory exists
    os.makedirs("data/latents", exist_ok=True)
    
    nn.state.safe_save({'data': latent}, latent_path, metadata=metadata)
    
    # Update tracking CSV
    latents_df = pd.concat([latents_df, pd.DataFrame([{'prompt': prompt_str, 'latent_path': latent_path}])])
    latents_df.to_csv(CSV_PATH, index=False)
