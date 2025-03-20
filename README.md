```
git clone git@github.com:spikedoanz/diffusion-sparse-autoencoding.git
cd diffusion-sparse-autoencoding
python3 -m venv .venv
source .venv/bin/activate
pip install tinygrad

python sdxl.py # or stable_diffusion.py
```

fetch prompts from https://www.kaggle.com/datasets/tanreinama/900k-diffusion-prompts-dataset/data

and put into ./data/

then run sample.py to dedup the prompts



obtain_latents.py will run through the prompts and generate latents.




