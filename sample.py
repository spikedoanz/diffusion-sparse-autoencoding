import pandas as pd
import random

random.seed(42)

df = pd.read_csv("data/diffusion_prompts.csv")
df['length'] = df['prompt'].str.len()
avg_length = df['length'].mean()

# Filter to shorter than average prompts
short_prompts = df[df['length'] < avg_length]

# Sample 10,000 (or all if less than 10,000 available)
sample_size = min(10000, len(short_prompts))
sampled = short_prompts.sample(n=sample_size, random_state=42)

# Save results
sampled.to_csv("data/sampled_prompts.csv", index=False)
print(f"Sampled {len(sampled)} prompts with average length {sampled['length'].mean():.1f}")
