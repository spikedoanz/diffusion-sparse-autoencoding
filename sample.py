import pandas as pd
import random

random.seed(42)
df = pd.read_csv("data/diffusion_prompts.csv")
df['length'] = df['prompt'].str.len()
avg_length = df['length'].mean()

# Filter to shorter than average prompts
short_prompts = df[df['length'] < avg_length]

# Deduplicate prompts
deduped_prompts = short_prompts.drop_duplicates(subset=['prompt'])

# Save only the prompt column
deduped_prompts[['prompt']].to_csv("data/sampled_prompts.csv", index=False)
print(f"Saved {len(deduped_prompts)} deduplicated prompts with average length {deduped_prompts['length'].mean():.1f}")
