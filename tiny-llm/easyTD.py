
from datasets import load_dataset
ds = load_dataset("roneneldan/TinyStories", split="train")
[print(f'-----------------\n{d}') for d in ds]