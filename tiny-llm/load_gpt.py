import torch
import torch.nn as nn
import torch.nn.functional as F
from gpt_model import GPT, Head, MultiHeadAttention, context_len

# Set device
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'Using device: {device}')

# Load checkpoint with proper device mapping
checkpoint = torch.load(
    '/Users/peternyman/Documents/GitHub/StreamRL/tiny-llm/gpt_checkpoint.pt',
    map_location=device
)

print(f'Loaded checkpoint from training:')
print(f'  Steps: {checkpoint["training_stats"]["steps"]}')
print(f'  Final loss: {checkpoint["training_stats"]["final_loss"]:.4f}')

# Create model with saved hyperparameters
gpt = GPT(
    checkpoint['hyperparameters']['n_emb'],
    checkpoint['hyperparameters']['num_heads'],
    checkpoint['hyperparameters']['head_size'],
    checkpoint['hyperparameters']['vocab_size']
).to(device)

# Load weights
gpt.load_state_dict(checkpoint['model_state_dict'])
gpt.eval()  # Set to evaluation mode

# Get vocabulary for encoding/decoding
stoi = checkpoint['vocab']['stoi']
itos = checkpoint['vocab']['itos']
decoder = lambda l: ''.join([itos[i] for i in l])

print('\nGenerating text...\n')

# Generate text
context = torch.tensor([[stoi['\n']]], dtype=torch.long, device=device)
generated_indices = gpt.generate(context, max_new_tokens=500)
generated_text = decoder(generated_indices[0].tolist())

print(generated_text)
