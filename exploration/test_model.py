"""
Sanity check for the DARNet model: feeds one real batch from the
train_loader through the model and verifies the output shape/dtype
make sense before we move on to the training loop.
"""
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from dataset import get_dataloaders
from model import DARNet

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "preprocessed")

# Build loaders (same as before)
train_loader, val_loader, train_ids, val_ids = get_dataloaders(DATA_DIR, batch_size=64)

# Pull one real batch
batch_x, batch_y = next(iter(train_loader))
print(f"Input batch shape: {batch_x.shape}")   # expect [64, 32, 128]

# Build the model using this batch's shape to fill config['Data_shape']
config = {'Data_shape': batch_x.shape}   # (64, 32, 128) -- only [1] and [2] matter
model = DARNet(config)

# Forward pass -- no training yet, just checking wiring is correct
model.eval()
output = model(batch_x)

print(f"Output shape: {output.shape}")   # expect [64, 2]
print(f"Output dtype: {output.dtype}")
print()
print("Sample raw logits (first 3 rows):")
print(output[:3])

# Total parameter count, for context on model size
n_params = sum(p.numel() for p in model.parameters())
print()
print(f"Total trainable parameters: {n_params:,}")