"""
Sanity check for get_dataloaders(): verifies the full pipeline
(split -> Dataset -> DataLoader) produces correctly shaped batches
before we move on to building the model / training loop.
"""
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from dataset import get_dataloaders

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "preprocessed")

train_loader, val_loader, train_ids, val_ids = get_dataloaders(DATA_DIR, batch_size=64)

print(f"Train subjects ({len(train_ids)}): {train_ids}")
print(f"Val subjects ({len(val_ids)}): {val_ids}")
print(f"Number of train batches: {len(train_loader)}")   # expect ceil(157920 / 64)
print(f"Number of val batches: {len(val_loader)}")        # expect ceil(39480 / 64)
print()

# Pull one batch from each loader and inspect shapes/dtypes
train_batch_x, train_batch_y = next(iter(train_loader))
print(f"Train batch x shape: {train_batch_x.shape}")   # expect [64, 32, 128]
print(f"Train batch x dtype: {train_batch_x.dtype}")    # expect float32
print(f"Train batch y shape: {train_batch_y.shape}")   # expect [64]
print(f"Train batch y dtype: {train_batch_y.dtype}")    # expect int64
print()

val_batch_x, val_batch_y = next(iter(val_loader))
print(f"Val batch x shape: {val_batch_x.shape}")
print(f"Val batch y shape: {val_batch_y.shape}")
print()

# Sanity check on label balance within one batch (not guaranteed exact,
# but should be roughly balanced given the dataset is 50/50 overall)
print(f"Train batch label distribution: {train_batch_y.bincount()}")