import os
import sys

import torch
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.dirname(__file__))
from splits import get_train_val_split


class AADDataset(Dataset):
    """
    Wraps an already-stacked (data, labels) pair -- as returned by
    get_train_val_split() -- and serves individual (window, label)
    samples in the shape DARNet expects: (channels, time).

    Doesn't know or care about subjects; that logic lives in splits.py.
    """

    def __init__(self, data, labels):
        """
        Args:
            data   : np.ndarray, shape (N, 128, 32) -- (trials, time, channels)
            labels : np.ndarray, shape (N, 1) or (N,)
        """
        data = data.transpose(0, 2, 1)
        labels = labels.squeeze()

        self.data = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def get_dataloaders(data_dir, batch_size=64, n_subjects=30, n_val_subjects=6, seed=42):
    """
    One-call front door to the whole data pipeline: gets the subject split,
    wraps train/val data in AADDataset, and returns ready-to-use DataLoaders.

    Returns:
        train_loader, val_loader, train_ids, val_ids
    """
    train_data, train_labels, val_data, val_labels, train_ids, val_ids = (
        get_train_val_split(data_dir, n_subjects=n_subjects,
                             n_val_subjects=n_val_subjects, seed=seed)
    )

    train_ds = AADDataset(train_data, train_labels)
    val_ds = AADDataset(val_data, val_labels)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, train_ids, val_ids