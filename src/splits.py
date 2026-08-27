"""
splits.py

Handles splitting the EEG-AAD dataset into training and validation sets
by SUBJECT (never by individual trial/flashcard), to avoid leaking a
subject's brain-signal quirks between train and validation.

Usage:
    from splits import get_train_val_split

    train_data, train_labels, val_data, val_labels, train_ids, val_ids = get_train_val_split(
        data_dir="data/preprocessed",
        n_subjects=30,
        n_val_subjects=6,
        seed=42,
    )
"""

import numpy as np
import os


def get_train_val_split(data_dir, n_subjects=30, n_val_subjects=6, seed=42):
    """
    Randomly selects `n_val_subjects` subjects (out of `n_subjects`) to
    hold out as validation, using a fixed random seed for reproducibility.
    All other subjects go to training. Splitting happens at the SUBJECT
    level -- every flashcard/trial belonging to a given subject goes
    entirely into one group, never split across train and val.

    Returns:
        train_data   : np.ndarray, all training trials stacked together
        train_labels : np.ndarray, matching labels
        val_data     : np.ndarray, all validation trials stacked together
        val_labels   : np.ndarray, matching labels
        train_ids    : list of subject IDs used for training
        val_ids      : list of subject IDs used for validation
    """
    rng = np.random.default_rng(seed)

    all_ids = list(range(1, n_subjects + 1))  # [1, 2, ..., 30]
    shuffled_ids = all_ids.copy()
    rng.shuffle(shuffled_ids)

    val_ids = sorted(shuffled_ids[:n_val_subjects])
    train_ids = sorted(shuffled_ids[n_val_subjects:])

    # Sanity check: no subject should appear in both lists
    overlap = set(train_ids) & set(val_ids)
    assert len(overlap) == 0, f"Subject overlap detected between train/val: {overlap}"

    def load_subjects(subject_ids):
        data_list = []
        label_list = []
        for sid in subject_ids:
            data = np.load(os.path.join(data_dir, "data", f"S{sid}.npy"))
            label = np.load(os.path.join(data_dir, "label", f"S{sid}.npy"))
            data_list.append(data)
            label_list.append(label)
        stacked_data = np.concatenate(data_list, axis=0)
        stacked_labels = np.concatenate(label_list, axis=0)
        return stacked_data, stacked_labels

    train_data, train_labels = load_subjects(train_ids)
    val_data, val_labels = load_subjects(val_ids)

    return train_data, train_labels, val_data, val_labels, train_ids, val_ids


if __name__ == "__main__":
    # Quick sanity-check run when this file is executed directly
    train_data, train_labels, val_data, val_labels, train_ids, val_ids = get_train_val_split(
        data_dir="data/preprocessed",
        n_subjects=30,
        n_val_subjects=6,
        seed=42,
    )

    print("=" * 60)
    print("SPLIT SANITY CHECK")
    print("=" * 60)
    print(f"Train subjects ({len(train_ids)}): {train_ids}")
    print(f"Val subjects   ({len(val_ids)}): {val_ids}")
    print(f"Overlap between train/val subject IDs: {set(train_ids) & set(val_ids)}")
    print()
    print(f"Train data shape:   {train_data.shape}")
    print(f"Train labels shape: {train_labels.shape}")
    print(f"Val data shape:     {val_data.shape}")
    print(f"Val labels shape:   {val_labels.shape}")
    print()
    print(f"Total train flashcards: {len(train_labels)}")
    print(f"Total val flashcards:   {len(val_labels)}")
    print("=" * 60)