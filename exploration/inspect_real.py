import numpy as np

BASE = "data"  
def inspect(label, data_path, label_path):
    eeg = np.load(data_path)
    lab = np.load(label_path)
    print(f"--- {label} ---")
    print(f"EEG shape: {eeg.shape}, dtype: {eeg.dtype}")
    print(f"EEG value range: min={eeg.min():.3f}, max={eeg.max():.3f}, mean={eeg.mean():.3f}, std={eeg.std():.3f}")
    print(f"Label shape: {lab.shape}, dtype: {lab.dtype}")
    print(f"Unique label values: {np.unique(lab, return_counts=True)}")
    print()

inspect("RAW - Subject 1", f"{BASE}/raw/data/S1.npy", f"{BASE}/raw/label/S1.npy")
inspect("PREPROCESSED - Subject 1", f"{BASE}/preprocessed/data/S1.npy", f"{BASE}/preprocessed/label/S1.npy")