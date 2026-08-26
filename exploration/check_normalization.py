import numpy as np

BASE = "data"

subjects = [1, 2, 3, 4, 5]

print(f"{'Subject':<10}{'Mean':>12}{'Std':>12}{'Min':>12}{'Max':>12}")
for sid in subjects:
    data = np.load(f"{BASE}/preprocessed/data/S{sid}.npy")
    print(f"S{sid:<9}{data.mean():>12.4f}{data.std():>12.4f}{data.min():>12.4f}{data.max():>12.4f}")

# Also check per-channel stats within one subject, to see if normalization
# was done per-channel or across all channels pooled together
print("\nPer-channel stats for Subject 1 (first 5 channels):")
s1 = np.load(f"{BASE}/preprocessed/data/S1.npy")  # shape (trials, time, channels)
for ch in range(5):
    ch_data = s1[:, :, ch]
    print(f"  Channel {ch}: mean={ch_data.mean():.4f}, std={ch_data.std():.4f}")