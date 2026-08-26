import numpy as np
import matplotlib.pyplot as plt

BASE = "data"

# Pick one subject, one trial, one channel — keep it simple
subject = 1
trial = 0     # first trial
channel = 0   # first channel

raw = np.load(f"{BASE}/raw/data/S{subject}.npy")            # shape (trials, time, channels)
prep = np.load(f"{BASE}/preprocessed/data/S{subject}.npy")  # shape (trials, time, channels)

raw_signal = raw[trial, :, channel]     # one trial, one channel, over time
prep_signal = prep[trial, :, channel]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=False)

ax1.plot(raw_signal, color='tab:red', linewidth=0.8)
ax1.set_title(f"RAW — Subject {subject}, Trial {trial}, Channel {channel}")
ax1.set_ylabel("Amplitude (raw units)")

ax2.plot(prep_signal, color='tab:blue', linewidth=0.8)
ax2.set_title(f"PREPROCESSED — Subject {subject}, Trial {trial}, Channel {channel}")
ax2.set_ylabel("Amplitude (normalized)")
ax2.set_xlabel("Time samples")

plt.tight_layout()
plt.savefig("raw_vs_preprocessed.png")
plt.show()
print("Saved as raw_vs_preprocessed.png")