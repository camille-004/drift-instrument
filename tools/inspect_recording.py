import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

sample_rate, audio = wavfile.read("drift_session.wav")
audio = audio.astype(np.float32) / 32767.0

duration = len(audio) / sample_rate
print(f"Loaded {duration:.1f}s of audio at {sample_rate} Hz ({len(audio)} samples)")

diff = np.abs(np.diff(audio))
click_threshold = 0.15
click_indices = np.where(diff > click_threshold)[0]
print(f"Found {len(click_indices)} suspected click samples (jump > {click_threshold})")
if len(click_indices) > 0:
    print(f"First click at sample {click_indices[0]} (t={click_indices[0]/sample_rate:.3f}s)")
    print(f"Last click at sample {click_indices[-1]} (t={click_indices[-1]/sample_rate:.3f}s)")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

t = np.arange(len(audio)) / sample_rate
ax1.plot(t, audio, linewidth=0.5)
ax1.scatter(click_indices / sample_rate, audio[click_indices], color="red", s=10, zorder=5, label=f"{len(click_indices)} Clicks")
ax1.set_title("Full Recording (Red = Sample Jumps > 0.15)")
ax1.set_xlabel("Seconds")
ax1.set_ylabel("Amplitude")
ax1.legend()

if len(click_indices) > 0:
    center = click_indices[0]
    window = 200
    start = max(0, center - window)
    end = min(len(audio), center + window)
    ax2.plot(np.arange(start, end), audio[start:end], marker=".", markersize=2)
    ax2.axvline(center, color="red", linestyle="--", label=f"Click at Sample {center}")
    ax2.set_title(f"First Click Zoomed in (±{window} Samples)")
    ax2.set_xlabel("Sample Index")
    ax2.set_ylabel("Amplitude")
    ax2.legend()
else:
    ax2.text(0.5, 0.5, "No clicks detected", ha="center", va="center", transform=ax2.transAxes, fontsize=16)
    ax2.set_axis_off()

plt.tight_layout()
plt.show()