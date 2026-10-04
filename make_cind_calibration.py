"""
Build the -cind calibration file for TTM's past_values input, and print
the exact onnx2tf command to try.

onnx2tf's -oiqt calibration data must be stored pre-normalized to [0,1],
with mean/std such that (stored - mean) / std reconstructs the real model
input. Our calibration data (exports/ttm_calibration_windows.npy) is
already z-scored, NOT in [0,1], so we store a linearly-rescaled version
and derive mean/std to undo the rescaling exactly:
    scaled = (z - zmin) / (zmax - zmin)
    mean   = -zmin / (zmax - zmin)
    std    = 1 / (zmax - zmin)
Verified: (scaled - mean) / std == z for both zmin and zmax.

freq_token is int64, constant value 3 -- NOT included here. Per onnx2tf's
docs, -cind for -oiqt requires Float32 inputs; untested whether an int64
constant input needs its own -cind entry at all. Try WITHOUT one first.
"""
import numpy as np

z = np.load("exports/ttm_calibration_windows.npy")  # shape (200, 52, 1)
zmin, zmax = float(z.min()), float(z.max())

scaled = (z - zmin) / (zmax - zmin)
mean = -zmin / (zmax - zmin)
std = 1.0 / (zmax - zmin)

# sanity check before saving anything
reconstructed = (scaled - mean) / std
assert np.allclose(reconstructed, z, atol=1e-6), "Remap derivation is wrong -- do not proceed"

np.save("exports/ttm_calibration_scaled_0to1.npy", scaled.astype(np.float32))
print(f"zmin={zmin:.6f} zmax={zmax:.6f}")
print(f"mean={mean:.6f} std={std:.6f}")
print("Saved exports/ttm_calibration_scaled_0to1.npy, shape", scaled.shape)
