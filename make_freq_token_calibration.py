"""
Build the -cind calibration array for freq_token: a constant integer
index (always 3 in this pipeline), not a real distribution to calibrate
against. onnx2tf's -oiqt still requires a -cind entry for every graph
input, including this one.

Per onnx2tf's documented -cind contract for -oiqt: calibration data must
be stored as Float32, pre-normalized to [0, 1], with mean/std such that
(stored - mean) / std reconstructs the real input value. For a true
constant, the stored value can be anything in [0,1] as long as mean/std
reconstruct 3.0 exactly -- using scaled=0.5 as an arbitrary, intentionally
dead-center choice:

    scaled = 0.5
    mean   = scaled - FREQ_TOKEN = 0.5 - 3 = -2.5
    std    = 1.0
    (scaled - mean) / std = (0.5 - (-2.5)) / 1.0 = 3.0  ✓

The assert below verifies this before anything is written, so a bad
derivation fails loudly rather than silently shipping a mis-calibrated
constant into the quantizer.
"""
import numpy as np

N = 200  # must match past_values calibration sample count exactly
FREQ_TOKEN = 3

scaled = np.full((N, 1), 0.5, dtype=np.float32)  # arbitrary midpoint, verified below
mean = 0.5 - FREQ_TOKEN  # = -2.5
std = 1.0

reconstructed = (scaled - mean) / std
assert np.allclose(reconstructed, FREQ_TOKEN), f"Got {reconstructed[0]}, expected {FREQ_TOKEN}"

np.save("exports/ttm_calibration_freq_token.npy", scaled)
print(f"mean={mean} std={std}")
print("Saved exports/ttm_calibration_freq_token.npy, shape", scaled.shape)
