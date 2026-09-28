"""
Minimal reproducible test: isolate whether ST Edge AI Core v4.0.1's
failure to compress int8 weights on STM32F4 (seen on both the ONNX and
TFLite exports of SmallTCN -- weights stayed within a few percent of
float size across 8 configurations, plus a severe accuracy regression
on the TFLite path) is:
  (a) a general tool/target limitation, unrelated to SmallTCN, or
  (b) specific to SmallTCN's causal-padding / dilated-conv pattern.

Three tiny models, otherwise identical, quantized identically:
  A: plain Conv1D (padding='valid', dilation=1)      -- simplest possible
  B: causal-padded Conv1D (padding='causal', dilation=1) -- isolates causal padding
  C: causal-padded, DILATED Conv1D (padding='causal', dilation=2) -- isolates dilation

Random weights are fine here -- this test only checks the TOOL's ability
to compress/execute the STRUCTURE correctly, not task accuracy.

If A already fails to compress -> general tool/target limitation,
unrelated to anything about SmallTCN. If A succeeds but B and/or C
don't -> narrows the cause to causal padding and/or dilation
specifically, which may be workaroundable (e.g. replacing padding='causal'
with an explicit ZeroPadding1D + 'valid' conv).

Usage:
    python test_minimal_quant.py
    Then for each of the 3 generated .tflite files:
      /Applications/ST/STEdgeAI/4.0/Utilities/macarm/stedgeai analyze \\
          --model <path> --target stm32f4
    Compare the 'weights (ro)' and '% vs float model' lines across A, B, C.
"""

import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

from torch_to_tflite_tcn import convert_to_tflite_int8, verify_tflite_int8

OUT_DIR = "exports/minimal_repro"
os.makedirs(OUT_DIR, exist_ok=True)

CONTEXT_LENGTH = 20
IN_CHANNELS = 3
OUT_CHANNELS = 8
KERNEL_SIZE = 3
FORECAST_LENGTH = 4


def build_model(padding, dilation, name):
    inputs = layers.Input(shape=(CONTEXT_LENGTH, IN_CHANNELS), name="input")
    x = layers.Conv1D(OUT_CHANNELS, KERNEL_SIZE, dilation_rate=dilation, padding=padding, name="conv")(inputs)
    x = layers.ReLU(name="act")(x)
    last = layers.Lambda(lambda t: t[:, -1, :], name="gather_last")(x)
    out = layers.Dense(FORECAST_LENGTH, name="head")(last)
    return Model(inputs, out, name=name)


def random_calibration_windows(n=64):
    rng = np.random.default_rng(0)
    return rng.standard_normal((n, CONTEXT_LENGTH, IN_CHANNELS)).astype(np.float32)


def build_and_export(padding, dilation, tag):
    model = build_model(padding, dilation, tag)
    calib = random_calibration_windows()
    out_path = os.path.join(OUT_DIR, f"{tag}_int8.tflite")
    convert_to_tflite_int8(model, calib, out_path)
    kb = os.path.getsize(out_path) / 1024
    print(f"\n=== {tag} (padding={padding}, dilation={dilation}) ===")
    print(f"Saved: {out_path} ({kb:.2f} KB)")
    verify_tflite_int8(out_path)
    return out_path


if __name__ == "__main__":
    print("Building 3 minimal test models to isolate the ST Edge AI Core compression issue...")
    a = build_and_export("valid", 1, "A_plain")
    b = build_and_export("causal", 1, "B_causal_no_dilation")
    c = build_and_export("causal", 2, "C_causal_dilated")

    print("\n\nDone. Now run, for EACH file:")
    print("  /Applications/ST/STEdgeAI/4.0/Utilities/macarm/stedgeai analyze \\")
    print("      --model <path> --target stm32f4")
    print("\nCompare the 'weights (ro)' and '% vs float model' lines across A, B, C.")
    for p in (a, b, c):
        print(f"  {p}")
