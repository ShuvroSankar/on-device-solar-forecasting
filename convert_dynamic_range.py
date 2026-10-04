"""
Dynamic-range quantization: int8 weights, float32 activations computed
at runtime. No representative_dataset needed -- this sidesteps both
failure modes found so far:

  - Full int8 (TFLITE_BUILTINS_INT8): Erf/GELU decomposition's DIV node
    collapses ~77% of its divisor activations to the int8 zero-point
    (literal 0.0 after dequantization) regardless of input data --
    confirmed via Node 70 inspection, same pattern across all 33 DIV
    nodes (one per Erf instance).
  - int16x8 (ACTIVATIONS_INT16_WEIGHTS_INT8): converter cannot calibrate
    a Cast op's min/max range -- a TFLite converter limitation, not fixable
    with better calibration data.

Dynamic-range quantization only converts weights post-training; activations
stay float32 at inference time, so neither failure mode applies.
"""
import os

import tensorflow as tf

SAVED_MODEL_DIR = "exports/ttm_solar_savedmodel"
OUT_PATH = "exports/ttm_solar_quant/ttm_solar_dynamic_range.tflite"

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# No representative_dataset, no target_spec.supported_ops restriction,
# no inference_input_type/output_type override -- this is what makes it
# "dynamic range" rather than full-integer: weights only.

tflite_model = converter.convert()
with open(OUT_PATH, "wb") as f:
    f.write(tflite_model)
print(f"Wrote {OUT_PATH} ({len(tflite_model)/1024:.1f} KiB)")
