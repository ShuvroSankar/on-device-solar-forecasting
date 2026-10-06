"""
Use TF's QuantizationDebugger to identify ops with high quantization error,
then build a selectively-quantized model: int8 everywhere except the
flagged ops (expected: the Erf/GELU decomposition), which stay float32.

Targets the root cause (Erf's DIV denominator's dynamic range doesn't
survive int8's 256 levels) without needing to redesign the activation's
math.

API notes:
  - TFLiteConverter.representative_dataset expects a dict keyed by
    input name -> array.
  - QuantizationDebugger.debug_dataset expects a list/tuple of arrays in
    the same order as interpreter.get_input_details(). Yielding a dict
    crashes inside _set_input_tensors with "'str' object has no attribute
    'shape'".
  - The layer-statistics DataFrame is debugger.layer_statistics, not the
    return value of debugger.layer_statistics_dump().
"""
import os

import numpy as np
import tensorflow as tf

SAVED_MODEL_DIR = "exports/ttm_solar_savedmodel"
CALIB_PAST_VALUES = "exports/ttm_calibration_scaled_0to1_tf.npy"
OUT_CSV = "exports/ttm_solar_quant/quant_debug_layers.csv"
ZMIN, ZMAX = -0.569221, 5.799404
FREQ_TOKEN = 3

os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)


def _load_z():
    past = np.load(CALIB_PAST_VALUES).astype(np.float32)
    return past * (ZMAX - ZMIN) + ZMIN  # back to z-scored model input


def rep_dataset_for_converter():
    """Dict form: required by TFLiteConverter.representative_dataset."""
    z = _load_z()
    for i in range(z.shape[0]):
        yield {
            "past_values": z[i:i + 1].astype(np.float32),
            "freq_token": np.array([FREQ_TOKEN], dtype=np.int64),
        }


def rep_dataset_for_debugger():
    """List form: required by QuantizationDebugger.debug_dataset.
    Order must match interpreter.get_input_details(), which for this
    SavedModel is [past_values, freq_token] in declaration order (as
    observed in the SavedModel signature check earlier)."""
    z = _load_z()
    for i in range(z.shape[0]):
        yield [
            z[i:i + 1].astype(np.float32),
            np.array([FREQ_TOKEN], dtype=np.int64),
        ]


converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = rep_dataset_for_converter
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# Match the full-int8 setup that we know crashes at runtime, so the
# debugger reports on THAT quantization scheme, not partial one.
# freq_token stays int64 automatically -- the converter does not force
# index inputs to int8.
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

debug_options = tf.lite.experimental.QuantizationDebugOptions(
    denylisted_nodes=[],
)
debugger = tf.lite.experimental.QuantizationDebugger(
    converter=converter,
    debug_dataset=rep_dataset_for_debugger,
    debug_options=debug_options,
)
debugger.run()

# Attribute, not method. DataFrame with one row per tensor.
results = debugger.layer_statistics
print("Columns:", list(results.columns))
print(f"Total tensors analyzed: {len(results)}\n")

# The error metric column name varies; find it defensively.
metric_col = next(
    (c for c in results.columns if "rmse" in c.lower() and "scale" in c.lower()),
    None,
)
if metric_col is None:
    print("No rmse/scale column found; dumping first 20 rows instead.")
    print(results.head(20).to_string())
else:
    print(f"Sorting by: {metric_col}\n")
    top = results.sort_values(metric_col, ascending=False, na_position="last").head(25)
    print(top.to_string())

results.to_csv(OUT_CSV, index=False)
print(f"\nWrote full stats -> {OUT_CSV}")
