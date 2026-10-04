"""
Try int16-activation / int8-weight TFLite conversion from the onnx2tf
SavedModel. If the calibrator aborts with the known
'Empty min/max for tensor Cast' error, fall back to plain float32 TFLite.

Why int8 fails: TTM's Erf/GELU decomposition produces 33 DIV nodes whose
divisor tensor quantizes to the int8 zero-point (0.0) for ~77% of its
activations, because the divisor's real dynamic range (~0..7.4) is too
wide for int8's 256 levels. TFLite's int8 DIV kernel refuses to divide by
a code that represents exactly zero. Structural, not data-dependent.

Why int16 might fix it: int16 activations give 65,536 levels, so small
legitimate values no longer collapse to the exact zero-point.

Why int16 might still fail: the int64 freq_token input forces a Cast op
in the graph; the TFLite int16 calibrator requires min/max on every
intermediate tensor and aborts on integer-valued Cast outputs. If this
fires, there is no in-converter fix -- the fallback is float32.

Layouts:
  calibration windows are z-scored, ONNX order  (N, 52, 1)
  model input past_values expects TF order      (1, 1, 52)
  ZMIN/ZMAX come from make_cind_calibration.py, used to undo the [0,1]
  rescaling that the onnx2tf -cind contract requires.
"""
import os
import traceback

import numpy as np
import tensorflow as tf

SAVED_MODEL_DIR   = "exports/ttm_solar_savedmodel"
CALIB_PAST_VALUES = "exports/ttm_calibration_scaled_0to1_tf.npy"
FREQ_TOKEN        = 3
ZMIN, ZMAX        = -0.569221, 5.799404

OUT_DIR     = "exports/ttm_solar_quant"
OUT_INT16X8 = os.path.join(OUT_DIR, "ttm_solar_int16x8_quant.tflite")
OUT_FLOAT32 = os.path.join(OUT_DIR, "ttm_solar_float32_fallback.tflite")


def representative_dataset():
    """Yield one sample per call with correct dtype per input.

    past_values: float32 z-scored model input (the [0,1] rescaling is undone
                 so the converter calibrates on the real distribution).
    freq_token:  int64 constant 3 -- required for the Cast op in the graph
                 to be traced correctly during calibration.
    """
    past = np.load(CALIB_PAST_VALUES).astype(np.float32)  # (200, 1, 52), [0,1]
    z = past * (ZMAX - ZMIN) + ZMIN                       # back to z-scored
    for i in range(z.shape[0]):
        yield {
            "past_values": z[i:i + 1].astype(np.float32),          # (1, 1, 52)
            "freq_token":  np.array([FREQ_TOKEN], dtype=np.int64), # (1,)
        }


def try_int16x8():
    converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset

    # 16x8: int16 activations, int8 weights.
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8
    ]

    # Deliberately NOT setting inference_input_type / inference_output_type:
    # that would force ALL inputs to one dtype and break freq_token (int64).
    # The converter assigns per-tensor dtypes on its own.
    return converter.convert()


def float32_fallback():
    """Plain float32 TFLite -- no quantization, always works, known-correct."""
    converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
    return converter.convert()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=== Trying int16x8 (int16 activations, int8 weights) ===")
    try:
        model = try_int16x8()
        with open(OUT_INT16X8, "wb") as f:
            f.write(model)
        print(f"int16x8 succeeded -> {OUT_INT16X8} ({len(model)/1024:.1f} KiB)")
        print("\nNext: verify on-device int16 kernel support before committing.")
        print("      run: python test_quantized_model.py " + OUT_INT16X8)
        return
    except Exception as e:
        print("int16x8 conversion FAILED:")
        print(f"  {type(e).__name__}: {e}")
        msg = str(e)
        if "Empty min/max for tensor" in msg or "Cast" in msg:
            print("  -> Known Cast/min-max limitation of int16 calibration.")
        else:
            traceback.print_exc()

    print("\n=== Falling back to float32 TFLite ===")
    model = float32_fallback()
    with open(OUT_FLOAT32, "wb") as f:
        f.write(model)
    print(f"float32 fallback -> {OUT_FLOAT32} ({len(model)/1024:.1f} KiB)")
    print("\nThis is the deployment target for ESP32-C6 (TFLite Micro float32).")
    print("Verify with: python test_quantized_model.py " + OUT_FLOAT32)


if __name__ == "__main__":
    main()
