"""
Manual int8 TFLite conversion from the onnx2tf-produced SavedModel.

Why not onnx2tf -oiqt -cind? Because freq_token is int64 in the graph,
and onnx2tf's -cind path only accepts float32 calibration data. The TFLite
interpreter rejects setting an int64 tensor with a float32 array, so there
is no way to satisfy freq_token through -cind. This is the same limitation
documented in onnx2tf issue #248 (BERT-Squad INT8 quantization).

The workaround, per the maintainer's own recommendation there: emit a
SavedModel with -osd, then run tf.lite.TFLiteConverter manually with a
representative_dataset that yields the correct dtype per input -- float32
for past_values, int64 for freq_token.
"""
import os

import numpy as np
import tensorflow as tf

SAVED_MODEL_DIR = "exports/ttm_solar_savedmodel"
CALIB_PAST_VALUES = "exports/ttm_calibration_scaled_0to1_tf.npy"  # (200, 1, 52) float32
N_CALIB = 200
FREQ_TOKEN = 3
OUT_PATH = "exports/ttm_solar_int8/ttm_solar_full_integer_quant.tflite"


def representative_dataset():
    """Yield one sample at a time, with the correct dtype for each input.

    past_values comes from the saved calibration .npy (already in TF order,
    range [0,1], pre-normalized per onnx2tf's -cind contract -- but here we
    feed it directly to TFLiteConverter, which does not apply the mean/std
    remap the way -cind does. So we must undo the [0,1] rescaling ourselves
    to recover the real z-scored model input).

    freq_token is the constant integer 3, shaped (1,), dtype int64.
    """
    past = np.load(CALIB_PAST_VALUES).astype(np.float32)  # (200, 1, 52), [0,1]

    # Undo the [0,1] rescaling: scaled = (z - zmin)/(zmax - zmin), so
    # z = scaled*(zmax - zmin) + zmin. The zmin/zmax values are the ones
    # printed by make_cind_calibration.py.
    zmin, zmax = -0.569221, 5.799404
    z = past * (zmax - zmin) + zmin  # back to z-scored model input

    for i in range(past.shape[0]):
        yield {
            "past_values": z[i:i+1].astype(np.float32),      # (1, 1, 52)
            "freq_token": np.array([FREQ_TOKEN], dtype=np.int64),  # (1,)
        }


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    with open(OUT_PATH, "wb") as f:
        f.write(tflite_model)

    print(f"Wrote {OUT_PATH} ({len(tflite_model) / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
