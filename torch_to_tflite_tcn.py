"""
Convert the trained SmallTCN PyTorch checkpoint into a genuinely
int8-quantized TFLite model, as an alternative path to the ONNX route
that seven independent configurations failed to get ST Edge AI Core to
actually compress (see project notes: weights stayed at ~146 KiB / -1.6%
vs float, regardless of dynamic/static/QDQ/QOperator/per-channel/
per-tensor/compression-flag variations, despite verified genuine int8
weights at the ONNX graph level).

Pipeline:
  1. Rebuild the exact SmallTCN architecture in Keras.
  2. Port the trained PyTorch weights into it (with the correct axis
     transposes -- PyTorch and Keras use different weight layouts).
  3. HARD PARITY CHECK: run the same inputs through both models and
     assert the outputs match closely. If this fails, STOP -- do not
     proceed to quantization on top of a broken port.
  4. Convert to TFLite with full-integer (int8 weights + int8
     activations) post-training quantization, using real calibration
     windows (reusing the same preprocessing as training: capacity
     normalize + z-score the generation channel, leave sin/cos alone).
  5. Verify the resulting .tflite file's actual weight storage is int8
     (not just claimed), and cross-check accuracy against the fp32
     Keras model on real data -- all without needing the board.

Usage:
    python torch_to_tflite_tcn.py --checkpoint checkpoints/small_tcn_solar.pt \\
        --out_prefix small_tcn_solar --calib_dir ./processed/val
"""

import argparse
import os

import numpy as np
import torch

from tcn_model import SmallTCN
from solar_data_pipeline import load_site_capacities, METADATA_PATH
from export_quantize_tcn_v2 import build_calibration_windows


def build_keras_tcn(context_length, forecast_length, in_channels, channels=(32, 32, 64, 64), kernel_size=5):
    import tensorflow as tf
    from tensorflow.keras import layers, Model

    inputs = layers.Input(shape=(context_length, in_channels), name="input")
    x = inputs
    for i, out_ch in enumerate(channels):
        dilation = 2 ** i
        # padding='causal' is Keras's built-in equivalent of PyTorch's
        # "pad both sides by (kernel_size-1)*dilation, then trim the
        # right" trick -- left-pads only, never looks at future steps.
        x = layers.Conv1D(
            out_ch, kernel_size, dilation_rate=dilation, padding="causal",
            name=f"blocks_{i}_conv",
        )(x)
        x = layers.ReLU(name=f"blocks_{i}_act")(x)
    # take the last timestep -- same as PyTorch's x[:, :, -1] after
    # its internal channels-first representation (here channels-last,
    # so it's x[:, -1, :]).
    last = layers.Lambda(lambda t: t[:, -1, :], name="gather_last")(x)
    out = layers.Dense(forecast_length, name="head")(last)
    return Model(inputs, out, name="SmallTCN_keras")


def port_pytorch_weights_to_keras(keras_model, torch_state_dict, n_blocks=4):
    for i in range(n_blocks):
        w = torch_state_dict[f"blocks.{i}.conv.conv.weight"].numpy()  # (out, in, k)
        b = torch_state_dict[f"blocks.{i}.conv.conv.bias"].numpy()    # (out,)
        w_keras = np.transpose(w, (2, 1, 0))  # -> (k, in, out), Keras Conv1D layout
        keras_model.get_layer(f"blocks_{i}_conv").set_weights([w_keras, b])

    w_head = torch_state_dict["head.weight"].numpy()  # (out_features, in_features)
    b_head = torch_state_dict["head.bias"].numpy()
    keras_model.get_layer("head").set_weights([w_head.T, b_head])  # Keras Dense wants (in, out)


def parity_check(torch_model, keras_model, context_length, in_channels, n_samples=32, atol=1e-4, rtol=1e-3):
    """Run identical random inputs through both models and require the
    outputs to match closely. This is the gate -- if it fails, the port
    has a bug (wrong axis order, wrong padding convention, etc.) and
    nothing downstream (quantization, deployment) should proceed."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((n_samples, context_length, in_channels)).astype(np.float32)

    torch_model.eval()
    with torch.no_grad():
        torch_out = torch_model(torch.tensor(x)).numpy()

    keras_out = keras_model.predict(x, verbose=0)

    max_abs_diff = np.max(np.abs(torch_out - keras_out))
    close = np.allclose(torch_out, keras_out, atol=atol, rtol=rtol)

    print(f"\n--- Parity check (PyTorch vs Keras, {n_samples} random inputs) ---")
    print(f"Max absolute difference: {max_abs_diff:.8f}")
    print(f"Within tolerance (atol={atol}, rtol={rtol}): {'YES' if close else 'NO -- STOP, DO NOT PROCEED'}")
    if not close:
        raise RuntimeError(
            "Parity check FAILED. The Keras port does not match the PyTorch "
            "model's outputs. Do not quantize or deploy this model until "
            "this is fixed -- likely a weight transpose or padding-convention bug."
        )
    return max_abs_diff


def convert_to_tflite_int8(keras_model, calib_windows, out_path):
    import tensorflow as tf

    def representative_dataset():
        for i in range(len(calib_windows)):
            yield [calib_windows[i : i + 1].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    with open(out_path, "wb") as f:
        f.write(tflite_model)
    return out_path


def verify_tflite_int8(tflite_path):
    """Load the .tflite file back and inspect it directly -- don't trust
    file size or converter claims, check the actual tensor dtypes."""
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"\n--- TFLite model inspection: {tflite_path} ---")
    print(f"Input: dtype={input_details[0]['dtype']}, shape={input_details[0]['shape']}, "
          f"quant={input_details[0]['quantization']}")
    print(f"Output: dtype={output_details[0]['dtype']}, shape={output_details[0]['shape']}, "
          f"quant={output_details[0]['quantization']}")

    # Inspect every tensor's dtype -- this is the ground-truth check for
    # whether weights are ACTUALLY stored as int8, not just claimed.
    tensor_details = interpreter.get_tensor_details()
    dtype_counts = {}
    int8_weight_bytes = 0
    float_weight_bytes = 0
    for t in tensor_details:
        dt = str(t["dtype"])
        dtype_counts[dt] = dtype_counts.get(dt, 0) + 1
        try:
            arr = interpreter.get_tensor(t["index"])
            if arr.ndim >= 2:  # heuristic: weight-like tensors, not scalars/biases-only
                nbytes = arr.nbytes
                if arr.dtype == np.int8:
                    int8_weight_bytes += nbytes
                elif arr.dtype == np.float32:
                    float_weight_bytes += nbytes
        except Exception:
            pass

    print(f"Tensor dtype counts: {dtype_counts}")
    print(f"Weight-like tensor bytes -- int8: {int8_weight_bytes:,}  float32: {float_weight_bytes:,}")
    if float_weight_bytes > int8_weight_bytes:
        print("[!] WARNING: more float32 weight bytes than int8 -- quantization may not have "
              "applied to the conv layers. Investigate before trusting file-size compression.")
    else:
        print("Conv/Dense weights appear genuinely int8-stored. Good.")

    return interpreter


def accuracy_cross_check(keras_model, tflite_interpreter, calib_windows, n_samples=20):
    """Compare fp32 Keras output vs dequantized TFLite output on the same
    real data -- same spirit as ST's validate step, but runnable locally."""
    input_details = tflite_interpreter.get_input_details()[0]
    output_details = tflite_interpreter.get_output_details()[0]
    in_scale, in_zero = input_details["quantization"]
    out_scale, out_zero = output_details["quantization"]

    n = min(n_samples, len(calib_windows))
    fp32_preds = keras_model.predict(calib_windows[:n], verbose=0)

    tflite_preds = []
    for i in range(n):
        x = calib_windows[i : i + 1].astype(np.float32)
        x_q = np.round(x / in_scale + in_zero).astype(np.int8)
        tflite_interpreter.set_tensor(input_details["index"], x_q)
        tflite_interpreter.invoke()
        y_q = tflite_interpreter.get_tensor(output_details["index"])
        y = (y_q.astype(np.float32) - out_zero) * out_scale
        tflite_preds.append(y[0])
    tflite_preds = np.array(tflite_preds)

    mae = np.mean(np.abs(fp32_preds - tflite_preds))
    rmse = np.sqrt(np.mean((fp32_preds - tflite_preds) ** 2))
    cos = np.mean([
        np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        for a, b in zip(fp32_preds, tflite_preds)
    ])
    print(f"\n--- Accuracy cross-check (fp32 Keras vs dequantized TFLite, {n} real samples) ---")
    print(f"MAE: {mae:.6f}  RMSE: {rmse:.6f}  mean cosine similarity: {cos:.6f}")


def main(checkpoint_path, out_prefix, out_dir, calib_dir, num_calib_samples, calib_stride):
    os.makedirs(out_dir, exist_ok=True)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    context_length = ckpt["context_length"]
    forecast_length = ckpt["forecast_length"]
    in_channels = ckpt.get("in_channels", 1)
    norm_mean = ckpt.get("norm_mean")
    norm_std = ckpt.get("norm_std")
    print(f"Loaded checkpoint: context_length={context_length}, "
          f"forecast_length={forecast_length}, in_channels={in_channels}")

    torch_model = SmallTCN(context_length=context_length, forecast_length=forecast_length, in_channels=in_channels)
    torch_model.load_state_dict(ckpt["model_state_dict"])
    torch_model.eval()

    print("\nBuilding Keras model and porting weights...")
    keras_model = build_keras_tcn(context_length, forecast_length, in_channels)
    port_pytorch_weights_to_keras(keras_model, ckpt["model_state_dict"])

    parity_check(torch_model, keras_model, context_length, in_channels)

    print("\nBuilding calibration windows...")
    capacities = load_site_capacities(METADATA_PATH)
    calib_windows = build_calibration_windows(
        calib_dir, context_length, forecast_length, norm_mean, norm_std,
        num_samples=num_calib_samples, stride=calib_stride,
    )

    print("\nConverting to full-integer TFLite...")
    tflite_path = os.path.join(out_dir, f"{out_prefix}_int8.tflite")
    convert_to_tflite_int8(keras_model, calib_windows, tflite_path)
    tflite_kb = os.path.getsize(tflite_path) / 1024
    print(f"Saved: {tflite_path} ({tflite_kb:.1f} KB)")

    interpreter = verify_tflite_int8(tflite_path)
    accuracy_cross_check(keras_model, interpreter, calib_windows)

    print(f"\nDone. Bring {tflite_path} into AI Studio as a new network (type: tflite).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out_prefix", required=True)
    parser.add_argument("--out_dir", default="exports")
    parser.add_argument("--calib_dir", default="./processed/val")
    parser.add_argument("--num_calib_samples", type=int, default=128)
    parser.add_argument("--calib_stride", type=int, default=6)
    args = parser.parse_args()

    main(
        args.checkpoint, args.out_prefix, args.out_dir,
        args.calib_dir, args.num_calib_samples, args.calib_stride,
    )
