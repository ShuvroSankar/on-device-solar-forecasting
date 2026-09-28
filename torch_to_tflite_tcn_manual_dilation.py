"""
Rebuild SmallTCN in Keras WITHOUT using Conv1D's native dilation_rate,
which the minimal-repro test proved triggers a SpaceToBatchND/
BatchToSpaceND decomposition that ST Edge AI Core's STM32F4 backend
cannot execute as true int8 -- it silently falls back to float32 for
that Conv2D, which is why every prior attempt (ONNX, TFLite, every
quantization format/config) never actually shrank the weights: 3 of
SmallTCN's 4 blocks use dilation >= 2 and together hold ~94% of the
model's parameters.

A dilated causal convolution is mathematically a sum of shifted,
ordinary (non-dilated) pointwise (kernel_size=1) convolutions:

    out[t] = sum_j W[:, :, j] @ x[t - (k-1-j)*dilation]

The minimal-repro test proved Pad, Slice/Crop, plain Conv2D (dilation=1)
and elementwise ops all quantize correctly on this backend -- so we
implement each dilated block as: left-pad, k static crops (one per
tap), k pointwise (1x1) convs, summed. No SpaceToBatchND anywhere.

Usage:
    python torch_to_tflite_tcn_manual_dilation.py \\
        --checkpoint checkpoints/small_tcn_solar.pt \\
        --out_prefix small_tcn_solar_manual --calib_dir ./processed/val
"""

import argparse
import os

import numpy as np
import torch

from tcn_model import SmallTCN
from solar_data_pipeline import load_site_capacities, METADATA_PATH
from export_quantize_tcn_v2 import build_calibration_windows
from torch_to_tflite_tcn import (
    parity_check,
    convert_to_tflite_int8,
    verify_tflite_int8,
    accuracy_cross_check,
)


def build_keras_tcn_manual_dilation(context_length, forecast_length, in_channels,
                                     channels=(32, 32, 64, 64), kernel_size=5):
    import tensorflow as tf
    from tensorflow.keras import layers, Model

    inputs = layers.Input(shape=(context_length, in_channels), name="input")
    x = inputs
    for i, out_ch in enumerate(channels):
        dilation = 2 ** i
        pad_amount = (kernel_size - 1) * dilation

        x_pad = layers.ZeroPadding1D(padding=(pad_amount, 0), name=f"blocks_{i}_pad")(x)

        taps = []
        for j in range(kernel_size):
            crop_front = j * dilation
            crop_back = pad_amount - crop_front
            x_slice = layers.Cropping1D(
                cropping=(crop_front, crop_back), name=f"blocks_{i}_tap{j}_crop"
            )(x_pad)
            use_bias = (j == kernel_size - 1)  # bias added exactly once, on the last tap
            tap_out = layers.Conv1D(
                out_ch, 1, use_bias=use_bias, name=f"blocks_{i}_tap{j}_conv"
            )(x_slice)
            taps.append(tap_out)

        summed = layers.Add(name=f"blocks_{i}_sum")(taps) if len(taps) > 1 else taps[0]
        x = layers.ReLU(name=f"blocks_{i}_act")(summed)

    last = layers.Lambda(lambda t: t[:, -1, :], name="gather_last")(x)
    out = layers.Dense(forecast_length, name="head")(last)
    return Model(inputs, out, name="SmallTCN_manual_dilation")


def port_pytorch_weights_manual(keras_model, torch_state_dict, channels=(32, 32, 64, 64), kernel_size=5):
    for i in range(len(channels)):
        w = torch_state_dict[f"blocks.{i}.conv.conv.weight"].numpy()  # (out, in, k)
        b = torch_state_dict[f"blocks.{i}.conv.conv.bias"].numpy()    # (out,)
        for j in range(kernel_size):
            w_tap = w[:, :, j]  # (out, in)
            w_tap_keras = np.transpose(w_tap, (1, 0))[np.newaxis, :, :]  # (1, in, out)
            layer = keras_model.get_layer(f"blocks_{i}_tap{j}_conv")
            if j == kernel_size - 1:
                layer.set_weights([w_tap_keras, b])
            else:
                layer.set_weights([w_tap_keras])

    w_head = torch_state_dict["head.weight"].numpy()
    b_head = torch_state_dict["head.bias"].numpy()
    keras_model.get_layer("head").set_weights([w_head.T, b_head])


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

    print("\nBuilding Keras model with MANUAL dilation (no SpaceToBatchND) and porting weights...")
    keras_model = build_keras_tcn_manual_dilation(context_length, forecast_length, in_channels)
    port_pytorch_weights_manual(keras_model, ckpt["model_state_dict"])

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

    print(f"\nDone. Now run locally first (fast, no board needed):")
    print(f"  /Applications/ST/STEdgeAI/4.0/Utilities/macarm/stedgeai analyze \\")
    print(f"      --model {tflite_path} --target stm32f4")
    print(f"Check for NO SpaceToBatchND/BatchToSpaceND in the op list, and 'weights (ro)' "
          f"close to the ~38K int8 params (should be roughly 35-45 KB, not ~146 KB).")


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
