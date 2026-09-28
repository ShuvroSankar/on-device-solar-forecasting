"""
Prepare two things needed for the ESP32-C6 / ESP-IDF / TFLite Micro deployment:

1. model_data.h/.cc -- the verified small_tcn_solar_manual_int8.tflite file,
   converted into a C byte array (TFLite Micro loads models from memory,
   not a filesystem).

2. test_vector.h -- a REAL preprocessed+quantized input window (pulled from
   your actual val split, same preprocessing as training: capacity-normalize
   + z-score the generation channel, sin/cos channels untouched) plus its
   expected int8 output, computed by running the same .tflite file through
   the TFLite interpreter on the host. This lets the ESP32 firmware check
   its own output against a known-correct answer, instead of just printing
   numbers with nothing to compare them to.

Usage:
    python prepare_esp32_deployment.py \\
        --tflite_path exports/small_tcn_solar_manual_int8.tflite \\
        --checkpoint checkpoints/small_tcn_solar.pt \\
        --calib_dir ./processed/test \\
        --out_dir esp32_deploy_files
"""

import argparse
import os

import numpy as np
import torch

from solar_data_pipeline import load_site_capacities, METADATA_PATH
from export_quantize_tcn_v2 import build_calibration_windows


def tflite_to_c_array(tflite_path, out_dir, var_name="g_model_data"):
    with open(tflite_path, "rb") as f:
        data = f.read()

    header_path = os.path.join(out_dir, "model_data.h")
    cc_path = os.path.join(out_dir, "model_data.cc")

    with open(header_path, "w") as f:
        f.write("#pragma once\n#include <cstdint>\n\n")
        f.write(f"extern const unsigned char {var_name}[];\n")
        f.write(f"extern const unsigned int {var_name}_len;\n")

    with open(cc_path, "w") as f:
        f.write('#include "model_data.h"\n\n')
        # alignas(16) matters for TFLite Micro's flatbuffer alignment requirements
        f.write(f"alignas(16) const unsigned char {var_name}[] = {{\n")
        for i in range(0, len(data), 12):
            chunk = data[i : i + 12]
            line = ", ".join(f"0x{b:02x}" for b in chunk)
            f.write(f"    {line},\n")
        f.write("};\n")
        f.write(f"const unsigned int {var_name}_len = {len(data)};\n")

    print(f"Wrote {header_path} and {cc_path} ({len(data):,} bytes)")


def get_real_test_sample(checkpoint_path, calib_dir, seed=0):
    """Pull one real preprocessed window + run it through the .tflite
    interpreter to get a known-correct expected output."""
    import tensorflow as tf

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    context_length = ckpt["context_length"]
    forecast_length = ckpt["forecast_length"]
    norm_mean = ckpt["norm_mean"]
    norm_std = ckpt["norm_std"]

    capacities = load_site_capacities(METADATA_PATH)
    windows = build_calibration_windows(
        calib_dir, context_length, forecast_length, norm_mean, norm_std,
        num_samples=1, stride=6, seed=seed,
    )
    return windows[0:1]  # (1, context_length, in_channels) float32


def quantize_and_get_expected_output(tflite_path, sample_float):
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(
    model_path=tflite_path,
    experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF,
)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    in_scale, in_zero = input_details["quantization"]
    out_scale, out_zero = output_details["quantization"]

    sample_q = np.round(sample_float / in_scale + in_zero).astype(np.int8)

    interpreter.set_tensor(input_details["index"], sample_q)
    interpreter.invoke()
    output_q = interpreter.get_tensor(output_details["index"])[0]  # (forecast_length,) int8

    return sample_q[0], output_q, in_scale, in_zero, out_scale, out_zero


def write_test_vector_header(out_dir, input_q, expected_output_q, in_scale, in_zero, out_scale, out_zero):
    path = os.path.join(out_dir, "test_vector.h")
    context_length, in_channels = input_q.shape
    flat_input = input_q.flatten()

    with open(path, "w") as f:
        f.write("#pragma once\n#include <cstdint>\n\n")
        f.write(f"// Real preprocessed+quantized window pulled from the test split.\n")
        f.write(f"// Input quantization: scale={in_scale}, zero_point={in_zero}\n")
        f.write(f"// Output quantization: scale={out_scale}, zero_point={out_zero}\n\n")
        f.write(f"constexpr int kTestInputContextLength = {context_length};\n")
        f.write(f"constexpr int kTestInputChannels = {in_channels};\n")
        f.write(f"constexpr int kTestOutputLength = {len(expected_output_q)};\n")
        f.write(f"constexpr float kInputScale = {in_scale}f;\n")
        f.write(f"constexpr int kInputZeroPoint = {in_zero};\n")
        f.write(f"constexpr float kOutputScale = {out_scale}f;\n")
        f.write(f"constexpr int kOutputZeroPoint = {out_zero};\n\n")

        f.write("const int8_t kTestInput[] = {\n    ")
        f.write(", ".join(str(int(v)) for v in flat_input))
        f.write("\n};\n\n")

        f.write("// Expected output from running this exact input through the\n")
        f.write("// verified .tflite model on host -- compare on-device output against this.\n")
        f.write("const int8_t kExpectedOutput[] = {\n    ")
        f.write(", ".join(str(int(v)) for v in expected_output_q))
        f.write("\n};\n")

    print(f"Wrote {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tflite_path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--calib_dir", default="./processed/test")
    parser.add_argument("--out_dir", default="esp32_deploy_files")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    tflite_to_c_array(args.tflite_path, args.out_dir)

    sample = get_real_test_sample(args.checkpoint, args.calib_dir)
    input_q, output_q, in_scale, in_zero, out_scale, out_zero = quantize_and_get_expected_output(
        args.tflite_path, sample
    )
    write_test_vector_header(args.out_dir, input_q, output_q, in_scale, in_zero, out_scale, out_zero)

    print(f"\nDone. Files in {args.out_dir}/: model_data.h, model_data.cc, test_vector.h")
