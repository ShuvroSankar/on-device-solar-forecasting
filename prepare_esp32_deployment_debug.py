"""
DEBUG VARIANT -- adds an intermediate-tensor checkpoint on top of the
original prepare_esp32_deployment.py, so the ESP32 firmware can compare an
early activation (not just the final output) against the host reference.
Use this to localize where int8 divergence first appears in the graph.
The original prepare_esp32_deployment.py is untouched -- fall back to it
any time by just using it instead of this file.

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


def _pick_intermediate_tensor_index(interpreter, input_index, output_index, explicit_index=None):
    """
    Choose which intermediate tensor to use as an early checkpoint.

    If --intermediate_tensor_index is given explicitly, use that. Otherwise,
    print every tensor (index/name/shape/dtype) so you can eyeball the graph
    and re-run with an explicit index if the auto-pick isn't what you want,
    then auto-pick the first tensor whose name suggests an Add op (matching
    the first EltwiseInt(add) stage the ST Edge AI Core report showed), or
    -- if no "add"-named tensor is found -- fall back to the lowest-index
    tensor that isn't the model's own input or output.
    """
    details = interpreter.get_tensor_details()
    print("\n[debug] All tensors in this .tflite graph:")
    for d in sorted(details, key=lambda x: x["index"]):
        print(f"    idx={d['index']:>3}  shape={str(d['shape']):<16}  "
              f"dtype={d['dtype'].__name__:<8}  name={d['name']}")

    if explicit_index is not None:
        print(f"[debug] Using explicitly requested intermediate tensor index={explicit_index}")
        return explicit_index

    candidates = [d for d in details if "add" in d["name"].lower()]
    candidates = sorted(candidates, key=lambda x: x["index"])
    if candidates:
        idx = candidates[0]["index"]
        print(f"[debug] Auto-picked intermediate tensor index={idx} "
              f"(name='{candidates[0]['name']}', first tensor matching 'add')")
        return idx

    fallback = [d["index"] for d in details
                if d["index"] not in (input_index, output_index)]
    idx = min(fallback)
    print(f"[!] [debug] No tensor with 'add' in its name found -- falling back to "
          f"lowest non-I/O tensor index={idx}. Check the printed list above and "
          f"pass --intermediate_tensor_index explicitly if this isn't the checkpoint "
          f"you want (e.g. the tensor right after the first Pad->Slice->Conv2D_PW block).")
    return idx


def quantize_and_get_expected_output(tflite_path, sample_float, intermediate_tensor_index=None):
    import tensorflow as tf

    # BUILTIN_REF: use TFLite's plain reference kernels (not the XNNPACK
    # delegate) -- ruled out as the cause of the ESP32 divergence already,
    # but kept here since it's still the more apples-to-apples comparison
    # against TFLite Micro's own reference-style kernel implementations.
    interpreter = tf.lite.Interpreter(
        model_path=tflite_path,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF,
        experimental_preserve_all_tensors=True,  # needed to read intermediates, not just I/O
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

    mid_index = _pick_intermediate_tensor_index(
        interpreter, input_details["index"], output_details["index"],
        explicit_index=intermediate_tensor_index,
    )
    mid_detail = next(d for d in interpreter.get_tensor_details() if d["index"] == mid_index)
    mid_value = interpreter.get_tensor(mid_index)
    mid_scale, mid_zero = mid_detail["quantization"]

    intermediate = {
        "index": mid_index,
        "name": mid_detail["name"],
        "shape": list(mid_value.shape),
        "scale": float(mid_scale) if mid_scale else 1.0,
        "zero_point": int(mid_zero) if mid_zero else 0,
        "values": mid_value.flatten().astype(np.int8),
    }

    return sample_q[0], output_q, in_scale, in_zero, out_scale, out_zero, intermediate


def write_test_vector_header(out_dir, input_q, expected_output_q, in_scale, in_zero,
                              out_scale, out_zero, intermediate=None):
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

        if intermediate is not None:
            f.write(f"\n// --- DEBUG: intermediate checkpoint, for localizing divergence ---\n")
            f.write(f"// tensor index={intermediate['index']}  name=\"{intermediate['name']}\"\n")
            f.write(f"// shape={intermediate['shape']}  "
                    f"scale={intermediate['scale']}  zero_point={intermediate['zero_point']}\n")
            f.write(f"constexpr int kIntermediateTensorIndex = {intermediate['index']};\n")
            f.write(f"constexpr int kIntermediateLen = {len(intermediate['values'])};\n\n")
            f.write("const int8_t kIntermediateExpected[] = {\n    ")
            f.write(", ".join(str(int(v)) for v in intermediate["values"]))
            f.write("\n};\n")

    print(f"Wrote {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tflite_path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--calib_dir", default="./processed/test")
    parser.add_argument("--out_dir", default="esp32_deploy_files")
    parser.add_argument(
        "--intermediate_tensor_index", type=int, default=None,
        help="DEBUG: force a specific tensor index as the intermediate checkpoint. "
        "If omitted, the script prints every tensor in the graph and auto-picks "
        "the first one whose name suggests an Add op.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    tflite_to_c_array(args.tflite_path, args.out_dir)

    sample = get_real_test_sample(args.checkpoint, args.calib_dir)
    input_q, output_q, in_scale, in_zero, out_scale, out_zero, intermediate = quantize_and_get_expected_output(
        args.tflite_path, sample, intermediate_tensor_index=args.intermediate_tensor_index
    )
    write_test_vector_header(
        args.out_dir, input_q, output_q, in_scale, in_zero, out_scale, out_zero,
        intermediate=intermediate,
    )

    print(f"\nDone. Files in {args.out_dir}/: model_data.h, model_data.cc, test_vector.h")
