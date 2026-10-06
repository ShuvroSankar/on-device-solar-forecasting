"""
TTM variant of prepare_esp32_deployment.py: prepare model_data.h/.cc and
test_vector.h for the ESP32-C6 deployment, for the float32 TTM model
(ttm_solar_float32_fallback.tflite) rather than SmallTCN's int8 model.

Three real differences from the SmallTCN version, not just parameter swaps:

1. FLOAT32, not int8. TTM's deployment format is float32 (see README --
   full int8, int16x8, and dynamic-range quantization were all tried and
   ruled out). No quantize/dequantize step: test_vector.h holds real
   float32 values directly, scale=1.0/zero_point=0 placeholders kept only
   so firmware code written against the SmallTCN header shape still
   compiles if shared, but should be treated as "no quantization" markers.

2. TWO inputs, not one: past_values (float32) AND freq_token (int64,
   always 3 in this pipeline -- see export_ttm_onnx.py). SmallTCN's
   single-input assumption throughout the original script does not carry
   over; test_vector.h now emits both.

3. Input layout is (1, 1, 52), NOT (1, context_length, in_channels) /
   (1, 52, 1) like SmallTCN. onnx2tf transposed this during ONNX->TF
   conversion (confirmed in onnx_to_tf_ttm.py's parity check) -- TTM's
   single generation channel is also NOT capacity-normalized + z-scored
   with sin/cos time features appended like SmallTCN; it is capacity-
   normalized + z-scored ALONE (TTM is univariate -- see finetune_ttm.py).

Usage:
    python prepare_esp32_deployment_ttm.py \\
        --tflite_path exports/ttm_solar_quant/ttm_solar_float32_fallback.tflite \\
        --calib_dir ./processed/test \\
        --out_dir esp32_deploy_files_ttm
"""

import argparse
import os

import numpy as np

from solar_data_pipeline import load_solar_split, load_site_capacities, METADATA_PATH
from memory_safe_windows import build_windows_channel0_lite

# Must match finetune_ttm.py / export_ttm_onnx.py exactly.
CONTEXT_LENGTH = 52
FORECAST_LENGTH = 16
FREQ_TOKEN = 3

# The train-fit Normalizer stats, reproduced in build_ttm_calibration_data.py
# and confirmed there (printed mean/std). Hardcoded here rather than
# recomputed, since recomputing requires rebuilding 300K train windows --
# see build_ttm_calibration_data.py's docstring for why this is unavoidable
# without a cached value. If finetune_ttm.py is ever changed to save these
# into its own checkpoint (flagged as a TODO in this session), read them
# from there instead of trusting this constant.
TTM_NORM_MEAN = 9.020441
TTM_NORM_STD = 15.846984


def tflite_to_c_array(tflite_path, out_dir, var_name="g_model_data"):
    """Unchanged from prepare_esp32_deployment.py -- reading raw bytes and
    emitting a C array is identical regardless of model architecture or
    quantization. Reused verbatim."""
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
        f.write(f"alignas(16) const unsigned char {var_name}[] = {{\n")
        for i in range(0, len(data), 12):
            chunk = data[i : i + 12]
            line = ", ".join(f"0x{b:02x}" for b in chunk)
            f.write(f"    {line},\n")
        f.write("};\n")
        f.write(f"const unsigned int {var_name}_len = {len(data)};\n")

    print(f"Wrote {header_path} and {cc_path} ({len(data):,} bytes)")
    if len(data) > 300_000:
        print(f"  NOTE: {len(data)/1024:.1f} KiB is substantially larger than "
              f"SmallTCN's int8 model (40.7 KiB) -- confirm this compiles and "
              f"fits before assuming it will; flash budget was already tight "
              f"enough that TTM int8 didn't fit on STM32 at a smaller size.")


def get_real_ttm_test_sample(calib_dir, seed=0):
    """Pull ONE real preprocessed window from the test split, matching
    TTM's actual preprocessing: capacity-normalize generation, then
    z-score with the TRAIN-fit Normalizer stats (not fit fresh here).

    Returns (past_values, freq_token):
      past_values: (1, 1, CONTEXT_LENGTH) float32 -- TF-order layout,
                   matching what onnx2tf's conversion expects (see module
                   docstring).
      freq_token:  (1,) int64, constant FREQ_TOKEN.
    """
    capacities = load_site_capacities(METADATA_PATH)
    test_sites = load_solar_split(calib_dir)
    X, cap = build_windows_channel0_lite(
        test_sites, CONTEXT_LENGTH, FORECAST_LENGTH, stride=24,
        capacities=capacities, max_samples=1, seed=seed, label="esp32-test-sample",
    )
    del test_sites

    X_cap = X / cap[:, None]  # (1, CONTEXT_LENGTH)
    z = (X_cap - TTM_NORM_MEAN) / TTM_NORM_STD

    past_values = z.reshape(1, 1, CONTEXT_LENGTH).astype(np.float32)
    freq_token = np.array([FREQ_TOKEN], dtype=np.int64)
    return past_values, freq_token


def get_expected_output(tflite_path, past_values, freq_token):
    """Run the float32 TTM TFLite model on host to get a known-correct
    expected output -- no quantization involved, unlike SmallTCN."""
    from ai_edge_litert.interpreter import Interpreter

    interpreter = Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    in_details = {d["name"]: d for d in interpreter.get_input_details()}
    out_details = interpreter.get_output_details()[0]

    pv_name = next(n for n in in_details if "past_values" in n)
    ft_name = next(n for n in in_details if "freq_token" in n)

    interpreter.set_tensor(in_details[pv_name]["index"], past_values)
    interpreter.set_tensor(in_details[ft_name]["index"], freq_token)
    interpreter.invoke()
    output = interpreter.get_tensor(out_details["index"])  # (1, FORECAST_LENGTH, 1) float32

    return output.flatten()


def write_test_vector_header(out_dir, past_values, freq_token, expected_output):
    path = os.path.join(out_dir, "test_vector.h")
    flat_input = past_values.flatten()

    with open(path, "w") as f:
        f.write("#pragma once\n#include <cstdint>\n\n")
        f.write("// Real preprocessed window pulled from the test split.\n")
        f.write("// TTM is deployed as FLOAT32 (see README: full int8, int16x8,\n")
        f.write("// and dynamic-range quantization were all tried and ruled out).\n")
        f.write("// No quantization: values below are real float32 model inputs/outputs.\n\n")
        f.write(f"constexpr int kTestContextLength = {CONTEXT_LENGTH};\n")
        f.write(f"constexpr int kTestForecastLength = {FORECAST_LENGTH};\n")
        f.write(f"constexpr int64_t kFreqToken = {int(freq_token[0])};\n\n")

        f.write("// past_values, TF-order layout (1, 1, 52) -- NOT (1, 52, 1).\n")
        f.write("// onnx2tf transposed this during ONNX -> TF conversion; confirm\n")
        f.write("// your input-preparation code on-device matches this exact layout.\n")
        f.write("const float kTestInput[] = {\n    ")
        f.write(", ".join(f"{v:.8f}f" for v in flat_input))
        f.write("\n};\n\n")

        f.write("// Expected output from running this exact input through the\n")
        f.write("// verified float32 .tflite model on host -- compare on-device\n")
        f.write("// output against this (float32, no dequantization needed).\n")
        f.write("const float kExpectedOutput[] = {\n    ")
        f.write(", ".join(f"{v:.8f}f" for v in expected_output))
        f.write("\n};\n")

    print(f"Wrote {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tflite_path", required=True)
    parser.add_argument("--calib_dir", default="./processed/test")
    parser.add_argument("--out_dir", default="esp32_deploy_files_ttm")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    tflite_to_c_array(args.tflite_path, args.out_dir)

    past_values, freq_token = get_real_ttm_test_sample(args.calib_dir)
    expected_output = get_expected_output(args.tflite_path, past_values, freq_token)
    write_test_vector_header(args.out_dir, past_values, freq_token, expected_output)

    print(f"\nDone. Files in {args.out_dir}/: model_data.h, model_data.cc, test_vector.h")
    print("\nNOTE: unlike SmallTCN's esp32_project, this float32 model needs the TFLite")
    print("Micro ops resolver to include Erf, LayerNorm, Softmax, and friends --")
    print("confirm esp-tflite-micro's AllOpsResolver (or an explicit op list) covers")
    print("these before assuming the firmware will even load the model, let alone run it.")
