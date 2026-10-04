"""
Convert the parity-verified TTM ONNX export to a TensorFlow Lite (float32)
model using onnx2tf's flatbuffer_direct path, with a hard parity check
against the ONNX model before trusting it.

onnx2tf is explicitly untested against this architecture (adaptive-patching
reshape path, see export_ttm_onnx.py's docstring on the dynamo=True finding).
Known onnx2tf risk areas, checked explicitly below rather than assumed:
  - Axis/channel-order transposes: onnx2tf's main source of silent bugs is
    inserting NCHW->NHWC transposes for 4D+ tensors. TTM's internal tensors
    are 4D (batch, nvars, num_patch, d_model) at points in the patch mixer,
    so this is a real risk here, not a theoretical one.
  - Output naming/ordering: onnx2tf may rename or reorder signature keys
    relative to the ONNX graph's output_names.
  - Static vs dynamic shapes: this ONNX model is static (batch=1), which
    actually works in onnx2tf's favor -- less for it to get wrong.

Order of operations:
  1. onnx_sanity_check   -- run the ONNX model, capture its output as the
                             reference. If this fails, the ONNX file itself
                             is suspect (shouldn't happen -- it's the file
                             export_ttm_onnx.py already verified -- but we
                             don't assume a file on disk is still the one
                             we think it is).
  2. convert              -- onnx2tf ONNX -> float32 TFLite (flatbuffer_direct)
  3. parity_check         -- run the TFLite model, compare to the ONNX
                             reference captured in stage 1. Hard gate.

Usage:
    python onnx_to_tf_ttm.py \
        --onnx_path exports/ttm_solar.onnx \
        --out_dir exports/ttm_solar_tf
"""

import argparse
import os

import numpy as np

CONTEXT_LENGTH = 52
FORECAST_LENGTH = 16
FREQ_TOKEN = 3


def onnx_sanity_check(onnx_path, n_samples=16):
    """Run the ONNX model standalone and capture its outputs as the
    reference for the parity check below. Returns (inputs_list, outputs_list)
    so the exact same inputs can be replayed against the TFLite model.
    """
    import onnxruntime as ort

    print(f"\n--- ONNX sanity check: {onnx_path} ---")
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_names = [i.name for i in sess.get_inputs()]
    output_names = [o.name for o in sess.get_outputs()]
    print(f"Inputs: {input_names}")
    print(f"Outputs: {output_names}")

    rng = np.random.default_rng(0)
    inputs_list, outputs_list = [], []
    for _ in range(n_samples):
        past_values = rng.standard_normal((1, CONTEXT_LENGTH, 1)).astype(np.float32)
        freq_token = np.full((1,), FREQ_TOKEN, dtype=np.int64)
        out = sess.run(None, {"past_values": past_values, "freq_token": freq_token})[0]
        inputs_list.append((past_values, freq_token))
        outputs_list.append(out)

    print(f"ONNX sanity check OK. Captured {n_samples} reference outputs, "
          f"shape {outputs_list[0].shape}.")
    return inputs_list, outputs_list


def convert(onnx_path, out_dir):
    """ONNX -> float32 TFLite via onnx2tf's flatbuffer_direct path.

    Deliberately uses onnx2tf's Python API (not the CLI via subprocess) so
    conversion warnings/errors surface in this process and the script can
    fail loudly rather than silently continuing past a problem buried in
    captured CLI output.

    flatbuffer_direct avoids an intermediate SavedModel, which is the
    intended pipeline for this project (see parity_check docstring).

    NOTE: parameter is ``tflite_backend``, not ``tflite_converter``.
    flatbuffer_direct is the default since onnx2tf v2.4.0, but we set it
    explicitly so the script's intent is unambiguous.
    """
    import onnx2tf

    print(f"\n--- Converting {onnx_path} -> {out_dir} (onnx2tf flatbuffer_direct) ---")
    os.makedirs(out_dir, exist_ok=True)
    onnx2tf.convert(
        input_onnx_file_path=onnx_path,
        output_folder_path=out_dir,
        copy_onnx_input_output_names_to_tflite=True,  # keep names aligned with ONNX
        tflite_backend="flatbuffer_direct",           # direct TFLite, no SavedModel
        non_verbose=False,  # see every transformation onnx2tf applies -- do not suppress this
    )
    print(f"Converted. Contents of {out_dir}:")
    for f in sorted(os.listdir(out_dir)):
        print(f"  {f}")


def _coerce_to_shape(arr, target_shape, label):
    """Reshape/transpose ``arr`` so it matches ``target_shape``.

    onnx2tf frequently inserts NCHW->NHWC-style transposes for 3D/4D
    tensors, so the shape the TFLite model actually expects can differ
    from the ONNX graph's shape by a permutation of the non-batch axes.
    If the shapes match exactly, return as-is. If they differ only by a
    swap of the last two axes (the common case), apply that transpose.
    Otherwise, fail loudly rather than guessing.
    """
    target_shape = tuple(int(d) for d in target_shape)
    if arr.shape == target_shape:
        return arr
    # Common onnx2tf case: last two axes swapped (e.g. (1,52,1) -> (1,1,52))
    if len(arr.shape) == len(target_shape) and len(arr.shape) >= 2:
        perm = list(range(len(arr.shape)))
        perm[-1], perm[-2] = perm[-2], perm[-1]
        if tuple(arr.shape[p] for p in perm) == target_shape:
            print(f"  NOTE: {label} shape {arr.shape} != TFLite expected "
                  f"{target_shape}; applying last-two-axes transpose.")
            return np.transpose(arr, perm)
    raise ValueError(
        f"{label} shape {arr.shape} does not match TFLite expected "
        f"{target_shape} and is not a simple last-two-axes swap. "
        f"onnx2tf may have inserted a different layout transform -- inspect "
        f"the TFLite input/output details and the tensor correspondence report."
    )


def parity_check(out_dir, reference_inputs, reference_outputs, atol=1e-3, rtol=1e-2):
    """Run the converted float32 TFLite model (produced directly by onnx2tf's
    flatbuffer_direct path -- no intermediate SavedModel in this version),
    compare against the ONNX reference captured earlier. Hard gate.
    """
    from ai_edge_litert.interpreter import Interpreter

    tflite_path = os.path.join(out_dir, "ttm_solar_float32.tflite")
    print(f"\n--- Parity check: TFLite (float32) vs ONNX reference ({tflite_path}) ---")

    interpreter = Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"TFLite inputs: {[(d['name'], d['shape'], d['dtype']) for d in input_details]}")
    print(f"TFLite outputs: {[(d['name'], d['shape'], d['dtype']) for d in output_details]}")

    max_abs_diff = 0.0
    scale = 0.0
    for (past_values, freq_token), onnx_out in zip(reference_inputs, reference_outputs):
        for detail in input_details:
            name = detail["name"]
            if "past_values" in name:
                pv = _coerce_to_shape(past_values, detail["shape"], "past_values")
                interpreter.set_tensor(detail["index"], pv.astype(detail["dtype"]))
            elif "freq_token" in name:
                ft = _coerce_to_shape(freq_token, detail["shape"], "freq_token")
                interpreter.set_tensor(detail["index"], ft.astype(detail["dtype"]))

        interpreter.invoke()
        tf_out = interpreter.get_tensor(output_details[0]["index"])

        # The TFLite output may itself be in a permuted layout relative to ONNX.
        tf_out = _coerce_to_shape(tf_out, onnx_out.shape, "prediction_outputs")

        max_abs_diff = max(max_abs_diff, float(np.max(np.abs(tf_out - onnx_out))))
        scale = max(scale, float(np.max(np.abs(onnx_out))))

    close = max_abs_diff <= atol + rtol * scale
    print(f"\nMax absolute difference (TFLite float32 vs ONNX): {max_abs_diff:.6f}")
    print(f"Within tolerance (atol={atol}, rtol={rtol}, scale={scale:.6f}): "
          f"{'YES' if close else 'NO -- STOP, DO NOT PROCEED to int8 conversion'}")
    if not close:
        raise RuntimeError(
            "Parity check FAILED. The float32 TFLite model does not match the "
            "ONNX model's outputs. Do not proceed to int8 quantization."
        )
    return max_abs_diff


def main(onnx_path, out_dir):
    inputs, outputs = onnx_sanity_check(onnx_path)
    convert(onnx_path, out_dir)
    parity_check(out_dir, inputs, outputs)
    print(f"\nDone. Next step: int8 TFLite conversion from {out_dir}, "
          f"same converter approach as SmallTCN -- verify carefully, this is still untested ground.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_path", default="exports/ttm_solar.onnx")
    parser.add_argument("--out_dir", default="exports/ttm_solar_tf")
    args = parser.parse_args()
    main(args.onnx_path, args.out_dir)
