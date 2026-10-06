"""
Export the fine-tuned TTM-R2 checkpoint (TinyTimeMixerForPrediction) to ONNX,
with a hard parity check against the PyTorch model before trusting the export
-- same discipline as the SmallTCN PyTorch-to-Keras port.

This is step 1 of the ESP32 deployment path for TTM:
  1. (this script) PyTorch -> ONNX, parity-checked
  2. ONNX -> TensorFlow SavedModel, via onnx2tf (separate step -- genuinely
     untested against this architecture, verify carefully once there)
  3. TensorFlow -> int8 TFLite (same converter approach as SmallTCN)
  4. Deploy via the same ESP-IDF/TFLite Micro project as SmallTCN

Unlike SmallTCN, there's no feasible hand-port to Keras for a model this
size/complexity -- onnx2tf is a reasonable, standard choice, but untested
here, so steps 2-3 should be verified carefully once we get there rather
than assumed to work.

Note on export shape: the ONNX graph is exported STATIC (batch=1), matching
the deployment target -- TFLite Micro does not support dynamic batch, and
tracing with batch=1 while declaring dynamic_axes can silently bake a
literal `1` into internal reshape/view calls, producing a graph that only
runs at batch=1 anyway (and fails loudly at batch=N). So we don't pretend
it's dynamic. See parity_check for the batch-1 loop.

Note on checkpoint sanity: before doing anything ONNX-related, we run the
loaded PyTorch model eagerly on a single batch-1 input. This isolates
checkpoint/config problems (e.g. a config whose `num_patches` doesn't match
what `context_length / patch_length` actually produces, or mixer weights
sized for a different patch count than the input pipeline computes) from
export/tracing problems. If the eager forward pass fails, the ONNX step
never runs -- no point tracing a model that can't even run in eager mode.

Order of operations in main():
  1. load checkpoint
  2. checkpoint_sanity_check  (eager PyTorch forward -- is the checkpoint OK?)
  3. export to ONNX
  4. onnx_shape_check         (did the export produce sane, resolved shapes?)
  5. parity_check             (does the ONNX output match PyTorch numerically?)
Each stage fails loudly and stops the run -- nothing downstream is trusted
if something upstream is broken.

Usage:
    python export_ttm_onnx.py --checkpoint ttm_full_finetuned_lr1e-4/ttm_finetuned \
        --out_path exports/ttm_solar.onnx
"""

import argparse
import os

import numpy as np
import torch
from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction

CONTEXT_LENGTH = 52
FORECAST_LENGTH = 16
FREQ_TOKEN = 3


class TTMExportWrapper(torch.nn.Module):
    """Returns only prediction_outputs as a plain tensor, so the ONNX
    graph has exactly one output matching output_names.

    TinyTimeMixerForPrediction.forward() returns a ModelOutput dataclass,
    which the tracer flattens into six graph outputs (prediction_outputs,
    hidden states, loc, scale, ...). Declaring a single output_name then
    breaks the ONNX export's output-count expectation.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, past_values, freq_token):
        return self.model(
            past_values=past_values,
            freq_token=freq_token,
        ).prediction_outputs


def checkpoint_sanity_check(model):
    """Run the loaded PyTorch model eagerly, no ONNX, no tracing.

    Purpose: isolate checkpoint/config problems from export problems. If
    the eager forward pass fails (shape mismatch inside the patch mixer,
    etc.), the checkpoint itself is inconsistent -- e.g. `num_patches` in
    config.json disagreeing with `context_length / patch_length`, or mixer
    weights saved for a different patch count than the input pipeline
    computes. No amount of export-side fiddling fixes that; the fix is on
    the checkpoint side.

    If this succeeds but the ONNX export then fails, the problem is in
    tracing/export, and we look there instead.
    """
    print("\n--- Checkpoint sanity check (eager PyTorch, no ONNX) ---")
    model.eval()
    past_values = torch.randn(1, CONTEXT_LENGTH, 1)
    freq_token = torch.tensor([FREQ_TOKEN], dtype=torch.int32)
    with torch.no_grad():
        out = model(past_values=past_values, freq_token=freq_token)
    pred = out.prediction_outputs
    print(f"Eager forward OK. prediction_outputs shape: {tuple(pred.shape)}")
    expected = (1, FORECAST_LENGTH, 1)
    if tuple(pred.shape) != expected:
        print(
            f"WARNING: expected shape {expected} but got {tuple(pred.shape)}. "
            f"Continuing, but treat downstream results with suspicion -- this "
            f"may indicate a config/checkpoint mismatch."
        )


def onnx_shape_check(onnx_path):
    """Run ONNX shape inference on the just-exported graph and print the
    declared shapes of all graph inputs and outputs.

    This is diagnostic, not a gate: shape inference in non-strict mode may
    legitimately leave some intermediate dims unresolved. What we care about
    is that the four boundary tensors -- past_values, freq_token, and
    prediction_outputs -- have shapes matching what the ESP32-side code will
    assume: [1, CONTEXT_LENGTH, 1], [1], [1, FORECAST_LENGTH, 1].

    Uses non-strict mode deliberately: strict_mode raises on any inference
    hiccup, including benign ones, and we do not want an import-time or
    export-time crash from a diagnostic step. parity_check is the hard gate.
    """
    import onnx
    from onnx import shape_inference

    m = onnx.load(onnx_path)
    try:
        inferred = shape_inference.infer_shapes(m, strict_mode=False)
    except Exception as e:
        print(f"\nWARNING: ONNX shape inference raised: {e}")
        print("Continuing -- parity_check is the authoritative gate.")
        return

    def dims_of(value_info):
        return [
            d.dim_value if d.HasField("dim_value") else (d.dim_param or "?")
            for d in value_info.type.tensor_type.shape.dim
        ]

    print("\n--- ONNX shape inference (boundary tensors) ---")
    for vi in list(inferred.graph.input) + list(inferred.graph.output):
        print(f"  {vi.name}: {dims_of(vi)}")


def parity_check(model, onnx_path, n_samples=16, atol=1e-3, rtol=1e-2):
    """Run n_samples individual (batch=1) inputs through the PyTorch model
    and the exported ONNX model, require outputs to match closely.

    We loop over batch-1 samples rather than feeding one batched call: the
    ONNX graph is exported static (batch=1), so a batched call would fail
    internally at the first reshape that has `1` baked in. If this fails,
    STOP -- do not proceed to the TFLite conversion step on a broken export.
    """
    import onnxruntime as ort

    rng = np.random.default_rng(0)
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    print(f"ONNX model inputs: {[i.name for i in sess.get_inputs()]}")

    model.eval()
    max_abs_diff = 0.0
    scale = 0.0
    for _ in range(n_samples):
        past_values = rng.standard_normal((1, CONTEXT_LENGTH, 1)).astype(np.float32)
        freq_token = np.full((1,), FREQ_TOKEN, dtype=np.int32)

        with torch.no_grad():
            torch_pred = model(
                past_values=torch.tensor(past_values),
                freq_token=torch.tensor(freq_token),
            ).prediction_outputs.numpy()

        onnx_pred = sess.run(None, {"past_values": past_values, "freq_token": freq_token})[0]
        max_abs_diff = max(max_abs_diff, float(np.max(np.abs(torch_pred - onnx_pred))))
        scale = max(scale, float(np.max(np.abs(torch_pred))))

    close = max_abs_diff <= atol + rtol * scale

    print(f"\n--- Parity check (PyTorch vs ONNX, {n_samples} random batch-1 inputs) ---")
    print(f"Max absolute difference: {max_abs_diff:.6f}")
    print(f"Within tolerance (atol={atol}, rtol={rtol}, scale={scale:.6f}): "
          f"{'YES' if close else 'NO -- STOP, DO NOT PROCEED'}")
    if not close:
        raise RuntimeError(
            "Parity check FAILED. The ONNX export does not match the PyTorch "
            "model's outputs. Do not proceed to TFLite conversion on this export."
        )
    return max_abs_diff


def main(checkpoint_path, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print(f"Loading {checkpoint_path} ...")
    model = TinyTimeMixerForPrediction.from_pretrained(checkpoint_path)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded. Parameters: {n_params:,}")

    # Stage 1: is the checkpoint internally consistent? If not, stop here --
    # the fix is on the checkpoint side, not the export side.
    checkpoint_sanity_check(model)

    dummy_past_values = torch.randn(1, CONTEXT_LENGTH, 1)
    dummy_freq_token = torch.tensor([FREQ_TOKEN], dtype=torch.int32)

    # Wrap so only prediction_outputs is traced into the ONNX graph.
    export_model = TTMExportWrapper(model)
    export_model.eval()

    # Stage 2: export. Static (batch=1); no dynamic_axes -- see module docstring.
    print(f"\nExporting to ONNX -> {out_path}")
    torch.onnx.export(
        export_model,
        (dummy_past_values, dummy_freq_token),
        out_path,
        input_names=["past_values", "freq_token"],
        output_names=["prediction_outputs"],
        opset_version=17,
        dynamo=True,  # REQUIRED for TTM: the legacy JIT tracer (dynamo=False) bakes
              # TinyTimeMixerAdaptivePatchingBlock's shape-arithmetic reshape
              # (hidden.shape[2] * adaptive_patch_factor, etc.) into incorrect
              # literal constants, producing a graph where mixers.0's traced
              # activation shape disagrees with its own weight shape
              # (observed: declared output [1,16,13] instead of [1,16,1],
              # ONNX Runtime failing at session-creation shape inference).
              # The torch.export-based exporter (dynamo=True) traces this
              # symbolically and is correct; verified via eager-mode hook
              # confirming weight in_features (14) matches runtime shape (14),
              # ruling out the checkpoint/config as the cause.
    )
    onnx_kb = os.path.getsize(out_path) / 1024
    print(f"Exported. File size: {onnx_kb:.1f} KB")

    # Stage 3: diagnostic -- did the exported graph get sane boundary shapes?
    # Non-fatal by design; parity_check below is the authoritative gate.
    onnx_shape_check(out_path)

    # Stage 4: hard gate -- does the ONNX output actually match PyTorch?
    # Pass the ORIGINAL model (not the wrapper); parity_check reads
    # torch_out.prediction_outputs from the ModelOutput.
    parity_check(model, out_path)

    print(f"\nDone. Next step: convert {out_path} to a TensorFlow SavedModel via onnx2tf, "
          f"then to int8 TFLite -- verify each stage before trusting it.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out_path", default="exports/ttm_solar.onnx")
    args = parser.parse_args()
    main(args.checkpoint, args.out_path)
