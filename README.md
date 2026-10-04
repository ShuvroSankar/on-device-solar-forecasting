# On-Device Solar Power Forecasting Using Compressed Time-Series Models

MSCS thesis project, AIUB. Shuvro Sankar Sen (ID 25-93776-2). Supervisor: Dr. Rajarshi Roy Chowdhury.

A small dilated-causal-CNN forecaster (**SmallTCN**, 38K parameters) is trained on the public UK PV dataset
and deployed in int8 on two microcontrollers, **STM32F446RE** and **ESP32-C6**. In parallel, IBM's
**Tiny Time Mixer (TTM)**, a pretrained time-series foundation model, is being fine-tuned and compressed
for the same hardware, to compare a purpose-built small model against a pretrained one. The goal is
short-term (15 to 90 min) solar output forecasting on genuinely constrained hardware.

Full progress write-up: [`PROGRESS_REPORT.md`](PROGRESS_REPORT.md)

## Results (SmallTCN)

Forecasting accuracy (test set 2024, 703 sites; naive-persistence MAE is 12.31 Wh):

| Metric | SmallTCN |
|---|---|
| MASE | 0.808 |
| MAE / RMSE | 9.95 Wh / 22.25 Wh |
| sMAPE (daylight only) | 44.77% |

Deployment (int8, measured on hardware):

| | STM32F446RE | ESP32-C6 |
|---|---|---|
| Inference time | 18.2 ms | 97.7 ms |
| Model weights (flash) | 40.7 KiB | 100 KiB model file embedded |
| Working RAM | 16.4 KiB activations | 144 KiB arena (upper bound, see report) |
| Output vs reference | exact | exact |

Not yet measured: forecast accuracy of the int8 model on the full test set. See the report for
caveats and open items.

## TTM: status

A fine-tuned TTM-R2 checkpoint (70,972 parameters, context=52, forecast=16) exports cleanly to ONNX
with a verified parity check (max abs diff 0.000001 vs. PyTorch, 16 random samples), converts to a
float32 TFLite model via `onnx2tf` (parity vs. ONNX: max abs diff 0.000002), and quantizes via
dynamic-range quantization (int8 weights, float32 activations) to 375.8 KiB, 18% smaller than the
float32 model (relative max error 0.49% vs. ONNX, 200 real test-split windows). Full int8 and
int16x8 quantization were both attempted and both fail, for two distinct, isolated, documented
reasons -- see Key findings below. Still open: whether `esp-tflite-micro` supports dynamic-range
weight dequantization at inference, and the model's real flash footprint once embedded in an
ESP-IDF build.

### TTM checkpoints

Three fine-tuning runs are tracked, from separate invocations of `finetune_ttm.py`:

| Directory | Run | Status |
|---|---|---|
| `ttm_finetuned_models/` | default settings | loads correctly, not yet used downstream |
| `ttm_full_finetuned_models/` | full fine-tune, default LR | loads correctly, not yet used downstream |
| `ttm_full_finetuned_lr1e-4/` | full fine-tune, lr=1e-4 | **used by `export_ttm_onnx.py`; verified end-to-end (ONNX parity check passed)** |

If you're picking up this repo later: use `ttm_full_finetuned_lr1e-4/` unless you have a specific
reason to compare against the other two.

## Repo layout

```
preprocess_ukpv.py                       filter / clean / split the dataset (3 stages)
data_pipeline.py                         REFIT pipeline + Normalizer (reused)
solar_data_pipeline.py                   multi-site solar windowing, capacity loading
tcn_model.py                             SmallTCN (configurable input channels)
train_tcn.py                             original REFIT training (proxy baseline)
train_solar_tcn.py                       solar training (current model)
torch_to_tflite_tcn_manual_dilation.py   PyTorch -> Keras port, dilation rewritten as shifted
                                          1x1 convs, parity-checked -> int8 TFLite (final, SmallTCN)
test_minimal_quant.py                    minimal repro of the STM32 dilation issue
prepare_esp32_deployment.py              model + test vector as C arrays for the ESP32 firmware
esp32_project/                           ESP-IDF app (TFLite Micro)
models/                                  final int8 TFLite model (SmallTCN)
checkpoints/                             trained SmallTCN checkpoint

finetune_ttm.py                          fine-tune TTM-R2 on solar data
zeroshot_ttm.py                          zero-shot TTM baseline
export_ttm_onnx.py                       TTM -> ONNX, parity-checked (dynamo=True required -- see below)
evaluate_int8_accuracy.py                int8 vs fp32 accuracy comparison
ttm_full_finetuned_lr1e-4/               verified TTM checkpoint (see table above)
debug/check_patch_mixer_shapes.py        diagnostic: confirms TTM's adaptive-patching
                                          layer shapes are correct in eager mode
debug/check_ttm_checkpoint_paths.py      diagnostic: sanity-loads all three TTM checkpoints

onnx_to_tf_ttm.py                        TTM ONNX -> TF SavedModel / float32 TFLite (onnx2tf), parity-checked
build_ttm_calibration_data.py            real-data calibration windows (not synthetic noise),
                                          reproduces the exact train-fit Normalizer
memory_safe_windows.py                   two-pass window counting/sampling (avoids OOM on
                                          the full 6.7M-window train split)
convert_int8_manual.py                   full int8 TFLite conversion attempt -- fails, see below
inspect_int8_flatbuffer.py               raw flatbuffer weight-byte-by-dtype inspection
inspect_node70.py                        diagnostic: isolates the failing int8 DIV node
int8_parity_check.py / isolate_div_crash.py   confirm the int8 failure is structural, not data-dependent
convert_int16x8_or_fallback.py           int16x8 quantization attempt -- also fails, see below
convert_dynamic_range.py                 dynamic-range quantization -- works (deployment candidate)
test_quantized_model.py / verify_float32_runs.py   parity checks used across all TFLite variants
make_cind_calibration.py, make_freq_token_calibration.py   abandoned onnx2tf -oiqt/-cind
                                          approach, kept as a documented dead end (see below)
```

## Reproduce

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 1. dataset (5-minutely files only): huggingface_hub snapshot_download with
#    allow_patterns=['5_minutely/*','metadata.csv','bad_data.csv','README.md']
python3 preprocess_ukpv.py --stage filter
python3 preprocess_ukpv.py --stage split

# 2. train SmallTCN
python3 train_solar_tcn.py --stride 24

# 3. SmallTCN -> int8 TFLite (with PyTorch vs Keras parity check)
python3 torch_to_tflite_tcn_manual_dilation.py \
    --checkpoint checkpoints/small_tcn_solar.pt --out_prefix small_tcn_solar_manual

# 4. ESP32-C6 (separate terminal, ESP-IDF environment only, never pip install here)
cd esp32_project && . ~/esp/esp-idf/export.sh
idf.py set-target esp32c6 && idf.py build && idf.py -p <PORT> flash monitor

# 5. TTM -> ONNX (parity-checked)
python3 export_ttm_onnx.py \
    --checkpoint ttm_full_finetuned_lr1e-4/ttm_finetuned \
    --out_path exports/ttm_solar.onnx
```

STM32: import `models/small_tcn_solar_manual_int8.tflite` into STM32Cube AI Studio (type: TFLite,
target stm32f4), then Analyze / Generate / Validate on target.

## Key findings

**STM32 / SmallTCN:** ST Edge AI Core v4.0.1 (STM32F4 backend) runs dilated convolutions in float32
even when the model file is int8, so weights stayed near float size. Rewriting each dilated conv as
k shifted pointwise convs (identical math) fixes it. Details and the minimal repro are in the report.

**TTM ONNX export:** the legacy TorchScript tracer (`torch.onnx.export(..., dynamo=False)`) incorrectly
traces TTM's adaptive-patching reshape (`hidden.shape[2] * adaptive_patch_factor`), producing a graph
where the patch-mixer's traced activation shape disagrees with its own weight shape (exported output
declared `[1, 16, 13]` instead of the correct `[1, 16, 1]`; fails at ONNX Runtime session creation,
not at inference). Confirmed via `debug/check_patch_mixer_shapes.py`: the checkpoint and eager PyTorch
forward pass are correct (weight `in_features` matches the real runtime shape); the bug is isolated to
the tracer. Fix: `dynamo=True` (the `torch.export`-based exporter), which traces the reshape
symbolically. **Always export TTM with `dynamo=True`** -- the legacy tracer silently produces an
unusable graph for this architecture.

**TTM int8/int16 quantization: both fail, for two distinct, structural reasons.**

- *Full int8* (`TFLITE_BUILTINS_INT8`) converts and loads without error, but crashes on 100% of 200
  real test-split windows with `tflite/kernels/div.cc:242 data[i] != 0` at the same graph node,
  regardless of input variance (ruling out a data-dependent cause -- see `isolate_div_crash.py`).
  Root cause, confirmed via direct node/tensor inspection (`inspect_node70.py`,
  `inspect_int8_flatbuffer.py`): TTM's Erf/GELU activation decomposes into a rational approximation
  (`abs`/`sign`/`exp`/`div`/`rsqrt`), and the divisor tensor's real dynamic range (~0 to 7.4) is too
  wide for int8's 256 levels -- about 77% of its real values quantize down to the exact zero-point,
  which dequantizes to literal `0.0`. TFLite's int8 `DIV` kernel correctly refuses to divide by a
  zero-point code. This is structural to the architecture (33 DIV nodes, one per Erf instance, all
  fail identically), not fixable with more or better calibration data.
- *int16x8* (`ACTIVATIONS_INT16_WEIGHTS_INT8`, meant to fix the above by giving activations 65,536
  levels instead of 256) fails for a different reason: the TFLite converter cannot calibrate a
  `Cast` op's min/max range (`Empty min/max for tensor Cast`), caused by `freq_token`'s int64 path.
  This is a known TFLite converter limitation, not something a better representative dataset fixes.
- *Working alternative:* dynamic-range quantization (int8 weights, float32 activations, no
  calibration data needed) sidesteps both failure modes, since activations are never quantized.
  375.8 KiB vs. 458.7 KiB float32 (18% smaller -- more modest than SmallTCN's 73%, since only
  weights shrink). Parity vs. ONNX: max abs diff 0.024185, relative max error 0.49%, Pearson
  0.999996 (200 real windows) -- real quantization error, not bit-exact like SmallTCN's int8.
- *Abandoned path, kept for the record:* `onnx2tf`'s own `-oiqt`/`-cind` quantization flags were
  tried first and abandoned -- `-cind` requires float32 calibration data for every graph input, with
  no documented way to supply a non-quantized integer input like `freq_token`. Switched to a manual
  `tf.lite.TFLiteConverter` script instead, which handles mixed input dtypes correctly.

## Status

- [x] Dataset pipeline
- [x] SmallTCN trained on solar data
- [x] int8 deployment on STM32F446RE and ESP32-C6
- [x] TTM fine-tuned and exported to ONNX (parity-checked)
- [x] TTM: ONNX -> TensorFlow -> TFLite, with a working quantized candidate (dynamic-range,
      375.8 KiB, 0.49% relative error vs. ONNX); full int8 and int16x8 both fail for documented
      structural reasons (see Key findings)
- [ ] int8 accuracy of SmallTCN on the full test set
- [ ] Confirm `esp-tflite-micro` supports dynamic-range quantized ops; measure TTM's real flash
      footprint once embedded in the ESP-IDF build; deploy on ESP32-C6
- [ ] Hardware benchmark vs published works (TinyHAR-Net)
- [ ] Accuracy comparison vs published cloud-side PV forecasting models
- [ ] MQTT network cost and live dashboard (needs solar panel)
- [ ] Repo cleanup: remove superseded/debug scripts (`export_quantize_tcn.py`,
      `export_quantize_tcn_v2.py`, `torch_to_tflite_tcn.py`, `*_debug.py`, `*.broken`)
