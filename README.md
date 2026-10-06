# On-Device Solar Power Forecasting Using Compressed Time-Series Models

MSCS thesis project, AIUB. Shuvro Sankar Sen (ID 25-93776-2). Supervisor: Dr. Rajarshi Roy Chowdhury.

A small dilated-causal-CNN forecaster (**SmallTCN**, 38K parameters) is trained on the public UK PV dataset
and deployed in int8 on two microcontrollers, **STM32F446RE** and **ESP32-C6**. In parallel, IBM's
**Tiny Time Mixer (TTM)**, a pretrained time-series foundation model, is fine-tuned and studied for
the same hardware, to compare a purpose-built small model against a pretrained one. The goal is
short-term (15 to 90 min) solar output forecasting on genuinely constrained hardware.

Full progress write-up: [`PROGRESS_REPORT.md`](PROGRESS_REPORT.md)

## Results (SmallTCN)

Forecasting accuracy on the full test set (2024, 703 sites, 969,445 windows;
naive-persistence MAE is 12.31 Wh). The fp32 column is the trained PyTorch model;
the int8 column is the deployed TFLite model that actually runs on both MCUs.

| Metric | SmallTCN fp32 | SmallTCN int8 (deployed) |
|---|---|---|
| MASE | 0.808 | **0.846** |
| MAE / RMSE | 9.95 Wh / 22.25 Wh | 10.42 Wh / 22.84 Wh |
| sMAPE (daylight only) | 44.77% | 47.06% |

**Quantization cost:** int8 deployment costs **+0.038 MASE** (+4.7% relative),
+0.47 Wh MAE, +2.29 percentage-points sMAPE (daylight). Daylight-only cosine
similarity between int8 and fp32 predictions is **0.971** — computed on the
538,069 test windows with actual generation > 1 Wh, since the all-windows
cosine (0.745) is dominated by numerical noise on near-zero night-time
vectors. An earlier 20-window host check reported cosine 0.998 / normalized
MAE 0.025; that sample was too small to estimate the tail of the quantization
error distribution and overstated fidelity — the full-test numbers above are
the honest figures. The int8 cost is at the higher end of typical for a TCN,
likely because the 38K-parameter model has less redundancy to absorb
quantization noise and the manual dilation rewrite introduces more
quantization breakpoints than a fused dilated op.

Deployment (int8, measured on hardware):

| | STM32F446RE | ESP32-C6 |
|---|---|---|
| Inference time | 18.2 ms | 97.7 ms |
| Model weights (flash) | 40.7 KiB | 100 KiB model file embedded |
| Working RAM | 16.4 KiB activations | 144 KiB arena (upper bound, see report) |
| Output vs reference | exact | exact |

## TTM: status

A fine-tuned TTM-R2 checkpoint (70,972 parameters, context=52, forecast=16) exports cleanly to ONNX
with a verified parity check (max abs diff 0.000001 vs. PyTorch, 16 random samples), and converts to
a float32 TFLite model via `onnx2tf` (parity vs. ONNX: max abs diff 0.000002, Pearson 1.000000 on
200 real test windows). **Deployment to ESP32-C6 was attempted and is blocked -- TTM does not fit on
the target in its current form.** The investigation reached a well-isolated, evidence-backed
negative result. Full report: [`thesis_artifacts/ttm_esp32/REPORT.md`](thesis_artifacts/ttm_esp32/REPORT.md).

### What worked

The conversion chain is verified end-to-end:
- PyTorch → ONNX, parity max abs diff **1e-6**
- ONNX → TFLite float32 via `onnx2tf`, parity max abs diff **2e-6**
- TFLite loads on the ESP32-C6, allocates a **160,152 / 245,760 byte** tensor arena, correctly
  wires the two inputs (`past_values` float32 (1,1,52), `freq_token` int64 (1,)) and one output
  (float32 (1,16,1)), and executes the first ~8 operations of the graph cleanly

### What failed on-device

**The model aborts inside `tflite-micro`'s greedy memory planner** at the 3rd `GATHER` node, with
a reproducible RISC-V store/AMO access fault (`MCAUSE=0x7`). Debug instrumentation confirmed that
the same node with the same tensor pointers succeeds on the 1st invocation and fails on the 3rd
within a single `Invoke()` -- a lifetime-tracking bug triggered by TTM's 1600-tensor graph.

**The linear planner** (which avoids the greedy planner's buffer-reuse logic entirely) requires
**2.9 MB of activation memory** for the same graph -- **~5.7x the ESP32-C6's total 512 KB SRAM**.
TTM fits only because the greedy planner's lifetime-based buffer reuse compresses the 2.9 MB
worst-case footprint down to 160 KB.

**Three quantization approaches were tried, all three ruled out:**

- *Full int8* (`TFLITE_BUILTINS_INT8`) converts but crashes on 100% of 200 real test-split windows
  with `tflite/kernels/div.cc:242 data[i] != 0` at the same five DIV nodes, regardless of input
  variance. Root cause: TTM's Erf/GELU activation decomposes into a rational approximation
  (`abs`/`sign`/`exp`/`div`/`rsqrt`), and the divisor tensor's dynamic range (~0 to 7.4) is too
  wide for int8's 256 levels -- ~77% of its activations quantize to the exact zero-point.
  Confirmed pervasive (TF's own `QuantizationDebugger` crashes at a *different* DIV node on its
  own traversal, plus `NaN`/`Inf` statistics at multiple layers).
- *int16x8* (`ACTIVATIONS_INT16_WEIGHTS_INT8`, meant to fix the above) fails at conversion --
  `freq_token`'s int64 path forces a `Cast`, and the int16 calibrator cannot record min/max for
  integer-valued Cast outputs.
- *Dynamic-range* (int8 weights, float32 activations) converts cleanly and passes parity
  (375.8 KiB, 18% smaller than float32; relative max error 0.49%) but **TFLite Micro does not
  support dynamic-range quantization at all** -- it only ships kernels for pure float32 or
  full-integer inference.

**Conclusion:** TTM resists int8 quantization structurally (Erf/GELU dynamic range), is blocked by
a TFLite converter limitation at int16x8 (Cast calibration), and -- even in float32 -- does not fit
on the ESP32-C6 in its current form because the model requires the greedy memory planner (2.9 MB
linear footprint vs. 512 KB SRAM) and that planner has a reproducible lifetime-tracking bug on TTM's
1600-tensor graph. The finding is documented in `thesis_artifacts/ttm_esp32/` with the exact tensor
pointers, arena size, register dump, and every kernel modification made during investigation.

### Kernel patches required (for the record)

TTM's TFLite graph uses ops that `esp-tflite-micro` 1.4.1 either does not ship or does not
dispatch on int64. Seven kernel modifications were made to get the model loading and executing:

| File | Change |
|---|---|
| `sign.cc` | New SIGN kernel + `ParseSign` no-op parser (was missing upstream) |
| `select_extra.cc` | New `Register_SELECT` wrapper around `Register_SELECT_V2` |
| `cast.cc` | int64 input and output branches |
| `select.cc` | int64 coords dispatch |
| `add.cc` / `esp_nn/add.cc` | int64 branch (esp-nn variant is the one actually linked) |
| `gather.cc` | int64 coords dispatch |
| `micro_mutable_op_resolver.h` | `AddSign`, `AddSelect`, `ParseSign` forward declarations |

Full patched sources in `thesis_artifacts/ttm_esp32/patched_kernels/`.

### TTM checkpoints

Three fine-tuning runs are tracked, from separate invocations of `finetune_ttm.py`:

| Directory | Run | Status |
|---|---|---|
| `ttm_finetuned_models/` | default settings | loads correctly, not yet used downstream |
| `ttm_full_finetuned_models/` | full fine-tune, default LR | loads correctly, not yet used downstream |
| `ttm_full_finetuned_lr1e-4/` | full fine-tune, lr=1e-4 | **used by `export_ttm_onnx.py`; verified end-to-end** |

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
evaluate_int8_accuracy.py                full-test-set int8 vs fp32 accuracy comparison
esp32_project/                           ESP-IDF app (TFLite Micro)
models/                                  final int8 TFLite model (SmallTCN)
checkpoints/                             trained SmallTCN checkpoint

finetune_ttm.py                          fine-tune TTM-R2 on solar data
zeroshot_ttm.py                          zero-shot TTM baseline
export_ttm_onnx.py                       TTM -> ONNX, parity-checked (dynamo=True required -- see below)
ttm_full_finetuned_lr1e-4/               verified TTM checkpoint (see table above)
debug/check_patch_mixer_shapes.py        diagnostic: confirms TTM's adaptive-patching
                                          layer shapes are correct in eager mode
debug/check_ttm_checkpoint_paths.py      diagnostic: sanity-loads all three TTM checkpoints

onnx_to_tf_ttm.py                        TTM ONNX -> TF SavedModel / float32 TFLite (onnx2tf), parity-checked
build_ttm_calibration_data.py            real-data calibration windows (not synthetic noise),
                                          reproduces the exact train-fit Normalizer
memory_safe_windows.py                   two-pass window counting/sampling (avoids OOM on
                                          the full 6.7M-window train split)

# TTM quantization investigation (see Key findings; all preserved for the record)
convert_int8_manual.py                   full int8 TFLite conversion attempt -- fails, see below
inspect_int8_flatbuffer.py               raw flatbuffer weight-byte-by-dtype inspection
inspect_node70.py                        diagnostic: isolates the failing int8 DIV node
int8_parity_check.py / isolate_div_crash.py   confirm the int8 failure is structural, not data-dependent
convert_int16x8_or_fallback.py           int16x8 quantization attempt -- also fails, see below
convert_dynamic_range.py                 dynamic-range quantization -- works numerically, unsupported by TFLM
test_quantized_model.py / verify_float32_runs.py   parity checks used across all TFLite variants
make_cind_calibration.py, make_freq_token_calibration.py   abandoned onnx2tf -oiqt/-cind
                                          approach, kept as a documented dead end
quant_debug_erf.py                       TF QuantizationDebugger run -- confirms the int8 DIV
                                          failure is pervasive, not a single fixable node

# TTM ESP32-C6 deployment investigation (see thesis_artifacts/ttm_esp32/)
main_ttm.cpp                             TTM firmware: two-input handling, de-norm, timing loop
list_ttm_ops.py                          op-set audit for the TFLite graph
prepare_esp32_deployment_ttm.py          model + test vector as C arrays for TTM firmware

thesis_artifacts/ttm_esp32/              Full TTM-on-ESP32 investigation archive
  REPORT.md                                detailed technical write-up
  main_ttm.cpp                             TTM firmware source
  model_data.cc.snapshot                   TTM model as embedded C array (469,696 bytes)
  patched_kernels/                         all kernel modifications (7 files + resolver header)
  ttm_ge.log, ttm_steps.log, ttm_data.log  on-device boot traces with debug instrumentation
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

# 4. int8 accuracy on the full test set (969,445 windows)
python3 evaluate_int8_accuracy.py

# 5. ESP32-C6 (separate terminal, ESP-IDF environment only, never pip install here)
cd esp32_project && . ~/esp/esp-idf/export.sh
idf.py set-target esp32c6 && idf.py build && idf.py -p <PORT> flash monitor

# 6. TTM -> ONNX (parity-checked)
python3 export_ttm_onnx.py \
    --checkpoint ttm_full_finetuned_lr1e-4/ttm_finetuned \
    --out_path exports/ttm_solar.onnx

# 7. TTM ONNX -> TFLite float32 (parity-checked)
python3 onnx_to_tf_ttm.py --onnx_path exports/ttm_solar.onnx --out_dir exports/ttm_solar_tf
```

To reproduce the TTM-on-ESP32 deployment attempt (blocks at the greedy memory planner):

```bash
# See thesis_artifacts/ttm_esp32/REPORT.md, section 6 "Reproducibility"
```

## Key findings

**STM32 / SmallTCN:** ST Edge AI Core v4.0.1 (STM32F4 backend) runs dilated convolutions in float32
even when the model file is int8, so weights stayed near float size. Rewriting each dilated conv as
k shifted pointwise convs (identical math) fixes it. Details and the minimal repro are in the report.

**SmallTCN int8 accuracy cost:** deployment in int8 costs +0.038 MASE (0.808 → 0.846, +4.7%
relative), +0.47 Wh MAE (9.95 → 10.42 Wh), and +2.29 pp sMAPE (daylight, 44.77% → 47.06%), measured
on the full 969,445-window test set. Daylight-only cosine similarity vs fp32 predictions is 0.971.
This is at the higher end of typical for a TCN, likely due to the small parameter count (less
redundancy to absorb quantization noise) and the manual dilation rewrite (more quantization
breakpoints than a fused dilated op). The deployed int8 model is bit-exact to its own host
reference; the accuracy cost is inherent to the quantization step, not to the deployment.

**TTM ONNX export:** the legacy TorchScript tracer (`torch.onnx.export(..., dynamo=False)`) incorrectly
traces TTM's adaptive-patching reshape (`hidden.shape[2] * adaptive_patch_factor`), producing a graph
where the patch-mixer's traced activation shape disagrees with its own weight shape (exported output
declared `[1, 16, 13]` instead of the correct `[1, 16, 1]`; fails at ONNX Runtime session creation,
not at inference). Confirmed via `debug/check_patch_mixer_shapes.py`. Fix: `dynamo=True` (the
`torch.export`-based exporter), which traces the reshape symbolically. **Always export TTM with
`dynamo=True`** -- the legacy tracer silently produces an unusable graph for this architecture.

**TTM quantization: three approaches tried, all three ruled out.**

- *Full int8* converts and loads but crashes on 100% of test windows with `data[i] != 0` at five
  fixed DIV nodes. Root cause: TTM's Erf/GELU activation decomposes into a rational approximation
  whose divisor tensor (~0-7.4 dynamic range) loses ~77% of its values to the int8 zero-point. The
  int8 `DIV` kernel correctly refuses to divide by a zero-point code. Not data-dependent, not
  single-node-fixable -- confirmed independently by TF's `QuantizationDebugger` crashing at a
  *different* DIV node during its own traversal, and flagging `NaN`/`Inf` at multiple layers.
  Rewriting the activation (like the STM32 dilation fix) is out of scope -- no known equivalent
  rewrite of Erf's rational approximation avoids the same dynamic-range problem.
- *int16x8* fails at conversion -- `freq_token`'s int64 path forces a `Cast`, and the int16
  calibrator can't record min/max for integer-valued Cast outputs. TFLite converter limitation.
- *Dynamic-range* (int8 weights, float32 activations) is numerically clean and passes parity
  (relative max error 0.49%, Pearson 0.999996) but **TFLite Micro does not support dynamic-range
  quantization by design** -- it only ships kernels for pure float32 or full-integer inference.
- *The `-oiqt`/`-cind` onnx2tf path was tried first and abandoned*: `-cind` requires float32
  calibration for every graph input, with no documented way to supply a non-quantized integer
  input like `freq_token`. A manual `tf.lite.TFLiteConverter` script handles mixed dtypes correctly.

**TTM does not fit on the ESP32-C6 in float32 either, once actually deployed.**

- TTM loads, allocates 160 KB / 245 KB arena, and executes the first ~8 ops cleanly.
- The model then aborts inside `tflite-micro`'s greedy memory planner at the 3rd Gather node --
  same tensor pointers on the 1st and 3rd invocations, but a RISC-V store/AMO access fault on the
  3rd. Reproducible lifetime-tracking bug triggered by TTM's 1600-tensor graph.
- The linear planner (avoids the bug) needs **2.9 MB** of activation memory -- **5.7x** the
  ESP32-C6's 512 KB SRAM. TTM fits at all only because the greedy planner's lifetime-based
  buffer reuse compresses the 2.9 MB worst-case footprint down to 160 KB.
- TTM's TFLite graph uses several ops that `esp-tflite-micro` 1.4.1 does not ship or does not
  dispatch on int64. Seven kernel patches were required just to reach the point of running.
- **Conclusion: TTM would need model compression (reduced context length, reduced d_model, or
  pruning) to fit on the ESP32-C6.** The specific failure modes are documented in
  `thesis_artifacts/ttm_esp32/REPORT.md` and are architecturally distinct from the int8 issue --
  even a perfectly quantizable TTM would still not fit in float32 on this chip.

**Architectural contrast worth stating plainly:** SmallTCN (purpose-built, no GELU-family
activations) quantizes to int8 cleanly after one targeted toolchain fix and runs at 97.7 ms on the
ESP32-C6, at a measured accuracy cost of +4.7% MASE. TTM (a pretrained foundation model with
Erf/GELU throughout) resists int8 structurally across three independent approaches, and -- even in
float32 -- does not fit on the same hardware without model compression. That is the thesis result.

## Status

- [x] Dataset pipeline
- [x] SmallTCN trained on solar data
- [x] int8 deployment on STM32F446RE and ESP32-C6
- [x] int8 accuracy of SmallTCN on the full test set (MASE 0.846 vs fp32 0.808)
- [x] TTM fine-tuned and exported to ONNX (parity-checked)
- [x] TTM: ONNX → TFLite float32 (parity-checked)
- [x] TTM: full int8, int16x8, and dynamic-range quantization tried and ruled out, each for a
      distinct confirmed reason (see Key findings)
- [x] TTM: ESP32-C6 deployment attempted; model loads and executes but aborts in the greedy
      memory planner; linear-planner footprint 2.9 MB (5.7x SRAM). **Does not fit as-is.**
- [x] Full TTM-on-ESP32 investigation archived in `thesis_artifacts/ttm_esp32/`
- [ ] TTM model compression experiment (context=26 or reduced d_model) -- the obvious next step
      toward making TTM fit
- [ ] Hardware benchmark vs published works (TinyHAR-Net)
- [ ] Accuracy comparison vs published cloud-side PV forecasting models
- [ ] MQTT network cost and live dashboard (needs solar panel)
- [ ] Repo cleanup: remove superseded/debug scripts (`export_quantize_tcn.py`,
      `export_quantize_tcn_v2.py`, `torch_to_tflite_tcn.py`, `*_debug.py`, `*.broken`)
