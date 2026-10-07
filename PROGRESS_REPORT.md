# Progress Report — On-Device Solar Power Forecasting

**Student:** Shuvro Sankar Sen (ID 25-93776-2), AIUB
**Supervisor:** Dr. Rajarshi Roy Chowdhury
**Repo:** `<paste GitHub URL here>`
**Covers:** dataset pipeline → SmallTCN training → int8 deployment on STM32F446RE and ESP32-C6 → TTM investigation

---

## 1. Summary

- A **SmallTCN** forecaster (38K parameters) is trained on the public UK PV dataset and **runs on both boards** (STM32F446RE and ESP32-C6) in int8, with on-device output matching the reference model **bit-exactly**.
- On the test set it beats naive persistence: **fp32 MASE 0.808; deployed int8 MASE 0.846** — i.e. int8 quantization costs +0.038 MASE (+4.7% relative, MAE 9.95 → 10.42 Wh) on the full 969,445-window test set. That cost is now measured, not estimated (§5.3).
- Getting a genuinely compressed int8 model onto the STM32 took a long investigation. The root cause was a **toolchain limitation with dilated convolutions**, fixed with a mathematically equivalent rewrite. After the fix: **40.7 KiB of weights, 18.2 ms per inference**.
- The ESP32-C6 deployment initially produced *wrong* outputs. A debugging pass fixed it (details in §5.4), but **the RAM figure there is inflated** and the root-cause attribution is still unconfirmed.
- **TTM (Tiny Time Mixer) was fine-tuned, exported, converted, and attempted on the ESP32-C6.** The conversion chain (PyTorch → ONNX → TFLite float32) is fully parity-verified. Three quantization approaches (int8, int16x8, dynamic-range) were each tried and each ruled out for a distinct, confirmed, architectural reason (§5.5). The resulting float32 model was deployed to real ESP32-C6 hardware, where it loaded and began executing before faulting inside the `tflite-micro` greedy memory planner; the fallback (linear) planner would need roughly 2.9 MB of activation memory against the chip's 512 KB SRAM. The C6 also has **no hardware FPU**, so float32 inference there would be software-emulated even if it fit. **TTM does not fit on the ESP32-C6 in its current form** (§5.6). This is a complete, evidence-backed negative result, not an unfinished one. SmallTCN remains the deployed model on both boards.
- **Not yet done:** the ESP32 buffer-reuse ablation, the hardware and accuracy comparison tables against published works (extraction in progress, §7), MQTT/dashboard work (waiting on the solar panel), and adaptive updating (TEDA-RLS).

---

## 2. Status against the proposal

| Item (proposal timeline) | Status |
|---|---|
| Solar dataset pipeline (Months 1–2) | ✅ Done |
| Retrain SmallTCN on solar data (Months 1–2) | ✅ Done (fp32 MASE 0.808) |
| Quantize + deploy SmallTCN on both boards (Months 2–3) | ✅ Done (both boards, verified; int8 accuracy cost measured at +0.038 MASE) |
| Quantize + deploy TTM (Months 2–3) | 🟡 Fine-tuned, exported, converted, and attempted on hardware. Does not fit on ESP32-C6 (§5.5–5.6) — documented negative result, not pending work |
| Hardware benchmark vs TinyHAR-Net (Month 4) | ✅ SmallTCN vs TinyHAR-Net compared on the same STM32F446RE; see §7 and `thesis_artifacts/hardware_comparison_tinyhar.md`. TTM excluded (does not run on-device) |
| Accuracy comparison vs published cloud models (Month 4) | ✅ First paper compared (Zhou et al. 2019, single-site ALSTM); full per-horizon + per-site breakdown in §7. Broader comparison across more published works remains open. |
| MQTT network cost + live dashboard (Month 5) | ⬜ Waiting on solar panel |
| Adaptive updating (TEDA-RLS) | ⬜ Not started |

---

## 3. Data pipeline

- **Source:** Open Climate Fix UK PV (`openclimatefix/uk_pv`, 5-minute resolution), 2018–2024, 83 monthly parquet files, **708.6M raw rows**.
- **Cleaning:** masked the flagged bad-data date ranges (`bad_data.csv` flags time windows per site, not whole sites); kept sites with more than 50% non-zero readings; forward-filled gaps up to 30 min.
- **Result:** **703 sites, 231.4M rows** kept (32.7% of rows).
- **Split by year (no shuffling):** train 2018–2022, validation 2023, test 2024.
- **Note:** the dataset contains 1,905 sites in total, not the 1,311 quoted in the proposal. The proposal figure appears to come from a different snapshot.
- **Windowing:** 3 h of context (36 steps) → 90 min forecast (18 steps). Windows never cross a real gap or a site boundary.
- **Input features (5 channels):** generation, sin/cos of hour-of-day, sin/cos of day-of-year. Generation is divided by each site's installed capacity (kWp), then z-scored.

Two bugs were caught along the way and fixed. The first was a pandas index-misalignment that silently dropped about 60% of rows. It was found by cross-checking the site count against the coverage pass. The second was an out-of-memory crash from loading all 708M rows at once, fixed with file-by-file streaming.

---

## 4. Model and forecasting accuracy

**SmallTCN:** 4 dilated causal conv blocks (channels 32/32/64/64, kernel 5, dilations 1/2/4/8) plus a linear head, 38,002 parameters.

Test-set results (naive persistence baseline: MAE 12.31 Wh):

| Configuration | MASE ↓ | MAE (Wh) | RMSE (Wh) | sMAPE, daylight only |
|---|---|---|---|---|
| Baseline (generation channel only) | 0.852 | – | – | – |
| + hour-of-day features | 0.823 | 10.13 | 22.44 | 47.10% |
| + capacity normalization | 0.816 | 10.04 | 22.50 | 46.98% |
| + day-of-year features | **0.807** | 9.93 | 22.37 | 46.15% |
| Tried: 6 h context instead of 3 h | 0.812 | 10.02 | 22.26 | 43.96% |
| **Deployed checkpoint (retrain, 3 h context), fp32** | **0.808** | 9.95 | 22.25 | 44.77% |
| **Deployed checkpoint, int8 (full test set)** | **0.846** | 10.42 | 22.84 | 47.06% |

Notes:
- The 6 h context was **not adopted**: MASE did not improve, and two sites dropped out because their contiguous runs were too short.
- Plain sMAPE over all points is about 123%. That number is inflated by the many night-time zeros (near-zero denominators), so **daylight-only sMAPE and MAE/RMSE are reported instead**.
- Run-to-run variance is about ±0.001 MASE (0.807 vs 0.808 for the same configuration).
- RMSE is about 2.2× MAE, so the error is dominated by a tail of large misses. My interpretation is that these are short-horizon cloud effects a history-only model cannot anticipate. **This is a hypothesis; I have not tested it** (for example by breaking errors down by time of day or by variability).

---

## 5. Deployment results

### 5.1 Final numbers (measured on hardware)

| | STM32F446RE (ST Edge AI Core v4.0.1) | ESP32-C6 (TFLite Micro, esp-tflite-micro 1.4.1) |
|---|---|---|
| Clock / core | 180 MHz, Cortex-M4 (with FPU) | 160 MHz, RISC-V, no SIMD, **no hardware FPU** (see §5.5) |
| Inference time | **18.2 ms** (54.95 inf/s) | **97.7 ms** (10.23 inf/s) |
| Cycles per MAC (derived) | 2.43 | ≈ 11.6 (using ST's MAC count for both) |
| Model weights in flash | **41,708 B (40.7 KiB)**, −73.1% vs float | model file embedded: 102,536 B (flatbuffer incl. graph metadata) |
| Total AI flash (runtime + weights) | 78,386 B (76.5 KiB) | not separable; app image is 366,240 B including ESP-IDF |
| Working RAM | activations 16,768 B (16.4 KiB); AI total 26,296 B | tensor arena **147,836 B (144.4 KiB)** ⚠ see caveat |
| Output vs reference model | rmse 0, cos 1.0 | max abs diff 0 (all 18 outputs and an intermediate tensor) |

Comparison caveats:
- The two boards use **different runtimes**. ST generates static C code; TFLite Micro interprets the model at run time. The 5.4× latency gap therefore cannot be attributed to the hardware alone. My guess is that ESP32-C6 uses portable reference int8 kernels, but I have not verified which kernels are active.
- The flash figures are not directly comparable. The ESP32 image includes the ESP-IDF framework; the STM32 figure covers only the AI runtime and weights. (The full STM32 validation image, including HAL and ST's validation stack, is 111 KB.)
- The STM32 on-target check compares the generated C code against the TFLite reference on random inputs. It verifies **deployment fidelity, not forecast accuracy**.

### 5.2 Finding 1 — STM32: dilated convolution is silently not quantized

**Symptom.** Every int8 export reported `weights ≈ 146 KiB (−1.6% vs float)`, when about 40 KiB was expected. Eight configurations gave the identical number: ONNX dynamic quantization, static QDQ, static QOperator, per-channel and per-tensor, head layer excluded, ST's compression flag, and a TFLite export. I confirmed by inspecting the graphs that the *files* contained genuine int8 weights, so the quantization itself was correct.

**Isolation.** I built three tiny test models (plain conv, causal conv, causal + dilated conv):

| Test model | Flash vs float | Conv execution |
|---|---|---|
| Plain conv | −65.2% | int8 |
| Causal-padded conv | −65.2% | int8 |
| Causal-padded **dilated** conv | −17.0% | **float32 fallback** |

**Root cause.** TFLite lowers a dilated conv to `SpaceToBatchND → Conv2D → BatchToSpaceND`. ST Edge AI Core (STM32F4 backend, v4.0.1) then runs that Conv2D in float32 and stores its weights in float32. Three of SmallTCN's four blocks have dilation ≥ 2 and hold about 95% of the parameters, which is why nothing we changed upstream made a difference.

**Fix.** Implement each dilated conv without the `dilation_rate` attribute: left-pad, take *k* static crops (one per tap), apply *k* pointwise (1×1) convs, and sum them. This is mathematically identical to the dilated conv. I checked the identity numerically (max difference 1.8e-15) and checked the PyTorch→Keras port (max difference 8e-7). After the fix, all ops run as int8, and latency dropped from 84.8 ms to 18.2 ms.

### 5.3 Finding 2 — int8 accuracy cost on the full test set

The deployed int8 model was evaluated on the full test set (969,445 windows, 279 sites). This fills the previously open item: **the int8 forecast accuracy is now measured**, not estimated.

| Metric | fp32 (PyTorch checkpoint) | int8 (deployed TFLite) | Delta |
|---|---|---|---|
| MASE | 0.808 | **0.846** | **+0.038** (+4.7% relative) |
| MAE | 9.95 Wh | 10.42 Wh | +0.47 Wh |
| RMSE | 22.25 Wh | 22.84 Wh | +0.59 Wh |
| sMAPE (daylight only) | 44.77% | 47.06% | +2.29 pp |

Daylight-only cosine similarity between int8 and fp32 predictions: **0.971**, computed on the 538,069 test windows with actual generation > 1 Wh. The all-windows cosine (0.745) is dominated by numerical noise on near-zero night-time vectors and is not a meaningful figure.

**Correction to an earlier claim.** An earlier host-only check on 20 real windows reported "cosine 0.998, normalized MAE 0.025 vs. the fp32 model". The full-test-set daylight number is 0.971 (MAE 2.74 Wh vs. fp32 predictions). The 20-window estimate was too small a sample to estimate the tail of the quantization-error distribution, and overstated fidelity. The full-test-set numbers above are the ones to cite.

**Interpretation.** The +0.038 MASE cost is at the higher end of typical for a TCN. Two plausible causes: (1) the model is very small (38K parameters), so there is less redundancy for the quantizer to work with; (2) the manual-dilation rewrite replaces each dilated conv with *k* separate pointwise convs, which introduces more quantization breakpoints than a single fused dilated op would. I have not tested these as separate hypotheses; they are offered as context, not claims.

### 5.4 Finding 3 — ESP32-C6: wrong outputs until the memory settings were changed

**Symptom.** The first run gave outputs that drifted smoothly away from the reference (up to 36 int8 units by the last forecast step), even though the same `.tflite` file matched exactly on STM32.

**What fixed it.** Two changes were made at the same time: `preserve_all_tensors=true` on the interpreter and a larger arena (80 KiB → 176 KiB). After both, output and an intermediate tensor matched bit-exactly.

**What is not established.** I suspect the cause is TFLite Micro's memory planner reusing buffers that this graph (many parallel branches reading one padded tensor) still needs. **This is an inference, not a confirmed cause**, because both settings changed together. `preserve_all_tensors=true` disables buffer reuse, which is why the arena is 144 KiB instead of roughly the 16 KiB STM32 needs. The reported ESP32 RAM figure is therefore **an upper bound, not the model's true requirement.**

**Planned ablation:** keep the 176 KiB arena and set `preserve_all_tensors=false`. If the output breaks, buffer reuse is confirmed. If it stays correct, the arena size was the issue and RAM can be shrunk substantially.

---

### 5.5 TTM: conversion verified, deployment blocked by independent, architectural constraints

TTM-R2 (70,972 parameters after resizing to context=52/forecast=16, frequency-prefix-tuned) was fine-tuned on the same solar data and capacity-normalization convention as SmallTCN (train-fit Normalizer: mean=9.020441, std=15.846984, within ~1% of SmallTCN's own fitted stats on overlapping data — cross-checked as a units sanity check, not a coincidence).

**Conversion chain — fully verified:**

| Stage | Artifact | Verification |
|---|---|---|
| PyTorch → ONNX | `exports/ttm_solar.onnx` | Max abs diff vs. PyTorch: 0.000001 |
| ONNX → TFLite (float32) | `ttm_solar_float32_fallback.tflite` (458.7 KiB) | Max abs diff vs. ONNX: 0.000002 |
| TFLite parity | 200 real test-split windows | Pearson correlation 1.000000 |

One toolchain bug was found and fixed along the way: the legacy ONNX tracer (`torch.onnx.export(..., dynamo=False)`) incorrectly bakes TTM's adaptive-patching reshape into literal constants, producing a graph whose declared output shape (`[1, 16, 13]`) disagreed with the model's own weights (`[1, 16, 1]` expected) — confirmed via eager-mode forward hooks that the checkpoint itself was correct, so the bug was isolated to the tracer. Fix: `dynamo=True` (the `torch.export`-based exporter), which traces the reshape symbolically.

**Quantization — three approaches tried, all three ruled out, each for a distinct, confirmed reason:**

- **Full int8** converts and loads without error, but crashes on 100% of 200 real test windows with `tflite/kernels/div.cc:242 data[i] != 0`, at the same handful of graph nodes regardless of input variance (ruling out a data-dependent cause). Root cause, confirmed via direct node/tensor inspection: TTM's Erf/GELU activation decomposes into a rational approximation (`abs`/`sign`/`exp`/`div`/`rsqrt`), and the divisor tensor's real dynamic range (~0 to 7.4) is too wide for int8's 256 levels — about 77% of real values quantize down to the exact zero-point, which dequantizes to literal 0.0. TFLite's int8 `DIV` kernel correctly refuses to divide by a zero-point code. Confirmed pervasive, not one fixable node: TensorFlow's own `QuantizationDebugger` independently crashes at a *different* DIV node while walking the graph in its own order, and separately flags a `NaN`/`Inf` layer — evidence of fragility at multiple points in the Erf/LayerNorm-heavy mixer stack, not an isolated bug.
- **int16x8** (int16 activations, int8 weights — meant to fix the above with 65,536 levels instead of 256) fails for an unrelated reason: the TFLite converter cannot calibrate a `Cast` op's min/max range (`Empty min/max for tensor Cast`), caused by `freq_token`'s int64 path. A known TFLite converter limitation, not fixable with better calibration data.
- **Dynamic-range quantization** (int8 weights, float32 activations) converts cleanly and passes parity (375.8 KiB, 18% smaller than float32; relative max error 0.49% vs. ONNX on 200 real windows) — but TFLite Micro does not support dynamic-range quantization at all, by explicit upstream design (TFLM ships kernels only for pure float32 or full-integer inference, never runtime int8-weight dequantization).

**Consequence.** Float32 is the only TTM format that TFLite Micro can run, so it is the format that was attempted on the ESP32-C6.

**FPU note (important for §5.6).** The ESP32-C6 has **no hardware floating-point unit** (RV32IMAC core). ESP-IDF's capability headers define `SOC_CPU_HAS_FPU` for `esp32`, `esp32s3` and `esp32p4`, but the `esp32c6` header does not define it. Float32 inference on the C6 therefore runs through libgcc software emulation (`__mulsf3`, `__addsf3`, …). This means the float32 route is a poor match for the C6 on compute as well as on memory. (An earlier draft of this report stated that the C6 has an FPU; that was wrong and is corrected here.)

### 5.6 TTM on ESP32-C6: loads and begins executing, then faults in the `tflite-micro` greedy planner — does not fit

Deploying the float32 model required patching real gaps in `esp-tflite-micro`'s kernel set: a missing `SIGN` kernel (written from scratch), a missing `Register_SELECT` exposure, and missing int64 input/output branches in `cast.cc`, `select.cc`, `add.cc` (both the standard and `esp_nn` variants), and `gather.cc`. All patches are preserved under `thesis_artifacts/ttm_esp32/patched_kernels/`.

With those patches in place, the model loaded on real ESP32-C6 hardware (schema verified, tensor arena allocated at 160,152 of 245,760 bytes), with correctly-shaped inputs/outputs, and executed its first several operations cleanly — then aborted inside TFLite Micro's greedy memory planner at a `Gather` node, with a confirmed write fault (`MCAUSE=0x7`, store/AMO access fault): the same tensor pointer is reused correctly on an earlier call and overflows on a later one. **The fault type and location are measured; the mechanism is an inference** — it is consistent with the planner's lifetime analysis mis-accounting for a tensor that appears twice in TTM's graph (once early, once late), but I have not confirmed it independently.

The linear memory planner (which never reuses buffers, and so avoids this specific bug) was tried as an alternative, and its own allocator reports the real cost of that safety:

```
Failed to resize buffer. Requested: 2,945,744, available: 136,160, missing: 2,809,584
```

TTM's true activation footprint without buffer reuse is roughly 2.9 MB — about **21.6×** the RAM actually available for the tensor arena at that point in the firmware (136 KB, after WiFi stack, app code, and the bootloader already claim their share of the chip's 512 KB SRAM), or about 5.7× the chip's *entire* SRAM budget even in the (unrealistic) case of reserving every byte for the arena alone. Even a bug-free greedy planner could not close a gap this large without real model compression (e.g. a shorter context length) on TTM's side.

**Conclusion:** TTM does not fit on the ESP32-C6 in its current form. The finding rests on two independent grounds: memory (≈2.9 MB without buffer reuse vs 512 KB total SRAM) and compute (no hardware FPU, so float32 inference would be software-emulated). It is a complete, well-isolated negative result, backed by on-device evidence at every stage (conversion parity, kernel patches, the fault type and location, the linear planner's measured memory requirement). SmallTCN remains the deployed model on both boards; see §5.1 for its verified hardware numbers.

---

## 6. What is verified vs. not yet verified

| Claim | Verified? |
|---|---|
| int8 models on both boards reproduce the reference model's outputs exactly | ✅ measured on hardware |
| STM32 flash/latency/RAM numbers in §5.1 | ✅ measured on hardware |
| Dilation is the cause of the STM32 non-compression | ✅ isolated with a minimal test; the fix worked |
| Manual-dilation rewrite equals real dilated conv | ✅ checked numerically |
| Forecast accuracy of the int8 model on the full test set | ✅ measured. int8 MASE 0.846 vs fp32 0.808, delta +0.038 (+4.7%). MAE 10.42 vs 9.95 Wh. Daylight-only cosine vs fp32: 0.971 on 538,069 windows (§5.3) |
| TTM's PyTorch → ONNX → TFLite(float32) conversion is numerically correct | ✅ parity-checked at every stage (max abs diff ≤ 0.000002) |
| TTM int8 quantization fails due to Erf/GELU's DIV-zero-point collision | ✅ isolated via node inspection; confirmed pervasive via `QuantizationDebugger` |
| TTM int16x8 quantization fails due to a Cast-calibration limitation | ✅ reproduced; matches a known TFLite converter limitation |
| TTM dynamic-range quantization is numerically valid | ✅ parity-checked (0.49% relative error vs. ONNX) |
| TTM does not fit on the ESP32-C6 | ✅ measured on hardware: store access fault in the greedy planner; linear planner's real requirement (≈2.9 MB) measured directly |
| ESP32-C6 has no hardware FPU | ✅ confirmed via ESP-IDF `soc_caps.h` (no `SOC_CPU_HAS_FPU` for esp32c6) |
| Greedy-planner fault is caused by a tensor appearing twice in the graph | ❌ hypothesis; fault type/location measured, mechanism not isolated |
| ESP32 SmallTCN root cause is buffer reuse | ❌ hypothesis, ablation pending |
| ESP32 slowdown is due to lack of SIMD / reference kernels | ❌ hypothesis |
| Errors are dominated by cloud variability | ❌ hypothesis |
| int8 quantization cost is at the higher end of typical because of small model size + manual dilation rewrite | ❌ hypothesis, two plausible mechanisms not separated |

---

## 7. Next steps

- [x] **Evaluate the int8 model on the full test set (MASE/MAE)** and compare with fp32. Done: int8 MASE 0.846 vs fp32 0.808 (+0.038, +4.7%). Full-test-set daylight cosine vs fp32: 0.971. See §5.3.
- [ ] ESP32 ablation (`preserve_all_tensors=false`) to confirm root cause and recover RAM.
- [x] Fine-tune **TTM** on the solar data, compress it, and deploy it. Done: fine-tuned, exported, converted, and attempted on ESP32-C6 hardware. TTM does not fit — three quantization approaches ruled out, and the float32 model is blocked by the planner fault and the ≈2.9 MB linear footprint (§5.5–5.6). No further deployment work planned against TTM-on-ESP32-C6 unless scope changes.
- [x] Hardware comparison table vs TinyHAR-Net (both on STM32F446RE). Full comparison in
      `thesis_artifacts/hardware_comparison_tinyhar.md`. SmallTCN: 40.7 KiB flash (int8),
      18.2 ms, 16.4 KiB RAM. TinyHAR-Net: 43.78–51.30 KiB flash (float32), 41.31–51.57 ms,
      30.12 KiB RAM. SmallTCN is 2.3–2.8× faster on the same MCU; the two models make
      opposite choices on int8 quantization, each justified by starting footprint. TTM
      excluded since it does not run on-device.
- [x] Accuracy comparison against the cited cloud-side PV forecasting results. First paper (Zhou et al. 2019) compared: SmallTCN fp32 MAPE at 5/15/30/60 min = 31.45 / 47.10 / 58.11 / 73.19% vs their 24.65 / 28.81 / 32.18 / 37.82%. Per-site breakdown across 276 test sites: median 28.45% at 5 min, best sites 23.2–24.2% (matching Zhou zero-shot). Long-horizon gap remains (best-site 60 min 55.38% vs Zhou 37.82%). Logs: `thesis_artifacts/smalltcn_per_horizon_mape_full.log`, `thesis_artifacts/smalltcn_per_site_mape.log`.
- [ ] *(Possible follow-up, not committed to timeline)* If TTM remains in scope: re-export with a shorter context length to shrink the ≈2.9 MB linear-planner requirement, or evaluate ESP32-S3 with PSRAM as a larger target. The S3 has an FPU per ESP-IDF's capability headers, which would also address the compute concern from §5.5. Neither attempted yet.
- [ ] When the solar panel is available: sensor → MCU → MQTT → dashboard, plus network cost measurements.
- [ ] Adaptive updating (TEDA-RLS) — scope to be discussed given the overall workload.

---

## 8. Questions for the supervisor

1. **Manual-dilation model:** the deployed graph is a rewrite of the trained model (identical math, different structure). Is this acceptable as the "SmallTCN" deployment, with the toolchain limitation documented as a finding?
2. **Scope:** TTM does not fit on the ESP32-C6 (§5.5–5.6) — is the negative result, as documented, sufficient for the thesis, or is further follow-up (context-length reduction, ESP32-S3/PSRAM) expected within the remaining timeline?
3. **Latency comparison:** the cross-board gap mixes hardware and runtime effects. Is it worth adding a like-for-like TFLite Micro run on the STM32 to separate the two?
4. **int8 accuracy cost interpretation:** the +0.038 MASE (+4.7%) cost is at the higher end of typical for a TCN. Two plausible mechanisms (small model size, manual dilation rewrite introducing more quantization breakpoints) are not separated by the current experiment. Should I invest in separating them, or is reporting the aggregate cost with the hypotheses noted sufficient?

---

## 9. Reproducing

```
preprocess_ukpv.py                 # filter / clean / split the dataset
train_solar_tcn.py                 # train SmallTCN (--stride 24)
torch_to_tflite_tcn_manual_dilation.py   # port to Keras (with parity check) + int8 TFLite
evaluate_int8_accuracy.py          # full-test-set int8 vs fp32 accuracy comparison
prepare_esp32_deployment.py        # model + test vector as C arrays
esp32_project/                     # ESP-IDF app (TFLite Micro) -- currently builds SmallTCN
test_minimal_quant.py              # minimal repro of the dilation issue

finetune_ttm.py                    # fine-tune TTM-R2 on solar data
export_ttm_onnx.py                 # TTM -> ONNX, parity-checked (dynamo=True required)
onnx_to_tf_ttm.py                  # ONNX -> TF SavedModel / float32 TFLite (onnx2tf)
convert_int8_manual.py, convert_int16x8_or_fallback.py, convert_dynamic_range.py
                                    # the three quantization attempts, all ruled out (see §5.5)
prepare_esp32_deployment_ttm.py    # TTM model + test vector as C arrays
thesis_artifacts/ttm_esp32/        # full on-device investigation archive: patched kernels,
                                    # boot logs, REPORT.md (detailed write-up of §5.6)
```

The final SmallTCN int8 model is in `models/small_tcn_solar_manual_int8.tflite` (the deployed model on both boards). The TTM float32 model is in `exports/ttm_solar_quant/ttm_solar_float32_fallback.tflite` (verified by parity, not deployable on the ESP32-C6 -- see §5.6).
