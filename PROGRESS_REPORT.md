# Progress Report — On-Device Solar Power Forecasting

**Student:** Shuvro Sankar Sen (ID 25-93776-2), AIUB
**Supervisor:** Dr. Rajarshi Roy Chowdhury
**Repo:** `<paste GitHub URL here>`
**Covers:** dataset pipeline → SmallTCN training → int8 deployment on STM32F446RE and ESP32-C6

---

## 1. Summary

- A **SmallTCN** forecaster (38K parameters) is trained on the public UK PV dataset and **runs on both boards** (STM32F446RE and ESP32-C6) in int8, with on-device output matching the reference model **bit-exactly**.
- On the test set it beats naive persistence: **MASE 0.808** (about 19% lower error than persistence).
- Getting a genuinely compressed int8 model onto the STM32 took a long investigation. The root cause was a **toolchain limitation with dilated convolutions**, fixed with a mathematically equivalent rewrite. After the fix: **40.7 KiB of weights, 18.2 ms per inference**.
- The ESP32-C6 deployment initially produced *wrong* outputs. A debugging pass fixed it (details in §5), but **the RAM figure there is inflated** and the root-cause attribution is still unconfirmed.
- **Not yet done:** TTM adaptation, full-test-set accuracy of the int8 model, MQTT/dashboard work (waiting on the solar panel), comparison against published works.

---

## 2. Status against the proposal

| Item (proposal timeline) | Status |
|---|---|
| Solar dataset pipeline (Months 1–2) | ✅ Done |
| Retrain SmallTCN on solar data (Months 1–2) | ✅ Done (MASE 0.808) |
| Quantize + deploy SmallTCN on both boards (Months 2–3) | ✅ Done (both boards, verified) |
| Quantize + deploy TTM (Months 2–3) | ⬜ Not started |
| Hardware benchmark vs TinyHAR-Net (Month 4) | 🟡 SmallTCN numbers ready; TTM and comparison table pending |
| Accuracy comparison vs published cloud models (Month 4) | ⬜ Not started |
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
| **Deployed checkpoint (retrain, 3 h context)** | **0.808** | 9.95 | 22.25 | 44.77% |

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
| Clock | 180 MHz, Cortex-M4 | 160 MHz, RISC-V (no SIMD) |
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

### 5.3 Finding 2 — ESP32-C6: wrong outputs until the memory settings were changed

**Symptom.** The first run gave outputs that drifted smoothly away from the reference (up to 36 int8 units by the last forecast step), even though the same `.tflite` file matched exactly on STM32.

**What fixed it.** Two changes were made at the same time: `preserve_all_tensors=true` on the interpreter and a larger arena (80 KiB → 176 KiB). After both, output and an intermediate tensor matched bit-exactly.

**What is not established.** I suspect the cause is TFLite Micro's memory planner reusing buffers that this graph (many parallel branches reading one padded tensor) still needs. **This is an inference, not a confirmed cause**, because both settings changed together. `preserve_all_tensors=true` disables buffer reuse, which is why the arena is 144 KiB instead of roughly the 16 KiB STM32 needs. The reported ESP32 RAM figure is therefore **an upper bound, not the model's true requirement.**

**Planned ablation:** keep the 176 KiB arena and set `preserve_all_tensors=false`. If the output breaks, buffer reuse is confirmed. If it stays correct, the arena size was the issue and RAM can be shrunk substantially.

---

## 6. What is verified vs. not yet verified

| Claim | Verified? |
|---|---|
| int8 models on both boards reproduce the reference model's outputs exactly | ✅ measured on hardware |
| STM32 flash/latency/RAM numbers in §5.1 | ✅ measured on hardware |
| Dilation is the cause of the STM32 non-compression | ✅ isolated with a minimal test; the fix worked |
| Manual-dilation rewrite equals real dilated conv | ✅ checked numerically |
| **Forecast accuracy of the int8 model on the full test set** | ❌ **not measured.** Only a host check on 20 real windows (cosine 0.998, normalized MAE 0.025 vs the fp32 model) |
| ESP32 root cause is buffer reuse | ❌ hypothesis, ablation pending |
| ESP32 slowdown is due to lack of SIMD / reference kernels | ❌ hypothesis |
| Errors are dominated by cloud variability | ❌ hypothesis |

---

## 7. Next steps

- [ ] **Evaluate the int8 model on the full test set (MASE/MAE)** and compare with fp32. This gives the "accuracy cost of compression" the proposal asks for.
- [ ] ESP32 ablation (`preserve_all_tensors=false`) to confirm root cause and recover RAM.
- [ ] Fine-tune **TTM** on the solar data, compress it, and deploy it. Note that TTM at int8 did not fit the STM32F446RE (1,068.7 KB vs 512 KB flash) in the preliminary result, so the ESP32-C6 is the expected target.
- [ ] Complete the hardware comparison table (including TinyHAR-Net's published figures).
- [ ] Accuracy comparison against the cited cloud-side PV forecasting results (needs care: different datasets and horizons).
- [ ] When the solar panel is available: sensor → MCU → MQTT → dashboard, plus network cost measurements.
- [ ] Adaptive updating (TEDA-RLS) — scope to be discussed given the overall workload.

---

## 8. Questions for the supervisor

1. **Manual-dilation model:** the deployed graph is a rewrite of the trained model (identical math, different structure). Is this acceptable as the "SmallTCN" deployment, with the toolchain limitation documented as a finding?
2. **Scope:** with the ESP32 results in, should TTM be the next priority, or the full-test-set int8 accuracy evaluation first?
3. **Latency comparison:** the cross-board gap mixes hardware and runtime effects. Is it worth adding a like-for-like TFLite Micro run on the STM32 to separate the two?

---

## 9. Reproducing

```
preprocess_ukpv.py                 # filter / clean / split the dataset
train_solar_tcn.py                 # train SmallTCN (--stride 24)
torch_to_tflite_tcn_manual_dilation.py   # port to Keras (with parity check) + int8 TFLite
prepare_esp32_deployment.py        # model + test vector as C arrays
esp32_project/                     # ESP-IDF app (TFLite Micro)
test_minimal_quant.py              # minimal repro of the dilation issue
```

The final int8 model is in `models/small_tcn_solar_manual_int8.tflite`.
