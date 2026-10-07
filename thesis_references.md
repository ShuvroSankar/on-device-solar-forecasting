## [Abushahla et al. 2025] — Neural Network Quantization for Microcontrollers: A Comprehensive Survey of Methods, Platforms, and Applications

**Venue:** arXiv preprint (2508.15008v4) — survey paper

**Task:** Survey — quantization methods, hardware platforms, software frameworks, and real-world applications of QNNs on MCUs (covers classification, detection, speech, HAR, healthcare, etc.)

**Hardware:** Covers ARM Cortex-M (M0+/M4/M4F/M7/M33/M55), RISC-V (GAP8/GAP9, ESP32-C3/C6/P4, RP2350, CH32V003, BL616/618), and hybrid/NPU MCUs (MAX78000/78002, STM32N6, HX6538-WE2, NXP MCXN947). Representative specs: Cortex-M4 @ 64–480 MHz, 256 KB–1 MB RAM, 1–8 MB flash; GAP9 @ 250–370 MHz, 1.6 MB RAM; MAX78000 @ 100 MHz, 512 KB flash/RAM, 30 GOPS NPU.

**Framework:** Reviews TFLM, STM32Cube.AI, Edge Impulse, MinUn, ai8x-synthesis, ONNX Runtime, TinyMaix, NNoM, TinyNeuralNetwork, GAPflow, HEEPstor, ExecuTorch.

**Quantization:** Comprehensive — INT8 PTQ/QAT (dominant), sub-8-bit (4/2/1-bit, binary/ternary), mixed precision, data-free quantization, redistribution techniques, alternative formats (FP16, BF16, TF32, Posit, Zero-Skew, F8Net, VS-Quant).

**Key numbers:**

| Metric | Value |

|---|---|

| Accuracy | Survey reports deployed ranges: 43.5–99.95% (task-dependent; e.g., HAR 91–98%, environment 98.5–99.6%, healthcare 84.8–94.6% AUC/Acc) |

| F1 | 70.55–99.09% (spectrum sensing); 79.67–86.53% (environment) |

| MASE | Not reported (not standard in surveyed works) |

| Flash | 12.8 KB–3.6 MB across surveyed deployments (typical TinyML footprint < 1 MB) |

| RAM | ~30–500 KB typical; larger NPU platforms up to 4.2 MB (STM32N6) |

| Latency | 0.08 ms–6.3 s (wide range; NPU-backed works: 104 µs–13 ms) |

| Params | Not aggregated (model sizes range from <10 KB to several MB) |

| Energy | 5 µJ–11200 µJ per inference; power 1.43 mW–2.5 W |

**Comparable to my work on:** Same hardware (ARM Cortex-M, RISC-V, NPU-augmented MCUs), same task (TinyML deployment across vision/audio/HAR/healthcare), same framework (TFLM, STM32Cube.AI, ai8x), same quantization (INT8 PTQ/QAT, mixed precision, sub-8-bit).

**Notable claim:** This is the first hardware-oriented survey that systematically bridges quantization methods, MCU hardware platforms, software deployment toolchains, and real-world applications, while framing sustainability and system-level energy as first-class objectives for TinyML — concluding that INT8 (PTQ/QAT) remains the practical standard, but sub-8-bit and mixed precision will dominate once toolchains and hardware natively support them.

**Caveat:** As a survey, it aggregates numbers from heterogeneous works with inconsistent metrics, datasets, and reporting conventions; the "key numbers" above are ranges extracted across many papers, not results from a single controlled experiment. Direct comparison to any single work is imperfect because hardware, task, dataset, and measurement methodology vary substantially across the surveyed literature.

## [APEX Authors 2026] — APEX: Network-Native Time-Series Foundation Model for AP Failure Detection

**Venue:** arXiv preprint (2606.11553v1), 2026

**Task:** Multivariate time-series forecasting (regression) + anomaly detection — DHCP degradation in wireless access points

**Hardware:** Raspberry Pi 5 (quad-core Arm Cortex-A76, 1–2 GB RAM) as proxy for AP-class ARM hardware; not an MCU

**Framework:** Custom PyTorch-style decoder-only patched transformer (AMP, DDP, AdamW)

**Quantization:** None — float32 training/inference; INT8 mentioned only as future work for integrated edge AI

**Key numbers:**

| Metric | Value |

|---|---|

| Accuracy | Not reported (forecasting task) |

| F1 | 0.93 (APEX-Large MC-dropout); 0.89 (APEX-Edge MC-dropout); 0.94 (VAR-Mahalanobis) |

| MASE | Not reported |

| MAE | 2.98 (APEX-Large multi); 3.87 (APEX-Edge multi); 3.64 (Toto); 4.82 (SARIMA) |

| RMSE | 4.61 (APEX-Large multi); 5.83 (APEX-Edge multi); 5.52 (Toto) |

| MAPE | 3.1% (APEX-Large multi); 4.1% (APEX-Edge multi); 3.8% (Toto) |

| Flash / model size | ~40 MB checkpoint (APEX-Edge) |

| RAM | 428 MB peak |

| Latency | 202 ms single 96-step forecast (median; P95 = 205 ms); 11.4 s for 50 MC-dropout samples; sub-second for 5 samples |

| Params | 10.5M (APEX-Edge); 269M (APEX-Large) |

| Energy | Not reported |

**Comparable to my work on:** Same hardware class (ARM edge), same task (time-series forecasting/anomaly detection), same framework (custom transformer), but not same quantization (none vs INT8) and not same hardware tier (AP-class Cortex-A76 vs MCU-class Cortex-M).

**Notable claim:** Network-native pretraining on co-collected wireless telemetry lets a single compact decoder-only transformer (APEX-Edge, 10.5M params) jointly forecast DHCP degradation and detect anomalies, matching/beating general-purpose TSFMs and classical baselines while running on AP-class ARM hardware with no cloud dependency and raw telemetry staying on-device.

**Caveat:** Not a microcontroller deployment — Raspberry Pi 5 (Cortex-A76, 1–2 GB RAM) is far above MCU class; no quantization is used (float32 only); anomaly labels are consensus pseudo-ground-truth rather than human annotations; evaluation is limited to DHCP degradation; latency is measured on a proxy, not production AP SoC.

## [Grover et al. 2026] — Embodied Foundation Models at the Edge: A Survey of Deployment Constraints and Mitigation Strategies

**Venue:** arXiv preprint (2603.16952v2) — survey paper (ACM-style submission)

**Task:** Survey — deployment constraints and mitigation strategies for embodied foundation models (VLA policies, diffusion policies, vision encoders, 3D/LiDAR encoders, multimodal fusion stacks) on edge robotic platforms

**Hardware:** Embedded SoCs and edge platforms — NVIDIA Jetson AGX Orin, Jetson Orin NX, Qualcomm RB5, Apple M-series, Raspberry Pi 5, and MCU-class devices (referenced via MCUNet, MLPerf Tiny). Covers ARM Cortex-A76 class, unified-memory LPDDR platforms (25–204 GB/s), but not MCU-class deployment in depth.

**Framework:** Reviews ROS 2, DDS, Iceoryx, Agnocast, NITROS, MMEdge, TVM/MATCH, TinyIREE, Dora, PREEMPT_RT, PiCAS, FlashAttention, vLLM/PagedAttention, etc.

**Quantization:** Reviews as one of several mitigation strategies — LLM.int8, GPTQ, AWQ, ZeroQuant, SmoothQuant, QuaRot, KV-cache quantization (KIVI, KVQuant, XQuant), mixed 2-bit/4-bit (Apple Intelligence), structured pruning, low-bit NPU execution. Not a quantization-specific paper.

**Key numbers** (extracted from surveyed works — not the authors' own experiments):

| Metric | Value |

|---|---|

| Accuracy | Not reported (survey; no unified accuracy metric across surveyed works) |

| F1 | Not reported |

| MASE | Not reported |

| Flash | Not applicable (no MCU flash measurements reported) |

| RAM | OpenVLA-7B: ~14 GB FP16; APEX-Large: 6–8 GB weights; edge SoCs typically 8–32 GB unified LPDDR |

| Latency | VLA autoregressive decoding: 30–60% of end-to-end latency from kernel launch overhead; cross-device tensor migration: 4–15 ms; ROS 2 latency penalty: up to 50% vs. shared-memory; jitter spikes: 5–20 ms on Jetson; diffusion policies: 1–2 Hz naive → 60+ Hz with OneDP |

| Params | OpenVLA-7B (7B), RT-2 (up to 55B), APEX-Large (269M), APEX-Edge (10.5M), NanoVLA, TinyVLA, MobileBERT |

| Energy | Zero-sum power competition: 10–15 W accelerator inference reduces aerial endurance by several minutes; memory access dominates arithmetic energy |

**Comparable to my work on:** Same hardware class (ARM edge SoCs, unified-memory platforms), same task (edge inference of foundation models), same framework (ROS 2, TVM, TFLite Micro referenced), same quantization (INT8/low-bit quantization as mitigation). However, this survey focuses on higher-tier embedded SoCs (Jetson Orin, 8–32 GB RAM) rather than MCU-class (<1 MB RAM) devices — a key distinction for TinyML work.

**Notable claim:** Deploying foundation models on embodied edge platforms is fundamentally a systems problem — not just model compression — and the dominant failures arise from eight coupled barriers (the "Deployment Gauntlet"): sensor fusion tax, heterogeneous compute mismatch, unified memory bottleneck, energy/thermal ceiling, long-horizon drift, safety/verification gap, OS/scheduling bottleneck, and I/O/communication bottleneck. The survey argues that quantization, pruning, and compression improve average-case throughput but do not resolve worst-case deployment failures, and that architectural decomposition (separating fast control from slow semantic reasoning) is a more promising direction than monolithic end-to-end inference.

**Caveat:** As a survey, it aggregates findings from heterogeneous works with inconsistent metrics, hardware, and experimental setups; the numbers above are ranges or representative values extracted across many papers, not controlled results. The survey explicitly focuses on higher-compute embedded platforms (Jetson-class, 8–32 GB RAM) and does not deeply address MCU-class (<1 MB RAM) deployment — making direct comparison to MCU-focused TinyML quantization work imperfect. No new experimental results are presented.

## [Hossan et al. 2025] — TinyHAR-Net: TinyML-Based Human Activity Recognition on STM32

**Venue:** IEEE Embedded Systems Letters (ESL) — letter

**Task:** Classification — human activity recognition (HAR) from wearable accelerometer/gyroscope/magnetometer data (6/6/12 activity classes across WISDM, MotionSense, MHEALTH)

**Hardware:** STM32 Nucleo-F446RE — ARM Cortex-M4F @ up to 180 MHz, 128 KB SRAM, 512 KB flash

**Framework:** TensorFlow/Keras training → TensorFlow Lite (.tflite) → C header for STM32CubeIDE / TFLite Micro

**Quantization:** Float32 (unquantized) primary model; full-integer INT8 PTQ evaluated but not used for final deployment

**Key numbers:**

| Metric | Value |

|---|---|

| Accuracy | WISDM 97.90%, MotionSense 98.39%, MHEALTH 99.74% (input shape 120×f) |

| F1 | WISDM 96.75%, MotionSense 97.73%, MHEALTH 99.76% |

| MASE | Not reported |

| Flash | WISDM 43.78 KB, MotionSense 47.16 KB, MHEALTH 51.30 KB |

| RAM | 30.12 KB (consistent across all datasets at optimal input shape) |

| Latency | WISDM 41.31 ms, MotionSense 46.37 ms, MHEALTH 51.57 ms |

| Params | WISDM ~11.69K, MotionSense ~12.61K, MHEALTH ~13.54K |

| Energy | Not reported |

**Comparable to my work on:** Same MCU (STM32F446RE, Cortex-M4F), same task (HAR classification), same framework (TFLite Micro / STM32CubeIDE), same quantization (evaluates full-INT8 PTQ vs float32).

**Notable claim:** A hybrid depthwise-separable CNN + single-head Linformer self-attention model achieves high-accuracy HAR on an STM32F446RE without quantization or pruning, matching or beating larger compressed baselines while using only ~44–51 KB flash and ~30 KB RAM.

**Caveat:** Direct comparison is imperfect because datasets/activity classes and input shapes differ (WISDM/MotionSense 6 classes, MHEALTH 12; window length 120), only a single 80/20 train-test split is used, energy is not reported, and the final model is unquantized while many baselines rely on INT8 compression — though the paper’s own INT8 PTQ reduces flash to ~12–14 KB and latency to ~38–40 ms at the cost of ~1–2.4 F1 points.

## [Andrade et al. 2025] — TEDA-Forecasting: Unsupervised TinyML Incremental Learning for Outlier Processing and Forecasting

**Venue:** Computing (Springer), 2025, 107:162

**Task:** Time-series forecasting + unsupervised outlier detection/correction — univariate data streams (electricity production and real-time vehicle speed)

**Hardware:** OBD-II Freematics ONE+ (ESP32-based edge device); Arduino Nano 33 BLE Sense (64 MHz ARM Cortex-M4, 256 KB RAM, 1 MB flash — specs not fully reported in paper); plus Python/C++ on desktop for comparison

**Framework:** Custom C++ and Python implementations; Arduino deployment; no TFLite Micro or neural-network inference framework

**Quantization:** None — float32/incremental learning; no quantization or pruning applied

**Key numbers:**

| Metric | Value |

|---|---|

| Accuracy | Outlier detection: ~99.4% at TEDA threshold m=2.0 (electricity dataset); forecasting accuracy not reported |

| F1 | Outlier detection: 0.833 at m=2.0 (electricity dataset); not reported for vehicle data |

| MASE | Not reported |

| Flash | Not reported |

| RAM | Not reported |

| Latency | Total wall time: 207.07 µs per iteration (mean); TEDA: 85.58 µs; RLS: 78.72 µs; virtual RLS: 42.76 µs. At 1 Hz sampling, ~4829× free time |

| Params | Not applicable (non-neural, TEDA-RLS algorithm) |

| Energy | 4.68×10⁻⁵ (as reported in mW; 406× less than TEDA-LSTM, 35.25× less than TEDA-kNN) |

**Forecasting errors (electricity dataset, best config):**

- TEDA-Forecasting: RMSE 15.571, MAE 5.319

- TEDA-LSTM: RMSE 18.99 (1.22×), MAE 9.86 (1.86×)

- TEDA-kNN: RMSE 22.69 (1.48×), MAE 13.93 (2.62×)

**Vehicle speed data (real edge deployment):**

- Python: RMSE 2.422, MAE 1.623

- Freematics (ESP32): RMSE 5.326, MAE 3.884

- C++ / Arduino: RMSE 5.446, MAE 4.034

- Virtual forecasts (multi-step): Python RMSE 3.358, MAE 2.411; others ~5.3–5.4 RMSE, ~3.9–4.0 MAE

**Comparable to my work on:** Same MCU (Arduino Nano 33 BLE Sense, ESP32-based Freematics), same task (time-series forecasting and anomaly detection), same framework (custom embedded C++/Arduino), same quantization (none/float32). Not directly comparable if my work uses quantized neural networks.

**Notable claim:** TEDA-Forecasting combines the TEDA unsupervised anomaly detector with an RLS adaptive filter to perform online outlier detection, correction, and multi-step forecasting on microcontrollers without any training or quantization, achieving lower error and orders-of-magnitude lower energy than LSTM and kNN baselines.

**Caveat:** Only univariate time series are evaluated; no flash/RAM/parameter counts are reported; energy units are ambiguously labeled (mW rather than mWh); performance is sensitive to the TEDA threshold m; forecasting accuracy degrades over longer horizons; no statistical significance tests; and comparisons against LSTM/kNN do not include their training energy or quantized variants.

## [Andrade et al. 2024] — TEDA-RLS: TinyML Incremental Learning for Outlier Detection and Correction

**Venue:** IEEE Sensors Journal (2024)

**Task:** Unsupervised outlier detection and correction in univariate data streams — evaluated on simulated power substation data and real vehicle speed data

**Hardware:** Freematics ONE+ OBD-II scanner (ESP32 MCU); also simulated on Arduino Nano 33 BLE (64 MHz ARM Cortex-M4); Python/C++ desktop comparison

**Framework:** Custom C++ implementation on Freematics (PlatformIO); no neural-network inference framework (no TFLite Micro, no STM32Cube.AI)

**Quantization:** None — float32 arithmetic; TEDA-RLS is a statistical/adaptive-filter algorithm, not a quantized neural network

**Key numbers:**

| Metric | Value |

|---|---|

| Accuracy | Outlier detection (power dataset): ~0.98 at best threshold m=2.5 (Fig. 2) |

| F1 | Outlier detection (power dataset): 0.83 at m=2.5 (best); lower at m=2.0–2.35 |

| MASE | Not reported |

| Flash | Freematics ONE+: 9952 kB / 16 MB (62.2% per PlatformIO estimate) |

| RAM | Freematics ONE+: ~1064 kB / 8 MB (13.3% estimate); average runtime 3.75 MB (3.74–3.77 MB) |

| Latency | Wall time mean 243.63 µs (std 14.32 µs); TEDA 74.25 µs; RLS 169.38 µs |

| Params | Not applicable (non-neural algorithm) |

| Energy | Power consumption ~200 mA with algorithm + GPS + 4G active; ~10 mA standby (not energy per inference) |

| Prediction RMSE (power) | RMSEpred 126.95; RMSEsaved 124.39 (best config) |

| Prediction MAE (power) | MAEpred 27.11; MAEsaved 22.23 (best config) |

| Vehicle speed RMSE/MAE | Python ~1.75/0.85; Freematics/Arduino/C++ ~2.06/1.15 (Fig. 8, approximate) |

**Comparable to my work on:** Same hardware (ESP32-based Freematics ONE+, Arduino Nano 33 BLE), same task (real-time outlier detection/correction in time-series data), same framework (custom embedded C++/Arduino), same quantization (none/float32). Not directly comparable if my work uses quantized neural networks or TFLite Micro.

**Notable claim:** TEDA-RLS is the first TinyML algorithm to combine TEDA unsupervised outlier detection with an RLS adaptive filter for simultaneous detection *and* correction of outliers in data streams, requiring no training or prior data knowledge and achieving sub-millisecond execution on resource-constrained edge hardware.

**Caveat:** Only univariate time series are evaluated; no neural network or quantization is used; the reported Freematics memory footprint is MB-level (not typical MCU KB-level) due to ESP32 PSRAM and peripherals; detection performance is sensitive to the TEDA threshold m; the real vehicle speed dataset has no labels, so detection quality cannot be quantified on real data; no energy-per-inference or statistical significance tests are reported; no comparison against quantized DNN-based TinyML baselines.

## [Wangmek et al. 2024] — Enhancing Short-Term Solar PV Power Forecasting Using a Hybrid Deep Learning Approach

**Venue:** IEEE Access, vol. 12, 2024

**Task:** Regression — short-term (3-hour ahead) solar photovoltaic (PV) power forecasting at 1-minute intervals, using data from a 45 MW hydro-floating solar plant in Thailand

**Hardware:** MacBook Pro M2 (10-core CPU @ 3.4 GHz, 16-core GPU, 16 GB RAM) — training and inference on a laptop, **not** on a microcontroller or edge device

**Framework:** TensorFlow / Keras (Python 3.8)

**Quantization:** None — full float32 (FP32) model; no quantization, pruning, or edge deployment

**Key numbers:**

| Metric | Value |

|---|---|

| Accuracy | Not reported as classification accuracy; forecasting task uses RMSE/MAE/R² |

| F1 | Not applicable |

| MASE | Not reported |

| RMSE (normalized) | Winter: 0.025; Summer: 0.050; Rainy: 0.094 |

| MAE (normalized) | Winter: 0.014; Summer: 0.031; Rainy: 0.048 |

| R² | Winter: 0.994; Summer: 0.956; Rainy: 0.891 |

| Flash | Not applicable (no MCU deployment) |

| RAM | Not reported |

| Latency | Not reported (inference time not measured) |

| Params | Not reported |

| Energy | Not reported |

| Training time | 1,038.6 s (proposed model); CNN 8,354.1 s; LSTM 3,622.9 s; GRU 3,163.3 s; CNN-GRU 8,530.5 s |

**Comparable to my work on:** Same task (time-series forecasting), but **not** same hardware (no MCU), **not** same framework (no TFLite Micro / STM32Cube.AI), **not** same quantization (no INT8 PTQ/QAT). If my work focuses on MCU-based solar forecasting, this paper provides a high-accuracy deep learning baseline but no embedded deployment insights.

**Notable claim:** A hybrid 1D CNN–GRU model combined with SHAP-based feature selection, EMA smoothing, and Gaussian noise augmentation achieves state-of-the-art short-term solar PV forecasting accuracy across all Thai seasons, with significantly shorter training time than CNN, LSTM, GRU, and 2D CNN-GRU baselines.

**Caveat:** This paper does not deploy on any microcontroller or edge device, uses no quantization or model compression, and reports no energy, latency, flash, or RAM metrics. Direct comparison to TinyML/MCU work is therefore imperfect — it represents a cloud/laptop-level deep learning solution rather than an embedded one. Additionally, RMSE/MAE values are on normalized data (0–1 range), so absolute error magnitudes are not directly comparable to papers reporting errors in physical units (e.g., MW).

## [Zhou et al. 2019] — Short-Term Photovoltaic Power Forecasting Based on LSTM and Attention Mechanism

**Venue:** IEEE Access, vol. 7, 2019

**Task:** Regression — short-term PV power forecasting at 7.5, 15, 30, and 60 min horizons

**Hardware:** Not specified (desktop/GPU for training); no MCU or edge deployment

**Framework:** Keras 2.1.6 / TensorFlow

**Quantization:** None — full float32; no quantization, pruning, or embedded optimization

**Key numbers:**

| Metric | Value |

|---|---|

| Accuracy | Not reported (regression task) |

| F1 | Not applicable |

| MASE | Not reported |

| MAPE (ALSTM overall) | 24.65% @ 7.5 min; 28.81% @ 15 min; 32.18% @ 30 min; 37.82% @ 60 min |

| RMSE (ALSTM overall) | 1.39 @ 7.5 min; 1.60 @ 15 min; 1.81 @ 30 min; 2.09 @ 60 min |

| MAE | Reported in paper (Table V), but not extracted here; follows same trend as RMSE/MAPE |

| Flash | Not applicable (no MCU) |

| RAM | Not reported |

| Latency | Not reported |

| Params | Not reported |

| Energy | Not reported |

| Dataset | 20 kW rooftop PV station, Shaoxing, China; 7.5-min interval; 2014–2018 (train 2014–2016, test 2017–2018) |

| Baseline models | PM, ARIMAX, MLP, LSTM |

| Statistical test | Two-sample t-test confirms ALSTM significantly outperforms LSTM (p < 0.05 for 15–60 min annual forecasts) |

**Comparable to my work on:** Same task (short-term PV power forecasting), but **not** same hardware (no MCU), **not** same framework (no TFLite Micro / STM32Cube.AI), **not** same quantization (no INT8 PTQ/QAT). Provides a high-accuracy deep learning baseline but no embedded or TinyML insights.

**Notable claim:** An ensemble of two LSTM networks with attention mechanisms (ALSTM) — one for PV power and one for module temperature — outperforms PM, ARIMAX, MLP, and plain LSTM across all four seasons and all forecasting horizons (7.5–60 min), with the largest gains at horizons ≥15 min.

**Caveat:** This paper does not deploy on any microcontroller or edge device, uses no quantization or model compression, and reports no energy, latency, flash, or RAM metrics. Direct comparison to TinyML/MCU work is therefore imperfect — it represents a workstation-level deep learning solution rather than an embedded one. Additionally, RMSE/MAE are reported in kW for a 20 kW plant, so absolute error magnitudes are not directly comparable to normalized or MCU-deployed results.

## [Tegon et al. 2026] — FEMBA on the Edge: Physiologically-Aware Pre-Training, Quantization, and Deployment of a Bidirectional Mamba EEG Foundation Model on an Ultra-low Power Microcontroller

**Venue:** IEEE Transactions on Biomedical Engineering (TBME), 2026

**Task:** Regression/classification — EEG foundation model (pre-trained on 21,000+ hours of EEG) fine-tuned for downstream biosignal analysis (epilepsy, sleep disorders)

**Hardware:** GAP9 — a parallel ultra-low-power RISC-V MCU. Same ISA family as the ESP32-C6, but not an equivalent platform: GAP9 is a 9-core parallel cluster with 1.6 MB RAM and hardware accelerators purpose-built for this class of workload, versus the ESP32-C6's single RISC-V core and 512 KB SRAM — roughly 3× more memory plus dedicated parallel compute. This asymmetry matters when comparing why FEMBA fit on-device and TTM did not (see Caveat below).

**Framework:** Custom quantization-aware training pipeline + double-buffered memory streaming for embedded deployment

**Quantization:** Quantization-Aware Training (QAT) to compress the model to 2-bit weights, explicitly because standard post-training quantization fails

**Key numbers:**

| Metric | Value |
| --- | --- |
| Accuracy cost, PTQ | standard post-training quantization degrades accuracy by approximately 30% |
| Accuracy cost, QAT | QAT successfully compresses weights with negligible performance loss |
| Memory reduction | reduces the memory footprint by 74% (to approx 2 MB) |
| Latency | deterministic real-time inference (1.70 s per 5 s window) |
| Compute advantage | up to 27 times fewer FLOPs than Transformer benchmarks |

**Comparable to my work on:** Same core problem — deploying a pretrained foundation model (not a purpose-built small model) on resource-constrained hardware, where naive post-training quantization fails for architecture-specific reasons tied to the model's internal activation statistics.

**Notable claim:** This is, to the best of the authors' knowledge, the first deployment of an SSM-based foundation model on a parallel ultra-low-power RISC-V MCU, and it demonstrates that the quantization failure mode is specific to the architecture's activation behavior, not a general TinyML limitation.

**Caveat, and why it's worth citing directly against the TTM finding:** FEMBA's PTQ failure (~30% accuracy loss, a degradation, not a hard crash) is a softer failure than TTM's (a structural, 100%-reproducible runtime crash from an Erf/GELU decomposition's divisor collapsing to the int8 zero-point — not a precision loss, an inference-time fault). FEMBA's authors had a path forward (QAT) and the hardware headroom to use it; my investigation shows TTM's int8 failure is a TFLite-kernel-level hard stop, and even int16x8 (a step toward the precision QAT would buy) fails for an unrelated converter limitation (Cast calibration) before QAT would even become relevant. This is a genuinely useful contrast to draw explicitly: two pretrained foundation models, both resist naive PTQ, but for different failure classes (graceful degradation vs. hard crash) with different available remedies (QAT retraining, which FEMBA's larger/more capable hardware could absorb and I did not attempt, vs. a kernel-level toolchain gap that retraining would not have fixed). The hardware gap above means this is not a controlled comparison — FEMBA's success and TTM's failure are not attributable to QAT alone.

## [Ling et al. 2024] — Integer-only Quantized Transformers for Embedded FPGA-based Time-series Forecasting in AIoT

**Venue:** IEEE Annual Congress on Artificial Intelligence of Things (AIoT), 2024

**Task:** Regression — single-step-ahead time-series forecasting, univariate and multivariate

**Hardware:** Xilinx Spartan-7 XC7S15 (embedded FPGA, not an MCU — a relevant hardware-class difference to note)

**Framework:** Custom integer-only quantization with Quantization-Aware Training, realizing 6-bit and 4-bit quantized Transformer models

**Quantization:** QAT, integer-only, down to 4-bit

**Key numbers:**

| Metric | Value |
| --- | --- |
| Accuracy cost (4-bit vs. 8-bit baseline) | increases test loss by only 0.63% |
| Speed | operates up to 132.33x faster than the 8-bit baseline |
| Energy | consumes 48.19x less energy |

**Comparable to my work on:** Same task (time-series forecasting), same deployment concern — an attention/Transformer-family architecture's numerically awkward ops ("this computation, involving division and square root operations, is computationally expensive for embedded" hardware) directly echoes my own Erf/GELU DIV finding — but a different hardware class (FPGA, not MCU).

**Notable claim:** Sub-8-bit integer-only quantization is achievable for Transformer-based forecasters when the quantization scheme and training procedure (QAT, not PTQ) are built around the architecture's specific numerically difficult ops from the start.

**Caveat:** FPGA, not MCU — power/resource trade-offs and toolchain constraints differ substantially from TFLite Micro on an ESP32-C6. A follow-up paper by the same group (Ling et al. 2025, below) extends this to mixed-precision quantization and a broader task set, worth citing alongside it for completeness.

## [Ling et al. 2025] — Automating Versatile Time-Series Analysis with Tiny Transformers on Embedded FPGAs

**Venue:** arXiv preprint (2505.17662), 2025

**Task:** Forecasting, classification, and anomaly detection — three time-series tasks, six public datasets

**Hardware:** Two embedded FPGA platforms

**Framework:** A unified and fully automated deployment framework for Tiny Transformers on embedded FPGAs, combining quantization-aware training (down to 4 bits), hardware-aware hyperparameter search using Optuna, and automatic VHDL generation

**Quantization:** QAT, down to 4-bit

**Comparable to my work on:** Same task family (forecasting among others), same deployment target class (resource-constrained embedded hardware). The paper explicitly contrasts itself against MCU-targeted prior work, noting that "prior work targeting Microcontroller Units (MCUs) has explored hardware-specific optimizations, [but] such approaches are often task-specific and limited to 8-bit fixed-point precision" — this framing identifies exactly the ceiling (8-bit, PTQ-style) my own TTM investigation hit and could not get past on real MCU hardware.

**Caveat:** Same FPGA-vs-MCU caveat as above; also uses a custom compact Transformer architecture designed for quantizability from the start, unlike TTM, which is a pretrained model not designed with embedded int8 deployment in mind.