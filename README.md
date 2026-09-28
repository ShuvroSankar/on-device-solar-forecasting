# On-Device Solar Power Forecasting Using Compressed Time-Series Foundation Models

MSCS Thesis Project — AIUB
Shuvro Sankar Sen | ID: 25-93776-2 | Supervisor: Dr. Rajarshi Roy Chowdhury

## Overview

Can IBM's Tiny Time Mixer (TTM), a pretrained time-series foundation model, be compressed and deployed on bare-metal microcontrollers (STM32F446RE, ESP32-C6) for short-term (15–90 min) solar power forecasting? TTM is benchmarked against a purpose-built small model (SmallTCN) trained from scratch for the same hardware.

So far, SmallTCN is trained on solar data and runs in int8 on **both boards**, with on-device output matching the reference model bit-exactly. TTM adaptation is the next step.

## Dataset

Public UK PV generation dataset (Open Climate Fix), 5-minute resolution, 2018–2024. After cleaning, **703 sites and 231.4M rows** are kept (from 1,905 sites and 708.6M raw rows). Split by year with no shuffling: train 2018–2022, validation 2023, test 2024. The dataset is not committed to this repo; regenerate it locally with the steps below.

Source: https://huggingface.co/datasets/openclimatefix/uk_pv

## Repo structure

```
.
├── preprocess_ukpv.py                       # raw parquet -> filtered/cleaned/split dataset
├── data_pipeline.py                         # Original REFIT (load-data) pipeline + Normalizer (reused)
├── tcn_model.py                             # SmallTCN architecture (configurable in_channels)
├── solar_data_pipeline.py                   # Solar multi-site windowing, capacity loading
├── train_tcn.py                             # Original REFIT training script (proxy baseline)
├── train_solar_tcn.py                       # Solar SmallTCN training script (current model)
├── torch_to_tflite_tcn_manual_dilation.py   # PyTorch -> Keras port (with parity check) + int8 TFLite
├── prepare_esp32_deployment.py              # Model + test vector as C arrays
├── esp32_project/                           # ESP-IDF app (TFLite Micro)
├── test_minimal_quant.py                    # Minimal repro of the STM32 dilated-conv issue
├── models/
│   └── small_tcn_solar_manual_int8.tflite   # Final int8 model
├── requirements.txt
└── checkpoints/                             # Small (tens of KB) trained model checkpoints
```

## Reproducing

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Download the dataset (5-minutely files only) from
# https://huggingface.co/datasets/openclimatefix/uk_pv
# <add your huggingface_hub snapshot_download command with allow_patterns here>

# Phase 1: filter, clean, split into train/val/test by year
python3 preprocess_ukpv.py --stage inspect
python3 preprocess_ukpv.py --stage filter
python3 preprocess_ukpv.py --stage split

# Train SmallTCN on solar data
python3 train_solar_tcn.py --stride 24

# Port to Keras (manual-dilation form) and export int8 TFLite
python3 torch_to_tflite_tcn_manual_dilation.py

# Generate C arrays for the ESP32 project
python3 prepare_esp32_deployment.py
```

## Model and forecasting results (SmallTCN, solar)

SmallTCN: 4 dilated causal conv blocks (channels 32/32/64/64, kernel 5, dilations 1/2/4/8) plus a linear head, 38,002 parameters.

Config: context = 36 steps (3 h), forecast = 18 steps (90 min), 5 input channels (generation + sin/cos hour-of-day + sin/cos day-of-year), capacity-normalized.

Deployed checkpoint, test set (2024). Naive persistence baseline: MAE 12.31 Wh.

| Metric | Value |
|---|---|
| MASE | 0.808 (about 19% lower error than persistence) |
| MAE | 9.95 Wh |
| RMSE | 22.25 Wh |
| sMAPE (daylight only) | 44.77% |

Run-to-run variance is about ±0.001 MASE. Plain sMAPE over all points (about 123%) is inflated by night-time zeros, so daylight-only sMAPE is reported.

## Deployment results (int8, measured on hardware)

| | STM32F446RE | ESP32-C6 |
|---|---|---|
| Toolchain | ST Edge AI Core v4.0.1 (static C code) | TFLite Micro (esp-tflite-micro 1.4.1) |
| Clock | 180 MHz, Cortex-M4 | 160 MHz, RISC-V |
| Inference time | 18.2 ms | 97.7 ms |
| Model weights | 41,708 B (40.7 KiB), −73.1% vs float | 102,536 B model file (flatbuffer incl. metadata) |
| Working RAM | 16.4 KiB activations (26.3 KiB AI total) | 144.4 KiB tensor arena (upper bound, see below) |
| Output vs reference | rmse 0, cosine 1.0 | max abs diff 0 |

Caveats:
- The boards use different runtimes (generated C code vs. an interpreter), so the latency gap is not attributable to hardware alone.
- The ESP32 RAM figure is inflated: `preserve_all_tensors=true` disables buffer reuse. An ablation is pending.
- The on-target checks verify deployment fidelity, not forecast accuracy.

### Finding: STM32 silently skips quantization of dilated convolutions

ST Edge AI Core (STM32F4 backend, v4.0.1) runs the `SpaceToBatchND → Conv2D → BatchToSpaceND` pattern that TFLite generates for dilated convs in float32, so int8 exports showed almost no compression (−1.6%). The fix rewrites each dilated conv as *k* static crops plus *k* pointwise convs, summed. This is mathematically identical (max difference 1.8e-15) and brought latency from 84.8 ms to 18.2 ms. `test_minimal_quant.py` reproduces the issue.

## Status

- [x] Dataset pipeline (filter, clean, split)
- [x] SmallTCN trained on solar data (MASE 0.808)
- [x] Quantization + deployment of SmallTCN (STM32F446RE, ESP32-C6), verified bit-exact
- [ ] Full-test-set accuracy of the int8 model (MASE/MAE vs. fp32)
- [ ] ESP32 ablation (`preserve_all_tensors=false`) to confirm root cause and recover RAM
- [ ] TTM adaptation on solar data, quantization and deployment
- [ ] Hardware benchmarking vs. TinyHAR-Net (SmallTCN numbers ready; TTM and comparison table pending)
- [ ] Accuracy comparison vs. published cloud-side PV forecasting models (Thipwangmek et al., 2024; Zhou et al., 2019)
- [ ] Network cost validation (MQTT vs. on-device) + live dashboard demo (waiting on solar panel)
- [ ] Adaptive updating (TEDA-RLS), scope to be discussed
