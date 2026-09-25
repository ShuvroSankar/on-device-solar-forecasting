# On-Device Solar Power Forecasting Using Compressed Time-Series Foundation Models

MSCS Thesis Project — AIUB
Shuvro Sankar Sen | ID: 25-93776-2 | Supervisor: Dr. Rajarshi Roy Chowdhury

## Overview

Deploying compressed time-series foundation models (IBM's Tiny Time Mixer,
TTM) on true bare-metal microcontrollers (STM32F446RE, ESP32-C6) for
short-term (15–90 min) solar power forecasting, benchmarked against a
purpose-built small model (SmallTCN) trained from scratch for the same
hardware.

## Dataset

Public UK PV generation dataset (Open Climate Fix), 5-minute resolution
across ~700+ usable sites after filtering. Not committed to this repo —
regenerate locally with the steps below.

Source: https://huggingface.co/datasets/openclimatefix/uk_pv

## Repo structure

```
.
├── preprocess_ukpv.py       # Phase 1: raw parquet -> filtered/cleaned/split dataset
├── data_pipeline.py         # Original REFIT (load-data) pipeline + Normalizer (reused)
├── tcn_model.py              # SmallTCN architecture (configurable in_channels)
├── solar_data_pipeline.py    # Solar-specific multi-site windowing, capacity loading
├── train_tcn.py              # Original REFIT training script (proxy baseline)
├── train_solar_tcn.py        # Solar SmallTCN training script (current model)
├── requirements.txt
└── checkpoints/               # Small (tens of KB) trained model checkpoints
```

## Reproducing

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Download the dataset (5-minutely only; see chat history / notes for the
# full huggingface_hub snapshot_download command with allow_patterns)

# Phase 1: filter, clean, split into train/val/test by year
python3 preprocess_ukpv.py --stage inspect
python3 preprocess_ukpv.py --stage filter
python3 preprocess_ukpv.py --stage split

# Train SmallTCN on solar data
python3 train_solar_tcn.py --stride 24
```

## Current results (SmallTCN, solar)

Config: context=36 (3h), forecast=18 (90min), 5 input channels
(generation + sin/cos hour-of-day + sin/cos day-of-year), capacity-normalized.

| Metric | Value |
|---|---|
| MASE | 0.807 |
| MAE | 9.93 Wh |
| RMSE | 22.37 Wh |
| sMAPE (daylight only) | 46.15% |

## Status

- [x] Phase 1: dataset pipeline (filter, clean, split)
- [x] SmallTCN trained on solar data (replacing load-data proxy result)
- [ ] TTM adaptation on solar data
- [ ] Quantization + deployment (STM32F446RE, ESP32-C6)
- [ ] Hardware benchmarking vs. TinyHAR-Net
- [ ] Accuracy comparison vs. Kaas et al. / PV-forecasting benchmark
- [ ] Network cost validation (MQTT vs on-device) + live dashboard demo
