# On-Device Solar Power Forecasting Using Compressed Time-Series Models

MSCS thesis project, AIUB. Shuvro Sankar Sen (ID 25-93776-2). Supervisor: Dr. Rajarshi Roy Chowdhury.

A small dilated-causal-CNN forecaster (**SmallTCN**, 38K parameters) is trained on the public UK PV dataset
and deployed in int8 on two microcontrollers, **STM32F446RE** and **ESP32-C6**. The goal is short-term
(15 to 90 min) solar output forecasting on genuinely constrained hardware.

Full progress write-up: [`PROGRESS_REPORT.md`](PROGRESS_REPORT.md)

## Results

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

## Repo layout

```
preprocess_ukpv.py                       filter / clean / split the dataset (3 stages)
data_pipeline.py                         REFIT pipeline + Normalizer (reused)
solar_data_pipeline.py                   multi-site solar windowing, capacity loading
tcn_model.py                             SmallTCN (configurable input channels)
train_tcn.py                             original REFIT training (proxy baseline)
train_solar_tcn.py                       solar training (current model)
export_quantize_tcn_v2.py                ONNX export + int8 quantization (dynamic/static)
diagnose_onnx.py                         inspect which ops in an ONNX graph are truly int8
torch_to_tflite_tcn.py                   PyTorch -> Keras port with parity check -> int8 TFLite
torch_to_tflite_tcn_manual_dilation.py   same, with dilation rewritten as shifted 1x1 convs (final)
test_minimal_quant.py                    minimal repro of the STM32 dilation issue
prepare_esp32_deployment.py              model + test vector as C arrays for the ESP32 firmware
esp32_project/                           ESP-IDF app (TFLite Micro)
models/                                  final int8 TFLite model
checkpoints/                             trained SmallTCN checkpoint
```

## Reproduce

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 1. dataset (5-minutely files only): huggingface_hub snapshot_download with
#    allow_patterns=['5_minutely/*','metadata.csv','bad_data.csv','README.md']
python3 preprocess_ukpv.py --stage filter
python3 preprocess_ukpv.py --stage split

# 2. train
python3 train_solar_tcn.py --stride 24

# 3. int8 TFLite (with PyTorch vs Keras parity check)
python3 torch_to_tflite_tcn_manual_dilation.py \
    --checkpoint checkpoints/small_tcn_solar.pt --out_prefix small_tcn_solar_manual

# 4. ESP32-C6 (separate terminal, ESP-IDF environment only, never pip install here)
cd esp32_project && . ~/esp/esp-idf/export.sh
idf.py set-target esp32c6 && idf.py build && idf.py -p <PORT> flash monitor
```

STM32: import `models/small_tcn_solar_manual_int8.tflite` into STM32Cube AI Studio (type: TFLite,
target stm32f4), then Analyze / Generate / Validate on target.

## Key finding

ST Edge AI Core v4.0.1 (STM32F4 backend) runs dilated convolutions in float32 even when the model file is
int8, so weights stayed near float size. Rewriting each dilated conv as k shifted pointwise convs (identical
math) fixes it. Details and the minimal repro are in the report.

## Status

- [x] Dataset pipeline
- [x] SmallTCN trained on solar data
- [x] int8 deployment on STM32F446RE and ESP32-C6
- [ ] int8 accuracy on the full test set
- [ ] TTM adaptation and deployment
- [ ] Hardware benchmark vs published works
- [ ] MQTT network cost and live dashboard (needs solar panel)
