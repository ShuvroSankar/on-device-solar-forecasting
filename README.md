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
with a verified parity check (max abs diff 0.000001 vs. PyTorch, 16 random samples). TensorFlow
conversion (`onnx2tf`) and int8 TFLite export are the next step -- not yet done.

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

## Status

- [x] Dataset pipeline
- [x] SmallTCN trained on solar data
- [x] int8 deployment on STM32F446RE and ESP32-C6
- [x] TTM fine-tuned and exported to ONNX (parity-checked)
- [ ] int8 accuracy of SmallTCN on the full test set
- [ ] TTM: ONNX -> TensorFlow (`onnx2tf`) -> int8 TFLite -> deploy on ESP32-C6
- [ ] Hardware benchmark vs published works (TinyHAR-Net)
- [ ] Accuracy comparison vs published cloud-side PV forecasting models
- [ ] MQTT network cost and live dashboard (needs solar panel)
- [ ] Repo cleanup: remove superseded/debug scripts (`export_quantize_tcn.py`,
      `export_quantize_tcn_v2.py`, `torch_to_tflite_tcn.py`, `*_debug.py`, `*.broken`)
