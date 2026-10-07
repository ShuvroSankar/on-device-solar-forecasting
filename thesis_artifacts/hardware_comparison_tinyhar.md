# Hardware comparison: SmallTCN vs TinyHAR-Net (both on STM32F446RE)

Both models are deployed on the **same MCU** — the STM32 Nucleo-F446RE
(ARM Cortex-M4F, 180 MHz, 128 KB SRAM, 512 KB flash). This makes the
flash, RAM, and latency columns directly comparable across hardware.
The *task* differs (regression vs classification), so no accuracy column
is included; see the "Reading the numbers" section for what the
comparison does and does not show.

## Side-by-side

| | **SmallTCN (this work)** | **TinyHAR-Net (Hossan et al. 2025)** |
|---|---|---|
| Venue | — | IEEE Embedded Systems Letters |
| Task | Solar generation forecasting (regression) | Human activity recognition (classification) |
| MCU | STM32F446RE, Cortex-M4F @ 180 MHz | STM32 Nucleo-F446RE, Cortex-M4F @ 180 MHz |
| SRAM / flash | 128 KB / 512 KB | 128 KB / 512 KB |
| Parameters | 38,002 | 11,686 – 13,535 (dataset-dependent) |
| Arithmetic ops / inference | 3.17 M ops (1.585 M MACs) | 1.14 M FLOPs @ (120, f) input |
| Numerical precision | int8 (post-training quantized, after manual dilation rewrite) | float32 (unquantized by design) |
| Inference framework | ST Edge AI Core v4.0.1 (static codegen) | TFLite Micro (interpreter, converted to C header) |
| **Model weights in flash** | **40.7 KiB** (int8) | **43.78 – 51.30 KiB** (float32) |
| **Inference latency** | **18.2 ms** | **41.31 – 51.57 ms** |
| **Working RAM** | 16.4 KiB activations (26.3 KiB total AI) | **30.12 KiB** |
| Output vs reference | bit-exact (int8, both on-device and intermediate-tensor check) | not reported |
| Quantization applied? | yes — int8, required the manual-dilation rewrite to work | no — quantized variant tested and rejected |

## Reading the numbers

### Latency: SmallTCN is 2.3–2.8× faster despite doing ~2.8× more MACs

Three compounding factors explain the gap, not one:

1. **Precision.** SmallTCN runs int8; TinyHAR-Net runs float32.
2. **Runtime.** SmallTCN is compiled as static C by ST Edge AI Core; TinyHAR-Net
   runs through the TFLite Micro interpreter. Static codegen eliminates per-op
   dispatch overhead and enables whole-graph scheduling; interpretation does not.
3. **Architecture.** SmallTCN is a pure conv TCN. TinyHAR-Net is a
   depthwise-separable CNN + single-head Linformer self-attention. Attention
   materializes intermediate Q/K/V-like tensors, which increases both peak
   RAM and per-layer latency.

The relative contribution of each is not separable from the reported data —
doing so would require porting SmallTCN to TFLite Micro or TinyHAR-Net to
ST Edge AI Core, both outside the scope of this comparison.

### Flash: comparable in size, opposite in cost

Both models land in the 40–51 KiB range, but by very different routes:

- **TinyHAR-Net** hits ~44 KB **unquantized** — a natively efficient
  architecture.
- **SmallTCN** needs int8 to hit 40.7 KiB (float32 would be ~150 KiB), and
  int8 only worked after the manual-dilation rewrite documented in the
  progress report's §5.2.

Same ballpark size, but SmallTCN's compression path required a real
toolchain investigation; TinyHAR-Net's did not.

### Working RAM: SmallTCN uses roughly half

16.4 KiB activations vs 30.12 KiB. Both fit comfortably in the F446RE's
128 KB SRAM. The gap is consistent with the same architectural reason as
latency: attention requires more intermediate tensor storage than a pure
convolutional stack.

### Accuracy: no direct comparison

SmallTCN reports MASE/MAE/sMAPE (regression). TinyHAR-Net reports
accuracy/F1 (classification). Different output spaces, different metrics.
No accuracy column is included.

### Design-choice contrast: both models saw the int8 option, chose differently

This is the most instructive part of the comparison, and worth stating
plainly in the thesis:

| | TinyHAR-Net | SmallTCN |
|---|---|---|
| Float32 size | 43.78 KB | ~150 KB |
| After int8 PTQ | 12.38 KB (−72%) | 40.7 KB (−73%) |
| Accuracy cost | −1.62 pp F1 (WISDM) | +0.038 MASE (+4.7%) |
| Quantized version deployed? | **No** | **Yes** |
| Stated reason | "further compression offers little practical benefit while degrading performance" | float32 would not fit alongside the runtime |

The same quantization trade is right or wrong depending on **starting
footprint and application**. TinyHAR-Net already fits at float32 on a
512 KB part; quantizing it saves flash nobody needs and costs accuracy.
SmallTCN does not fit at float32 without pushing the runtime budget;
quantizing it is essential and the accuracy cost is acceptable.

This is a real engineering lesson, not a hedge.

## What the comparison does not claim

- **It does not claim SmallTCN is "better."** Faster and lower-RAM on
  the same MCU, yes. But the tasks are fundamentally different, and
  TinyHAR-Net's unquantized float32 accuracy is not the same thing as
  SmallTCN's quantized regression accuracy.
- **It does not claim ST Edge AI Core is faster than TFLite Micro in
  general.** Runtime and precision differ between the two deployments;
  this comparison cannot isolate the runtime effect.
- **It does not claim parameter efficiency.** TinyHAR-Net has ~3× fewer
  parameters but higher latency and RAM. Efficiency per parameter is
  architecture-dependent and not what this table is about.

## Reference

Hossan, I., Mary, M. N. J., & Motin, M. A. (2025). *TinyHAR-Net: Design
and Implementation of a TinyML-Based Human Activity Recognition Framework
on STM32.* IEEE Embedded Systems Letters. DOI: 10.1109/LES.2025.3639944.

Extracted figures from Tables II, III, IV, and V of the letter:
- Table II:  F1 per input shape across WISDM / MotionSense / MHEALTH
- Table III: Flash usage and inference latency per input shape
- Table IV:  Effect of full-integer quantization on F1, flash, latency
- Table V:   Comparison against CNN-Transformer and CNN-Linformer variants
- Table VI:  State-of-the-art comparison (parameters, flash, latency)

The 43.78 KB / 41.31 ms / 30.12 KB figures are for the optimal input
shape (120, f), which the paper identifies as the best overall
configuration across all three datasets.

