"""
Export a SmallTCN checkpoint to ONNX, then apply int8 dynamic quantization
using the SAME methodology already confirmed for your TTM export
(ttm_512_96_int8.onnx: producer "onnx.quantize 0.1.0", built from
DynamicQuantizeLinear + MatMulInteger ops -- i.e.
onnxruntime.quantization.quantize_dynamic).

IMPORTANT DIFFERENCE FROM TTM:
TTM is transformer-based (MatMul/Gemm-heavy), so ONNX Runtime's default
dynamic-quantization op set already covers nearly all of its parameters.
SmallTCN is a CNN -- its dilated causal Conv1d layers hold almost all of
its ~38K parameters. If 'Conv' isn't explicitly included in
op_types_to_quantize, dynamic quantization only touches the tiny final
Linear head and leaves the conv weights (the vast majority of the model)
in fp32 -- silently producing a misleading "int8" model that barely
shrank. This script always passes op_types_to_quantize explicitly so
that doesn't happen quietly.

Usage:
    python export_quantize_tcn.py --checkpoint checkpoints/small_tcn_solar.pt --out_prefix small_tcn_solar
"""

import argparse
import os

import torch
from onnxruntime.quantization import quantize_dynamic, QuantType

from tcn_model import SmallTCN


def export_and_quantize(checkpoint_path, out_prefix, out_dir="exports"):
    os.makedirs(out_dir, exist_ok=True)

    # weights_only=False: safe here since this is your own checkpoint,
    # produced locally by train_solar_tcn.py -- newer PyTorch defaults
    # torch.load to weights_only=True, which refuses to unpickle the
    # numpy scalars (norm_mean/norm_std) this checkpoint also carries.
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    context_length = ckpt["context_length"]
    forecast_length = ckpt["forecast_length"]
    in_channels = ckpt.get("in_channels", 1)  # defaults to 1 for old REFIT checkpoints

    print(f"Loaded checkpoint: context_length={context_length}, "
          f"forecast_length={forecast_length}, in_channels={in_channels}")

    model = SmallTCN(
        context_length=context_length,
        forecast_length=forecast_length,
        in_channels=in_channels,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dummy = torch.randn(1, context_length, in_channels)

    fp32_path = os.path.join(out_dir, f"{out_prefix}.onnx")
    int8_path = os.path.join(out_dir, f"{out_prefix}_int8.onnx")

    # Fixed batch size of 1 -- deployment is single-sample inference on
    # MCU, no reason to carry dynamic-batch overhead into the graph.
    torch.onnx.export(
        model,
        dummy,
        fp32_path,
        input_names=["input"],
        output_names=["forecast"],
        opset_version=17,
        dynamo=False,  # use the stable legacy exporter -- the newer
                        # dynamo-based exporter (now default on recent
                        # PyTorch) requires the separate onnxscript
                        # package and isn't needed for a model this simple
    )
    fp32_kb = os.path.getsize(fp32_path) / 1024
    print(f"Exported fp32 ONNX -> {fp32_path} ({fp32_kb:.1f} KB)")

    quantize_dynamic(
        fp32_path,
        int8_path,
        weight_type=QuantType.QUInt8,
        # Explicit on purpose -- see module docstring. Conv MUST be
        # included or SmallTCN's conv weights (almost all its params)
        # are silently left in fp32.
        op_types_to_quantize=["Conv", "MatMul", "Gemm"],
    )
    int8_kb = os.path.getsize(int8_path) / 1024
    print(f"Exported int8 ONNX -> {int8_path} ({int8_kb:.1f} KB)")

    print(f"\nCompression: {fp32_kb:.1f} KB -> {int8_kb:.1f} KB ({fp32_kb / int8_kb:.2f}x)")

    # Sanity check: confirm Conv actually got quantized, not just skipped.
    import onnx
    m = onnx.load(int8_path)
    op_types = {n.op_type for n in m.graph.node}
    has_conv_quant = "ConvInteger" in op_types or "QLinearConv" in op_types
    print(f"Conv layers quantized: {'YES' if has_conv_quant else 'NO -- CHECK THIS, something is wrong'}")
    print(f"Quantization-related ops present: "
          f"{sorted(op for op in op_types if 'Quant' in op or 'Integer' in op)}")

    return fp32_path, int8_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out_prefix", required=True)
    parser.add_argument("--out_dir", default="exports")
    args = parser.parse_args()
    export_and_quantize(args.checkpoint, args.out_prefix, args.out_dir)
