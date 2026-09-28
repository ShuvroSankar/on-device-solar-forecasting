"""
Diagnostic: dump every node in an ONNX graph, with its op_type and whether
its weight input (if any) is stored as int8/uint8 or still float32.

We've had two quantization attempts (QDQ, QOperator) both report summary
stats consistent with Conv layers NOT actually being quantized, despite
onnxruntime.quantization claiming to include Conv in op_types_to_quantize.
This script gives ground truth instead of inferring from file size or a
quant-op-type count.

Usage:
    python diagnose_onnx.py exports/small_tcn_solar_qop_int8.onnx
"""

import sys

import onnx
from onnx import numpy_helper


def diagnose(path):
    m = onnx.load(path)
    graph = m.graph

    # Map initializer name -> its dtype, so we can check each node's
    # weight input's actual stored type.
    init_dtype = {}
    for init in graph.initializer:
        arr = numpy_helper.to_array(init)
        init_dtype[init.name] = str(arr.dtype)

    print(f"Model: {path}")
    print(f"Total nodes: {len(graph.node)}")
    print(f"Total initializers (weights/constants): {len(graph.initializer)}\n")

    print(f"{'op_type':<20} {'name':<20} {'input dtypes':<40}")
    print("-" * 85)
    for node in graph.node:
        input_dtypes = []
        for inp in node.input:
            if inp in init_dtype:
                input_dtypes.append(f"{inp.split('/')[-1][:15]}:{init_dtype[inp]}")
        dtype_str = ", ".join(input_dtypes) if input_dtypes else "(no weight input)"
        print(f"{node.op_type:<20} {node.name[:20]:<20} {dtype_str:<40}")

    print("\n--- Conv/MatMul/Gemm nodes specifically ---")
    target_ops = {"Conv", "QLinearConv", "MatMul", "QLinearMatMul", "Gemm", "QGemm"}
    found_any_quantized_compute_op = False
    for node in graph.node:
        if node.op_type in target_ops:
            weight_dtypes = [init_dtype.get(inp, "?") for inp in node.input if inp in init_dtype]
            print(f"  {node.op_type} ({node.name}): weight dtypes = {weight_dtypes}")
            if node.op_type in {"QLinearConv", "QLinearMatMul", "QGemm"}:
                found_any_quantized_compute_op = True
            elif any(d in ("int8", "uint8") for d in weight_dtypes):
                found_any_quantized_compute_op = True

    print()
    if found_any_quantized_compute_op:
        print("CONFIRMED: at least one compute-heavy op has genuinely int8 weights.")
    else:
        print("PROBLEM CONFIRMED: no Conv/MatMul/Gemm op has int8 weights. "
              "The model's compute-heavy layers are still fp32, regardless "
              "of what the file size or op-type summary suggested.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagnose_onnx.py <path_to_onnx>")
        sys.exit(1)
    diagnose(sys.argv[1])
