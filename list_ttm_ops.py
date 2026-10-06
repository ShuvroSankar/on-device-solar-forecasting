"""
List every op in a TFLite model, sorted, with counts. Run this on
exports/ttm_solar_final.tflite to get the exact set of Add*() calls
your MicroMutableOpResolver needs.
"""
import sys
from collections import Counter

from ai_edge_litert.interpreter import Interpreter


def main(path):
    interp = Interpreter(model_path=path)
    interp.allocate_tensors()

    print(f"Model: {path}\n")
    ins = interp.get_input_details()
    outs = interp.get_output_details()
    print("Inputs :", [(d["name"], d["dtype"].__name__, tuple(d["shape"])) for d in ins])
    print("Outputs:", [(d["name"], d["dtype"].__name__, tuple(d["shape"])) for d in outs])

    ops = interp._get_ops_details()
    counts = Counter(o["op_name"] for o in ops if "DELEGATE" not in o["op_name"].upper())

    print(f"\nDistinct ops: {len(counts)}\n")
    for name, c in sorted(counts.items()):
        print(f"  {name:<24} x{c}")

    # Print the op names in a ready-to-paste C++ form
    print("\n--- Suggested resolver additions (paste after existing Add*()s) ---")
    # Map TFLite op names to tflite::MicroMutableOpResolver method names.
    # Add more entries if the inspector reports ops not covered here.
    mapping = {
        "ADD": "AddAdd",
        "SUB": "AddSub",
        "MUL": "AddMul",
        "DIV": "AddDiv",
        "ABS": "AddAbs",
        "SIGN": "AddSign",
        "EXP": "AddExp",
        "RSQRT": "AddRsqrt",
        "SQRT": "AddSqrt",
        "POW": "AddPow",
        "MEAN": "AddMean",
        "SOFTMAX": "AddSoftmax",
        "TRANSPOSE": "AddTranspose",
        "CONCATENATION": "AddConcatenation",
        "RESHAPE": "AddReshape",
        "GATHER": "AddGather",
        "FULLY_CONNECTED": "AddFullyConnected",
        "FULLY_CONNECTED_INT8": "AddFullyConnected",
        "CAST": "AddCast",
        "LOGISTIC": "AddLogistic",
        "TANH": "AddTanh",
        "MUL": "AddMul",
        "MAXIMUM": "AddMaximum",
        "MINIMUM": "AddMinimum",
        "PAD": "AddPad",
        "STRIDED_SLICE": "AddStridedSlice",
        "EXPAND_DIMS": "AddExpandDims",
        "QUANTIZE": "AddQuantize",
        "DEQUANTIZE": "AddDequantize",
    }
    unresolved = []
    for name in sorted(counts):
        key = name.upper()
        if key in mapping:
            print(f"  resolver.{mapping[key]}();")
        else:
            unresolved.append(name)
    if unresolved:
        print(f"\n  // Unmapped op names (check names against micro_mutable_op_resolver.h):")
        for name in unresolved:
            print(f"  //   {name}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "exports/ttm_solar_final.tflite"
    main(path)
