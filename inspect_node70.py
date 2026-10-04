"""
Inspect node 70 (the universally-failing DIV) directly:
  - what ops is it, what are its inputs/outputs
  - are any inputs constants (not data-dependent)
  - what quantization params were assigned to each tensor

Hypothesis under test: one of node 70's two inputs is a constant weight
or architectural scalar whose real float value quantizes to int8 0, which
would explain universal failure independent of past_values.
"""
import numpy as np
from ai_edge_litert.interpreter import Interpreter

PATH = "exports/ttm_solar_int8/ttm_solar_full_integer_quant.tflite"
NODE_INDEX = 70

interp = Interpreter(model_path=PATH)
interp.allocate_tensors()

ops = interp._get_ops_details()

# _get_ops_details() indices are not guaranteed to match 'index' field;
# find by matching op_name/index if present, else fall back to position.
node = None
for i, o in enumerate(ops):
    if o.get("index", i) == NODE_INDEX:
        node = o
        break
if node is None and NODE_INDEX < len(ops):
    node = ops[NODE_INDEX]
    print(f"(No 'index' field matched; using positional ops[{NODE_INDEX}])")

print(f"Node {NODE_INDEX}: {node}\n")

tensors = {t["index"]: t for t in interp.get_tensor_details()}

print(f"--- Inputs to node {NODE_INDEX} ---")
for idx in node["inputs"]:
    t = tensors[idx]
    scale, zp = t["quantization"]
    print(f"  tensor {idx}:")
    print(f"    name  = {t['name']}")
    print(f"    dtype = {t['dtype'].__name__}  shape = {t['shape']}")
    print(f"    quant = scale={scale!r} zero_point={zp!r}")
    try:
        v = interp.get_tensor(idx)
        flat = v.flatten()
        print(f"    value (first 8): {flat[:8]}")
        print(f"    value min={v.min()} max={v.max()} "
              f"all_zero={(flat == 0).all()} any_zero={(flat == 0).any()}")
        if scale is not None and scale > 0:
            # dequantize to see real-space value
            deq = (v.astype(np.float32) - zp) * scale
            print(f"    dequantized min={deq.min():.6f} max={deq.max():.6f} "
                  f"mean={deq.mean():.6f}")
    except Exception as e:
        print(f"    (not readable: {e})")

print(f"\n--- Outputs of node {NODE_INDEX} ---")
for idx in node["outputs"]:
    t = tensors[idx]
    print(f"  tensor {idx}: name={t['name']} dtype={t['dtype'].__name__} "
          f"shape={t['shape']} quant={t['quantization']}")

# Also dump every DIV node in the graph, so we can see whether all 4+
# reported failing nodes (70, 238, 154, 195) have the same structure.
print(f"\n--- All DIV nodes in the graph ---")
div_nodes = [o for o in ops if o["op_name"].upper() == "DIV" or "DIV" in o["op_name"].upper()]
for o in div_nodes:
    ins = o["inputs"]
    oidx = o["outputs"][0] if o["outputs"] else None
    print(f"  node {o.get('index', '?')}: inputs={ins} output={oidx}")
    for idx in ins:
        t = tensors[idx]
        scale, zp = t["quantization"]
        try:
            v = interp.get_tensor(idx)
            zero_count = int((v.flatten() == 0).sum())
            total = v.size
            tag = f"ZEROS={zero_count}/{total}" if zero_count else ""
        except Exception:
            tag = "(unreadable)"
        print(f"    in  {idx}: {t['name']}  q=({scale}, {zp})  {tag}")
