"""
Read the int8 TFLite flatbuffer directly, bypassing any interpreter/delegate,
so we see the TRUE on-disk weight-byte breakdown by dtype -- not the
delegate-colored runtime view that XNNPACK would give us on desktop.

If schema_generated.py doesn't import cleanly from one of the search
directories below, this falls back to printing the interpreter's tensor
dtype counts, which is still useful (just includes runtime delegate noise).
"""
import os
import sys
from collections import Counter

TFLITE_PATH = "exports/ttm_solar_int8/ttm_solar_full_integer_quant.tflite"

# onnx2tf exports a schema_generated.py alongside every TFLite it writes.
for d in ("exports/ttm_solar_savedmodel", "exports/ttm_solar_tf"):
    if os.path.exists(os.path.join(d, "schema_generated.py")):
        sys.path.insert(0, d)
        break

dtype_map = {
    0: "float32", 1: "float16", 2: "int32", 3: "uint8",
    4: "int64", 6: "bool", 7: "int16", 9: "int8", 10: "float64",
}

try:
    from schema_generated import Model

    with open(TFLITE_PATH, "rb") as f:
        buf = f.read()

    model = Model.GetRootAsModel(buf, 0)
    subgraph = model.Subgraphs(0)

    dtype_bytes, dtype_count = {}, {}
    for i in range(subgraph.TensorsLength()):
        t = subgraph.Tensors(i)
        b = model.Buffers(t.Buffer())
        nbytes = b.DataLength() if not b.DataIsNone() else 0
        dt = dtype_map.get(t.Type(), f"other({t.Type()})")
        dtype_bytes[dt] = dtype_bytes.get(dt, 0) + nbytes
        dtype_count[dt] = dtype_count.get(dt, 0) + 1

    total = sum(dtype_bytes.values())
    print(f"Raw flatbuffer: {TFLITE_PATH}")
    print(f"Total weight bytes: {total:,}\n")
    print(f"{'dtype':<12} {'bytes':>14} {'%':>7} {'tensors':>9}")
    for dt in sorted(dtype_bytes, key=lambda x: -dtype_bytes[x]):
        n, c = dtype_bytes[dt], dtype_count[dt]
        pct = 100 * n / total if total else 0
        print(f"{dt:<12} {n:>14,} {pct:>6.1f}% {c:>9}")

except Exception as e:
    print(f"Raw flatbuffer parse failed ({e}); falling back to interpreter view.")
    from ai_edge_litert.interpreter import Interpreter
    interp = Interpreter(model_path=TFLITE_PATH)
    interp.allocate_tensors()
    dtypes = Counter(str(t["dtype"].__name__) for t in interp.get_tensor_details())
    print("Interpreter tensor dtype counts (delegate-tainted on desktop):", dict(dtypes))
