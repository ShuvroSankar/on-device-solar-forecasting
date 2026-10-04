"""
Parity check: int8 TFLite vs ONNX reference, on the same 200 z-scored
calibration windows used to build the int8 calibration file.

This is the authoritative test. Op lists and file-size percentages are
diagnostic; only numerical agreement proves the int8 model is usable.

Layouts:
  exports/ttm_calibration_windows.npy  (200, 52, 1)  ONNX order, z-scored
  ONNX expects past_values             (1, 52, 1)
  int8 TFLite expects past_values      (1, 1, 52)    TF order
"""
import numpy as np
import onnxruntime as ort
from ai_edge_litert.interpreter import Interpreter

ONNX_PATH         = "exports/ttm_solar.onnx"
INT8_TFLITE_PATH  = "exports/ttm_solar_int8/ttm_solar_full_integer_quant.tflite"
CALIB_WINDOWS     = "exports/ttm_calibration_windows.npy"
FREQ_TOKEN        = 3
RTOL              = 0.10   # 10% relative error budget for int8


def run_onnx_reference(windows):
    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    outs = []
    for z in windows:
        pv = z.reshape(1, 52, 1).astype(np.float32)
        ft = np.array([FREQ_TOKEN], dtype=np.int64)
        out = sess.run(None, {"past_values": pv, "freq_token": ft})[0]
        outs.append(out)
    return outs


def find_input(details, needle):
    for d in details:
        if needle in d["name"]:
            return d
    raise KeyError(f"No input matching {needle!r} in {[d['name'] for d in details]}")


def run_int8_tflite(windows):
    interp = Interpreter(model_path=INT8_TFLITE_PATH)
    interp.allocate_tensors()
    ins = interp.get_input_details()
    outs = interp.get_output_details()
    print("int8 inputs :", [(d["name"], d["dtype"].__name__, d["shape"]) for d in ins])
    print("int8 outputs:", [(d["name"], d["dtype"].__name__, d["shape"]) for d in outs])

    pv = find_input(ins, "past_values")
    ft = find_input(ins, "freq_token")
    out_d = outs[0]

    pv_scale, pv_zp = pv["quantization"]
    out_scale, out_zp = out_d["quantization"]
    print(f"past_values  quant: scale={pv_scale!r} zp={pv_zp!r}")
    print(f"output       quant: scale={out_scale!r} zp={out_zp!r}")

    results = []
    for z in windows:
        z_tf = z.reshape(1, 1, 52).astype(np.float32)
        if pv["dtype"] == np.int8:
            q = np.clip(np.round(z_tf / pv_scale + pv_zp), -128, 127).astype(np.int8)
        else:
            q = z_tf.astype(pv["dtype"])
        interp.set_tensor(pv["index"], q)
        interp.set_tensor(ft["index"], np.array([FREQ_TOKEN], dtype=ft["dtype"]))
        interp.invoke()
        out_q = interp.get_tensor(out_d["index"])
        if out_d["dtype"] == np.int8:
            results.append((out_q.astype(np.float32) - out_zp) * out_scale)
        else:
            results.append(out_q.astype(np.float32))
    return results


def main():
    windows = np.load(CALIB_WINDOWS)
    print(f"Loaded {windows.shape[0]} windows, shape {windows.shape} (z-scored, ONNX order).\n")

    print("--- ONNX reference ---")
    onnx_outs = run_onnx_reference(windows)
    print(f"Captured {len(onnx_outs)}, shape {onnx_outs[0].shape}.\n")

    print("--- int8 TFLite ---")
    tflite_outs = run_int8_tflite(windows)

    # Flatten both to compare element-wise regardless of layout
    onnx_arr   = np.array([o.reshape(-1) for o in onnx_outs])
    tflite_arr = np.array([o.reshape(-1) for o in tflite_outs])

    diff = np.abs(tflite_arr - onnx_arr)
    scale = float(np.max(np.abs(onnx_arr)))
    max_abs = float(np.max(diff))
    mean_abs = float(np.mean(diff))
    p95 = float(np.percentile(diff, 95))
    rel_max = max_abs / scale if scale else float("inf")
    corr = float(np.corrcoef(onnx_arr.flatten(), tflite_arr.flatten())[0, 1])

    print("\n=== Parity: int8 TFLite vs ONNX ===")
    print(f"Max |diff|         : {max_abs:.6f}")
    print(f"Mean |diff|        : {mean_abs:.6f}")
    print(f"95th pct |diff|    : {p95:.6f}")
    print(f"ONNX max |y|       : {scale:.6f}")
    print(f"Relative max error : {rel_max*100:.2f}%")
    print(f"Pearson correlation: {corr:.6f}")

    if rel_max <= RTOL:
        print(f"\nPASS: within rtol={RTOL*100:.0f}%")
    else:
        print(f"\nWARN: relative max error {rel_max*100:.2f}% > rtol={RTOL*100:.0f}%")
        print("Int8 quantization introduces more error than expected.")
        print("Check whether the Erf hybrid-quant path (DEQUANTIZE/QUANTIZE pairs)")
        print("is degrading quality. Correlation is the tie-breaker: high corr +")
        print("high max error usually means a few outlier samples, not a broken model.")


if __name__ == "__main__":
    main()
