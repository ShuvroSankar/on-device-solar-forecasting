"""
Test any TFLite model (int8, int16, or float32) against the ONNX reference
on the 200 calibration windows. Auto-detects dtypes and dequantizes outputs
so the comparison is always in real (float) space.

Usage:
    python test_quantized_model.py exports/ttm_solar_quant/ttm_solar_float32_fallback.tflite
    python test_quantized_model.py exports/ttm_solar_quant/ttm_solar_int16x8_quant.tflite
"""
import sys

import numpy as np
import onnxruntime as ort
from ai_edge_litert.interpreter import Interpreter

ONNX_PATH = "exports/ttm_solar.onnx"
WINDOWS = "exports/ttm_calibration_windows.npy"
FREQ_TOKEN = 3


def onnx_forward(windows):
    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    outs = []
    for z in windows:
        pv = z.reshape(1, 52, 1).astype(np.float32)
        ft = np.array([FREQ_TOKEN], dtype=np.int64)
        outs.append(sess.run(None, {"past_values": pv, "freq_token": ft})[0])
    return outs


def find_input(details, needle):
    for d in details:
        if needle in d["name"]:
            return d
    raise KeyError(f"No input matching {needle!r} in {[d['name'] for d in details]}")


def main(model_path):
    windows = np.load(WINDOWS)
    print(f"Loaded {windows.shape[0]} windows, shape {windows.shape}.\n")

    interp = Interpreter(model_path=model_path)
    interp.allocate_tensors()
    ins = interp.get_input_details()
    outs = interp.get_output_details()

    print("TFLite inputs :", [(d["name"], d["dtype"].__name__, d["shape"]) for d in ins])
    print("TFLite outputs:", [(d["name"], d["dtype"].__name__, d["shape"]) for d in outs])

    pv = find_input(ins, "past_values")
    ft = find_input(ins, "freq_token")
    od = outs[0]

    pv_scale, pv_zp = pv["quantization"]
    od_scale, od_zp = od["quantization"]
    print(f"past_values quant: scale={pv_scale!r} zp={pv_zp!r}")
    print(f"output      quant: scale={od_scale!r} zp={od_zp!r}\n")

    print("--- ONNX reference ---")
    onnx_outs = onnx_forward(windows)
    print(f"Captured {len(onnx_outs)}, shape {onnx_outs[0].shape}.\n")

    print("--- TFLite ---")
    tflite_outs = []
    ok, failed = 0, 0
    first_err = None
    for i, z in enumerate(windows):
        z_tf = z.reshape(1, 1, 52).astype(np.float32)

        if pv["dtype"] == np.int16:
            q = np.clip(np.round(z_tf / pv_scale + pv_zp), -32768, 32767).astype(np.int16)
        elif pv["dtype"] == np.int8:
            q = np.clip(np.round(z_tf / pv_scale + pv_zp), -128, 127).astype(np.int8)
        else:
            q = z_tf.astype(pv["dtype"])

        interp.set_tensor(pv["index"], q)
        interp.set_tensor(ft["index"], np.array([FREQ_TOKEN], dtype=ft["dtype"]))
        try:
            interp.invoke()
            out_q = interp.get_tensor(od["index"])
            if od["dtype"] in (np.int8, np.int16):
                out = (out_q.astype(np.float32) - od_zp) * od_scale
            else:
                out = out_q.astype(np.float32)
            tflite_outs.append(out)
            ok += 1
        except Exception as e:
            failed += 1
            if first_err is None:
                first_err = (i, str(e).splitlines()[0][:160])

    print(f"OK: {ok}/{len(windows)}  FAILED: {failed}/{len(windows)}")
    if first_err:
        print(f"First failure at window {first_err[0]}: {first_err[1]}")

    if not tflite_outs:
        print("\nNo outputs -- cannot compare.")
        return

    n = min(len(onnx_outs), len(tflite_outs))
    a = np.array([onnx_outs[i].reshape(-1) for i in range(n)])
    b = np.array([tflite_outs[i].reshape(-1) for i in range(n)])
    diff = np.abs(a - b)
    scale = float(np.max(np.abs(a)))
    print(f"\n=== TFLite vs ONNX (first {n} windows) ===")
    print(f"Max  |diff|: {diff.max():.6f}")
    print(f"Mean |diff|: {diff.mean():.6f}")
    print(f"ONNX max |y|: {scale:.6f}")
    print(f"Relative max error: {100 * diff.max() / scale:.4f}%")
    corr = float(np.corrcoef(a.flatten(), b.flatten())[0, 1])
    print(f"Pearson correlation: {corr:.6f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
