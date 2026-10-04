"""
Run the float32 TFLite model (produced directly by onnx2tf, before the
manual int8 step) on the same 200 calibration windows used by the int8
parity check. If this runs cleanly, the int8 failure is specific to
quantization -- not an architecture/conversion issue inherited from onnx2tf.

Also compares float32 TFLite output against the ONNX reference so we know
the float32 model is still numerically correct at this point in the
pipeline (its own parity check was done earlier, but this re-confirms on
the exact same 200 windows the int8 path crashed on).
"""
import numpy as np
import onnxruntime as ort
from ai_edge_litert.interpreter import Interpreter

ONNX_PATH        = "exports/ttm_solar.onnx"
FLOAT32_TFLITE   = "exports/ttm_solar_tf/ttm_solar_float32.tflite"
WINDOWS          = "exports/ttm_calibration_windows.npy"
FREQ_TOKEN       = 3


def run_onnx(windows):
    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    outs = []
    for z in windows:
        pv = z.reshape(1, 52, 1).astype(np.float32)
        ft = np.array([FREQ_TOKEN], dtype=np.int64)
        outs.append(sess.run(None, {"past_values": pv, "freq_token": ft})[0])
    return outs


def run_float32_tflite(windows):
    interp = Interpreter(model_path=FLOAT32_TFLITE)
    interp.allocate_tensors()
    ins = interp.get_input_details()
    outs = interp.get_output_details()
    print("float32 TFLite inputs :",
          [(d["name"], d["dtype"].__name__, d["shape"]) for d in ins])
    print("float32 TFLite outputs:",
          [(d["name"], d["dtype"].__name__, d["shape"]) for d in outs])

    pv = next(d for d in ins if "past_values" in d["name"])
    ft = next(d for d in ins if "freq_token" in d["name"])
    od = outs[0]

    results = []
    ok, failed = 0, 0
    first_err = None
    for i, z in enumerate(windows):
        # onnx2tf's float32 tflite expects TF order (1,1,52), same as int8
        z_tf = z.reshape(1, 1, 52).astype(np.float32)
        interp.set_tensor(pv["index"], z_tf)
        interp.set_tensor(ft["index"], np.array([FREQ_TOKEN], dtype=ft["dtype"]))
        try:
            interp.invoke()
            results.append(interp.get_tensor(od["index"]))
            ok += 1
        except Exception as e:
            failed += 1
            if first_err is None:
                first_err = (i, str(e).splitlines()[0][:160])

    print(f"\nfloat32 inference: OK={ok}/{len(windows)}  FAILED={failed}/{len(windows)}")
    if first_err:
        print(f"First failure at window {first_err[0]}: {first_err[1]}")
    return results


def main():
    windows = np.load(WINDOWS)
    print(f"Loaded {windows.shape[0]} windows, shape {windows.shape}.\n")

    print("--- ONNX reference ---")
    onnx_outs = run_onnx(windows)
    print(f"Captured {len(onnx_outs)}, shape {onnx_outs[0].shape}.\n")

    print("--- float32 TFLite ---")
    tf_outs = run_float32_tflite(windows)

    if not tf_outs:
        print("\nNo float32 outputs -- cannot compare.")
        return

    n = min(len(onnx_outs), len(tf_outs))
    a = np.array([onnx_outs[i].reshape(-1) for i in range(n)])
    b = np.array([tf_outs[i].reshape(-1) for i in range(n)])
    diff = np.abs(a - b)
    scale = float(np.max(np.abs(a)))
    print(f"\n=== float32 TFLite vs ONNX (first {n} windows) ===")
    print(f"Max  |diff|: {diff.max():.6f}")
    print(f"Mean |diff|: {diff.mean():.6f}")
    print(f"ONNX max |y|: {scale:.6f}")
    print(f"Relative max error: {100 * diff.max() / scale:.4f}%")
    corr = float(np.corrcoef(a.flatten(), b.flatten())[0, 1])
    print(f"Pearson correlation: {corr:.6f}")


if __name__ == "__main__":
    main()
