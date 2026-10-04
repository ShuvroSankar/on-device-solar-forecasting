"""
Isolate which calibration windows crash the int8 model, and check whether
each crash correlates with a near-zero-variance (constant/flat) input window
-- the DIV-by-zero hypothesis.

Also collects the distinct failing node numbers so we can tell whether one
LayerNorm instance is uniquely broken or whether many are.
"""
import re

import numpy as np
from ai_edge_litert.interpreter import Interpreter

windows = np.load("exports/ttm_calibration_windows.npy")  # (200, 52, 1), z-scored

interp = Interpreter(model_path="exports/ttm_solar_int8/ttm_solar_full_integer_quant.tflite")
interp.allocate_tensors()
in_details = {d['name']: d for d in interp.get_input_details()}
pv_detail = in_details['serving_default_past_values:0']
ft_detail = in_details['serving_default_freq_token:0']
scale, zero_point = pv_detail['quantization']

crashed_windows = []
crashed_nodes = set()
ok = 0

for i in range(windows.shape[0]):
    z = windows[i].reshape(1, 1, 52)
    q = np.round(z / scale + zero_point).astype(np.int8)
    interp.set_tensor(pv_detail['index'], q)
    interp.set_tensor(ft_detail['index'], np.array([3], dtype=np.int64))
    try:
        interp.invoke()
        ok += 1
    except RuntimeError as e:
        raw = z.flatten()
        crashed_windows.append(i)
        m = re.search(r"Node number (\d+)", str(e))
        if m:
            crashed_nodes.add(int(m.group(1)))
        print(f"CRASHED at window {i}")
        print(f"  raw z-scored values: min={raw.min():.4f} max={raw.max():.4f} "
              f"std={raw.std():.6f} unique_values={len(np.unique(raw))}")
        print(f"  error: {str(e).splitlines()[0][:140]}")
        continue

print(f"\n=== SUMMARY ===")
print(f"OK windows:      {ok} / {windows.shape[0]}")
print(f"Crashed windows: {len(crashed_windows)} / {windows.shape[0]}")
print(f"Distinct failing node numbers: {sorted(crashed_nodes)}")

if crashed_windows:
    stds = np.array([windows[i].std() for i in crashed_windows])
    uniques = np.array([len(np.unique(windows[i])) for i in crashed_windows])
    print(f"\nCrashed-window std:    min={stds.min():.6f} "
          f"median={np.median(stds):.6f} max={stds.max():.6f}")
    print(f"Crashed-window uniques: min={uniques.min()} "
          f"median={int(np.median(uniques))} max={uniques.max()}")
    print(f"Crashes with std < 1e-3: {(stds < 1e-3).sum()} / {len(stds)}")

    # Also look at the non-crashing population, for contrast
    ok_windows = [i for i in range(windows.shape[0]) if i not in set(crashed_windows)]
    if ok_windows:
        ok_stds = np.array([windows[i].std() for i in ok_windows])
        print(f"\nOK-window std:         min={ok_stds.min():.6f} "
              f"median={np.median(ok_stds):.6f} max={ok_stds.max():.6f}")
        print(f"OK windows with std < 1e-3: {(ok_stds < 1e-3).sum()} / {len(ok_stds)}")
