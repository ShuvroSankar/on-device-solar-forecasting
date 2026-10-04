"""
Memory-safe reimplementation of make_windows_multi_site's window selection,
for calibration-data purposes only.

make_windows_multi_site materializes EVERY window across all sites (5
channels each) before build_windows subsamples down to 300,000 -- for the
full train split this is tens of millions of windows, which OOM-killed on
first attempt. This two-pass version counts windows without storing them,
then extracts only the selected subsample, keeping only the generation
channel (the only one Normalizer.fit ever reads).

Equivalence, not approximation: pass 1 counts windows via the IDENTICAL
nested loop (sites -> contiguous runs -> window starts, same stride) that
make_windows_multi_site uses, giving the same total count. rng.choice is
called with the same seed against that same total, so it selects the
IDENTICAL global window indices the original full-materialization approach
would have. Pass 2 replays the same loop, keeping only windows whose
position matches a selected index. This depends on site_data's dict
iteration order being deterministic across passes (guaranteed: same dict
object, not rebuilt between passes) and matching the real training run's
order (guaranteed: load_solar_split's file glob is sorted, per
solar_data_pipeline.py).
"""

import numpy as np
from solar_data_pipeline import _contiguous_runs


def _count_total_windows(site_data, context_length, forecast_length, stride):
    total_len = context_length + forecast_length
    count = 0
    for ss_id, (timestamps, values) in site_data.items():
        for start, end in _contiguous_runs(timestamps):
            run_len = end - start
            if run_len < total_len:
                continue
            n_windows = (run_len - total_len) // stride + 1
            if n_windows > 0:
                count += n_windows
    return count


def build_windows_channel0_lite(site_data, context_length, forecast_length, stride,
                                  capacities, max_samples, seed, label):
    """Returns (X, cap) only -- X shape (n_selected, context_length) float32,
    generation channel only. y/sin/cos are not needed for Normalizer fitting
    or for TTM's univariate past_values input."""
    print(f"\n=== Counting {label} windows (pass 1, no storage) ===")
    total = _count_total_windows(site_data, context_length, forecast_length, stride)
    print(f"{label}: {total:,} total windows available.")

    if max_samples is not None and max_samples < total:
        rng = np.random.default_rng(seed)
        selected = set(np.sort(rng.choice(total, size=max_samples, replace=False)).tolist())
        n_out = max_samples
        print(f"Selecting {max_samples:,} windows (seed={seed}), matching the "
              f"original build_windows' subsampling exactly.")
    else:
        selected = set(range(total))
        n_out = total

    fallback_cap = float(np.median(list(capacities.values()))) if capacities else 1.0

    X_out = np.empty((n_out, context_length), dtype=np.float32)
    cap_out = np.empty((n_out,), dtype=np.float32)
    out_i = 0
    global_i = 0
    total_len = context_length + forecast_length

    print(f"=== Extracting selected {label} windows (pass 2) ===")
    for ss_id, (timestamps, values) in site_data.items():
        site_cap = capacities.get(ss_id, fallback_cap) if capacities else fallback_cap
        for start, end in _contiguous_runs(timestamps):
            run_vals = values[start:end]
            run_len = len(run_vals)
            if run_len < total_len:
                continue
            n_windows = (run_len - total_len) // stride + 1
            if n_windows <= 0:
                continue
            for i in range(n_windows):
                if global_i in selected:
                    s = i * stride
                    X_out[out_i] = run_vals[s : s + context_length].astype(np.float32)
                    cap_out[out_i] = site_cap
                    out_i += 1
                global_i += 1

    assert out_i == n_out, f"Expected {n_out} windows, extracted {out_i} -- enumeration mismatch."
    print(f"{label}: extracted {out_i:,} windows, shape {X_out.shape}.")
    return X_out, cap_out
