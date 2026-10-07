"""
Per-horizon MAPE for SmallTCN on the full test set, for direct comparison
against Zhou et al. 2019 (IEEE Access, vol. 7) "Short-Term Photovoltaic
Power Forecasting Based on LSTM and Attention Mechanism".

Zhou et al. report MAPE at 7.5 / 15 / 30 / 60-min horizons on a single
20 kW rooftop PV plant (7.5-min resolution, Zhejiang, China). We can't
match their 7.5-min resolution exactly -- ours is 5 min -- so we compute
MAPE at every one of our 18 forecast steps, then pair the closest steps:

    Zhou 7.5 min  <->  our step 1  (5 min ahead)
    Zhou 15 min   <->  our step 3  (15 min ahead)
    Zhou 30 min   <->  our step 6  (30 min ahead)
    Zhou 60 min   <->  our step 12 (60 min ahead)

MAPE is computed per-step, daylight-only (y_true[:, h] > threshold),
matching how the project's sMAPE-daylight metric is defined. Plain MAPE
over all windows is dominated by near-zero night-time denominators and
is not informative.

Usage:
    python evaluate_per_horizon_mape.py                 # full test set
    python evaluate_per_horizon_mape.py --max_samples 5000   # quick check
"""

import argparse

import numpy as np
import tensorflow as tf
import torch

from data_pipeline import Normalizer
from solar_data_pipeline import (
    METADATA_PATH,
    load_site_capacities,
    load_solar_split,
    make_windows_multi_site,
)
from tcn_model import SmallTCN


CONTEXT_LENGTH = 36
FORECAST_LENGTH = 18
STRIDE = 24
DAYLIGHT_THRESHOLD = 1.0   # Wh, same as smape_daylight default
RUN_INT8 = True


def per_horizon_metrics(pred, actual, label, thresholds=(1.0, 5.0)):
    """Print per-horizon MAPE / MAE / RMSE at each daylight threshold.
    Returns {threshold: {h: (mape, mae, rmse, n)}}."""
    out = {}
    for thr in thresholds:
        print(f"\n=== {label} -- per-horizon MAPE (daylight-only, y_true > {thr} Wh) ===")
        print(f"{'step':>4}  {'mins':>4}  {'n':>10}  {'MAPE %':>8}  {'MAE Wh':>8}  {'RMSE Wh':>8}")
        rows = {}
        for h in range(FORECAST_LENGTH):
            mask = actual[:, h] > thr
            n = int(mask.sum())
            if n == 0:
                continue
            err = np.abs(pred[mask, h] - actual[mask, h])
            mape = 100.0 * float(np.mean(err / actual[mask, h]))
            mae = float(np.mean(err))
            rmse = float(np.sqrt(np.mean((pred[mask, h] - actual[mask, h]) ** 2)))
            rows[h] = (mape, mae, rmse, n)
            print(f"{h + 1:>4}  {(h + 1) * 5:>4}  {n:>10,}  {mape:>8.2f}  "
                  f"{mae:>8.3f}  {rmse:>8.3f}")
        out[thr] = rows
    return out


def main(checkpoint_path, tflite_path, test_dir, max_samples, seed):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    norm = Normalizer()
    norm.mean = ckpt["norm_mean"]
    norm.std = ckpt["norm_std"]

    print("Loading test split ...")
    capacities = load_site_capacities(METADATA_PATH)
    test_sites = load_solar_split(test_dir)
    X_test, y_test, cap_test = make_windows_multi_site(
        test_sites, CONTEXT_LENGTH, FORECAST_LENGTH, STRIDE, capacities=capacities
    )
    del test_sites
    if max_samples is not None and max_samples < len(X_test):
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(X_test), size=max_samples, replace=False))
        X_test, y_test, cap_test = X_test[idx], y_test[idx], cap_test[idx]
        print(f"Subsampled to {max_samples:,} windows (seed={seed}).")
    print(f"Evaluating on {len(X_test):,} test windows")

    X_test_gen_cap = X_test[..., 0] / cap_test[:, None]
    X_test_n = X_test.copy()
    X_test_n[..., 0] = norm.transform(X_test_gen_cap)

    # --- fp32 (PyTorch) ---
    print("\nRunning fp32 SmallTCN ...")
    model = SmallTCN(
        context_length=CONTEXT_LENGTH,
        forecast_length=FORECAST_LENGTH,
        in_channels=ckpt.get("in_channels", 5),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_test_n)
        bs = 4096
        preds = []
        for i in range(0, X_t.size(0), bs):
            preds.append(model(X_t[i:i + bs]).numpy())
        pred_fp32_n = np.concatenate(preds, axis=0)
    pred_fp32 = norm.inverse_transform(pred_fp32_n) * cap_test[:, None]

    rows_fp32 = per_horizon_metrics(pred_fp32, y_test, "SmallTCN fp32")

    # --- int8 (TFLite) ---
    rows_int8 = None
    if RUN_INT8:
        print("\nRunning int8 TFLite SmallTCN (window-by-window) ...")
        interp = tf.lite.Interpreter(model_path=tflite_path)
        interp.allocate_tensors()
        in_det = interp.get_input_details()[0]
        out_det = interp.get_output_details()[0]
        in_scale, in_zero = in_det["quantization"]
        out_scale, out_zero = out_det["quantization"]

        pred_int8_n = np.empty((len(X_test_n), FORECAST_LENGTH), dtype=np.float32)
        for i in range(len(X_test_n)):
            x = X_test_n[i:i + 1].astype(np.float32)
            q = np.round(x / in_scale + in_zero).astype(np.int8)
            interp.set_tensor(in_det["index"], q)
            interp.invoke()
            yq = interp.get_tensor(out_det["index"])
            pred_int8_n[i] = (yq.astype(np.float32) - out_zero) * out_scale
            if (i + 1) % 50000 == 0:
                print(f"  int8 inference: {i + 1:,}/{len(X_test_n):,}")
        pred_int8 = norm.inverse_transform(pred_int8_n) * cap_test[:, None]
        rows_int8 = per_horizon_metrics(pred_int8, y_test, "SmallTCN int8")

    # --- Zhou et al. comparison table ---
    print("\n" + "=" * 78)
    print("Comparison to Zhou et al. 2019 (ALSTM, single 20 kW site, 7.5-min res)")
    print("Their numbers: overall MAPE per horizon, averaged over one year (Table III)")
    print("Our numbers:   per-horizon MAPE at the closest matching step, daylight-only")
    print("=" * 78)
    zhou = {7.5: 24.65, 15.0: 28.81, 30.0: 32.18, 60.0: 37.82}
    our_step_for_zhou = {7.5: 0, 15.0: 2, 30.0: 5, 60.0: 11}  # 0-indexed h
    header = f"{'Horizon':>10}  {'our step':>9}  {'ours fp32':>10}  {'ours int8':>10}  {'Zhou ALSTM':>12}"
    print(header)
    print("-" * len(header))
    for z_h, z_mape in zhou.items():
        h = our_step_for_zhou[z_h]
        ours_fp32 = rows_fp32[1.0].get(h, (float('nan'),))[0]
        ours_int8 = (rows_int8[1.0].get(h, (float('nan'),))[0] if rows_int8 else float('nan'))
        print(f"{z_h:>7} min  {h + 1:>9}  {ours_fp32:>9.2f}%  {ours_int8:>9.2f}%  "
              f"{z_mape:>11.2f}%")
    print("\nCAVEATS for the reader:")
    print("- Zhou et al.: single 20 kW rooftop site (Zhejiang, China), 7.5-min samples,")
    print("  3 years training / 2 years testing, MAPE averaged over 12 months of 2017-2018.")
    print("- This work: 703 UK sites, 5-min samples, train 2018-2022, test 2024,")
    print("  daylight-only (y_true > 1 Wh), MAPE over the full test set.")
    print("- Different sites, different climates, different resolutions, different")
    print("  train/test windows. Directional comparison only, NOT an apples-to-apples")
    print("  accuracy benchmark. The main takeaways are: (a) the shape of the")
    print("  horizon-degradation curve, and (b) that our single small MCU-deployable")
    print("  model lands in the same MAPE range as their single-site LSTM+attention")
    print("  ensemble -- neither as good as a multi-site dataset allows nor as bad as")
    print("  the worst published single-site results.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/small_tcn_solar.pt")
    p.add_argument("--tflite_model", default="models/small_tcn_solar_manual_int8.tflite")
    p.add_argument("--test_dir", default="./processed/test")
    p.add_argument("--max_samples", type=int, default=None,
                   help="Subsample the test set for a quick sanity check. "
                        "Default: use the full test set.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    main(args.checkpoint, args.tflite_model, args.test_dir,
         args.max_samples, args.seed)
