#!/usr/bin/env python3
"""
Tier 1 baselines for the PV forecasting comparison:
  - Persistence:       yhat_{t+h} = y_t    (last observed generation)
  - Climatology:       yhat_{t+h} = mean(y | month, hour-of-day) from training

Both are capacity-normalized, matching how the neural model predicts.
Window build (stride, context, forecast) matches evaluate_per_horizon_mape.py.

Usage:
    python evaluate_baselines.py
    python evaluate_baselines.py --max-samples 100000
    python evaluate_baselines.py --skip-climatology
"""
import argparse

import numpy as np

from solar_data_pipeline import (
    METADATA_PATH,
    load_site_capacities,
    load_solar_split,
    make_windows_multi_site,
)

CONTEXT_LENGTH = 36
FORECAST_LENGTH = 18
STRIDE = 24
DAYLIGHT_THRESHOLDS = (1.0, 5.0)


def recover_hour(X):
    """Recover hour-of-day (0..24) from sin/cos channels 1,2."""
    return (np.arctan2(X[..., 1], X[..., 2]) * 24.0 / (2 * np.pi)) % 24.0


def recover_doy(X):
    """Recover day-of-year (1..366) from sin/cos channels 3,4."""
    return (np.arctan2(X[..., 3], X[..., 4]) * 365.0 / (2 * np.pi)) % 365.0 + 1.0


def month_from_doy(doy):
    bounds = np.cumsum([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    return np.searchsorted(bounds, doy, side="right") + 1


def build_climatology(train_sites, capacities):
    """Return clim[month-1, hour] = mean capacity-normalized generation."""
    print("Building climatology from training split ...")
    X_tr, y_tr, cap_tr = make_windows_multi_site(
        train_sites, CONTEXT_LENGTH, FORECAST_LENGTH, STRIDE,
        capacities=capacities,
    )
    print(f"  training windows: {len(X_tr):,}")

    hour_end = recover_hour(X_tr[:, -1, :])
    doy_end = recover_doy(X_tr[:, -1, :])
    step_hours = 5.0 / 60.0
    step_days = 5.0 / (60 * 24)

    sums = np.zeros((12, 24))
    counts = np.zeros((12, 24))
    for h in range(FORECAST_LENGTH):
        hour_h = (hour_end + (h + 1) * step_hours) % 24.0
        doy_h = doy_end + (h + 1) * step_days
        gen_norm = y_tr[:, h] / cap_tr
        h_bin = np.floor(hour_h).astype(int) % 24
        m_bin = np.clip(month_from_doy(doy_h) - 1, 0, 11)
        flat = m_bin * 24 + h_bin
        sums += np.bincount(flat, weights=gen_norm, minlength=288).reshape(12, 24)
        counts += np.bincount(flat, minlength=288).reshape(12, 24)

    clim = sums / np.where(counts == 0, 1, counts)
    print(f"  climatology cells populated: {int((counts > 0).sum())}/288")
    del X_tr, y_tr, cap_tr
    return clim


def per_horizon_table(pred, actual, label, thresholds=DAYLIGHT_THRESHOLDS):
    for thr in thresholds:
        print(f"\n=== {label} -- per-horizon MAPE (daylight-only, y_true > {thr} Wh) ===")
        print(f"{'step':>4}  {'mins':>4}  {'n':>10}  {'MAPE %':>8}  {'MAE Wh':>8}  {'RMSE Wh':>8}")
        for h in range(FORECAST_LENGTH):
            mask = actual[:, h] > thr
            n = int(mask.sum())
            if n == 0:
                continue
            err = np.abs(pred[mask, h] - actual[mask, h])
            mape = 100.0 * float(np.mean(err / actual[mask, h]))
            mae = float(np.mean(err))
            rmse = float(np.sqrt(np.mean((pred[mask, h] - actual[mask, h]) ** 2)))
            print(f"{h + 1:>4}  {(h + 1) * 5:>4}  {n:>10,}  {mape:>8.2f}  "
                  f"{mae:>8.3f}  {rmse:>8.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-dir", default="./processed/train")
    ap.add_argument("--test-dir", default="./processed/test")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-climatology", action="store_true")
    ap.add_argument("--clim-max-sites", type=int, default=None,
                    help="limit climatology build to N training sites (memory)")
    args = ap.parse_args()

    print("Loading capacities and test split ...")
    capacities = load_site_capacities(METADATA_PATH)
    test_sites = load_solar_split(args.test_dir)
    X_test, y_test, cap_test = make_windows_multi_site(
        test_sites, CONTEXT_LENGTH, FORECAST_LENGTH, STRIDE, capacities=capacities
    )
    del test_sites

    if args.max_samples and args.max_samples < len(X_test):
        rng = np.random.default_rng(args.seed)
        idx = np.sort(rng.choice(len(X_test), size=args.max_samples, replace=False))
        X_test, y_test, cap_test = X_test[idx], y_test[idx], cap_test[idx]
        print(f"Subsampled to {len(X_test):,}")

    # --- Persistence ---
    last_obs = X_test[:, -1, 0]
    pred_persist = np.tile(last_obs[:, None], (1, FORECAST_LENGTH))
    per_horizon_table(pred_persist, y_test, "Persistence (last observed)")

    # --- Climatology ---
    if not args.skip_climatology:
        train_sites = load_solar_split(args.train_dir)
        if args.clim_max_sites and len(train_sites) > args.clim_max_sites:
            rng = np.random.default_rng(args.seed)
            keys = list(train_sites.keys())
            sel = rng.choice(len(keys), size=args.clim_max_sites, replace=False)
            train_sites = {keys[i]: train_sites[keys[i]] for i in sel}
            print(f"Limited climatology to {args.clim_max_sites} sites")
        clim = build_climatology(train_sites, capacities)
        del train_sites

        hour_end = recover_hour(X_test[:, -1, :])
        doy_end = recover_doy(X_test[:, -1, :])
        step_hours = 5.0 / 60.0
        step_days = 5.0 / (60 * 24)

        pred_clim = np.zeros_like(y_test)
        for h in range(FORECAST_LENGTH):
            hour_h = (hour_end + (h + 1) * step_hours) % 24.0
            doy_h = doy_end + (h + 1) * step_days
            h_bin = np.floor(hour_h).astype(int) % 24
            m_bin = np.clip(month_from_doy(doy_h) - 1, 0, 11)
            pred_clim[:, h] = clim[m_bin, h_bin] * cap_test

        per_horizon_table(pred_clim, y_test, "Climatology (month x hour-of-day)")


if __name__ == "__main__":
    main()
