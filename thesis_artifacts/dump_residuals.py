#!/usr/bin/env python3
"""
Dump per-window residuals for a SINGLE site -> CSV for TEDA-RLS Phase 0.

Mirrors evaluate_per_horizon_mape.py but:
  - restricts to one site (auto-picks the one with most rows, or --site-id)
  - builds windows at a configurable stride (default 6 = 30-min spacing)
  - runs fp32 PyTorch inference only (drift signal is cleaner than int8)
  - extracts residual = truth - prediction at a chosen horizon
  - optionally filters to daylight so nighttime zeros don't dominate
  - writes rows in TIME ORDER (make_windows_multi_site never crosses sites)

Output CSV columns: truth,prediction
The teda_rls_prototype.py loader auto-computes residual = truth - prediction.

Usage:
    python dump_residuals.py --checkpoint smalltcn_best.pt
    python dump_residuals.py --checkpoint smalltcn_best.pt --stride 1
    python dump_residuals.py --checkpoint smalltcn_best.pt --site-id 20001
"""
import argparse
import csv

import numpy as np
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
DAYLIGHT_THRESHOLD = 1.0


def pick_best_site(site_data: dict) -> str:
    return max(site_data.keys(), key=lambda k: len(site_data[k][0]))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True,
                    help="SmallTCN checkpoint (.pt). Find the default with: "
                         "grep -n checkpoint evaluate_per_horizon_mape.py")
    ap.add_argument("--test-dir", default="./processed/test")
    ap.add_argument("--site-id", default=None,
                    help="ss_id to dump (auto-pick site with most rows if omitted)")
    ap.add_argument("--stride", type=int, default=6,
                    help="stride between windows (6 = 30-min spacing, 1 = 5-min)")
    ap.add_argument("--horizon", type=int, default=0,
                    help="forecast step to extract (0 = 5-min ahead)")
    ap.add_argument("--no-daylight-filter", dest="daylight_only",
                    action="store_false", default=True,
                    help="keep all rows including nighttime zeros")
    ap.add_argument("--threshold", type=float, default=DAYLIGHT_THRESHOLD,
                    help="truth threshold (Wh) for daylight filter")
    ap.add_argument("--out", default=None,
                    help="output CSV (default residuals_site<ID>.csv)")
    args = ap.parse_args()

    # --- model ---
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    norm = Normalizer()
    norm.mean = ckpt["norm_mean"]
    norm.std = ckpt["norm_std"]
    model = SmallTCN(
        context_length=CONTEXT_LENGTH,
        forecast_length=FORECAST_LENGTH,
        in_channels=ckpt.get("in_channels", 5),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # --- test split, one site ---
    print("Loading test split ...")
    capacities = load_site_capacities(METADATA_PATH)
    test_sites = load_solar_split(args.test_dir)

    if args.site_id is None:
        site_id = pick_best_site(test_sites)
        print(f"Auto-picked site with most rows: {site_id} "
              f"({len(test_sites[site_id][0]):,} rows)")
    else:
        site_id = args.site_id
        if site_id not in test_sites:
            raise SystemExit(
                f"site {site_id!r} not in test split. "
                f"First few available: {list(test_sites)[:5]}")
        print(f"Using site {site_id} ({len(test_sites[site_id][0]):,} rows)")

    one_site = {site_id: test_sites[site_id]}

    # --- windows (time order, single site) ---
    X, y, cap = make_windows_multi_site(
        one_site, CONTEXT_LENGTH, FORECAST_LENGTH, args.stride,
        capacities=capacities,
    )
    if len(X) == 0:
        raise SystemExit("No windows built for this site.")
    print(f"Built {len(X):,} windows (stride={args.stride})")

    # --- preprocess (matches eval) ---
    X_gen_cap = X[..., 0] / cap[:, None]
    X_n = X.copy()
    X_n[..., 0] = norm.transform(X_gen_cap)

    # --- fp32 inference, batched ---
    with torch.no_grad():
        X_t = torch.tensor(X_n)
        bs = 4096
        preds = []
        for i in range(0, X_t.size(0), bs):
            preds.append(model(X_t[i:i + bs]).numpy())
        pred_n = np.concatenate(preds, axis=0)
    pred = norm.inverse_transform(pred_n) * cap[:, None]

    # --- extract horizon h, daylight filter ---
    h = args.horizon
    truth_h = y[:, h]
    pred_h = pred[:, h]
    if args.daylight_only:
        keep = truth_h > args.threshold
    else:
        keep = np.ones(len(truth_h), dtype=bool)
    n_kept = int(keep.sum())
    print(f"Horizon {h} ({5 * (h + 1)} min ahead): {n_kept:,} / "
          f"{len(truth_h):,} rows kept"
          + (f" (daylight: truth > {args.threshold} Wh)"
             if args.daylight_only else ""))

    res = truth_h[keep] - pred_h[keep]
    if len(res):
        print(f"Residual stats: mean={res.mean():+.4f}  std={res.std():.4f}  "
              f"min={res.min():+.3f}  max={res.max():+.3f}")

    # --- write in time order ---
    out = args.out or f"residuals_site{site_id}.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["truth", "prediction"])
        for t, p in zip(truth_h[keep], pred_h[keep]):
            w.writerow([float(t), float(p)])
    print(f"Wrote {n_kept:,} rows to {out}")


if __name__ == "__main__":
    main()
