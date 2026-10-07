"""
Per-site MAPE on the test set -- quantifies how much of the gap vs. Zhou et al.
2019 (single-site LSTM+attention) is 'multi-site task hardness' vs 'our model
underperforms a single-site baseline'.

For each site in the test split:
  - Build windows for that site only
  - Run fp32 SmallTCN inference on those windows
  - Compute per-horizon MAPE at step 1 (5 min) and step 12 (60 min), daylight-only

Then report the distribution across all 279 test sites (min / quartiles / median
/ max), and list the top-5 best-performing sites with their capacities.

CAVEAT: even at a single test site, SmallTCN was TRAINED across 703 sites. So
this is not truly apples-to-apples with Zhou's site-specific model -- but it is
the closest comparison available without retraining on one site, and it isolates
the effect of testing on a single homogeneous site vs the full 279-site mix.
"""

import argparse

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
STRIDE = 24
DAYLIGHT_THRESHOLD = 1.0
SITES_TO_REPORT = 5


def per_horizon_mape(pred, actual, horizon_idx, threshold=DAYLIGHT_THRESHOLD):
    """MAPE at a single forecast step, filtered to daylight."""
    mask = actual[:, horizon_idx] > threshold
    if mask.sum() == 0:
        return float('nan'), 0
    err = np.abs(pred[mask, horizon_idx] - actual[mask, horizon_idx])
    mape = 100.0 * float(np.mean(err / actual[mask, horizon_idx]))
    return mape, int(mask.sum())


def main(checkpoint_path, test_dir):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    norm = Normalizer()
    norm.mean = ckpt["norm_mean"]
    norm.std = ckpt["norm_std"]

    capacities = load_site_capacities(METADATA_PATH)
    test_sites = load_solar_split(test_dir)

    model = SmallTCN(
        context_length=CONTEXT_LENGTH,
        forecast_length=FORECAST_LENGTH,
        in_channels=ckpt.get("in_channels", 5),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    results = []
    for i, (ss_id, site_data) in enumerate(test_sites.items()):
        single = {ss_id: site_data}
        try:
            X, y, cap = make_windows_multi_site(
                single, CONTEXT_LENGTH, FORECAST_LENGTH, STRIDE,
                capacities=capacities,
            )
        except Exception as e:
            print(f"  site {ss_id}: windowing failed ({e})")
            continue
        if len(X) < 100:   # too few windows for a meaningful estimate
            continue

        X_cap = X[..., 0] / cap[:, None]
        X_n = X.copy()
        X_n[..., 0] = norm.transform(X_cap)

        with torch.no_grad():
            pred_n = model(torch.tensor(X_n)).numpy()
        pred = norm.inverse_transform(pred_n) * cap[:, None]

        m1, n1 = per_horizon_mape(pred, y, 0)
        m12, n12 = per_horizon_mape(pred, y, 11)

        results.append({
            "ss_id": ss_id,
            "n_windows": len(X),
            "capacity_kwp": float(capacities.get(ss_id, float('nan'))),
            "mape_step1": m1, "n1": n1,
            "mape_step12": m12, "n12": n12,
        })
        if (i + 1) % 25 == 0:
            print(f"  processed {i + 1}/{len(test_sites)} sites")

    if not results:
        print("No sites produced windows -- nothing to report.")
        return

    def dist(vals, label):
        a = np.array([v for v in vals if not np.isnan(v)])
        if len(a) == 0:
            print(f"{label}: all NaN"); return
        print(f"{label}: n={len(a):>3}  "
              f"min={a.min():>6.2f}%  "
              f"p25={np.percentile(a, 25):>6.2f}%  "
              f"median={np.median(a):>6.2f}%  "
              f"p75={np.percentile(a, 75):>6.2f}%  "
              f"max={a.max():>6.2f}%")

    print("\n" + "=" * 78)
    print("Per-site MAPE distribution across test sites (daylight-only, y_true > 1 Wh)")
    print("=" * 78)
    dist([r["mape_step1"] for r in results], "Step 1 (5 min)")
    dist([r["mape_step12"] for r in results], "Step 12 (60 min)")

    print(f"\nTop {SITES_TO_REPORT} best sites (lowest step-1 MAPE):")
    print(f"{'ss_id':>8}  {'n_win':>7}  {'kWp':>6}  {'MAPE step1':>11}  {'MAPE step12':>12}")
    top = sorted(results, key=lambda r: r["mape_step1"])[:SITES_TO_REPORT]
    for r in top:
        print(f"{r['ss_id']:>8}  {r['n_windows']:>7}  "
              f"{r['capacity_kwp']:>6.2f}  {r['mape_step1']:>10.2f}%  "
              f"{r['mape_step12']:>11.2f}%")

    print(f"\nReference -- Zhou et al. 2019 (single-site ALSTM):")
    print(f"  step 1  (7.5 min):  24.65%")
    print(f"  step 12 (60 min):   37.82%")
    print("(Their horizon steps aren't identical to ours; the closest match is")
    print("step 1 for 7.5 min and step 12 for 60 min. Between those, our steps")
    print("line up as: 15 min = step 3, 30 min = step 6.)")

    print("\nINTERPRETATION:")
    print("- If our BEST single site is close to Zhou's number, the multi-site")
    print("  task is the dominant reason for the gap, and the model generalizes.")
    print("- If even our BEST single site is far above Zhou's number, the model")
    print("  underperforms a site-specific LSTM+attention on this specific kind")
    print("  of easy task (single homogeneous site).")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/small_tcn_solar.pt")
    p.add_argument("--test_dir", default="./processed/test")
    args = p.parse_args()
    main(args.checkpoint, args.test_dir)
