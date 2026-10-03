"""
Zero-shot evaluation of pretrained TTM-R2 (no fine-tuning at all) on the solar
test set -- same context=52/forecast=16/freq_token=3 setup as finetune_ttm.py,
so this is directly comparable to both SmallTCN and the fine-tuned TTM runs.

This answers: how much did fine-tuning actually buy you, versus just using
IBM's pretrained weights cold? Expect zero-shot to be noticeably worse than
both fine-tuned variants (0.866-0.878 MASE) -- if it's close, that's itself
an interesting/surprising finding worth a sentence in the report.

No training loop, no Trainer -- just get_model() + batched inference. Should
run in a couple of minutes (no epochs, single pass over the test set).

Usage:
    python zeroshot_ttm.py \
        --train_dir ./processed/train \
        --test_dir ./processed/test
"""

import argparse

import numpy as np
import torch

from data_pipeline import Normalizer
from solar_data_pipeline import METADATA_PATH, load_site_capacities, load_solar_split, make_windows_multi_site
from train_solar_tcn import mae_rmse, mase, smape_daylight

TTM_MODEL_PATH = "ibm-granite/granite-timeseries-ttm-r2"
CONTEXT_LENGTH = 52   # same as finetune_ttm.py -- the only TTM-R2 branch close to SmallTCN's window
FORECAST_LENGTH = 16  # 80-min horizon (2 steps short of SmallTCN's 90-min), same caveat as before
FREQ_TOKEN = 3        # DEFAULT_FREQUENCY_MAPPING["5min"] -- required by this frequency-prefix-tuned checkpoint


def build_windows(split_dir, capacities, stride, max_samples, seed, label):
    print(f"\n=== Loading {label} split ===")
    sites = load_solar_split(split_dir)
    X, y, cap = make_windows_multi_site(sites, CONTEXT_LENGTH, FORECAST_LENGTH, stride, capacities=capacities)
    del sites

    if max_samples is not None and max_samples < len(X):
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(X), size=max_samples, replace=False))
        X, y, cap = X[idx], y[idx], cap[idx]
        print(f"Subsampled {label} to {max_samples:,} windows (seed={seed}).")

    print(f"{label}: {len(X):,} windows.")
    return X, y, cap


def main(args):
    from tsfm_public.toolkit.get_model import get_model  # local import: not needed unless this script runs

    device = "cpu" if args.force_cpu else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    capacities = load_site_capacities(METADATA_PATH)

    # Train split is only used for: (a) fitting the Normalizer's z-score stats
    # (standard practice -- scaling calibration, not predictive leakage, same
    # convention as finetune_ttm.py and TTM's own zero-shot benchmarks, which
    # also fit scaling on the target dataset's train split even when the model
    # itself is never updated), and (b) the naive-persistence baseline MAE.
    X_train, y_train, cap_train = build_windows(
        args.train_dir, capacities, args.stride, args.max_train_samples, args.seed, "train"
    )
    X_test, y_test, cap_test = build_windows(
        args.test_dir, capacities, args.stride, args.max_test_samples, args.seed, "test"
    )

    X_train_cap = X_train[..., 0] / cap_train[:, None]
    X_test_cap = X_test[..., 0] / cap_test[:, None]
    norm = Normalizer()
    norm.fit(X_train_cap)
    X_test_n = norm.transform(X_test_cap)

    naive_pred_train = np.repeat(X_train[:, -1, 0:1], FORECAST_LENGTH, axis=1)
    naive_mae = np.mean(np.abs(y_train - naive_pred_train))
    print(f"\nNaive persistence baseline MAE (train split, {FORECAST_LENGTH}-step horizon): {naive_mae:.2f} Wh")

    print(f"\n=== Loading {TTM_MODEL_PATH} (context={CONTEXT_LENGTH}, forecast={FORECAST_LENGTH}), ZERO-SHOT ===")
    model = get_model(
        TTM_MODEL_PATH,
        context_length=CONTEXT_LENGTH,
        prediction_length=FORECAST_LENGTH,
        freq_prefix_tuning=True,
        freq="5min",
        prefer_l1_loss=False,
        prefer_longer_context=True,
    )
    model.to(device)
    model.eval()

    X_test_t = torch.tensor(X_test_n, dtype=torch.float32).unsqueeze(-1)  # (n, context, 1)
    freq_token = torch.full((args.batch_size,), FREQ_TOKEN, dtype=torch.long)

    print(f"\n=== Running zero-shot inference on {len(X_test_t):,} test windows ===")
    preds = []
    with torch.no_grad():
        for i in range(0, X_test_t.size(0), args.batch_size):
            batch = X_test_t[i : i + args.batch_size].to(device)
            bsz = batch.size(0)
            ft = freq_token[:bsz].to(device)
            out = model(past_values=batch, freq_token=ft)
            pred = out[0] if isinstance(out, (tuple, list)) else out.prediction_outputs
            preds.append(pred.squeeze(-1).cpu().numpy())
            if (i // args.batch_size) % 2000 == 0:
                print(f"  {i + bsz:,}/{len(X_test_t):,}", end="\r")
    print()
    pred_n = np.concatenate(preds, axis=0)

    pred_cap = norm.inverse_transform(pred_n)
    pred_wh = pred_cap * cap_test[:, None]

    m = mase(y_test, pred_wh, naive_mae)
    mae_v, rmse_v = mae_rmse(y_test, pred_wh)
    smape_d = smape_daylight(y_test, pred_wh)

    print(f"\n--- TTM-R2 ZERO-SHOT (context={CONTEXT_LENGTH}, forecast={FORECAST_LENGTH}) vs ground truth "
          f"(n={len(y_test):,}) ---")
    print(f"MASE: {m:.3f}  MAE: {mae_v:.2f} Wh  RMSE: {rmse_v:.2f} Wh  sMAPE (daylight): {smape_d:.2f}%")
    print("\nFor reference, your existing results at this same (52,16) horizon:")
    print("  TTM head-only fine-tuned:        MASE 0.878  MAE 10.38 Wh  RMSE 23.68 Wh")
    print("  TTM full fine-tuned (LR too hi): MASE 0.867  MAE 10.24 Wh  RMSE 23.56 Wh")
    print("  TTM full fine-tuned (LR=1e-4):   MASE 0.866  MAE 10.23 Wh  RMSE 23.34 Wh")
    print("  SmallTCN (18-step/90min, not directly comparable horizon): MASE 0.808")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", default="./processed/train")
    parser.add_argument("--test_dir", default="./processed/test")
    parser.add_argument("--stride", type=int, default=24)
    parser.add_argument("--max_train_samples", type=int, default=300000,
                         help="Only used for Normalizer fitting + naive baseline; doesn't need to be huge.")
    parser.add_argument("--max_test_samples", type=int, default=None,
                         help="Default: full test set, for a real reportable MASE number.")
    parser.add_argument("--batch_size", type=int, default=256,
                         help="No training here, so a larger batch than finetune_ttm.py is fine/faster.")
    parser.add_argument("--force_cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(args)
