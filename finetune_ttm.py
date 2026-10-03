"""
Fine-tune IBM's pretrained TinyTimeMixer (TTM-R2) on the solar dataset,
using context_length=52 / forecast_length=16 -- the closest available TTM-R2
branch to SmallTCN's (36, 18) window. TTM-R2's model registry only ships
fixed (context, prediction) pairs (see tsfm_public's ttm.yaml); the short
ones are exactly (52,16), (90,30), (180,60), (360,60) -- everything else
starts at 512. So this is an 80-minute forecast, 2 steps short of
SmallTCN's 90-minute one -- close enough to report side by side in the
same table with that one caveat stated, not a different-horizon result.

This deliberately reuses your own solar_data_pipeline windowing (gap-safe,
capacity-normalized, respects site boundaries) and Normalizer, rather than
TimeSeriesPreprocessor/get_datasets, to keep preprocessing identical to
SmallTCN's for a fair comparison -- and to sidestep uncertainty around how
TSP's split_config/scaling behave across multiple sites with masked gaps.

v1 is univariate (generation channel only, capacity-normalized + z-scored),
matching SmallTCN's own "baseline (generation channel only)" ablation row
(MASE 0.852) as the fairest single-channel comparison point. The sin/cos
calendar features are left as a possible v2 extension (see note at bottom).

Install (core tsfm_public utilities, no notebook extras needed):
    pip install granite-tsfm

Usage:
    python finetune_ttm.py \
        --train_dir ./processed/train \
        --val_dir ./processed/val \
        --test_dir ./processed/test \
        --max_train_samples 300000

If you hit an MPS crash during eval/predict (you've seen this class of bug
before, per your "Fix MPS evaluation memory crash" commit for SmallTCN),
pass --force_cpu.
"""

import argparse
import math
import os

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed
from transformers.integrations import INTEGRATION_TO_CALLBACK

from tsfm_public import TrackingCallback, count_parameters
from tsfm_public.toolkit.get_model import get_model
from tsfm_public.toolkit.lr_finder import optimal_lr_finder

from data_pipeline import Normalizer
from solar_data_pipeline import METADATA_PATH, load_site_capacities, load_solar_split, make_windows_multi_site
from train_solar_tcn import mae_rmse, mase, smape_daylight

TTM_MODEL_PATH = "ibm-granite/granite-timeseries-ttm-r2"
# TTM-R2's model registry (tsfm_public/resources/model_paths_config/ttm.yaml) only offers
# FIXED (context, prediction) pairs -- not a freely resizable range. The short-context
# options are exactly: (52,16), (90,30), (180,60), (360,60); everything else starts at 512.
# (52, 16) is the closest available pair to SmallTCN's (36, 18): an 80-minute forecast
# instead of 90 minutes, 2 steps short -- a small, clearly-stated approximation, not the
# drastically different 8h horizon an earlier (wrong) attempt at 96 would have produced.
CONTEXT_LENGTH = 52
FORECAST_LENGTH = 16
# The 52-16-ft-r2.1 checkpoint is frequency-prefix-tuned (config.resolution_prefix_tuning=True),
# so forward() requires a freq_token per sample -- an integer index into
# tsfm_public's DEFAULT_FREQUENCY_MAPPING (time_series_preprocessor.py), which maps
# "5min" -> 3 for our data's native resolution.
FREQ_TOKEN = 3


class WindowDataset(Dataset):
    """Wraps pre-built (context, forecast) windows for TTM's forward() signature:
    past_values (context_length, 1) -> future_values (forecast_length, 1)."""

    def __init__(self, X_gen_n, y_gen_n):
        self.X = torch.tensor(X_gen_n, dtype=torch.float32).unsqueeze(-1)  # (n, context, 1)
        self.y = torch.tensor(y_gen_n, dtype=torch.float32).unsqueeze(-1)  # (n, forecast, 1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return {
            "past_values": self.X[idx],
            "future_values": self.y[idx],
            "freq_token": torch.tensor(FREQ_TOKEN, dtype=torch.long),
        }


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
    set_seed(args.seed)
    capacities = load_site_capacities(METADATA_PATH)

    X_train, y_train, cap_train = build_windows(
        args.train_dir, capacities, args.stride, args.max_train_samples, args.seed, "train"
    )
    X_val, y_val, cap_val = build_windows(
        args.val_dir, capacities, args.stride, args.max_val_samples, args.seed, "val"
    )
    X_test, y_test, cap_test = build_windows(
        args.test_dir, capacities, args.stride, args.max_test_samples, args.seed, "test"
    )

    # Capacity-normalize generation channel (channel 0), then z-score fit on TRAIN only,
    # same convention as train_solar_tcn.py, so both models see the same-shaped signal.
    X_train_cap = X_train[..., 0] / cap_train[:, None]
    y_train_cap = y_train / cap_train[:, None]
    X_val_cap = X_val[..., 0] / cap_val[:, None]
    y_val_cap = y_val / cap_val[:, None]
    X_test_cap = X_test[..., 0] / cap_test[:, None]
    y_test_cap = y_test / cap_test[:, None]  # kept for reference; ground truth stays in raw Wh below

    norm = Normalizer()
    norm.fit(X_train_cap)
    X_train_n, y_train_n = norm.transform(X_train_cap), norm.transform(y_train_cap)
    X_val_n, y_val_n = norm.transform(X_val_cap), norm.transform(y_val_cap)
    X_test_n = norm.transform(X_test_cap)

    dset_train = WindowDataset(X_train_n, y_train_n)
    dset_val = WindowDataset(X_val_n, y_val_n)
    dset_test = WindowDataset(X_test_n, np.zeros_like(y_test))  # future_values unused at predict time

    # --- Naive persistence baseline (train split), same convention as train_solar_tcn.py ---
    naive_pred_train = np.repeat(X_train[:, -1, 0:1], FORECAST_LENGTH, axis=1)
    naive_mae = np.mean(np.abs(y_train - naive_pred_train))
    print(f"\nNaive persistence baseline MAE (train split): {naive_mae:.2f} Wh")

    # --- Load pretrained TTM, resized to our context/forecast lengths ---
    print(f"\n=== Loading {TTM_MODEL_PATH} (context={CONTEXT_LENGTH}, forecast={FORECAST_LENGTH}) ===")
    model = get_model(
        TTM_MODEL_PATH,
        context_length=CONTEXT_LENGTH,
        prediction_length=FORECAST_LENGTH,
        freq_prefix_tuning=True,
        freq="5min",
        prefer_l1_loss=False,
        prefer_longer_context=True,
    )

    if args.freeze_backbone:
        print("Params before freezing backbone:", count_parameters(model))
        for param in model.backbone.parameters():
            param.requires_grad = False
        print("Params after freezing backbone:", count_parameters(model))

    # --- Learning rate ---
    if args.learning_rate is None:
        learning_rate, model = optimal_lr_finder(model, dset_train, batch_size=args.batch_size)
        print("Suggested learning rate:", learning_rate)
    else:
        learning_rate = args.learning_rate

    out_dir = args.output_dir
    training_args = TrainingArguments(
        output_dir=os.path.join(out_dir, "output"),
        learning_rate=learning_rate,
        num_train_epochs=args.num_epochs,
        do_eval=True,
        eval_strategy="epoch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        dataloader_num_workers=4,
        report_to="none",
        save_strategy="epoch",
        logging_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=args.seed,
        use_cpu=args.force_cpu,
    )

    early_stopping = EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=1e-5)
    tracking = TrackingCallback()

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = OneCycleLR(
        optimizer, learning_rate, epochs=args.num_epochs,
        steps_per_epoch=math.ceil(len(dset_train) / args.batch_size),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dset_train,
        eval_dataset=dset_val,
        callbacks=[early_stopping, tracking],
        optimizers=(optimizer, scheduler),
    )
    trainer.remove_callback(INTEGRATION_TO_CALLBACK["codecarbon"])

    print("\n=== Fine-tuning ===")
    trainer.train()

    print("\n=== Predicting on test set ===")
    trainer.model.loss = "mse"
    predictions = trainer.predict(dset_test)
    pred_n = predictions.predictions[0].squeeze(-1)  # (n, forecast_length)

    pred_cap = norm.inverse_transform(pred_n)
    pred_wh = pred_cap * cap_test[:, None]

    m = mase(y_test, pred_wh, naive_mae)
    mae_v, rmse_v = mae_rmse(y_test, pred_wh)
    smape_d = smape_daylight(y_test, pred_wh)
    print(f"\n--- TTM-R2 finetuned (context={CONTEXT_LENGTH}, forecast={FORECAST_LENGTH}) vs ground truth "
          f"(n={len(y_test):,}) ---")
    print(f"MASE: {m:.3f}  MAE: {mae_v:.2f} Wh  RMSE: {rmse_v:.2f} Wh  sMAPE (daylight): {smape_d:.2f}%")
    print("MASE is directly comparable to SmallTCN's 0.808 (MASE is self-normalized against its own "
          "horizon-matched naive baseline, printed above, so a 16-step vs 18-step horizon doesn't bias it). "
          "Raw MAE/RMSE in Wh are only approximately comparable to SmallTCN's 9.95/22.25 Wh, since this is "
          "an 80-minute forecast (16 steps) vs SmallTCN's 90-minute one (18 steps) -- state that caveat "
          "wherever you report the MAE/RMSE numbers side by side.")

    save_path = os.path.join(out_dir, "ttm_finetuned")
    trainer.save_model(save_path)
    print(f"\nSaved fine-tuned model to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", default="./processed/train")
    parser.add_argument("--val_dir", default="./processed/val")
    parser.add_argument("--test_dir", default="./processed/test")
    parser.add_argument("--stride", type=int, default=24)
    parser.add_argument("--max_train_samples", type=int, default=300000,
                         help="Subsample train windows for a manageable first-pass run time. "
                              "Set to a larger value or omit (via a very large number) for the full set.")
    parser.add_argument("--max_val_samples", type=int, default=50000)
    parser.add_argument("--max_test_samples", type=int, default=None,
                         help="Default: full test set, for a real reportable MASE number.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=None,
                         help="Omit to auto-detect via optimal_lr_finder (IBM's recommended approach).")
    parser.add_argument("--freeze_backbone", action="store_true", default=True)
    parser.add_argument("--full_finetune", dest="freeze_backbone", action="store_false",
                         help="Fine-tune the whole model (backbone + head) instead of head-only.")
    parser.add_argument("--force_cpu", action="store_true",
                         help="Use if you hit the MPS eval crash you saw before with SmallTCN.")
    parser.add_argument("--output_dir", default="ttm_finetuned_models/")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(args)
