"""
Training loop: SmallTCN on UKPV solar-forecasting windows.

Same model (tcn_model.py, UNCHANGED) and same training/eval logic as
train_tcn.py, but using solar_data_pipeline.py to load the pre-made
./processed/{train,val,test} splits instead of one REFIT house CSV.

Usage:
    python train_solar_tcn.py
    (paths default to ./processed/{train,val,test}; override with flags if needed)
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from solar_data_pipeline import (
    load_solar_split,
    make_windows_multi_site,
    load_site_capacities,
    METADATA_PATH,
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_FORECAST_LENGTH,
)
from data_pipeline import Normalizer  # reused as-is, it's generic
from tcn_model import SmallTCN, count_params


def mase(y_true, y_pred, y_train_naive_mae):
    mae = np.mean(np.abs(y_true - y_pred))
    return mae / (y_train_naive_mae + 1e-8)


def mae_rmse(y_true, y_pred):
    """Plain MAE/RMSE in Wh -- interpretable and not distorted by
    zero-actual points the way sMAPE is."""
    err = y_true - y_pred
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err ** 2))
    return mae, rmse


def smape_daylight(y_true, y_pred, threshold=1.0):
    """sMAPE computed only where actual generation > threshold (Wh).
    Plain sMAPE blows up at night, when actual generation is ~0 -- the
    (|actual|+|pred|)/2 denominator collapses toward zero and any small
    nonzero prediction produces a huge relative error, even though the
    absolute error is tiny and harmless. Restricting to daylight points
    (or a small positive threshold, to also dodge near-zero dawn/dusk
    readings) makes this metric mean something comparable across models."""
    mask = y_true > threshold
    if mask.sum() == 0:
        return float("nan")
    yt, yp = y_true[mask], y_pred[mask]
    denom = (np.abs(yt) + np.abs(yp)) / 2 + 1e-8
    return np.mean(np.abs(yt - yp) / denom) * 100


def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2 + 1e-8
    return np.mean(np.abs(y_true - y_pred) / denom) * 100


def naive_forecast_mae(y, X):
    last_values = X[:, -1:]
    naive_pred = np.repeat(last_values, y.shape[1], axis=1)
    return np.mean(np.abs(y - naive_pred))


def train(
    train_dir="./processed/train",
    val_dir="./processed/val",
    test_dir="./processed/test",
    context_length=DEFAULT_CONTEXT_LENGTH,
    forecast_length=DEFAULT_FORECAST_LENGTH,
    stride=6,
    epochs=30,
    batch_size=256,
    lr=1e-3,
    patience=5,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # --- Data ---
    print("\n=== Loading site capacities (kWp) ===")
    capacities = load_site_capacities(METADATA_PATH)

    print("\n=== Loading train split ===")
    train_sites = load_solar_split(train_dir)
    X_train, y_train, cap_train = make_windows_multi_site(
        train_sites, context_length, forecast_length, stride, capacities=capacities
    )
    del train_sites  # free per-site dict once windows are built

    print("\n=== Loading val split ===")
    val_sites = load_solar_split(val_dir)
    X_val, y_val, cap_val = make_windows_multi_site(
        val_sites, context_length, forecast_length, stride, capacities=capacities
    )
    del val_sites

    print("\n=== Loading test split ===")
    test_sites = load_solar_split(test_dir)
    X_test, y_test, cap_test = make_windows_multi_site(
        test_sites, context_length, forecast_length, stride, capacities=capacities
    )
    del test_sites

    # Extract just what's needed for the naive-persistence baseline (last
    # context timestep's raw generation value) BEFORE any normalization --
    # this is a tiny (n,) array, so we don't need to keep the full raw
    # X_train around just for this.
    last_val_train = X_train[:, -1, 0].copy()

    # Capacity-normalize the generation channel/target BEFORE z-scoring:
    # divide each window's values by its site's installed capacity (kWp),
    # so a 1.89 kWp site and a 3.36 kWp site are expressed on a comparable
    # scale and the model learns the shared shape of the solar curve
    # instead of partly re-learning "how big is this site" from 3 hours
    # of context. cap_* arrays broadcast against (n, context_length) /
    # (n, forecast_length) via [:, None].
    X_train_gen_cap = X_train[..., 0] / cap_train[:, None]
    y_train_cap = y_train / cap_train[:, None]
    X_val_gen_cap = X_val[..., 0] / cap_val[:, None]
    y_val_cap = y_val / cap_val[:, None]
    X_test_gen_cap = X_test[..., 0] / cap_test[:, None]

    norm = Normalizer()
    norm.fit(X_train_gen_cap)  # fit on capacity-normalized generation channel
    del X_train_gen_cap, X_val_gen_cap  # (X_test_gen_cap kept a moment longer, used below)

    def normalize_X_inplace(X, X_gen_cap):
        # X: (n, context_length, 5) -- MUTATES X in place, replacing
        # channel 0 with the capacity-normalized + z-scored generation
        # values; channels 1-4 (sin/cos hour, sin/cos day-of-year) are
        # left untouched (already bounded in [-1, 1]). In-place avoids
        # doubling memory with a full X.copy() -- at larger context
        # lengths that duplication is what pushes past available RAM.
        X[..., 0] = norm.transform(X_gen_cap)
        return X

    X_train_n = normalize_X_inplace(X_train, X_train[..., 0] / cap_train[:, None])
    del X_train
    y_train_n = norm.transform(y_train_cap)
    del y_train_cap

    X_val_n = normalize_X_inplace(X_val, X_val[..., 0] / cap_val[:, None])
    del X_val
    y_val_n = norm.transform(y_val_cap)
    del y_val_cap

    X_test_n = normalize_X_inplace(X_test, X_test_gen_cap)
    del X_test, X_test_gen_cap

    def to_loader(X, y, shuffle):
        X_t = torch.tensor(X)  # already (n, context_length, in_channels)
        y_t = torch.tensor(y)
        return DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=shuffle)

    train_loader = to_loader(X_train_n, y_train_n, shuffle=True)
    val_loader = to_loader(X_val_n, y_val_n, shuffle=False)

    # --- Model (in_channels=5: generation + sin/cos hour-of-day + sin/cos day-of-year) ---
    model = SmallTCN(context_length=context_length, forecast_length=forecast_length, in_channels=5).to(device)
    print(f"\nModel parameters: {count_params(model):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    print("\nStarting training...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_losses.append(criterion(pred, yb).item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        print(f"Epoch {epoch:3d}/{epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
                break

    model.load_state_dict(best_state)

    # --- Test evaluation ---
    model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test_n)  # already (n, context_length, 5)
        eval_batch_size = 4096
        preds_list = []
        for i in range(0, X_test_t.size(0), eval_batch_size):
            batch_x = X_test_t[i : i + eval_batch_size].to(device)
            batch_pred = model(batch_x)
            preds_list.append(batch_pred.cpu())
        pred_n = torch.cat(preds_list, dim=0).numpy()
    pred_cap = norm.inverse_transform(pred_n)      # back to capacity-normalized units
    pred = pred_cap * cap_test[:, None]             # back to absolute Wh

    naive_pred_train = np.repeat(last_val_train[:, None], y_train.shape[1], axis=1)
    naive_mae = np.mean(np.abs(y_train - naive_pred_train))
    test_mase = mase(y_test, pred, naive_mae)
    test_mae, test_rmse = mae_rmse(y_test, pred)
    test_smape_daylight = smape_daylight(y_test, pred)
    test_smape_all = smape(y_test, pred)

    print("\n--- Final Test Results (SOLAR) ---")
    print(f"Naive baseline MAE (persistence): {naive_mae:.2f} Wh")
    print(f"SmallTCN MASE: {test_mase:.3f}  (< 1.0 means it beats the naive baseline)")
    print(f"SmallTCN MAE: {test_mae:.2f} Wh")
    print(f"SmallTCN RMSE: {test_rmse:.2f} Wh")
    print(f"SmallTCN sMAPE (daylight only, actual > 1 Wh): {test_smape_daylight:.2f}%")
    print(f"SmallTCN sMAPE (all points, incl. night -- inflated, for reference only): {test_smape_all:.2f}%")

    out_dir = os.path.expanduser("~/thesis/checkpoints")
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "small_tcn_solar.pt")
    torch.save(
        {
            "model_state_dict": best_state,
            "norm_mean": norm.mean,
            "norm_std": norm.std,
            "context_length": context_length,
            "forecast_length": forecast_length,
            "in_channels": 5,
            "capacity_normalized": True,  # inference must divide by site kWp, then de-normalize by multiplying back
        },
        ckpt_path,
    )
    print(f"\nSaved checkpoint: {ckpt_path}")

    return model, norm, (test_mase, test_mae, test_rmse, test_smape_daylight)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", default="./processed/train")
    parser.add_argument("--val_dir", default="./processed/val")
    parser.add_argument("--test_dir", default="./processed/test")
    parser.add_argument("--context_length", type=int, default=DEFAULT_CONTEXT_LENGTH)
    parser.add_argument("--forecast_length", type=int, default=DEFAULT_FORECAST_LENGTH)
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    train(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        test_dir=args.test_dir,
        context_length=args.context_length,
        forecast_length=args.forecast_length,
        stride=args.stride,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
