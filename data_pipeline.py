"""
Data pipeline for REFIT dataset -> forecasting-ready windows.

Loads one house's cleaned REFIT CSV, extracts whole-house aggregate power,
resamples to a regular interval, and produces (context, forecast) window
pairs for training/evaluating both the SmallTCN and TTM models.

Expected input: CLEAN_House<N>.csv from the REFIT "Cleaned" dataset
(CLEAN_REFIT_081116.7z), which has columns including a timestamp and
an "Aggregate" whole-house power column (in Watts).

Usage:
    from data_pipeline import load_refit_house, make_windows

    series = load_refit_house("data/refit/CLEAN_House1.csv", resample="10min")
    X, y = make_windows(series, context_length=128, forecast_length=96)
"""

import numpy as np
import pandas as pd


def load_refit_house(csv_path: str, resample: str = "10min") -> pd.Series:
    """
    Load a single REFIT house CSV and return a clean, regularly-sampled
    whole-house aggregate power series (Watts).

    resample: pandas offset string, e.g. "10min" (10-minute avg), "1h" (hourly).
        REFIT is natively ~8-second sampling -- that's far too fine-grained
        and noisy for a small forecasting model / MCU deployment story, so
        we downsample to something more realistic for a smart-meter use case.
    """
    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path)

    # REFIT CLEAN files have a "Time" column (unix timestamp) and an
    # "Aggregate" column (whole-house power in Watts). Column names can
    # vary slightly by release -- handle common variants defensively.
    time_col = next((c for c in df.columns if c.lower() in ("time", "unix")), None)
    agg_col = next((c for c in df.columns if c.lower() == "aggregate"), None)
    if time_col is None or agg_col is None:
        raise ValueError(
            f"Could not find expected columns. Found: {list(df.columns)}. "
            f"Expected a time/unix column and an 'Aggregate' column."
        )

    df["timestamp"] = pd.to_datetime(df[time_col], unit="s")
    df = df.set_index("timestamp").sort_index()

    series = df[agg_col].astype(float)

    # Drop obviously invalid readings (REFIT docs: negative or absurd spikes)
    series = series.clip(lower=0, upper=20000)  # 20kW is a generous ceiling for a house

    # Resample to regular interval (mean aggregation), then interpolate any
    # small remaining gaps (REFIT clean version has most gaps handled, but
    # resampling can reintroduce a few NaNs at boundaries).
    series = series.resample(resample).mean()
    series = series.interpolate(limit=6)  # limit: don't fill long dropouts
    series = series.dropna()

    print(f"Loaded {len(series):,} points, from {series.index.min()} to {series.index.max()}")
    print(f"Resampled to: {resample}")
    print(f"Value range: {series.min():.1f}W - {series.max():.1f}W, mean {series.mean():.1f}W")

    return series


def make_windows(series: pd.Series, context_length: int, forecast_length: int, stride: int = 1):
    """
    Slice a 1D series into (context, forecast) window pairs for supervised
    forecasting training.

    Returns:
        X: np.ndarray, shape (n_windows, context_length)
        y: np.ndarray, shape (n_windows, forecast_length)
    """
    values = series.values.astype(np.float32)
    total_len = context_length + forecast_length
    n_windows = (len(values) - total_len) // stride + 1

    if n_windows <= 0:
        raise ValueError(
            f"Series too short ({len(values)} points) for context_length="
            f"{context_length} + forecast_length={forecast_length}."
        )

    X = np.zeros((n_windows, context_length), dtype=np.float32)
    y = np.zeros((n_windows, forecast_length), dtype=np.float32)

    for i in range(n_windows):
        start = i * stride
        X[i] = values[start : start + context_length]
        y[i] = values[start + context_length : start + total_len]

    print(f"Created {n_windows:,} windows (context={context_length}, forecast={forecast_length}, stride={stride})")
    return X, y


def train_val_test_split(X, y, val_frac=0.15, test_frac=0.15):
    """
    Chronological split (NOT random shuffling) -- critical for time series:
    train on the past, validate/test on the future, to avoid leakage.
    """
    n = len(X)
    test_start = int(n * (1 - test_frac))
    val_start = int(n * (1 - test_frac - val_frac))

    X_train, y_train = X[:val_start], y[:val_start]
    X_val, y_val = X[val_start:test_start], y[val_start:test_start]
    X_test, y_test = X[test_start:], y[test_start:]

    print(f"Split -> train: {len(X_train):,}, val: {len(X_val):,}, test: {len(X_test):,}")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


class Normalizer:
    """
    Simple z-score normalizer, fit on training data only (standard practice
    to avoid leakage from val/test statistics).
    """

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X_train):
        self.mean = X_train.mean()
        self.std = X_train.std() + 1e-6

    def transform(self, X):
        return (X - self.mean) / self.std

    def inverse_transform(self, X):
        return X * self.std + self.mean


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python data_pipeline.py <path_to_CLEAN_HouseN.csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    series = load_refit_house(csv_path, resample="10min")
    X, y = make_windows(series, context_length=128, forecast_length=96, stride=4)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = train_val_test_split(X, y)

    norm = Normalizer()
    norm.fit(X_train)
    print(f"\nNormalizer fit: mean={norm.mean:.1f}, std={norm.std:.1f}")

    print(f"\nX_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print("\nReady for training. Save arrays with np.save if you want to cache them:")
    print("  np.save('X_train.npy', X_train)  etc.")
