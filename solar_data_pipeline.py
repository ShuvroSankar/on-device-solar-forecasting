"""
Data pipeline for UKPV (solar) dataset -> forecasting-ready windows.

Adapts the REFIT single-house pipeline (data_pipeline.py) to multi-site
solar data already filtered/split by Phase 1 (preprocess_ukpv.py), which
lives under ./processed/{train,val,test}/5_minutely/year=YYYY/month=MM/*.parquet

Key differences from the REFIT pipeline:
  - Multi-site: windows are built PER SITE, never mixed across sites.
  - Real gaps: Phase 1 dropped rows where imputation wasn't possible, so a
    site's remaining rows are not guaranteed perfectly contiguous. We check
    actual timestamp deltas and only window within contiguous runs.
  - Data is already split into train/val/test (by year) -- no fractional
    split here, just load each split directory separately.

Usage:
    from solar_data_pipeline import load_solar_split, make_windows_multi_site

    site_data = load_solar_split("./processed/train")
    X, y = make_windows_multi_site(site_data, context_length=36, forecast_length=18, stride=6)
"""

import glob
import os

import numpy as np
import pandas as pd

# Defaults matching the proposal's stated 15-90 min forecast horizon at 5-min resolution.
DEFAULT_CONTEXT_LENGTH = 36   # 3 hours of history
DEFAULT_FORECAST_LENGTH = 18  # 90 minutes ahead
EXPECTED_STEP = pd.Timedelta(minutes=5)
METADATA_PATH = "./dataset/uk_pv_data/metadata.csv"


def load_solar_split(split_dir: str) -> dict:
    """
    Load all parquet files under a split directory (e.g. ./processed/train)
    and group into a dict: {ss_id: (timestamps_ndarray, values_ndarray)}.

    Streams file-by-file to keep memory bounded, same approach as Phase 1.
    """
    files = sorted(glob.glob(f"{split_dir}/**/*.parquet", recursive=True))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {split_dir}")
    print(f"Loading {len(files)} files from {split_dir} ...")

    buffers = {}  # ss_id -> list of (timestamps_chunk, values_chunk)
    for i, f in enumerate(files):
        df = pd.read_parquet(f, columns=["ss_id", "datetime_GMT", "generation_Wh"])
        for ss_id, grp in df.groupby("ss_id"):
            grp = grp.sort_values("datetime_GMT")
            buffers.setdefault(ss_id, []).append(
                (grp["datetime_GMT"].values, grp["generation_Wh"].values.astype(np.float32))
            )
        if (i + 1) % 10 == 0 or i == len(files) - 1:
            print(f"  loaded {i + 1}/{len(files)} files")

    site_data = {}
    for ss_id, chunks in buffers.items():
        ts = np.concatenate([c[0] for c in chunks])
        vals = np.concatenate([c[1] for c in chunks])
        order = np.argsort(ts)
        site_data[ss_id] = (ts[order], vals[order])

    print(f"Loaded {len(site_data)} sites, "
          f"{sum(len(v) for _, v in site_data.values()):,} total rows")
    return site_data


def _contiguous_runs(timestamps: np.ndarray):
    """Yield (start_idx, end_idx) slices where consecutive timestamps are
    exactly EXPECTED_STEP apart -- i.e. no real gap."""
    if len(timestamps) == 0:
        return
    deltas = np.diff(timestamps)
    step_ns = np.timedelta64(EXPECTED_STEP)
    breaks = np.where(deltas != step_ns)[0]  # index i means gap between i and i+1

    start = 0
    for b in breaks:
        yield start, b + 1  # end is exclusive
        start = b + 1
    yield start, len(timestamps)


def _hour_sin_cos(timestamps: np.ndarray):
    """Cyclical time-of-day encoding: sin/cos of fractional hour (0-24)."""
    dt = pd.DatetimeIndex(timestamps)
    hour_frac = dt.hour.values + dt.minute.values / 60.0
    angle = 2 * np.pi * hour_frac / 24.0
    return np.sin(angle).astype(np.float32), np.cos(angle).astype(np.float32)


def _doy_sin_cos(timestamps: np.ndarray):
    """
    Cyclical day-of-year encoding: sin/cos of day-of-year (1-366).
    Captures season -- at the same site and same hour-of-day, output
    differs substantially between e.g. a December morning (low sun angle)
    and a June morning (high sun angle). Hour-of-day alone can't express
    this; the raw generation magnitude only expresses it implicitly.
    Approximated with a fixed 365.25-day cycle (ignores leap-year jitter,
    which is a sub-day effect and not worth the complexity here).
    """
    dt = pd.DatetimeIndex(timestamps)
    doy = dt.dayofyear.values.astype(np.float32)
    angle = 2 * np.pi * doy / 365.25
    return np.sin(angle).astype(np.float32), np.cos(angle).astype(np.float32)


def load_site_capacities(metadata_path: str = METADATA_PATH) -> dict:
    """Load per-site installed capacity (kWp) from metadata.csv.
    Used to capacity-normalize generation across sites of different sizes
    before pooled training -- otherwise the model has to partly re-learn
    "how big is this site" instead of just the shared shape of the solar
    curve. Returns {ss_id: kWp}."""
    if not os.path.exists(metadata_path):
        print(f"[!] metadata.csv not found at {metadata_path} -- capacity normalization unavailable")
        return {}
    meta = pd.read_csv(metadata_path)
    meta = meta.dropna(subset=["ss_id", "kWp"])
    meta = meta[meta["kWp"] > 0]
    cap = meta.drop_duplicates(subset="ss_id").set_index("ss_id")["kWp"].to_dict()
    print(f"Loaded capacity (kWp) for {len(cap)} sites")
    return cap


def make_windows_multi_site(
    site_data: dict,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    forecast_length: int = DEFAULT_FORECAST_LENGTH,
    stride: int = 6,
    capacities: dict = None,
):
    """
    Build (context, forecast) windows across all sites, never spanning a
    real gap or a site boundary.

    Each context timestep carries 5 channels: [generation_Wh, sin(hour),
    cos(hour), sin(day_of_year), cos(day_of_year)]. The forecast target is
    generation_Wh only (still 1-D per step) -- we're not asking the model
    to predict future time/season, that's already known/computable at
    inference time.

    If `capacities` (dict ss_id -> kWp) is provided, a per-window capacity
    array is also returned so the caller can capacity-normalize generation
    at train time and de-normalize predictions back to Wh at eval time.
    Sites missing from `capacities` fall back to the median capacity among
    known sites (with a warning), rather than being silently dropped.

    Returns:
        X: np.ndarray, shape (n_windows, context_length, 5)
        y: np.ndarray, shape (n_windows, forecast_length)
        cap: np.ndarray, shape (n_windows,) -- kWp for each window's site
             (all 1.0 if capacities is None)
    """
    total_len = context_length + forecast_length
    X_chunks, y_chunks, cap_chunks = [], [], []
    n_sites_used = 0
    n_missing_capacity = 0

    fallback_cap = 1.0
    if capacities:
        fallback_cap = float(np.median(list(capacities.values())))

    for ss_id, (timestamps, values) in site_data.items():
        site_cap = fallback_cap
        if capacities is not None:
            if ss_id in capacities:
                site_cap = capacities[ss_id]
            else:
                n_missing_capacity += 1

        site_windows = 0
        for start, end in _contiguous_runs(timestamps):
            run_vals = values[start:end]
            run_ts = timestamps[start:end]
            if len(run_vals) < total_len:
                continue
            sin_h, cos_h = _hour_sin_cos(run_ts)
            sin_d, cos_d = _doy_sin_cos(run_ts)

            n_windows = (len(run_vals) - total_len) // stride + 1
            if n_windows <= 0:
                continue
            for i in range(n_windows):
                s = i * stride
                ctx_val = run_vals[s : s + context_length]
                ctx_sin_h = sin_h[s : s + context_length]
                ctx_cos_h = cos_h[s : s + context_length]
                ctx_sin_d = sin_d[s : s + context_length]
                ctx_cos_d = cos_d[s : s + context_length]
                X_chunks.append(
                    np.stack([ctx_val, ctx_sin_h, ctx_cos_h, ctx_sin_d, ctx_cos_d], axis=-1)
                )
                y_chunks.append(run_vals[s + context_length : s + total_len])
                cap_chunks.append(site_cap)
                site_windows += 1
        if site_windows > 0:
            n_sites_used += 1

    if not X_chunks:
        raise ValueError("No windows could be built -- check context/forecast lengths vs data length.")

    X = np.stack(X_chunks).astype(np.float32)  # (n, context_length, 5)
    y = np.stack(y_chunks).astype(np.float32)  # (n, forecast_length)
    cap = np.array(cap_chunks, dtype=np.float32)  # (n,)
    print(f"Built {len(X):,} windows from {n_sites_used}/{len(site_data)} sites "
          f"(context={context_length}, forecast={forecast_length}, stride={stride}, channels=5)")
    if capacities is not None and n_missing_capacity > 0:
        print(f"  [!] {n_missing_capacity} sites had no capacity in metadata.csv "
              f"-- used fallback median ({fallback_cap:.2f} kWp)")
    return X, y, cap


if __name__ == "__main__":
    import sys

    split_dir = sys.argv[1] if len(sys.argv) > 1 else "./processed/train"
    site_data = load_solar_split(split_dir)
    X, y = make_windows_multi_site(site_data)
    print(f"\nX shape: {X.shape}")
    print(f"y shape: {y.shape}")
