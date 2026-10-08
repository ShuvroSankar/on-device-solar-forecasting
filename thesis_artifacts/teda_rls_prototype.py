#!/usr/bin/env python3
"""
TEDA-RLS prototype: adaptive residual corrector for SmallTCN PV forecasts.

Idea: SmallTCN makes prediction yhat_t. Residual r_t = y_t - yhat_t is what
it got wrong. If the residual distribution drifts (seasonal shift, sensor
aging), a static model can't compensate. TEDA-RLS learns to predict r_t from
[r_{t-W}, ..., r_{t-1}] and produces a corrected forecast yhat_t + r_hat_t.

TEDA handles "when does the regime change": it clusters the residual stream
into DataClouds, each with its own RLS filter. New regime -> new cloud.
Similar regimes -> clouds merge. Each filter adapts via a forgetting factor.

Usage:
    python teda_rls_prototype.py --synthetic          # sanity check
    python teda_rls_prototype.py residuals.csv        # your data
    python teda_rls_prototype.py residuals.csv --window 5 --mu 0.95 --m 2.0

CSV format (header required, auto-detected):
    residual                      (1 column)
    truth,prediction              (2 columns)
    timestamp,truth,prediction    (3 columns)
"""
from __future__ import annotations
import argparse
import csv
import math
from typing import Optional

import numpy as np


# ---------------- DataCloud ----------------

class DataCloud:
    """Recursive moments + RLS filter for one operating regime."""

    __slots__ = ("n", "mean", "M2", "w", "P", "mu_rls", "window", "eps")

    def __init__(self, window: int, mu_rls: float,
                 delta: float = 1e2, eps: float = 1e-6):
        self.window = window
        self.mu_rls = mu_rls
        self.eps = eps
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0                       # sum of squared deviations
        self.w = np.zeros(window)           # RLS weights
        self.P = np.eye(window) * delta     # inverse correlation matrix

    @property
    def var(self) -> float:
        return self.M2 / self.n if self.n > 0 else 0.0

    def eccentricity(self, x: float) -> float:
        """TEDA eccentricity of x w.r.t. this cloud. Large = eccentric."""
        if self.n < 2:
            return math.inf
        v = self.M2 / self.n + self.eps
        return 1.0 / self.n + (x - self.mean) ** 2 / (self.n * v)

    def update_stats(self, x: float) -> None:
        """Welford's online mean/variance."""
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        d2 = x - self.mean
        self.M2 += d * d2

    def predict(self, x_vec: np.ndarray) -> float:
        return float(x_vec @ self.w)

    def update_rls(self, x_vec: np.ndarray, target: float) -> None:
        """RLS step with forgetting factor mu_rls."""
        Px = self.P @ x_vec
        denom = self.mu_rls + float(x_vec @ Px)
        if denom <= 0:
            return
        k = Px / denom
        err = target - float(x_vec @ self.w)
        self.w = self.w + k * err
        self.P = (self.P - np.outer(k, Px)) / self.mu_rls
        # Numerical guards: prevent runaway weights / P.
        self.w = np.clip(self.w, -200.0, 200.0)
        self.P = np.clip(self.P, -1e6, 1e6)


# ---------------- TEDA-RLS manager ----------------

class TEDARLS:
    def __init__(self, window: int = 3, mu_rls: float = 0.99, m: float = 3.0,
                 max_clouds: int = 8, merge_threshold: float = 0.1,
                 delta: float = 1e3, eps: float = 1e-6):
        self.window = window
        self.mu_rls = mu_rls
        self.m = m
        self.max_clouds = max_clouds
        self.merge_threshold = merge_threshold
        self.delta = delta
        self.eps = eps
        self.clouds: list[DataCloud] = []
        self.hist: list[float] = []
        self._current_cloud_idx = -1
        self.n_new_clouds = 0
        self.n_merges = 0
        self.min_samples_for_correction = 40
        self.max_correction_sigma = 3.0   # clamp |pred| to k * running_std
        # Stream-level stats for input normalization and clamp scale
        self.stream_n = 0
        self.stream_mean = 0.0
        self.stream_M2 = 0.0

    def _input_vec(self) -> Optional[np.ndarray]:
        if len(self.hist) < self.window:
            return None
        return np.array(self.hist[-self.window:])

    @property
    def stream_std(self) -> float:
        return math.sqrt(self.stream_M2 / self.stream_n) if self.stream_n > 1 else 1.0

    def _threshold(self, n: int) -> float:
        return (self.m ** 2 + 1) / (2 * n)

    def _find_cloud(self, x: float) -> int:
        best_i, best_xi = -1, math.inf
        for i, c in enumerate(self.clouds):
            xi = c.eccentricity(x)
            if xi <= self._threshold(c.n) and xi < best_xi:
                best_xi, best_i = xi, i
        return best_i

    def _merge_closest(self, force: bool = False) -> bool:
        """Merge the two closest clouds. If force=True, merge regardless of
        threshold. Returns True if a merge happened."""
        if len(self.clouds) < 2:
            return False
        best_pair, best_d = None, math.inf
        for i in range(len(self.clouds)):
            for j in range(i + 1, len(self.clouds)):
                d = abs(self.clouds[i].mean - self.clouds[j].mean)
                if d < best_d:
                    best_d, best_pair = d, (i, j)
        if best_pair is None:
            return False
        if not force and best_d > self.merge_threshold:
            return False
        i, j = best_pair
        c1, c2 = self.clouds[i], self.clouds[j]
        n = c1.n + c2.n
        if n == 0:
            self.clouds.pop(max(i, j))
            self.clouds.pop(min(i, j))
            self.n_merges += 1
            return True
        mean = (c1.n * c1.mean + c2.n * c2.mean) / n
        d = c1.mean - c2.mean
        M2 = c1.M2 + c2.M2 + (c1.n * c2.n / n) * d * d      # parallel axis
        w = (c1.n * c1.w + c2.n * c2.w) / n
        c1.n, c1.mean, c1.M2, c1.w = n, mean, M2, w
        del self.clouds[j]
        self.n_merges += 1
        # Index shift invalidates the cached current cloud.
        self._current_cloud_idx = -1
        return True

    def update(self, r_t: float) -> float:
        """Process residual r_t. Returns r_hat_t (the prediction made at t-1)."""
        x_vec = self._input_vec()

        # Normalize input by stream std so RLS always sees ~unit-scale features.
        if x_vec is not None and self.stream_n >= 30:
            x_norm = x_vec / (self.stream_std + 1e-6)
        else:
            x_norm = x_vec

        # Predict using the cloud that r_{t-1} was assigned to.
        if x_norm is not None and self._current_cloud_idx >= 0:
            c = self.clouds[self._current_cloud_idx]
            if c.n >= self.min_samples_for_correction:
                pred = float(c.predict(x_norm))
            else:
                pred = 0.0
        else:
            pred = 0.0

        # Clamp to k * stream std (raw units). Fires only once stream is warm.
        if self.stream_n >= 30:
            limit = self.max_correction_sigma * self.stream_std
            pred = float(np.clip(pred, -limit, limit))

        # Assign r_t to a cloud (or create one).
        idx = self._find_cloud(r_t)
        if idx < 0:
            if len(self.clouds) >= self.max_clouds:
                while (len(self.clouds) >= self.max_clouds
                       and self._merge_closest(force=True)):
                    pass
                idx = self._find_cloud(r_t)
            if idx < 0:
                self.clouds.append(
                    DataCloud(self.window, self.mu_rls, self.delta, self.eps))
                self.n_new_clouds += 1
                idx = len(self.clouds) - 1
        self._current_cloud_idx = idx

        # Update cloud with normalized input, raw target.
        cloud = self.clouds[idx]
        cloud.update_stats(r_t)
        if x_norm is not None:
            cloud.update_rls(x_norm, r_t)

        # Advance history.
        self.hist.append(r_t)
        if len(self.hist) > self.window:
            self.hist.pop(0)

        # Update stream-level moments (Welford).
        self.stream_n += 1
        d = r_t - self.stream_mean
        self.stream_mean += d / self.stream_n
        self.stream_M2 += d * (r_t - self.stream_mean)

        return pred

    def run(self, residuals: np.ndarray) -> np.ndarray:
        out = np.zeros(len(residuals))
        for t, r in enumerate(residuals):
            out[t] = self.update(float(r))
        return out


# ---------------- Metrics & report ----------------

def print_report(name: str, r: np.ndarray, r_hat: np.ndarray, n_windows: int = 10):
    static = np.abs(r)
    corrected = np.abs(r - r_hat)
    s_mae, c_mae = float(static.mean()), float(corrected.mean())
    s_rmse = float(np.sqrt(np.mean(r ** 2)))
    c_rmse = float(np.sqrt(np.mean((r - r_hat) ** 2)))
    impr_mae = (s_mae - c_mae) / s_mae * 100 if s_mae > 0 else 0.0
    impr_rmse = (s_rmse - c_rmse) / s_rmse * 100 if s_rmse > 0 else 0.0

    print(f"\n=== {name} ===")
    print(f"  n = {len(r)}")
    print(f"  Static    MAE={s_mae:9.4f}   RMSE={s_rmse:9.4f}")
    print(f"  TEDA-RLS  MAE={c_mae:9.4f}   RMSE={c_rmse:9.4f}")
    print(f"  Improvement: MAE {impr_mae:+6.2f}%   RMSE {impr_rmse:+6.2f}%")

    wsize = max(1, len(r) // n_windows)
    print(f"\n  Per-window ({n_windows} windows of ~{wsize}):")
    print(f"  {'win':>4}  {'static MAE':>11}  {'corr. MAE':>11}  {'impr %':>8}")
    for i in range(n_windows):
        s = i * wsize
        e = s + wsize if i < n_windows - 1 else len(r)
        if e <= s:
            continue
        sr = r[s:e]
        sh = r_hat[s:e]
        sm = float(np.mean(np.abs(sr)))
        cm = float(np.mean(np.abs(sr - sh)))
        imp = (sm - cm) / sm * 100 if sm > 0 else 0.0
        print(f"  {i:>4}  {sm:>11.4f}  {cm:>11.4f}  {imp:>+8.2f}")


# ---------------- Synthetic drift data ----------------

def make_synthetic(n: int = 3000, seed: int = 42) -> np.ndarray:
    """Three regimes: baseline, mean-shift (drift), variance-shift."""
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 0.5, n)
    r[n // 3 : 2 * n // 3] += 1.5     # drift in the mean
    r[2 * n // 3 :] *= 2.5            # drift in the variance
    return r


# ---------------- CSV loader ----------------

def load_csv(path: str):
    with open(path) as f:
        rows = list(csv.reader(f))
    if not rows:
        raise SystemExit(f"Empty file: {path}")
    header, rows = rows[0], rows[1:]
    if not rows:
        raise SystemExit(f"No data rows in {path}")
    ncol = len(rows[0])
    if ncol == 1:
        r = np.array([float(x[0]) for x in rows])
        return r, None, None
    if ncol == 2:
        truth = np.array([float(x[0]) for x in rows])
        pred = np.array([float(x[1]) for x in rows])
        return truth - pred, truth, pred
    truth = np.array([float(x[1]) for x in rows])
    pred = np.array([float(x[2]) for x in rows])
    return truth - pred, truth, pred


# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="?", help="CSV with residuals or (truth, pred)")
    ap.add_argument("--synthetic", action="store_true",
                    help="Run on synthetic drift data (no CSV needed)")
    ap.add_argument("--window", type=int, default=3, help="RLS window W")
    ap.add_argument("--mu", type=float, default=0.99,
                    help="RLS forgetting factor (0<mu<=1)")
    ap.add_argument("--m", type=float, default=3.0,
                    help="TEDA sensitivity (2=2-sigma, 3=2.5-sigma)")
    ap.add_argument("--max-clouds", type=int, default=8)
    ap.add_argument("--merge-threshold", type=float, default=0.1,
                    help="Merge two clouds whose means differ by less than this")
    args = ap.parse_args()

    if args.synthetic or not args.csv:
        r = make_synthetic()
        name = "synthetic drift"
    else:
        r, _truth, _pred = load_csv(args.csv)
        name = args.csv

    model = TEDARLS(window=args.window, mu_rls=args.mu, m=args.m,
                    max_clouds=args.max_clouds,
                    merge_threshold=args.merge_threshold)
    r_hat = model.run(r)

    print_report(name, r, r_hat, n_windows=10)
    print(f"\n  Clouds created: {model.n_new_clouds}")
    print(f"  Clouds merged : {model.n_merges}")
    print(f"  Final #clouds : {len(model.clouds)}")
    print(f"  Cloud means   : {[round(c.mean, 3) for c in model.clouds]}")
    print(f"  Cloud counts  : {[c.n for c in model.clouds]}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
        axes[0].plot(r, lw=0.5, label="r_t (raw)")
        axes[0].plot(r_hat, lw=0.8, label="r_hat_t (TEDA-RLS)")
        axes[0].legend(); axes[0].set_ylabel("residual")
        axes[1].plot(np.abs(r), lw=0.5, label="|r_t| static err")
        axes[1].plot(np.abs(r - r_hat), lw=0.8, label="|r_t - r_hat_t| corrected")
        axes[1].legend(); axes[1].set_ylabel("abs error")
        n = np.arange(1, len(r) + 1)
        axes[2].plot(np.cumsum(np.abs(r)) / n, label="cum. static MAE")
        axes[2].plot(np.cumsum(np.abs(r - r_hat)) / n, label="cum. corrected MAE")
        axes[2].legend(); axes[2].set_ylabel("cumulative MAE")
        axes[2].set_xlabel("t")
        plt.tight_layout()
        plt.savefig("teda_rls_diagnostics.png", dpi=120)
        print("\n  Plot saved: teda_rls_diagnostics.png")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
