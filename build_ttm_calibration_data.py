"""
Build int8 calibration data for TTM TFLite conversion, reproducing the
exact Normalizer fit used by the trained checkpoint, then sampling real
test-split windows and transforming them with that normalizer.

Uses memory_safe_windows.build_windows_channel0_lite instead of
solar_data_pipeline.make_windows_multi_site: the latter materializes every
window across all sites (5 channels each) before subsampling, which OOMs
on the full train split. The lite version is provably equivalent for the
subset of behavior this script depends on (same total count via identical
enumeration, same rng.choice call, same seed -> identical selected global
indices; only channel 0 kept, which is all Normalizer.fit reads).
"""

import argparse

import numpy as np

from solar_data_pipeline import METADATA_PATH, load_solar_split, load_site_capacities
from finetune_ttm import Normalizer, CONTEXT_LENGTH, FORECAST_LENGTH
from memory_safe_windows import build_windows_channel0_lite


def main(train_dir, test_dir, n_calibration, out_path):
    capacities = load_site_capacities(METADATA_PATH)

    print("Rebuilding train windows (channel-0 only, memory-safe) to reproduce "
          "the exact Normalizer fit used by ttm_full_finetuned_lr1e-4 "
          "(stride=24, seed=42, max_train_samples=300000).")
    train_sites = load_solar_split(train_dir)
    X_train, cap_train = build_windows_channel0_lite(
        train_sites, CONTEXT_LENGTH, FORECAST_LENGTH, stride=24,
        capacities=capacities, max_samples=300000, seed=42, label="train"
    )
    del train_sites

    X_train_cap = X_train / cap_train[:, None]
    norm = Normalizer()
    norm.fit(X_train_cap)
    print(f"Normalizer fit. mean={norm.mean:.6f}, std={norm.std:.6f}")

    print(f"\nPulling {n_calibration} real calibration windows from test split...")
    test_sites = load_solar_split(test_dir)
    X_test, cap_test = build_windows_channel0_lite(
        test_sites, CONTEXT_LENGTH, FORECAST_LENGTH, stride=24,
        capacities=capacities, max_samples=n_calibration, seed=42, label="test-calibration"
    )
    del test_sites

    X_test_cap = X_test / cap_test[:, None]
    X_test_n = norm.transform(X_test_cap)
    calibration_windows = X_test_n[..., None].astype(np.float32)

    np.save(out_path, calibration_windows)
    print(f"\nSaved {calibration_windows.shape[0]} calibration windows, "
          f"shape {calibration_windows.shape[1:]}, to {out_path}")
    print(f"Value range: min={calibration_windows.min():.4f}, "
          f"max={calibration_windows.max():.4f}, mean={calibration_windows.mean():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True,
                        help="directory of train-split site CSVs (same one finetune used)")
    parser.add_argument("--test_dir", required=True,
                        help="directory of test-split site CSVs (same one finetune used)")
    parser.add_argument("--n_calibration", type=int, default=200,
                        help="number of calibration windows to sample from test split")
    parser.add_argument("--out_path", default="exports/ttm_calibration_windows.npy",
                        help="where to save the (N, CONTEXT_LENGTH, 1) float32 array")
    args = parser.parse_args()
    main(args.train_dir, args.test_dir, args.n_calibration, args.out_path)
