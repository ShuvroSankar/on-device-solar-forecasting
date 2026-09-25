"""
Phase 1: UKPV dataset exploration + preprocessing.

Schema (confirmed from a sample parquet file):
    ss_id            int32                 -- site ID
    datetime_GMT     datetime64[ns, UTC]   -- timestamp, 5-min resolution
    generation_Wh    float32               -- generation reading

Run in stages:
    python3 preprocess_ukpv.py --stage inspect
    python3 preprocess_ukpv.py --stage filter
    python3 preprocess_ukpv.py --stage split
"""

import argparse
import glob
import os

import pandas as pd

DATA_DIR = "./dataset/uk_pv_data/5_minutely"
METADATA_PATH = "./dataset/uk_pv_data/metadata.csv"
BAD_DATA_PATH = "./dataset/uk_pv_data/bad_data.csv"
OUTPUT_DIR = "./processed"

NONZERO_COVERAGE_THRESHOLD = 0.5   # keep sites with >50% non-zero readings (per D'Aversa et al.)
MAX_GAP_MINUTES = 30               # forward-fill gaps up to this; longer gaps are left as NaN


def load_all_parquet(pattern=f"{DATA_DIR}/**/*.parquet"):
    files = sorted(glob.glob(pattern, recursive=True))
    print(f"Found {len(files)} parquet files")
    dfs = []
    for f in files:
        dfs.append(pd.read_parquet(f))
    df = pd.concat(dfs, ignore_index=True)
    print(f"Combined shape: {df.shape}")
    return df


def inspect(sample_only=True):
    """Stage 1: sanity-check schema, metadata, bad_data, and basic stats."""
    files = sorted(glob.glob(f"{DATA_DIR}/**/*.parquet", recursive=True))
    print(f"Total parquet files: {len(files)}")

    df = pd.read_parquet(files[0]) if sample_only else load_all_parquet()
    print("\n--- Sample data ---")
    print(df.head())
    print(df.dtypes)
    print(f"\nUnique sites in this file: {df['ss_id'].nunique()}")
    print(f"Date range: {df['datetime_GMT'].min()} to {df['datetime_GMT'].max()}")
    print(f"NaN count: {df['generation_Wh'].isna().sum()}")
    print(f"Zero-reading fraction: {(df['generation_Wh'] == 0).mean():.3f}")

    if os.path.exists(METADATA_PATH):
        meta = pd.read_csv(METADATA_PATH)
        print("\n--- metadata.csv ---")
        print(meta.head())
        print(meta.columns.tolist())
    else:
        print(f"\n[!] metadata.csv not found at {METADATA_PATH}")

    if os.path.exists(BAD_DATA_PATH):
        bad = pd.read_csv(BAD_DATA_PATH)
        print("\n--- bad_data.csv ---")
        print(bad.head())
        print(bad.columns.tolist())
        print(f"Flagged rows/sites: {len(bad)}")
    else:
        print(f"\n[!] bad_data.csv not found at {BAD_DATA_PATH}")


def _load_bad_ranges():
    """Load bad_data.csv into a list of (ss_id, start, end) tuples for masking."""
    if not os.path.exists(BAD_DATA_PATH):
        print(f"[!] bad_data.csv not found at {BAD_DATA_PATH} -- skipping bad-data masking")
        return []
    bad = pd.read_csv(BAD_DATA_PATH, parse_dates=["start_datetime_GMT", "end_datetime_GMT"])
    return list(bad[["ss_id", "start_datetime_GMT", "end_datetime_GMT"]].itertuples(index=False))


def _mask_bad(df, bad_ranges):
    """Mask out rows falling in any flagged (ss_id, start, end) window."""
    if not bad_ranges:
        return df
    mask = pd.Series(False, index=df.index)
    for ss_id, start, end in bad_ranges:
        site_mask = df["ss_id"] == ss_id
        if pd.notna(start):
            site_mask &= df["datetime_GMT"] >= start
        if pd.notna(end):
            site_mask &= df["datetime_GMT"] <= end
        mask |= site_mask
    return df[~mask]


def compute_coverage(bad_ranges):
    """Pass 1: stream through files, accumulate per-site (total, nonzero) counts
    WITHOUT holding the full dataset in memory."""
    files = sorted(glob.glob(f"{DATA_DIR}/**/*.parquet", recursive=True))
    totals = {}    # ss_id -> [total_count, nonzero_count]

    for i, f in enumerate(files):
        df = pd.read_parquet(f, columns=["ss_id", "datetime_GMT", "generation_Wh"])
        df = _mask_bad(df, bad_ranges)
        grp = df.groupby("ss_id")["generation_Wh"].agg(
            total="count", nonzero=lambda x: (x > 0).sum()
        )
        for ss_id, row in grp.iterrows():
            if ss_id not in totals:
                totals[ss_id] = [0, 0]
            totals[ss_id][0] += row["total"]
            totals[ss_id][1] += row["nonzero"]
        if (i + 1) % 10 == 0 or i == len(files) - 1:
            print(f"  coverage pass: {i + 1}/{len(files)} files")
        del df

    coverage = {ss_id: nz / tot for ss_id, (tot, nz) in totals.items() if tot > 0}
    keep_ids = {ss_id for ss_id, frac in coverage.items() if frac > NONZERO_COVERAGE_THRESHOLD}
    print(f"Sites seen: {len(totals)}")
    print(f"Sites kept (>{NONZERO_COVERAGE_THRESHOLD:.0%} non-zero): {len(keep_ids)}")
    return keep_ids


def filter_and_clean():
    """Stage 2 (memory-safe): two passes over the files on disk.
    Pass 1 computes per-site non-zero coverage across the WHOLE dataset.
    Pass 2 masks bad ranges, filters to kept sites, imputes short gaps,
    and writes one output parquet per input file (mirrors source layout)
    instead of ever holding the full dataset in memory at once."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    bad_ranges = _load_bad_ranges()

    print("Pass 1/2: computing non-zero coverage per site...")
    keep_ids = compute_coverage(bad_ranges)

    print("\nPass 2/2: masking, filtering, imputing, writing per-file outputs...")
    files = sorted(glob.glob(f"{DATA_DIR}/**/*.parquet", recursive=True))
    total_rows_in, total_rows_out = 0, 0

    for i, f in enumerate(files):
        df = pd.read_parquet(f, columns=["ss_id", "datetime_GMT", "generation_Wh"])
        total_rows_in += len(df)

        df = _mask_bad(df, bad_ranges)
        df = df[df["ss_id"].isin(keep_ids)]

        df = df.sort_values(["ss_id", "datetime_GMT"])
        df["generation_Wh"] = df.groupby("ss_id")["generation_Wh"].transform(
            lambda x: x.ffill(limit=MAX_GAP_MINUTES // 5)
        )
        df = df.dropna(subset=["generation_Wh"])
        total_rows_out += len(df)

        # mirror the source year=YYYY/month=MM structure under OUTPUT_DIR
        rel_path = os.path.relpath(f, DATA_DIR)
        out_path = os.path.join(OUTPUT_DIR, "5_minutely", rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_parquet(out_path, index=False)

        if (i + 1) % 10 == 0 or i == len(files) - 1:
            print(f"  filter pass: {i + 1}/{len(files)} files written")
        del df

    print(f"\nTotal rows in: {total_rows_in:,}  ->  rows out: {total_rows_out:,} "
          f"({total_rows_out / total_rows_in:.1%} kept)")
    print(f"Filtered files written under {OUTPUT_DIR}/5_minutely/")


def split_train_val_test():
    """Stage 3 (memory-safe): sort the already-filtered per-file parquet files
    (from stage 2, under OUTPUT_DIR/5_minutely/year=YYYY/month=MM/) into
    train/val/test folders BY YEAR, without ever loading the full dataset
    into a single DataFrame. Training code should read each split lazily
    (e.g. via pyarrow.dataset or a file-list DataLoader), not via one big
    pd.concat."""
    filtered_root = os.path.join(OUTPUT_DIR, "5_minutely")
    if not os.path.isdir(filtered_root):
        raise FileNotFoundError("Run --stage filter first.")

    files = sorted(glob.glob(f"{filtered_root}/**/*.parquet", recursive=True))
    # year=YYYY is a path component, e.g. .../year=2019/month=03/data.parquet
    def year_of(path):
        for part in path.split(os.sep):
            if part.startswith("year="):
                return int(part.split("=")[1])
        return None

    years = sorted({year_of(f) for f in files if year_of(f) is not None})
    print(f"Years present: {years}")

    train_years = set(years[:-2])
    val_years = {years[-2]} if len(years) >= 2 else set()
    test_years = {years[-1]} if len(years) >= 1 else set()

    print(f"Train years: {sorted(train_years)}")
    print(f"Val years:   {sorted(val_years)}")
    print(f"Test years:  {sorted(test_years)}")

    counts = {"train": 0, "val": 0, "test": 0}
    for f in files:
        y = year_of(f)
        split = "train" if y in train_years else "val" if y in val_years else "test"
        rel_path = os.path.relpath(f, filtered_root)
        out_path = os.path.join(OUTPUT_DIR, split, rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        os.replace(f, out_path)   # move, not copy -- avoids duplicating disk usage
        counts[split] += 1

    print(f"Moved files -> train: {counts['train']}, val: {counts['val']}, test: {counts['test']}")
    print(f"Splits are under {OUTPUT_DIR}/train/, {OUTPUT_DIR}/val/, {OUTPUT_DIR}/test/")
    print("Load each split lazily at train time, e.g.:")
    print("  import pyarrow.dataset as ds")
    print(f"  ds.dataset('{OUTPUT_DIR}/train', format='parquet')")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["inspect", "inspect_full", "filter", "split"],
        required=True,
    )
    args = parser.parse_args()

    if args.stage == "inspect":
        inspect(sample_only=True)
    elif args.stage == "inspect_full":
        inspect(sample_only=False)
    elif args.stage == "filter":
        filter_and_clean()
    elif args.stage == "split":
        split_train_val_test()
