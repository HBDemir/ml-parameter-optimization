"""
extract_qfactors.py

Batch-computes Q-factors for all z-map images and exports the result to CSV.
Uses qfactor.py for all processing.

Output
------
qfactors_{segment_width_um}um.csv   — all slices with Q-factor and diagnostics,
                                      joined with process metadata

Usage
-----
    python extract_qfactors.py                  # default 1.0 µm slices
    python extract_qfactors.py --slice 0.5      # 0.5 µm slices
    python extract_qfactors.py --slice 2.0 --out results.csv
"""

import argparse
import os
from pathlib import Path

import pandas as pd

from qfactor import QFactorParams, process_dataset

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR = Path("data/dataset_combined")
DATASETS = ["06082025", "26082025"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def collect_zmap_files(data_dir: Path, datasets: list[str]) -> tuple[list[tuple], dict]:
    """
    Return:
        files   : list of (zmap_dir, filename) tuples
        fn_map  : dict mapping filename → dataset name
    """
    files = []
    fn_map = {}
    for dataset in datasets:
        zmap_dir = data_dir / dataset / "zmap"
        if not zmap_dir.exists():
            print(f"Warning: {zmap_dir} not found, skipping.")
            continue
        tifs = sorted(f for f in os.listdir(zmap_dir) if f.endswith(".tif"))
        files.extend((zmap_dir, f) for f in tifs)
        fn_map.update({f: dataset for f in tifs})
        print(f"  {dataset}: {len(tifs)} files")
    return files, fn_map


def load_metadata(data_dir: Path, datasets: list[str]) -> pd.DataFrame:
    """Load and concatenate metadata xlsx files. Cleans junk columns."""
    junk = ["*", "Unnamed: 10", "Unnamed: 11"]
    frames = []
    for dataset in datasets:
        path = data_dir / dataset / f"{dataset}_metadata.xlsx"
        if not path.exists():
            print(f"Warning: metadata not found for {dataset}, skipping.")
            continue
        df = pd.read_excel(path)
        df["dataset"] = dataset
        df = df.drop(columns=[c for c in junk if c in df.columns])

        # Rename '#' to sample_id if it exists, otherwise use row position
        if "#" in df.columns:
            df = df.rename(columns={"#": "sample_id"})
            df["sample_id"] = df["sample_id"].astype(int)
        else:
            print(f"  Warning: '#' column not found in {dataset}, using row index as sample_id.")
            print(f"  Columns found: {df.columns.tolist()}")
            df["sample_id"] = range(1, len(df) + 1)

        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Batch Q-factor extraction")
    parser.add_argument("--slice", type=float, default=1.0,
                        metavar="UM", help="Slice width in µm (default: 1.0)")
    parser.add_argument("--out", type=str, default=None,
                        help="Output CSV path (default: qfactors_{slice}um.csv)")
    args = parser.parse_args()

    segment_width_um = args.slice
    output_path = Path(args.out) if args.out else Path(f"qfactors_{segment_width_um}um.csv")

    print(f"Slice width : {segment_width_um} µm")
    print(f"Output      : {output_path}")
    print()

    # Collect files
    print("Collecting z-map files...")
    zmap_files, fn_map = collect_zmap_files(DATA_DIR, DATASETS)
    print(f"Total: {len(zmap_files)} files\n")

    if not zmap_files:
        print("No files found. Check DATA_DIR path.")
        return

    # Load metadata
    print("Loading metadata...")
    df_metadata = load_metadata(DATA_DIR, DATASETS)
    print(f"Metadata rows: {len(df_metadata)}\n")

    # Compute Q-factors
    print("Computing Q-factors...")
    params = QFactorParams()
    df_q = process_dataset(zmap_files, segment_width_um=segment_width_um, params=params)
    print(f"\nTotal slices: {len(df_q):,}")
    print(f"Valid Q-factors: {df_q['q_factor'].notna().sum():,} "
          f"({df_q['q_factor'].notna().mean() * 100:.1f}%)")

    # Add sample_id and dataset columns for joining
    df_q["dataset"] = df_q["filename"].map(fn_map)
    df_q["sample_id"] = df_q["filename"].str.split("_").str[0].astype(int)

    # Join with metadata on sample_id + dataset
    if not df_metadata.empty:
        df_out = df_q.merge(df_metadata, on=["sample_id", "dataset"], how="left")
        matched = df_out["Material"].notna().sum()
        print(f"Metadata joined: {matched:,} / {len(df_out):,} rows matched")
    else:
        df_out = df_q

    # Save
    df_out.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}  ({len(df_out):,} rows, {len(df_out.columns)} columns)")


if __name__ == "__main__":
    main()
