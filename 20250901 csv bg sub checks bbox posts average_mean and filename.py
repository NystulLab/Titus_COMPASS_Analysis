#!/usr/bin/env python3
import os, glob
import pandas as pd
import numpy as np

# ====== CONFIG ======
SRC_DIR = "/home/el_tito/Documents/PD_img_data/test tiff/"      # <-- change this
DST_DIR = "/home/el_tito/Documents/PD_img_data/test tiff/test/" # <-- change this
PATTERN = "*.csv"
CLIP_NONNEG = True  # set False if you do NOT want to clip negatives to 0
# ====================

REQ_COLS = ["label", "bbox-0", "bbox-3", "intensity_mean", "intensity_max"]

os.makedirs(DST_DIR, exist_ok=True)
files = sorted(glob.glob(os.path.join(SRC_DIR, PATTERN)))
if not files:
    raise FileNotFoundError(f"No CSV files found in {SRC_DIR!r} matching {PATTERN!r}")

def process_one_csv(path):
    df = pd.read_csv(path)

    # Ensure required columns exist and are numeric
    missing = [c for c in REQ_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"{os.path.basename(path)} missing columns: {missing}")

    for c in REQ_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=REQ_COLS).copy()

    # Sort by label to make "last" unambiguous
    df = df.sort_values("label").reset_index(drop=True)

    # Identify background row(s): single-slice extent in Z
    extent = df["bbox-3"] - df["bbox-0"]
    bg_candidates = df.index[extent == 1]

    if len(bg_candidates) == 0:
        print(f"[WARN] {os.path.basename(path)}: no row with (bbox-3 - bbox-0) == 1; skipping.")
        return None

    # If multiple candidates, pick the one with the largest label (your “last label”)
    bg_idx = df.loc[bg_candidates, "label"].idxmax()
    bg_mean = float(df.loc[bg_idx, "intensity_mean"])
    bg_max  = float(df.loc[bg_idx, "intensity_max"])

    # Background-subtract columns
    df["intensity_mean"] = df["intensity_mean"] - bg_mean
    df["intensity_max"]  = df["intensity_max"]  - bg_max

    if CLIP_NONNEG:
        df["intensity_mean"] = df["intensity_mean"].clip(lower=0)
        df["intensity_max"]  = df["intensity_max"].clip(lower=0)

    # Drop the background row (so averages reflect foreground only)
    df = df.drop(index=bg_idx).reset_index(drop=True)

    # === NEW: add metadata/summary columns ===
    df["source_file"] = os.path.basename(path)

    # Average of intensity_mean for this file (post background subtraction)
    # If you want the pre-subtraction average, compute/snapshot it before the subtraction above.
    avg_mean = float(df["intensity_mean"].mean()) if len(df) else np.nan
    df["average_mean_intensity"] = avg_mean
    # =========================================

    # Save
    base = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(DST_DIR, f"{base}_bg_sub.csv")
    df.to_csv(out_path, index=False)
    return out_path
def main():
    for f in files:
        try:
            out = process_one_csv(f)
            if out:
                print(f"[OK] {os.path.basename(f)} -> {os.path.basename(out)}")
        except Exception as e:
            print(f"[ERROR] {os.path.basename(f)}: {e}")

if __name__ == "__main__":
    main()
