#!/usr/bin/env python3
import os, glob
import pandas as pd

# ====== CONFIG ======
SRC_DIR = "/home/el_tito/Documents/PD_img_data/test tiff/test/"      # <-- change this
DST_DIR = "/home/el_tito/Documents/PD_img_data/test tiff/test/stitch_out/"  # <-- change this
PATTERN = "*.csv"
OUT_FILE = "stitched.csv"
# ====================

os.makedirs(DST_DIR, exist_ok=True)
files = sorted(glob.glob(os.path.join(SRC_DIR, PATTERN)))
if not files:
    raise FileNotFoundError(f"No CSV files found in {SRC_DIR!r} matching {PATTERN!r}")

def stitch_csvs(files, out_path):
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df["source_file"] = os.path.basename(f)
            frames.append(df)
            print(f"[OK] Loaded {os.path.basename(f)} with {len(df)} rows")
        except Exception as e:
            print(f"[ERROR] Could not read {os.path.basename(f)}: {e}")
    if not frames:
        print("[WARN] No CSVs stitched; all failed?")
        return None
    stitched = pd.concat(frames, ignore_index=True)
    stitched.to_csv(out_path, index=False)
    print(f"[DONE] Wrote stitched CSV with {len(stitched)} rows -> {out_path}")
    return out_path

def main():
    out_path = os.path.join(DST_DIR, OUT_FILE)
    stitch_csvs(files, out_path)

if __name__ == "__main__":
    main()
