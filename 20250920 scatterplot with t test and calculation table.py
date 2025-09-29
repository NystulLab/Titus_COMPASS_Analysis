#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

# =========================
# ====== CONFIGURE ========
# =========================

csv_files = [
    ("Control", "/home/el_tito/Documents/PD_img_data/20250529 HCR in compass/tiffed/results/results/bg sub results/processed/trimmed/20250529 cr hcr-02(9)-ApoTome RAW Convert-09_edited_corrected_props_bg_subcombined_averaged.csv", "#4D4D4D"),
    ("Rbbp R", "/home/el_tito/Documents/PD_img_data/20250529 HCR in compass/tiffed/results/results/bg sub results/processed/trimmed/20250529 rr hcr_010_edited_corrected_props_bg_subcombined_averaged.csv", "#737373"),
    ("Rbbpg",  "/home/el_tito/Documents/PD_img_data/20250529 HCR in compass/tiffed/results/results/bg sub results/processed/trimmed/20250529 rg hcr_006_edited_corrected_props_bg_subcombined_averaged.csv", "#8C564B"),
    # add more as needed...
]

target_column = "intensity_mean"
control_label = "Control"

# Where to write the per-file summary table
OUT_TABLE_CSV = "/home/el_tito/Documents/PD_img_data/20250529 HCR in compass/tiffed/results/results/bg sub results/processed/trimmed/summary_per_file.csv"

# Where to save the figure (set to None to skip saving)
OUT_FIG_PATH = "/home/el_tito/Documents/PD_img_data/20250529 HCR in compass/tiffed/results/results/bg sub results/processed/trimmed/box_scatter.png"

# =========================
# ====== FUNCTIONS ========
# =========================

def pval_to_star(p: float) -> str:
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"

def load_and_stack(csv_specs, target_col):
    """Load each CSV, keep target column, and add genotype/color/source_file metadata."""
    frames = []
    for genotype, path, color in csv_specs:
        if not os.path.exists(path):
            raise FileNotFoundError(f"CSV not found: {path}")
        df = pd.read_csv(path)
        if target_col not in df.columns:
            raise ValueError(f"'{target_col}' not found in {path}")
        sub = df[[target_col]].copy()
        sub["genotype"] = genotype
        sub["color"] = color
        sub["source_file"] = os.path.basename(path)
        frames.append(sub)
    return pd.concat(frames, ignore_index=True)

def compute_tests_and_summary(combined_df, target_col, control_lab):
    """Welch tests vs control (row-level), and per-file averages + comparisons."""
    # Welch t-tests per genotype vs control (using raw row values)
    control_vals = combined_df.loc[combined_df["genotype"] == control_lab, target_col].values
    if control_vals.size == 0:
        raise ValueError(f"No rows found for control label '{control_lab}'.")

    overall_means = combined_df.groupby("genotype")[target_col].mean().rename("overall_mean").to_dict()
    control_mean = float(overall_means[control_lab])

    pvals = {}
    for g in combined_df["genotype"].unique():
        if g == control_lab:
            continue
        test_vals = combined_df.loc[combined_df["genotype"] == g, target_col].values
        if test_vals.size == 0:
            pvals[g] = np.nan
        else:
            _, pval = ttest_ind(control_vals, test_vals, equal_var=False)
            pvals[g] = float(pval)

    # Per-file average of intensity_mean
    per_file = (
        combined_df
        .groupby(["genotype", "source_file"], as_index=False)[target_col]
        .mean()
        .rename(columns={target_col: "average_mean_intensity"})
    )

    # Add comparisons vs control (genotype-level replicated per file)
    per_file["overall_mean"] = per_file["genotype"].map(overall_means).astype(float)
    per_file["diff_vs_control"] = per_file["overall_mean"] - control_mean
    per_file["pct_change_vs_control"] = np.where(
        control_mean != 0,
        100.0 * (per_file["overall_mean"] - control_mean) / control_mean,
        np.nan
    )
    per_file["p_value_vs_control"] = per_file["genotype"].map(lambda g: pvals.get(g, np.nan))

    return pvals, control_mean, overall_means, per_file

def plot_box_with_scatter(combined_df, target_col, control_lab, pvals, order, palette, save_path=None):
    """Boxplot with dots in FRONT and simple significance bars vs control."""
    plt.figure(figsize=(12, 6))

    # Boxes behind
    ax = sns.boxplot(
        data=combined_df, x="genotype", y=target_col,
        order=order, palette=palette,
        showfliers=False, zorder=1
    )

    # Some backends ignore zorder in seaborn; force the patches lower anyway
    for patch in ax.artists:
        patch.set_zorder(1)

    # Dots on top
    for i, label in enumerate(order):
        subset = combined_df[combined_df["genotype"] == label]
        x_jitter = np.random.normal(loc=i, scale=0.08, size=len(subset))
        plt.scatter(
            x_jitter, subset[target_col],
            color="black", alpha=0.7, s=14,
            zorder=5, clip_on=False
        )

    # Significance bars vs control
    ymax = combined_df[target_col].max()
    # Robust offset if all small
    if not np.isfinite(ymax) or ymax == 0:
        ymax = 1.0
    base_offset = 0.1 * ymax
    step = 0.08 * ymax
    line_height = 0.05 * ymax

    ctrl_idx = order.index(control_lab)
    current_y = ymax + base_offset

    for i, label in enumerate(order):
        if label == control_lab:
            continue
        p = pvals.get(label, np.nan)
        if not np.isfinite(p):
            continue
        star = pval_to_star(p)
        x1, x2 = ctrl_idx, i
        # horizontal bar with vertical ticks
        ax.plot(
            [x1, x1, x2, x2],
            [current_y, current_y + line_height, current_y + line_height, current_y],
            lw=1.5, c="black", zorder=6
        )
        ax.text(
            (x1 + x2) / 2.0, current_y + line_height * 1.1,
            star, ha="center", va="bottom",
            color="red", fontsize=11, zorder=7
        )
        current_y += step + line_height

    ax.set_xlabel("")
    ax.set_ylabel(target_col)
    plt.xticks(rotation=45)
    plt.title("HCR (Welch’s t-tests vs Control)")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"[OK] Figure saved -> {save_path}")

    plt.show()

# =========================
# ========= MAIN ==========
# =========================

def main():
    # Load & stack
    combined_df = load_and_stack(csv_files, target_column)

    # Order and palette (preserve the order in csv_files)
    order = [g for g, _, _ in csv_files]
    palette = {g: c for g, _, c in csv_files}

    # Stats & per-file summary table
    pvals, control_mean, overall_means, per_file = compute_tests_and_summary(
        combined_df, target_column, control_label
    )

    # Write summary CSV
    os.makedirs(os.path.dirname(OUT_TABLE_CSV), exist_ok=True)
    per_file.to_csv(OUT_TABLE_CSV, index=False)
    print(f"[OK] Wrote summary table -> {OUT_TABLE_CSV}")

    # Plot
    plot_box_with_scatter(
        combined_df=combined_df,
        target_col=target_column,
        control_lab=control_label,
        pvals=pvals,
        order=order,
        palette=palette,
        save_path=OUT_FIG_PATH
    )

if __name__ == "__main__":
    main()
