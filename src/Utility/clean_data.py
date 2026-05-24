"""clean_data.py — One-time X-outlier removal utility.

Reads raw scalar and pressure-curve CSVs for a given material, removes parts
whose process-parameter (X) values are outliers, and saves clean versions of
both files to the same directory.  All downstream models (M1, M2, Fusion,
RefModels) read from the clean CSVs so that their own outlier-removal code
only needs to handle y-outliers on the dev set.

Outlier rule (applied per feature, X features only):
  • IQR > 0  →  standard IQR rule:  outside [Q1 − k·IQR, Q3 + k·IQR]
  • IQR = 0  →  z-score fallback:   |z| > z_threshold
    (catches near-constant columns with a single extreme value, e.g.
     DXP_HoldingPressure2 = 200 bar when all other parts have 0 bar)

Usage
-----
    python src/Utility/clean_data.py --material ABS
    python src/Utility/clean_data.py --material PP
    python src/Utility/clean_data.py --material ALL
Tuning knobs (edit at the top of this file):
    IQR_MULTIPLIER   = 1.5   # k for the standard IQR rule
    ZSCORE_THRESHOLD = 3.0   # threshold for zero-IQR columns
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parents[2]
DATA_DIR  = BASE_DIR / "data" / "Fraunhofer_ProBayes_Dataset" / "extracted"

ID_COL     = "ID_Part"
TARGET_COL = "SCA_PartWeight"

IQR_MULTIPLIER   = 1.5   # k for standard IQR rule
ZSCORE_THRESHOLD = 3.0   # threshold for zero-IQR columns

MATERIALS = ["ABS", "PP"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _scalar_csv(mat: str) -> Path:
    return DATA_DIR / f"scalar_features_{mat}.csv"


def _pressure_csv(mat: str) -> Path:
    return DATA_DIR / f"pressure_curve_padded_downsampled_{mat}.csv"


def _scalar_clean_csv(mat: str) -> Path:
    return DATA_DIR / f"scalar_features_{mat}_clean.csv"


def _pressure_clean_csv(mat: str) -> Path:
    return DATA_DIR / f"pressure_curve_padded_downsampled_{mat}_clean.csv"


def _build_outlier_mask(df: pd.DataFrame, pp_cols: list,
                        iqr_k: float, z_thr: float) -> pd.Series:
    """Return a boolean mask (True = keep) based on X-feature outlier rules."""
    mask   = pd.Series(True, index=df.index)
    report = {}  # col → list of removed part IDs

    for col in pp_cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr    = q3 - q1

        if iqr > 0:
            lo = q1 - iqr_k * iqr
            hi = q3 + iqr_k * iqr
            col_mask = (df[col] >= lo) & (df[col] <= hi)
        else:
            # IQR = 0: column is (nearly) constant — use z-score fallback
            std = df[col].std()
            if std == 0:
                col_mask = pd.Series(True, index=df.index)
            else:
                z        = (df[col] - df[col].mean()) / std
                col_mask = z.abs() <= z_thr

        flagged = df.index[~col_mask].tolist()
        if flagged:
            report[col] = flagged
        mask &= col_mask

    return mask, report


# ── Core cleaning function ────────────────────────────────────────────────────

def clean_material(mat: str, iqr_k: float, z_thr: float,
                   overwrite: bool = True) -> dict:
    """Clean one material's CSVs.  Returns a summary dict for logging."""
    sc_in  = _scalar_csv(mat)
    pt_in  = _pressure_csv(mat)
    sc_out = _scalar_clean_csv(mat)
    pt_out = _pressure_clean_csv(mat)

    if not sc_in.exists():
        raise FileNotFoundError(f"Scalar CSV not found: {sc_in}")
    if not pt_in.exists():
        raise FileNotFoundError(f"Pressure CSV not found: {pt_in}")

    print(f"\n{'━'*60}")
    print(f"  Material : {mat}")
    print(f"{'━'*60}")

    df_sc = pd.read_csv(sc_in,  index_col=ID_COL)
    df_pt = pd.read_csv(pt_in,  index_col=ID_COL)
    print(f"  Loaded   : {len(df_sc)} scalar rows, {len(df_pt)} pressure rows")

    # Process-parameter columns (exclude target)
    pp_cols = [c for c in df_sc.columns if c != TARGET_COL]

    # Impute any missing values in pp_cols before computing IQR
    for col in pp_cols:
        if df_sc[col].isna().any():
            df_sc[col] = df_sc[col].fillna(df_sc[col].median())

    # Build outlier mask on X features only
    mask, report = _build_outlier_mask(df_sc, pp_cols, iqr_k, z_thr)

    removed_ids = df_sc.index[~mask].tolist()
    n_removed   = len(removed_ids)
    print(f"  Removed  : {n_removed} parts")
    if n_removed:
        print(f"    IDs     : {removed_ids}")
        for col, ids in report.items():
            method = "IQR" if df_sc[col].std() > 0 and \
                     (df_sc[col].quantile(0.75) - df_sc[col].quantile(0.25)) > 0 \
                     else "z-score"
            print(f"    ← {col} ({method}): {ids}")

    # Apply mask — intersect with pressure IDs
    surviving_sc_ids = df_sc.index[mask]
    surviving_ids    = surviving_sc_ids.intersection(df_pt.index)

    if len(surviving_ids) < len(surviving_sc_ids):
        missing_pt = set(surviving_sc_ids) - set(df_pt.index)
        print(f"  [warn] {len(missing_pt)} scalar IDs have no pressure curve "
              f"and are dropped: {sorted(missing_pt)[:10]}{'...' if len(missing_pt)>10 else ''}")

    df_sc_clean = df_sc.loc[surviving_ids]
    df_pt_clean = df_pt.loc[surviving_ids]
    print(f"  Kept     : {len(df_sc_clean)} parts")

    if not overwrite and sc_out.exists():
        print(f"  [SKIP] {sc_out.name} already exists (use --overwrite to regenerate)")
    else:
        df_sc_clean.to_csv(sc_out)
        df_pt_clean.to_csv(pt_out)
        print(f"  Saved    : {sc_out.name}")
        print(f"             {pt_out.name}")

    return {
        "material":      mat,
        "n_original":    len(df_sc),
        "n_removed":     n_removed,
        "n_clean":       len(df_sc_clean),
        "removed_ids":   removed_ids,
        "trigger_cols":  {col: ids for col, ids in report.items()},
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="One-time X-outlier removal for IM-ML datasets."
    )
    p.add_argument("--material", required=True,
                   choices=["ABS", "PP", "ALL", "abs", "pp", "all"],
                   help="Material to process (ABS, PP, or ALL)")
    p.add_argument("--overwrite", action="store_true", default=True,
                   help="Overwrite existing clean files (default: True)")
    return p.parse_args()


def main():
    args = parse_args()
    mat  = args.material.upper()
    mats = MATERIALS if mat == "ALL" else [mat]

    print(f"\n{'═'*60}")
    print(f"  clean_data.py  |  iqr_k={IQR_MULTIPLIER}  "
          f"z_thr={ZSCORE_THRESHOLD}")
    print(f"{'═'*60}")

    summaries = []
    for m in mats:
        s = clean_material(m, IQR_MULTIPLIER, ZSCORE_THRESHOLD,
                           overwrite=args.overwrite)
        summaries.append(s)

    # Save a cleaning log to data dir
    log_path = DATA_DIR / "clean_data_log.json"
    log = {
        "iqr_multiplier":   IQR_MULTIPLIER,
        "zscore_threshold": ZSCORE_THRESHOLD,
        "materials":        summaries,
    }
    log_path.write_text(json.dumps(log, indent=2))
    print(f"\n  Cleaning log → {log_path}")

    print(f"\n{'═'*60}")
    print("  Done.  Next steps:")
    print("    1. Run M1:      python src/M1_PPPT_to_W.py --material <mat>")
    print("    2. Run M2:      python src/M2_PP_to_F.py   --material <mat>")
    print("    3. Run Fusion:  python src/Fusion_M1M2_WPred.py --material <mat>")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
