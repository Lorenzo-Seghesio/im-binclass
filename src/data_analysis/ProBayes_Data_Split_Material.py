"""
ProBayes_Data_Split_Material.py
================================
Splits scalar_features.csv and pressure_curve_padded_downsampled.csv by
material (ABS / PP), keeping the ID_Part link between both files.

Outputs (same folder as inputs):
  scalar_features_ABS.csv
  scalar_features_PP.csv
  pressure_curve_padded_downsampled_ABS.csv
  pressure_curve_padded_downsampled_PP.csv

The MET_MaterialName column is dropped from the output scalar files since
the material is now implicit in the filename.
"""

import pandas as pd
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent.parent.parent
EXTRACTED    = BASE_DIR / "data" / "Fraunhofer_ProBayes_Dataset" / "extracted"

SCALAR_CSV   = EXTRACTED / "scalar_features.csv"
PRESSURE_CSV = EXTRACTED / "pressure_curve_padded_downsampled.csv"

ID_COL       = "ID_Part"
MAT_COL      = "MET_MaterialName"

# ── Load ──────────────────────────────────────────────────────────────────────
df_scalar   = pd.read_csv(SCALAR_CSV,   index_col=ID_COL)
df_pressure = pd.read_csv(PRESSURE_CSV, index_col=ID_COL)

materials = sorted(df_scalar[MAT_COL].dropna().unique())
print(f"Materials found : {materials}")
print(f"Scalar rows     : {len(df_scalar)}   |   Pressure rows : {len(df_pressure)}")

# ── Split and save ────────────────────────────────────────────────────────────
for mat in materials:
    mask        = df_scalar[MAT_COL] == mat
    ids         = df_scalar.index[mask]

    # Scalar — drop material column (implicit in filename)
    df_sc_mat   = df_scalar.loc[ids].drop(columns=[MAT_COL])
    sc_out      = EXTRACTED / f"scalar_features_{mat}.csv"
    df_sc_mat.to_csv(sc_out)

    # Pressure — filter by same IDs (inner join to be safe)
    common_ids  = ids.intersection(df_pressure.index)
    df_pt_mat   = df_pressure.loc[common_ids]
    pt_out      = EXTRACTED / f"pressure_curve_padded_downsampled_{mat}.csv"
    df_pt_mat.to_csv(pt_out)

    print(f"\n[{mat}]  {len(df_sc_mat)} parts")
    print(f"  → {sc_out.name}")
    print(f"  → {pt_out.name}")

print("\nDone.")
