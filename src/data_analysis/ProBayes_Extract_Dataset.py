import numpy as np
import pandas as pd
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent.parent.parent
PARQUET_PATH = BASE_DIR / "data/Fraunhofer_ProBayes_Dataset/dataset_V2.parquet"
OUTPUT_DIR   = BASE_DIR / "data/Fraunhofer_ProBayes_Dataset/extracted"

SCALAR_COLS = [
    "MET_MaterialName",
    "QUA_TransferStroke",
    "QUA_CylinderTemperature11",
    "QUA_CoolingTime",
    "QUA_InjectionSpeed1",
    "TCE_TemperatureMainLineMean",
    "TCN_TemperatureMainLineMean",
    "DXP_HoldingPressure1",
    "DXP_HoldingPressure2",
    "DXP_HoldingTime1",
    "DXP_HoldingTime2",
    "SET_BackPressure",
    "SCA_PartWeight",
]

TS_SIGNAL = "DXP_MldCavPrs1Act"
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
print(f"Loading {PARQUET_PATH.name} ...")
df = pd.read_parquet(PARQUET_PATH)
print(f"  {len(df)} parts  |  {len(df.columns)} columns")

# ── Validate requested columns ────────────────────────────────────────────────
missing = [c for c in SCALAR_COLS + [TS_SIGNAL] if c not in df.columns]
if missing:
    raise ValueError(f"Columns not found in dataset: {missing}")

# ID_Part: 0-based row index (matches part_index used in analysis scripts)
id_part = np.arange(len(df))

# ── Extract 1: scalar features ────────────────────────────────────────────────
print("\nExtracting scalar features ...")
scalar_df = df[SCALAR_COLS].copy()
scalar_df.insert(0, "ID_Part", id_part)

out_scalar = OUTPUT_DIR / "scalar_features.csv"
scalar_df.to_csv(out_scalar, index=False)
print(f"  Columns : ID_Part + {len(SCALAR_COLS)} features")
print(f"  Rows    : {len(scalar_df)}")
print(f"  Saved   : {out_scalar}")

# ── Extract 2: pressure curve time series (zero-padded) ──────────────────────
print(f"\nExtracting '{TS_SIGNAL}' time series ...")

max_len = int(df[TS_SIGNAL].apply(
    lambda x: len(x) if isinstance(x, np.ndarray) else 0
).max())
print(f"  Max signal length : {max_len}  (padding target)")

# Build padded matrix — shape (n_parts, max_len), dtype float32
n_parts = len(df)
matrix  = np.zeros((n_parts, max_len), dtype=np.float32)

for i, sig in enumerate(df[TS_SIGNAL]):
    if isinstance(sig, np.ndarray):
        sig_len         = min(len(sig), max_len)
        matrix[i, :sig_len] = sig[:sig_len].astype(np.float32)

# Build DataFrame: ID_Part + t_00000 … t_12078
col_names = [f"t_{j:05d}" for j in range(max_len)]
ts_df     = pd.DataFrame(matrix, columns=col_names)
ts_df.insert(0, "ID_Part", id_part)

out_ts = OUTPUT_DIR / "pressure_curve_padded.csv"
ts_df.to_csv(out_ts, index=False, float_format="%.4f")
print(f"  Shape   : {ts_df.shape}  (parts × [ID_Part + {max_len} timesteps])")
print(f"  Saved   : {out_ts}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n── Done ──────────────────────────────────────────────────────────────")
print(f"  {out_scalar.name:<35}  {out_scalar.stat().st_size / 1e6:.1f} MB")
print(f"  {out_ts.name:<35}  {out_ts.stat().st_size / 1e6:.1f} MB")
