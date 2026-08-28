import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent.parent.parent
PARQUET_PATH = BASE_DIR / "data/Fraunhofer_ProBayes_Dataset/dataset_V2.parquet"
OUTPUT_DIR   = BASE_DIR / "outputs/ProBayes/DataAnalysis"

REFERENCE = "QUA_CylinderTemperature11"
TARGETS   = [
    "QUA_CylinderTemperature01",
    "QUA_CylinderTemperature02",
    "QUA_CylinderTemperature03",
    "QUA_CylinderTemperature04",
]
# ─────────────────────────────────────────────────────────────────────────────

# Wong colorblind-friendly palette
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_parquet(PARQUET_PATH)

all_cols = [REFERENCE] + TARGETS
missing  = [c for c in all_cols if c not in df.columns]
if missing:
    raise ValueError(f"Columns not found in dataset: {missing}")

sub = df[all_cols].apply(pd.to_numeric, errors="coerce").dropna()
x   = sub[REFERENCE].values

# ── Print correlation table ───────────────────────────────────────────────────
print("=" * 72)
print(f"Reference : {REFERENCE}")
print(f"N (valid) : {len(sub)}")
print("=" * 72)
print(f"  {'Target':<35}  {'Pearson r':>9}  {'p-value':>10}  {'R²':>7}")
print("  " + "-" * 66)

results = []
for col in TARGETS:
    y       = sub[col].values
    r, p    = stats.pearsonr(x, y)
    rho, _  = stats.spearmanr(x, y)
    r2      = r ** 2
    results.append({"col": col, "r": r, "p": p, "rho": rho, "r2": r2})
    sig = "*" if p < 0.05 else " "
    print(f"  {col:<35}  {r:>+9.4f}  {p:>10.2e}  {r2:>7.4f} {sig}")

print("=" * 72)
print("  (* = p < 0.05)")

# ── Figure: scatter plots (2 × 2) ────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes_flat = axes.flatten()

ref_short = REFERENCE.split("_", 1)[-1]

for i, (res, ax, color) in enumerate(zip(results, axes_flat, COLORS)):
    col   = res["col"]
    y     = sub[col].values
    short = col.split("_", 1)[-1]

    ax.scatter(x, y, s=12, alpha=0.55, color=color, zorder=3)

    # regression line
    m, b   = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 200)
    ax.plot(x_line, m * x_line + b, color="black", linewidth=1.2, label=f"y = {m:.3f}x + {b:.1f}")

    ax.set_xlabel(ref_short, fontsize=8)
    ax.set_ylabel(short, fontsize=8)
    ax.set_title(
        f"{short}\nr = {res['r']:+.4f}   R² = {res['r2']:.4f}   p = {res['p']:.2e}",
        fontsize=8,
    )
    ax.legend(fontsize=6)
    ax.tick_params(labelsize=7)

fig.suptitle(
    f"Cylinder Temperature Correlation\n{REFERENCE}  vs  01 / 02 / 03 / 04",
    fontsize=11,
)
fig.tight_layout()

out = OUTPUT_DIR / "cylinder_temp_correlation.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {out}")
