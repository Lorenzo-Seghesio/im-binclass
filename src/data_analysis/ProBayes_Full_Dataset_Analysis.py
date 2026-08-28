import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent.parent.parent
PARQUET_PATH = BASE_DIR / "data/Fraunhofer_ProBayes_Dataset/dataset_V2.parquet"
OUTPUT_DIR   = BASE_DIR / "outputs/ProBayes/DataAnalysis"
ID_COL       = "MET_MachineCycleID"
DXP_SIGNAL   = "DXP_MldCavPrs1Act"   # time-series signal to derive features from
TARGET       = "SCA_PartWeight"        # target variable for correlation analysis
COLS_PER_ROW = 3                      # subplots per row in scalar grid
TOP_N        = 12                     # top N scalar features shown in scatter plots
# ─────────────────────────────────────────────────────────────────────────────

# Wong colorblind-friendly palette
C_BLUE      = "#0072B2"
C_VERMILION = "#D55E00"
C_TEAL      = "#009E73"
C_MAUVE     = "#CC79A7"

# ── Output directory ──────────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_parquet(PARQUET_PATH)

# ── Classify columns ──────────────────────────────────────────────────────────
ts_cols     = [col for col in df.columns
               if df[col].dropna().apply(lambda x: isinstance(x, np.ndarray)).any()]
scalar_cols = [col for col in df.columns if col not in ts_cols]

# ── Print feature summary ─────────────────────────────────────────────────────
print("=" * 60)
print(f"Dataset : {PARQUET_PATH.name}")
print(f"  Rows (parts/shots)  : {len(df)}")
print(f"  Total columns       : {len(df.columns)}")
print(f"  Scalar columns      : {len(scalar_cols)}")
print(f"  Time-series columns : {len(ts_cols)}")
print("=" * 60)

# ── Compute DXP derived features ──────────────────────────────────────────────
print(f"\nComputing derived features from '{DXP_SIGNAL}'...")

if DXP_SIGNAL not in ts_cols:
    print(f"  WARNING: '{DXP_SIGNAL}' not found in time-series columns. "
          f"DXP figure will be skipped.")
    dxp_df = None
else:
    records = []
    for idx, signal in enumerate(df[DXP_SIGNAL]):
        if isinstance(signal, np.ndarray) and len(signal) > 1:
            arr            = signal.astype(float)
            integral       = float(np.trapezoid(arr))
            peak           = float(np.max(arr))
            mean           = float(np.mean(arr))
            std            = float(np.std(arr))
            rise_time      = float(np.argmax(arr) / len(arr))          # normalised index of peak
            dur_above_half = float(np.sum(arr >= 0.5 * peak) / len(arr))  # fraction above half-peak
        else:
            integral = peak = mean = std = rise_time = dur_above_half = np.nan
        records.append({
            "part_index": idx, "integral": integral, "peak": peak,
            "mean": mean, "std": std, "rise_time": rise_time,
            "dur_above_half": dur_above_half,
        })

    dxp_df = pd.DataFrame(records)
    print(f"  integral — mean: {dxp_df['integral'].mean():.2f}  "
          f"std: {dxp_df['integral'].std():.2f}  "
          f"min: {dxp_df['integral'].min():.2f}  "
          f"max: {dxp_df['integral'].max():.2f}")
    print(f"  peak     — mean: {dxp_df['peak'].mean():.2f}  "
          f"std: {dxp_df['peak'].std():.2f}  "
          f"min: {dxp_df['peak'].min():.2f}  "
          f"max: {dxp_df['peak'].max():.2f}")

# ── Figure 1 — DXP Derived Features ──────────────────────────────────────────
if dxp_df is not None:
    fig_dxp, axes_dxp = plt.subplots(1, 2, figsize=(12, 4))

    x = dxp_df["part_index"].values

    ax = axes_dxp[0]
    ax.plot(x, dxp_df["integral"].values, color=C_BLUE, linewidth=0.7, alpha=0.7)
    ax.scatter(x, dxp_df["integral"].values, color=C_BLUE, s=6, zorder=3)
    ax.set_title("Integral (area under pressure curve)", fontsize=9)
    ax.set_xlabel("Part Index", fontsize=8)
    ax.set_ylabel("Integral", fontsize=8)
    ax.tick_params(labelsize=7)

    ax = axes_dxp[1]
    ax.plot(x, dxp_df["peak"].values, color=C_VERMILION, linewidth=0.7, alpha=0.7)
    ax.scatter(x, dxp_df["peak"].values, color=C_VERMILION, s=6, zorder=3)
    ax.set_title("Peak value", fontsize=9)
    ax.set_xlabel("Part Index", fontsize=8)
    ax.set_ylabel("Peak", fontsize=8)
    ax.tick_params(labelsize=7)

    fig_dxp.suptitle(f"{DXP_SIGNAL}  —  Derived Features per Part", fontsize=11)
    fig_dxp.tight_layout()

    out_dxp = OUTPUT_DIR / "DXP_derived_features.png"
    fig_dxp.savefig(out_dxp, dpi=300, bbox_inches="tight")
    plt.close(fig_dxp)
    print(f"\nSaved: {out_dxp}")

# ── Figure(s) — Scalar Features grouped by prefix ────────────────────────────
# Exclude the ID column (row identifier) and group remaining scalars by prefix
plot_scalars = [c for c in scalar_cols if c != ID_COL]

scalar_groups: dict[str, list[str]] = {}
for col in plot_scalars:
    prefix = col.split("_")[0]
    scalar_groups.setdefault(prefix, []).append(col)

part_idx = np.arange(len(df))

print()
for prefix, cols in sorted(scalar_groups.items()):
    n      = len(cols)
    n_rows = math.ceil(n / COLS_PER_ROW)

    fig_sc, axes_sc = plt.subplots(
        n_rows, COLS_PER_ROW,
        figsize=(COLS_PER_ROW * 4, n_rows * 2.8),
        squeeze=False,
    )
    axes_sc_flat = axes_sc.flatten()

    for i, col in enumerate(cols):
        ax = axes_sc_flat[i]
        values = pd.to_numeric(df[col], errors="coerce").values
        ax.plot(part_idx, values, color=C_TEAL, linewidth=0.6, alpha=0.6)
        ax.scatter(part_idx, values, color=C_TEAL, s=4, zorder=3)
        ax.set_title(col[len(prefix) + 1:], fontsize=7)
        ax.set_xlabel("Part Index", fontsize=6)
        ax.tick_params(labelsize=6)

    for j in range(n, len(axes_sc_flat)):
        axes_sc_flat[j].set_visible(False)

    fig_sc.suptitle(f"[{prefix}]  Scalar Features per Part  ({n} signals)", fontsize=11)
    fig_sc.tight_layout()

    out_sc = OUTPUT_DIR / f"scalar_{prefix}.png"
    fig_sc.savefig(out_sc, dpi=300, bbox_inches="tight")
    plt.close(fig_sc)
    print(f"Saved: {out_sc}")

# ── Correlation Analysis ──────────────────────────────────────────────────────
if TARGET not in scalar_cols:
    print(f"\nWARNING: '{TARGET}' not found. Correlation analysis skipped.")
else:
    print(f"\n── Correlation analysis  vs  '{TARGET}' ─────────────────")
    target_vals  = pd.to_numeric(df[TARGET], errors="coerce")
    valid_target = target_vals.notna()
    y_all        = target_vals[valid_target].values

    # ── Pearson r for every scalar vs TARGET ──────────────────────────────────
    corr_records = []
    for col in plot_scalars:
        if col == TARGET:
            continue
        x_raw = pd.to_numeric(df[col], errors="coerce")[valid_target]
        both  = x_raw.notna()
        if both.sum() < 10:
            continue
        x, y_sub = x_raw[both].values, y_all[both.values]
        try:
            r, p = stats.pearsonr(x, y_sub)
        except Exception:
            continue
        corr_records.append({"feature": col, "r": r, "p": p})

    corr_df = (pd.DataFrame(corr_records)
               .assign(abs_r=lambda d: d["r"].abs())
               .sort_values("abs_r", ascending=True))   # ascending → highest at top in barh

    # ── Figure: horizontal bar chart (all scalars) ────────────────────────────
    n_feats = len(corr_df)
    fig_bar, ax_bar = plt.subplots(figsize=(9, max(6, n_feats * 0.2)))
    bar_colors = [C_BLUE if r >= 0 else C_VERMILION for r in corr_df["r"]]
    ax_bar.barh(corr_df["feature"], corr_df["r"], color=bar_colors, alpha=0.75)
    for i, (r_val, p_val) in enumerate(zip(corr_df["r"], corr_df["p"])):
        if p_val < 0.05:
            ax_bar.text(r_val + 0.008 * (np.sign(r_val) if r_val != 0 else 1),
                        i, "*", fontsize=7, va="center")
    ax_bar.axvline(0, color="black", linewidth=0.8)
    ax_bar.set_xlabel("Pearson r", fontsize=9)
    ax_bar.set_title(
        f"Scalar Features — Pearson Correlation with {TARGET}\n"
        f"(blue = positive, red = negative,  * = p < 0.05)", fontsize=10)
    ax_bar.tick_params(axis="y", labelsize=5)
    ax_bar.tick_params(axis="x", labelsize=7)
    fig_bar.tight_layout()
    out = OUTPUT_DIR / "correlation_scalars_barplot.png"
    fig_bar.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig_bar)
    print(f"  Saved: {out}")

    # ── Figure: top N scatter plots with regression line ──────────────────────
    top_feats     = corr_df.sort_values("abs_r", ascending=False).head(TOP_N)
    n_sc_rows     = math.ceil(TOP_N / COLS_PER_ROW)
    fig_top, axes_top = plt.subplots(n_sc_rows, COLS_PER_ROW,
                                     figsize=(COLS_PER_ROW * 4, n_sc_rows * 3),
                                     squeeze=False)
    axes_top_flat = axes_top.flatten()

    for i, feat_row in enumerate(top_feats.itertuples()):
        ax = axes_top_flat[i]
        x_raw = pd.to_numeric(df[feat_row.feature], errors="coerce")[valid_target]
        both  = x_raw.notna()
        x, y_sub = x_raw[both].values, y_all[both.values]
        color = C_BLUE if feat_row.r >= 0 else C_VERMILION
        ax.scatter(x, y_sub, s=8, alpha=0.6, color=color)
        m, b = np.polyfit(x, y_sub, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, m * x_line + b, color="black", linewidth=1)
        short = feat_row.feature.split("_", 1)[-1]
        ax.set_title(f"{short}\nr = {feat_row.r:.3f}  p = {feat_row.p:.3f}", fontsize=7)
        ax.set_xlabel(short, fontsize=6)
        ax.set_ylabel(TARGET.split("_", 1)[-1], fontsize=6)
        ax.tick_params(labelsize=6)

    for j in range(len(top_feats), len(axes_top_flat)):
        axes_top_flat[j].set_visible(False)

    fig_top.suptitle(f"Top {TOP_N} Scalar Features vs {TARGET}", fontsize=11)
    fig_top.tight_layout()
    out = OUTPUT_DIR / "correlation_scalars_topN_scatter.png"
    fig_top.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig_top)
    print(f"  Saved: {out}")

    # ── DXP Derived Features Correlation ─────────────────────────────────────
    if dxp_df is not None:
        dxp_features = [c for c in dxp_df.columns if c != "part_index"]
        dxp_target   = pd.to_numeric(df[TARGET], errors="coerce").values

        dxp_corr = []
        for feat in dxp_features:
            x_raw = dxp_df[feat].values
            both  = ~np.isnan(x_raw) & ~np.isnan(dxp_target)
            if both.sum() < 10:
                continue
            try:
                r, p = stats.pearsonr(x_raw[both], dxp_target[both])
            except Exception:
                continue
            dxp_corr.append({"feature": feat, "r": r, "p": p})

        dxp_corr_df = (pd.DataFrame(dxp_corr)
                       .assign(abs_r=lambda d: d["r"].abs())
                       .sort_values("abs_r", ascending=False))
        n_dxp = len(dxp_corr_df)

        # Bar chart
        fig_db, ax_db = plt.subplots(figsize=(7, max(3, n_dxp * 0.6)))
        bar_c = [C_BLUE if r >= 0 else C_VERMILION for r in dxp_corr_df["r"]]
        ax_db.barh(dxp_corr_df["feature"], dxp_corr_df["r"], color=bar_c, alpha=0.8)
        for i, (r_val, p_val) in enumerate(zip(dxp_corr_df["r"], dxp_corr_df["p"])):
            if p_val < 0.05:
                ax_db.text(r_val + 0.008 * (np.sign(r_val) if r_val != 0 else 1),
                           i, "*", fontsize=8, va="center")
        ax_db.axvline(0, color="black", linewidth=0.8)
        ax_db.set_xlabel("Pearson r", fontsize=9)
        ax_db.set_title(
            f"{DXP_SIGNAL} Derived Features — Correlation with {TARGET}\n(* = p < 0.05)",
            fontsize=10)
        ax_db.tick_params(labelsize=8)
        fig_db.tight_layout()
        out = OUTPUT_DIR / "correlation_DXP_barplot.png"
        fig_db.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig_db)
        print(f"  Saved: {out}")

        # Scatter plots — one per derived feature
        n_dxp_rows = math.ceil(n_dxp / COLS_PER_ROW)
        fig_ds, axes_ds = plt.subplots(n_dxp_rows, COLS_PER_ROW,
                                       figsize=(COLS_PER_ROW * 4, n_dxp_rows * 3),
                                       squeeze=False)
        axes_ds_flat = axes_ds.flatten()

        for i, dxp_row in enumerate(dxp_corr_df.itertuples()):
            ax = axes_ds_flat[i]
            x_raw = dxp_df[dxp_row.feature].values
            both  = ~np.isnan(x_raw) & ~np.isnan(dxp_target)
            x, y_sub = x_raw[both], dxp_target[both]
            color = C_BLUE if dxp_row.r >= 0 else C_VERMILION
            ax.scatter(x, y_sub, s=10, alpha=0.6, color=color)
            m, b = np.polyfit(x, y_sub, 1)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, m * x_line + b, color="black", linewidth=1)
            ax.set_title(f"{dxp_row.feature}\nr = {dxp_row.r:.3f}  p = {dxp_row.p:.3f}", fontsize=8)
            ax.set_xlabel(dxp_row.feature, fontsize=7)
            ax.set_ylabel(TARGET.split("_", 1)[-1], fontsize=7)
            ax.tick_params(labelsize=7)

        for j in range(n_dxp, len(axes_ds_flat)):
            axes_ds_flat[j].set_visible(False)

        fig_ds.suptitle(f"{DXP_SIGNAL} Derived Features vs {TARGET}", fontsize=11)
        fig_ds.tight_layout()
        out = OUTPUT_DIR / "correlation_DXP_scatter.png"
        fig_ds.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig_ds)
        print(f"  Saved: {out}")
