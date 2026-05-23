import argparse
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────
PARQUET_PATH = "data/Fraunhofer_ProBayes_Dataset/dataset_V2.parquet"
ID_COL       = "MET_MachineCycleID"
PART_INDEX   = 100          # which part (row index) to plot; change as needed
COLS_PER_ROW = 3          # subplots per row inside each group figure
MUX_CHANNELS = 19   # number of channels in multiplexed signals (should be 16 but also 19 not bad)
# ─────────────────────────────────────────────────────────────────────────────

# ── CLI arguments ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="ProBayes dataset analysis")
parser.add_argument(
    "signals", nargs="*",
    help="Time-series signal name(s) to plot. If omitted, all signals are plotted."
)
args = parser.parse_args()
filter_signals = args.signals  # empty list = plot all
# ─────────────────────────────────────────────────────────────────────────────

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_parquet(PARQUET_PATH)

# ── Classify columns ──────────────────────────────────────────────────────────
ts_cols     = [col for col in df.columns
               if df[col].dropna().apply(lambda x: isinstance(x, np.ndarray)).any()]
scalar_cols = [col for col in df.columns if col not in ts_cols]

# Group time-series by sensor prefix (part before first '_')
prefix_groups: dict[str, list[str]] = {}
for col in ts_cols:
    prefix = col.split("_")[0]
    prefix_groups.setdefault(prefix, []).append(col)

# ── Print feature summary ─────────────────────────────────────────────────────
print("=" * 60)
print(f"Dataset: {PARQUET_PATH}")
print(f"  Rows (parts/shots) : {len(df)}")
print(f"  Total columns      : {len(df.columns)}")
print(f"  Scalar columns     : {len(scalar_cols)}")
print(f"  Time-series columns: {len(ts_cols)}")
print()

print("── Scalar features ──────────────────────────────────────")
for col in scalar_cols:
    dtype = df[col].dtype
    print(f"  {col:45s}  {str(dtype)}")

print()
print("── Time-series features (grouped by sensor prefix) ──────")
for prefix, cols in sorted(prefix_groups.items()):
    # get typical array length for this group
    sample_len = len(df[cols[0]].dropna().iloc[0])
    print(f"  [{prefix}]  {len(cols)} signals  |  {sample_len} time steps each")
    for col in cols:
        print(f"    {col}")

print("=" * 60)

# ── Plot time-series for one part ─────────────────────────────────────────────
row      = df.iloc[PART_INDEX]
part_id  = row[ID_COL]

print(f"\n── Scalar values for part index {PART_INDEX}  (ID: {part_id}) ─────────")
for col in scalar_cols:
    print(f"  {col:45s}  {row[col]}")
print("=" * 60)

print(f"\nPlotting time-series for part index {PART_INDEX}  (ID: {part_id})")

# Filter to requested signals (or keep all)
if filter_signals:
    unknown = [s for s in filter_signals if s not in ts_cols]
    if unknown:
        print(f"Warning: unknown signal(s) ignored: {unknown}")
    plot_groups: dict[str, list[str]] = {}
    for sig in filter_signals:
        if sig in ts_cols:
            prefix = sig.split("_")[0]
            plot_groups.setdefault(prefix, []).append(sig)
else:
    plot_groups = prefix_groups

# ─────────────────────────────────────────────────────────────────────────────


def is_mux(col: str) -> bool:
    """Return True if the column name indicates a multiplexed signal."""
    return "Mux" in col and "TmpMux" in col


def plot_group(prefix: str, cols: list[str], row: pd.Series, part_id) -> None:
    """Plot one figure per sensor group. Mux signals get their own demuxed figure."""
    regular = [c for c in cols if not is_mux(c)]
    mux_cols = [c for c in cols if is_mux(c)]

    # ── regular signals ──────────────────────────────────────────────────────
    if regular:
        n      = len(regular)
        n_rows = math.ceil(n / COLS_PER_ROW)
        fig, axes = plt.subplots(n_rows, COLS_PER_ROW,
                                 figsize=(COLS_PER_ROW * 4, n_rows * 2.5))
        axes = np.array(axes).flatten()
        for i, col in enumerate(regular):
            signal = row[col]
            ax = axes[i]
            if isinstance(signal, np.ndarray):
                ax.plot(signal, linewidth=0.8)
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes, color="grey")
            ax.set_title(col[len(prefix) + 1:], fontsize=7)
            ax.tick_params(labelsize=6)
        for j in range(n, len(axes)):
            axes[j].set_visible(False)
        fig.suptitle(f"[{prefix}]  Part ID: {part_id}  ({n} signals)", fontsize=10)
        fig.tight_layout()

    # ── multiplexed signals (one figure per mux column) ──────────────────────
    for col in mux_cols:
        arr = row[col]
        if not isinstance(arr, np.ndarray):
            continue
        arr = arr.astype(float)

        # Determine active channels: mean > 0.5 AND std > 0.5 across the dataset
        active_mask = []
        for ch in range(MUX_CHANNELS):
            means = df[col].apply(
                lambda a, c=ch: a[c::MUX_CHANNELS].mean() if isinstance(a, np.ndarray) else np.nan
            )
            stds = df[col].apply(
                lambda a, c=ch: a[c::MUX_CHANNELS].std() if isinstance(a, np.ndarray) else np.nan
            )
            active_mask.append(means.mean() > 0.5 and stds.mean() > 0.5)

        active_chs = [ch for ch in range(MUX_CHANNELS) if active_mask[ch]]
        dead_chs   = [ch for ch in range(MUX_CHANNELS) if not active_mask[ch]]
        print(f"  {col}: {len(active_chs)} active channels: {[c+1 for c in active_chs]}")
        if dead_chs:
            print(f"          {len(dead_chs)} dead channels (skipped): {[c+1 for c in dead_chs]}")

        n      = len(active_chs)
        n_rows = math.ceil(n / COLS_PER_ROW)
        fig, axes = plt.subplots(n_rows, COLS_PER_ROW,
                                 figsize=(COLS_PER_ROW * 4, n_rows * 2.5))
        axes = np.array(axes).flatten()

        for i, ch in enumerate(active_chs):
            # Extract non-zero samples only (envelope of mux pulse signal)
            raw = arr[ch::MUX_CHANNELS]
            nonzero_idx = np.where(np.abs(raw) > 0.1)[0]
            ax = axes[i]
            if len(nonzero_idx) > 1:
                ax.plot(nonzero_idx, raw[nonzero_idx], linewidth=0.9, marker=".", markersize=2)
            else:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, color="grey", fontsize=7)
            ax.set_title(f"ch{ch + 1:02d}", fontsize=8)
            ax.set_xlabel("scan index", fontsize=6)
            ax.tick_params(labelsize=6)

        for j in range(n, len(axes)):
            axes[j].set_visible(False)

        short = col[len(prefix) + 1:]
        fig.suptitle(
            f"[{prefix}]  {short}  — {n} active channels (non-zero envelope)  |  Part ID: {part_id}",
            fontsize=10)
        fig.tight_layout()


for prefix, cols in sorted(plot_groups.items()):
    plot_group(prefix, cols, row, part_id)

plt.show()
