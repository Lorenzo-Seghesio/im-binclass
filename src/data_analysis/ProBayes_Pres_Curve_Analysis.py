import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from scipy.signal import resample

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent.parent.parent
CSV_PATH      = BASE_DIR / "data/Fraunhofer_ProBayes_Dataset/extracted/pressure_curve_padded.csv"
OUTPUT_DIR    = BASE_DIR / "outputs/DataAnalysis"
SAMPLING_RATE = 200          # Hz  — acquisition frequency of the pressure sensor
THRESHOLD_PCT  = 0.01         # 1 % of per-signal FFT peak; signals are considered
                              # "near zero" below this fraction → tune as needed
NYQ_RATE_MULTIPLIER = 2.5     # downsampling rate = NYQ_RATE_MULTIPLIER × Nyquist_Rate
N_RANDOM_PLOTS = 30           # number of random signals shown in the downsample comparison plot
PLOT_COLS      = 6            # subplot columns in the comparison grid
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH, index_col=0)
n_signals, n_samples = df.shape
print(f"Loaded : {n_signals} signals × {n_samples} samples  (Fs = {SAMPLING_RATE} Hz)")
print(f"         frequency resolution = {SAMPLING_RATE / n_samples:.5f} Hz/bin")

# ── FFT ───────────────────────────────────────────────────────────────────────
freqs  = np.fft.rfftfreq(n_samples, d=1.0 / SAMPLING_RATE)   # one-sided freq axis [Hz]
n_freq = len(freqs)

magnitudes = np.empty((n_signals, n_freq), dtype=np.float64)
bandwidths = np.empty(n_signals,           dtype=np.float64)

for i, (_part_id, row) in enumerate(df.iterrows()):
    signal  = row.to_numpy(dtype=np.float64)
    fft_mag = np.abs(np.fft.rfft(signal))
    magnitudes[i] = fft_mag

    peak = fft_mag.max()
    if peak > 0.0:
        above  = np.where(fft_mag >= THRESHOLD_PCT * peak)[0]
        bw_idx = int(above[-1]) if len(above) else 0
    else:
        bw_idx = 0
    bandwidths[i] = freqs[bw_idx]

Bandwidth_Max = float(bandwidths.max())
Nyquist_Rate  = 2.0 * Bandwidth_Max

# ── Report ────────────────────────────────────────────────────────────────────
print()
print(f"Results  (threshold = {THRESHOLD_PCT * 100:.1f} % of per-signal FFT peak)")
print(f"  Bandwidth_Max  = {Bandwidth_Max:.4f} Hz")
print(f"  Nyquist_Rate   = {Nyquist_Rate:.4f} Hz")
print()

# per-signal bandwidth statistics
print(f"  Per-signal bandwidth  min  = {bandwidths.min():.4f} Hz")
print(f"                        mean = {bandwidths.mean():.4f} Hz")
print(f"                        max  = {bandwidths.max():.4f} Hz")

# ── Plot ──────────────────────────────────────────────────────────────────────
cmap = plt.colormaps["viridis"]
norm = mcolors.Normalize(vmin=0, vmax=n_signals - 1)

fig, ax = plt.subplots(figsize=(14, 6))

for i in range(n_signals):
    ax.plot(freqs, magnitudes[i],
            color=cmap(norm(i)), alpha=0.18, linewidth=0.5)

# Bandwidth_Max / Nyquist marker
ax.axvline(
    Bandwidth_Max,
    color="#D55E00", linewidth=2.0, linestyle="--",
    label=(f"Bandwidth_Max  = {Bandwidth_Max:.3f} Hz\n"
           f"Nyquist_Rate   = {Nyquist_Rate:.3f} Hz"),
)

ax.set_xlabel("Frequency [Hz]", fontsize=12)
ax.set_ylabel("FFT Magnitude", fontsize=12)
ax.set_title(
    f"FFT spectra — all {n_signals} pressure signals\n"
    f"(threshold = {THRESHOLD_PCT * 100:.1f} % of per-signal FFT peak,  "
    f"Fs = {SAMPLING_RATE} Hz)",
    fontsize=13,
)
ax.legend(fontsize=11, loc="upper right")
ax.set_xlim(left=0)

sm   = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.01)
cbar.set_label("Signal index", fontsize=10)

fig.tight_layout()
out_path = OUTPUT_DIR / "pressure_fft_all.png"
fig.savefig(out_path, dpi=150)
plt.close(fig)
print(f"\nPlot saved → {out_path}")

# ── Downsample ────────────────────────────────────────────────────────────────
# New sampling rate = NYQ_RATE_MULTIPLIER × Nyquist_Rate
DOWNSAMPLE_FS = NYQ_RATE_MULTIPLIER * Nyquist_Rate
n_ds          = round(n_samples * DOWNSAMPLE_FS / SAMPLING_RATE)

print(f"\nDownsampling to Fs_new = {NYQ_RATE_MULTIPLIER} × Nyquist_Rate = {DOWNSAMPLE_FS:.4f} Hz")
print(f"  Original samples : {n_samples}")
print(f"  Resampled samples: {n_ds}")
print(f"  Downsampling ratio: {SAMPLING_RATE / DOWNSAMPLE_FS:.2f}×")

# ── Resample all signals and save CSV ─────────────────────────────────────────
ds_cols     = [f"t_{k:05d}" for k in range(n_ds)]
ds_matrix   = np.empty((n_signals, n_ds), dtype=np.float64)
for i, (_part_id, row) in enumerate(df.iterrows()):
    ds_matrix[i] = resample(row.to_numpy(dtype=np.float64), n_ds)

df_ds        = pd.DataFrame(ds_matrix, index=df.index, columns=ds_cols)
ds_csv_path  = CSV_PATH.parent / "pressure_curve_padded_downsampled.csv"
df_ds.to_csv(ds_csv_path)
print(f"  Downsampled CSV saved → {ds_csv_path}")

# pick random signals (reproducible with a fixed seed)
rng        = np.random.default_rng(seed=42)
rand_idx   = rng.choice(n_signals, size=min(N_RANDOM_PLOTS, n_signals), replace=False)
rand_idx   = np.sort(rand_idx)
part_ids   = df.index.to_numpy()

# ── Comparison plot ───────────────────────────────────────────────────────────
C_BLUE      = "#0072B2"   # original signal
C_VERMILION = "#D55E00"   # downsampled signal

n_plots  = len(rand_idx)
plot_rows = int(np.ceil(n_plots / PLOT_COLS))

t_orig = np.arange(n_samples) / SAMPLING_RATE          # original time axis [s]
t_ds   = np.arange(n_ds)       / DOWNSAMPLE_FS          # downsampled time axis [s]

fig2, axes = plt.subplots(
    plot_rows, PLOT_COLS,
    figsize=(PLOT_COLS * 3.5, plot_rows * 2.4),
    sharex=False, sharey=False,
)
axes_flat = axes.flatten()

# ── Peak & integral comparison ────────────────────────────────────────────────
stats_rows = []

for ax_i, sig_i in enumerate(rand_idx):
    ax = axes_flat[ax_i]
    signal_orig = df.iloc[sig_i].to_numpy(dtype=np.float64)
    signal_ds   = ds_matrix[sig_i]                          # already resampled

    peak_orig = float(signal_orig.max())
    peak_ds   = float(signal_ds.max())
    intg_orig = float(np.trapezoid(signal_orig, t_orig))
    intg_ds   = float(np.trapezoid(signal_ds,   t_ds))

    stats_rows.append({
        "Part":          part_ids[sig_i],
        "Peak_orig":     peak_orig,
        "Peak_ds":       peak_ds,
        "Peak_diff_%":   (peak_ds   - peak_orig) / peak_orig * 100 if peak_orig != 0 else float("nan"),
        "Intg_orig":     intg_orig,
        "Intg_ds":       intg_ds,
        "Intg_diff_%":   (intg_ds   - intg_orig) / intg_orig * 100 if intg_orig != 0 else float("nan"),
    })

    ax.plot(t_orig, signal_orig,
            color=C_BLUE,      linewidth=0.8, alpha=0.85, label="Original")
    ax.plot(t_ds,   signal_ds,
            color=C_VERMILION, linewidth=1.2, alpha=0.90, label="Downsampled",
            linestyle="--")
    ax.set_title(f"Part {part_ids[sig_i]}", fontsize=8)
    ax.tick_params(axis="both", labelsize=7)
    ax.set_xlabel("Time [s]", fontsize=7)
    ax.set_ylabel("Pressure", fontsize=7)

# ── Print comparison table ────────────────────────────────────────────────────
stats_df = pd.DataFrame(stats_rows)
col_w = [10, 14, 14, 12, 16, 16, 12]
header = (f"{'Part':>{col_w[0]}}  {'Peak_orig':>{col_w[1]}}  {'Peak_ds':>{col_w[2]}}"
          f"  {'Peak_diff_%':>{col_w[3]}}  {'Intg_orig':>{col_w[4]}}  {'Intg_ds':>{col_w[5]}}"
          f"  {'Intg_diff_%':>{col_w[6]}}")
sep    = "-" * len(header)
print(f"\nPeak & Integral comparison  (original Fs={SAMPLING_RATE} Hz  vs  "
      f"downsampled Fs={DOWNSAMPLE_FS:.3f} Hz)")
print(sep)
print(header)
print(sep)
for r in stats_rows:
    print(f"{str(r['Part']):>{col_w[0]}}  "
          f"{r['Peak_orig']:>{col_w[1]}.4f}  "
          f"{r['Peak_ds']:>{col_w[2]}.4f}  "
          f"{r['Peak_diff_%']:>{col_w[3]}.2f}  "
          f"{r['Intg_orig']:>{col_w[4]}.4f}  "
          f"{r['Intg_ds']:>{col_w[5]}.4f}  "
          f"{r['Intg_diff_%']:>{col_w[6]}.2f}")
print(sep)
print(f"{'MEAN':>{col_w[0]}}  "
      f"{stats_df['Peak_orig'].mean():>{col_w[1]}.4f}  "
      f"{stats_df['Peak_ds'].mean():>{col_w[2]}.4f}  "
      f"{stats_df['Peak_diff_%'].mean():>{col_w[3]}.2f}  "
      f"{stats_df['Intg_orig'].mean():>{col_w[4]}.4f}  "
      f"{stats_df['Intg_ds'].mean():>{col_w[5]}.4f}  "
      f"{stats_df['Intg_diff_%'].mean():>{col_w[6]}.2f}")
print(sep)

# hide unused axes
for ax_i in range(n_plots, len(axes_flat)):
    axes_flat[ax_i].set_visible(False)

# single shared legend at the top
handles, labels = axes_flat[0].get_legend_handles_labels()
fig2.legend(
    handles, labels,
    loc="upper center", ncol=2, fontsize=10,
    title=(
        f"Original Fs = {SAMPLING_RATE} Hz   |   "
        f"Downsampled Fs = {DOWNSAMPLE_FS:.3f} Hz  ({NYQ_RATE_MULTIPLIER} × Nyquist_Rate)"
    ),
    title_fontsize=9,
)
fig2.suptitle(
    f"Pressure signal comparison — {n_plots} random shots\n"
    f"Bandwidth_Max = {Bandwidth_Max:.3f} Hz   Nyquist_Rate = {Nyquist_Rate:.3f} Hz   "
    f"Fs_new = {DOWNSAMPLE_FS:.3f} Hz",
    fontsize=12, y=1.01,
)
fig2.tight_layout(rect=[0, 0, 1, 0.97])

out_path2 = OUTPUT_DIR / "pressure_downsample_comparison.png"
fig2.savefig(out_path2, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"Plot saved → {out_path2}")
