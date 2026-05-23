"""
M1_PPPT_to_W_FullData.py
=========================
Dual-input neural network for predicting injection-moulded part weight (Full dataset).

Architecture
------------
  p(t) branch : Conv1D stack → GlobalAvgPool → Linear → n_f features  (f)
  pp   branch : MLP  (process parameters)    → n_f features
  merge       : concat(f, pp_out) → MLP → 1  (predicted weight)

Training
--------
  • 5-fold cross-validation on the full cleaned dataset
  • AdamW optimiser + MAE loss + early stopping
  • MinMax scaling fitted on each fold's training split only

Outputs
-------
  outputs/M1/FullDataset/plots/          scatter & metrics plots
  outputs/M1/FullDataset/run_best/        best model of this run (always written)
  outputs/M1/FullDataset/best_overall/    overall best model across all runs (updated when MAE improves)
  outputs/M1/FullDataset/pressure_features_f.csv   encoder features for every part
"""

import copy, json, math, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent
CFG_PATH  = BASE_DIR / "config" / "M1_config.json"
cfg       = json.loads(CFG_PATH.read_text())

data_cfg  = cfg["data"]
prep_cfg  = cfg["preprocessing"]
model_cfg = cfg["model"]
train_cfg = cfg["training"]

SEED   = prep_cfg["random_seed"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)
print(f"Device : {DEVICE}")
print(f"Config : {CFG_PATH}")

# ── Output directories ────────────────────────────────────────────────────────
OUT_DIR          = BASE_DIR / cfg["output"]["output_dir"]
PLOTS_DIR        = BASE_DIR / cfg["output"]["plots_dir"]
RUN_BEST_DIR     = BASE_DIR / cfg["output"]["run_best_dir"]
OVERALL_BEST_DIR = BASE_DIR / cfg["output"]["best_overall_dir"]
FEAT_CSV         = BASE_DIR / cfg["output"]["features_csv"]
for d in [OUT_DIR, PLOTS_DIR, RUN_BEST_DIR, OVERALL_BEST_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Data loading ──────────────────────────────────────────────────────────────
df_scalar   = pd.read_csv(BASE_DIR / data_cfg["scalar_csv"],
                           index_col=data_cfg["id_col"])
df_pressure = pd.read_csv(BASE_DIR / data_cfg["pressure_csv"],
                           index_col=data_cfg["id_col"])

df_scalar = df_scalar.drop(columns=data_cfg["drop_cols"], errors="ignore")
df        = df_scalar.join(df_pressure, how="inner")
print(f"\nJoined : {df.shape[0]} parts × {df.shape[1]} columns")

# ── Encode categorical feature ────────────────────────────────────────────────
mat_col   = data_cfg["material_col"]
materials = sorted(df[mat_col].dropna().unique())
mat_map   = {m: i for i, m in enumerate(materials)}
print(f"Material encoding : {mat_map}")
df[mat_col] = df[mat_col].map(mat_map)

# ── Column groups ─────────────────────────────────────────────────────────────
target_col    = data_cfg["target_col"]
pressure_cols = list(df_pressure.columns)
pp_cols       = [c for c in df_scalar.columns if c != target_col]

# ── Median imputation (pp columns only) ───────────────────────────────────────
for col in pp_cols:
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())

# ── Outlier removal – IQR on pp + target ─────────────────────────────────────
iqr_k = prep_cfg["iqr_multiplier"]
mask  = pd.Series(True, index=df.index)
for col in pp_cols + [target_col]:
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr    = q3 - q1
    if iqr > 0:
        mask &= (df[col] >= q1 - iqr_k * iqr) & (df[col] <= q3 + iqr_k * iqr)
n_removed = int((~mask).sum())
df = df[mask].copy()
print(f"Outliers removed  : {n_removed}  →  {len(df)} parts remaining")

# ── Extract arrays ────────────────────────────────────────────────────────────
part_ids     = df.index.to_numpy()
X_pp         = df[pp_cols].to_numpy(dtype=np.float32)
X_pt         = df[pressure_cols].to_numpy(dtype=np.float32)
y            = df[target_col].to_numpy(dtype=np.float32)
strat_labels = df[mat_col].to_numpy()          # encoded material (0=ABS, 1=PP)
n_pp     = X_pp.shape[1]
T        = X_pt.shape[1]
print(f"pp features : {n_pp}   |   time steps : {T}   |   samples : {len(y)}")
print(f"Target      : mean={y.mean():.3f}  std={y.std():.3f}  "
      f"min={y.min():.3f}  max={y.max():.3f}")

# ── Model definition ──────────────────────────────────────────────────────────

def _he_zero_bias(m: nn.Module) -> None:
    """Kaiming-normal weight init + zero bias."""
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(m.bias)


class PressureEncoder(nn.Module):
    """
    Conv1D stack → GlobalAvgPool1d → Linear → ReLU → n_f features (f).
    Input shape : (B, 1, T)
    Output shape: (B, n_f)
    """

    def __init__(self, channels: list, kernels: list, n_f: int):
        super().__init__()
        conv_layers = []
        for i in range(len(channels) - 1):
            conv_layers += [
                nn.Conv1d(channels[i], channels[i + 1],
                          kernel_size=kernels[i],
                          padding=kernels[i] // 2),   # "same" padding
                nn.ReLU(),
            ]
        self.convs = nn.Sequential(*conv_layers)
        self.pool  = nn.AdaptiveAvgPool1d(1)           # global average pooling
        self.fc    = nn.Linear(channels[-1], n_f)
        self.act   = nn.ReLU()
        self._init_weights()

    def _init_weights(self):
        for m in self.convs:
            if isinstance(m, nn.Conv1d):
                _he_zero_bias(m)
        _he_zero_bias(self.fc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convs(x)              # (B, C_last, T)
        x = self.pool(x).squeeze(-1)   # (B, C_last)
        return self.act(self.fc(x))    # (B, n_f)


class PPMLP(nn.Module):
    """
    Process-parameter MLP.
    All layers have ReLU; dropout on all except the output layer.
    Initialisation: He/Kaiming + zero bias.
    """

    def __init__(self, input_dim: int, hidden_dims: list, dropout: float):
        super().__init__()
        layers = []
        dims   = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            if i < len(dims) - 2:           # no dropout on last (output) layer
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                _he_zero_bias(m)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MergeHead(nn.Module):
    """
    Merge MLP: concat(f, pp_out) → hidden layers (ReLU+Dropout) → Linear(→1).
    Final output layer: no activation, PyTorch default init (bias enabled).
    """

    def __init__(self, input_dim: int, hidden_dims: list, dropout: float):
        super().__init__()
        layers = []
        dims   = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], 1))   # output — no activation
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        linears = [m for m in self.net if isinstance(m, nn.Linear)]
        for m in linears[:-1]:              # all hidden linears: He + zero bias
            _he_zero_bias(m)
        # final linear: keep PyTorch default (Kaiming uniform, non-zero bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)      # (B,)


class M1Model(nn.Module):
    """
    Full dual-input model.
      forward(x_pt, x_pp) → predicted weight (B,)
      get_f(x_pt)          → encoder features (B, n_f)   [no_grad]
    """

    def __init__(self, n_pp: int, mcfg: dict):
        super().__init__()
        self.encoder = PressureEncoder(
            channels=mcfg["conv_channels"],
            kernels=mcfg["conv_kernels"],
            n_f=mcfg["n_f_features"],
        )
        self.pp_mlp  = PPMLP(
            input_dim=n_pp,
            hidden_dims=mcfg["pp_hidden"],
            dropout=mcfg["dropout"],
        )
        merge_in     = mcfg["n_f_features"] + mcfg["pp_hidden"][-1]
        self.merge   = MergeHead(
            input_dim=merge_in,
            hidden_dims=mcfg["merge_hidden"],
            dropout=mcfg["dropout"],
        )

    def forward(self, x_pt: torch.Tensor, x_pp: torch.Tensor) -> torch.Tensor:
        f   = self.encoder(x_pt)
        pp  = self.pp_mlp(x_pp)
        return self.merge(torch.cat([f, pp], dim=1))

    @torch.no_grad()
    def get_f(self, x_pt: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.encoder(x_pt)


def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── Scaling utilities ─────────────────────────────────────────────────────────

def fit_scalers(X_pp_tr: np.ndarray, X_pt_tr: np.ndarray):
    """Fit MinMax scalers on training data only."""
    pp_sc = MinMaxScaler()
    pp_sc.fit(X_pp_tr)
    pt_min = float(X_pt_tr.min())
    pt_max = float(X_pt_tr.max())
    return pp_sc, pt_min, pt_max


def apply_scalers(X_pp: np.ndarray, X_pt: np.ndarray,
                  pp_sc: MinMaxScaler, pt_min: float, pt_max: float):
    """Apply pre-fitted scalers."""
    X_pp_s = pp_sc.transform(X_pp).astype(np.float32)
    X_pt_s = ((X_pt - pt_min) / (pt_max - pt_min + 1e-8)).astype(np.float32)
    return X_pp_s, X_pt_s


def to_tensors(X_pp: np.ndarray, X_pt: np.ndarray,
               y_arr: np.ndarray, device: torch.device):
    """Convert numpy arrays to PyTorch tensors on the target device."""
    t_pp = torch.tensor(X_pp,              dtype=torch.float32, device=device)
    t_pt = torch.tensor(X_pt[:, None, :],  dtype=torch.float32, device=device)
    t_y  = torch.tensor(y_arr,             dtype=torch.float32, device=device)
    return t_pp, t_pt, t_y


# ── Training & evaluation ─────────────────────────────────────────────────────

def train_fold(model: M1Model,
               Xpp_tr, Xpt_tr, y_tr,
               Xpp_val, Xpt_val, y_val,
               tcfg: dict, device: torch.device):
    """Train for one fold with early stopping; returns model & best val MAE."""
    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=tcfg["lr"],
                                  weight_decay=tcfg["weight_decay"])
    if tcfg["scheduler_factor"] < 1:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           patience=tcfg["scheduler_patience"],
                                                           factor=tcfg["scheduler_factor"])

    pp_tr,  pt_tr,  ty_tr  = to_tensors(Xpp_tr,  Xpt_tr,  y_tr,  device)
    pp_val, pt_val, ty_val = to_tensors(Xpp_val, Xpt_val, y_val, device)

    loader = DataLoader(TensorDataset(pt_tr, pp_tr, ty_tr),
                        batch_size=tcfg["batch_size"], shuffle=True,
                        drop_last=False)

    best_val_mae = float("inf")
    best_state   = copy.deepcopy(model.state_dict())
    wait         = 0

    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        for b_pt, b_pp, b_y in loader:
            optimizer.zero_grad()
            loss = criterion(model(b_pt, b_pp), b_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(b_y)
        epoch_loss /= len(y_tr)

        model.eval()
        with torch.no_grad():
            val_mae = criterion(model(pt_val, pp_val), ty_val).item()
        if tcfg["scheduler_factor"] < 1:
            scheduler.step(val_mae)

        if epoch % tcfg["print_every"] == 0:
            print(f"    ep {epoch:4d}  train_mae={epoch_loss:.4f}  val_mae={val_mae:.4f}")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state   = copy.deepcopy(model.state_dict())
            wait         = 0
        else:
            wait += 1
            if wait >= tcfg["patience"]:
                print(f"    Early stop  ep={epoch}  best_val_mae={best_val_mae:.4f}")
                break

    model.load_state_dict(best_state)
    return model, best_val_mae


def evaluate_model(model: M1Model,
                   Xpp: np.ndarray, Xpt: np.ndarray,
                   y_arr: np.ndarray, device: torch.device) -> dict:
    """Return MAE / MSE / RMSE / R² and raw predictions."""
    pp_t, pt_t, _ = to_tensors(Xpp, Xpt, y_arr, device)
    model.eval()
    with torch.no_grad():
        pred = model(pt_t, pp_t).cpu().numpy()
    mae  = float(mean_absolute_error(y_arr, pred))
    mse  = float(mean_squared_error(y_arr, pred))
    rmse = math.sqrt(mse)
    r2   = float(r2_score(y_arr, pred))
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2, "pred": pred}


# ── K-fold cross-validation ───────────────────────────────────────────────────
K  = train_cfg["k_folds"]
kf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)

# Print model size on first instantiation
_dummy = M1Model(n_pp, model_cfg)
print(f"\nModel parameters : {_count_params(_dummy):,}")
del _dummy

fold_results  = []
best_fold_mae = float("inf")    # MAE lower = better
best_fold_idx = -1
best_model_cv = None
best_scalers  = None
best_y_test   = None
best_preds_cv = None

for fold, (tv_idx, test_idx) in enumerate(kf.split(X_pp, strat_labels)):
    print(f"\n══ Fold {fold + 1}/{K} ══════════════════════════════════════════")

    # Stratified train / val split inside the non-test portion
    sss = StratifiedShuffleSplit(n_splits=1,
                                 test_size=train_cfg["val_fraction"],
                                 random_state=SEED + fold)
    _tr_rel, _val_rel = next(sss.split(tv_idx, strat_labels[tv_idx]))
    tr_idx  = tv_idx[_tr_rel]
    val_idx = tv_idx[_val_rel]
    print(f"  Train={len(tr_idx)}  Val={len(val_idx)}  Test={len(test_idx)}")

    # Fit scalers on training split only
    pp_sc, pt_min, pt_max = fit_scalers(X_pp[tr_idx], X_pt[tr_idx])
    Xpp_tr,  Xpt_tr  = apply_scalers(X_pp[tr_idx],   X_pt[tr_idx],   pp_sc, pt_min, pt_max)
    Xpp_val, Xpt_val = apply_scalers(X_pp[val_idx],  X_pt[val_idx],  pp_sc, pt_min, pt_max)
    Xpp_te,  Xpt_te  = apply_scalers(X_pp[test_idx], X_pt[test_idx], pp_sc, pt_min, pt_max)

    # Build and train
    torch.manual_seed(SEED + fold)
    model = M1Model(n_pp, model_cfg).to(DEVICE)
    model, _ = train_fold(model,
                          Xpp_tr, Xpt_tr, y[tr_idx],
                          Xpp_val, Xpt_val, y[val_idx],
                          train_cfg, DEVICE)

    # Evaluate on held-out test fold
    res = evaluate_model(model, Xpp_te, Xpt_te, y[test_idx], DEVICE)
    fold_results.append({k: v for k, v in res.items() if k != "pred"})
    print(f"  Test  MAE={res['MAE']:.4f}  RMSE={res['RMSE']:.4f}  "
          f"R²={res['R2']:.4f}  MSE={res['MSE']:.4f}")

    if res["MAE"] < best_fold_mae:
        best_fold_mae = res["MAE"]
        best_fold_idx = fold
        best_model_cv = copy.deepcopy(model)
        best_scalers  = (pp_sc, pt_min, pt_max)
        best_y_test   = y[test_idx]
        best_preds_cv = res["pred"]

# ── Summary ───────────────────────────────────────────────────────────────────
metrics_df = pd.DataFrame(fold_results)

print("\n" + "═" * 60)
print(f"5-fold CV summary  (best fold: {best_fold_idx + 1}  "
      f"MAE={best_fold_mae:.4f})")
print("─" * 60)
for m in ["MAE", "RMSE", "R2", "MSE"]:
    print(f"  {m:6s}  {metrics_df[m].mean():.4f} ± {metrics_df[m].std():.4f}")
print("═" * 60)

metrics_df.index = [f"Fold_{i+1}" for i in range(K)]
metrics_df.to_csv(OUT_DIR / "fold_metrics.csv")

# ── Plots ─────────────────────────────────────────────────────────────────────
C_BLUE, C_RED = "#0072B2", "#D55E00"

# Scatter: real vs predicted (best fold)
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(best_y_test, best_preds_cv, alpha=0.7,
           edgecolors="k", linewidths=0.3, color=C_BLUE, zorder=3)
lo = min(best_y_test.min(), best_preds_cv.min())
hi = max(best_y_test.max(), best_preds_cv.max())
ax.plot([lo, hi], [lo, hi], color=C_RED, linestyle="--",
        linewidth=1.5, label="Ideal (1:1)")
ax.set_xlabel("Real weight [g]", fontsize=12)
ax.set_ylabel("Predicted weight [g]", fontsize=12)
ax.set_title(
    f"Best fold (fold {best_fold_idx + 1}) — Real vs Predicted\n"
    f"MAE={best_fold_mae:.4f} g   "
    f"R²={metrics_df['R2'].iloc[best_fold_idx]:.4f}",
    fontsize=12,
)
ax.legend(fontsize=10)
fig.tight_layout()
fig.savefig(PLOTS_DIR / "scatter_best_fold.png", dpi=150)
plt.close(fig)

# Metrics bar chart (per fold + mean)
fig, axes = plt.subplots(1, 4, figsize=(14, 4))
fold_labels = [f"F{i + 1}" for i in range(K)] + ["Mean"]
bar_colors  = [C_BLUE] * K + [C_RED]
for ax, metric in zip(axes, ["MAE", "RMSE", "R2", "MSE"]):
    direction = "↑ higher = better" if metric == "R2" else "↓ lower = better"
    vals = list(metrics_df[metric]) + [metrics_df[metric].mean()]
    bars = ax.bar(fold_labels, vals, color=bar_colors,
                  edgecolor="k", linewidth=0.4)
    ax.set_title(f"{metric}\n{direction}", fontsize=10)
    ax.set_ylabel(metric)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{v:.3f}", ha="center", va="bottom", fontsize=8)
fig.suptitle(f"K-fold CV metrics (K={K})", fontsize=13)
fig.tight_layout()
fig.savefig(PLOTS_DIR / "metrics_folds.png", dpi=150)
plt.close(fig)

print(f"\nPlots saved → {PLOTS_DIR}")

# ── Model saving ─────────────────────────────────────────────────────────────
# Metrics of the best fold's model (the model that is actually saved)
best_fold_metrics = metrics_df.iloc[best_fold_idx]
run_info = {
    "MAE":       float(best_fold_metrics["MAE"]),
    "RMSE":      float(best_fold_metrics["RMSE"]),
    "R2":        float(best_fold_metrics["R2"]),
    "MSE":       float(best_fold_metrics["MSE"]),
    "best_fold": best_fold_idx + 1,
    "n_pp":      n_pp,
    "T":         T,
    "model_cfg": model_cfg,
}

# Always save the best model of this run
torch.save(best_model_cv.state_dict(), RUN_BEST_DIR / "best_model.pt")
(RUN_BEST_DIR / "best_metrics.json").write_text(json.dumps(run_info, indent=2))
print(f"\nRun best model saved  (MAE={best_fold_mae:.4f})  → {RUN_BEST_DIR}")

# Update overall best only when this run's best fold MAE beats the previous record
overall_path = OVERALL_BEST_DIR / "best_metrics.json"
prev_mae     = float("inf")
if overall_path.exists():
    prev_mae = json.loads(overall_path.read_text()).get("MAE", float("inf"))

if best_fold_mae < prev_mae:
    torch.save(best_model_cv.state_dict(), OVERALL_BEST_DIR / "best_model.pt")
    overall_path.write_text(json.dumps(run_info, indent=2))
    print(f"Overall best updated  ({prev_mae:.4f} → {best_fold_mae:.4f} MAE)  "
          f"→ {OVERALL_BEST_DIR}")
else:
    print(f"Overall best unchanged  "
          f"(saved MAE={prev_mae:.4f},  this run best MAE={best_fold_mae:.4f})")

# ── Extract pressure features f (full cleaned dataset, best model encoder) ───
pp_sc_all, pt_min_all, pt_max_all = best_scalers
_, X_pt_all_sc = apply_scalers(X_pp, X_pt, pp_sc_all, pt_min_all, pt_max_all)

pt_all_t   = torch.tensor(X_pt_all_sc[:, None, :],
                           dtype=torch.float32, device=DEVICE)
f_features = best_model_cv.get_f(pt_all_t).cpu().numpy()

f_cols = [f"f_{i}" for i in range(f_features.shape[1])]
df_f   = pd.DataFrame(f_features, columns=f_cols, index=part_ids)
df_f.index.name = data_cfg["id_col"]
df_f.to_csv(FEAT_CSV)
print(f"Pressure features f saved → {FEAT_CSV}  "
      f"({df_f.shape[0]} parts × {df_f.shape[1]} features)")
