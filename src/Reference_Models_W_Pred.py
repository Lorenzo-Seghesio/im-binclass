"""
Reference_Models_W_Pred.py
==========================
Four reference models that all predict injection-moulded part weight (W):

  1. Encoder  — Conv1D stack → GlobalAvgPool → MLP head    (input: pressure curves)
  2. MLP      — fully connected MLP                         (input: process parameters)
  3. LightGBM — gradient-boosted trees                      (input: process parameters)
  4. XGBoost  — gradient-boosted trees                      (input: process parameters)

All four models are architecture-searched with Optuna (TPE sampler).
NN models use HyperbandPruner; GBDT models use MedianPruner.
After HPO a full K-fold CV is run for each model to produce final metrics.

Usage
-----
  python src/Reference_Models_W_Pred.py --material PP
  python src/Reference_Models_W_Pred.py --material ABS
  python src/Reference_Models_W_Pred.py --material ALL

Config files
------------
  config/RefModels_PP_config.json
  config/RefModels_ABS_config.json
  config/RefModels_AllData_config.json

Outputs
-------
  outputs/RefModels/Encoder/[material]/run_best/   + best_overall/
  outputs/RefModels/MLP/[material]/run_best/        + best_overall/
  outputs/RefModels/LightGBM/[material]/run_best/   + best_overall/
  outputs/RefModels/XGBoost/[material]/run_best/    + best_overall/

  Each folder: best_model.pt|.joblib, best_metrics.json, fold_metrics.csv,
               scatter_best_fold.png, metrics_folds.png, residuals_hist_best_fold.png,
               hpo_best_config.json, hpo_trials.csv
"""

import argparse
from datetime import datetime
import copy, json, math, shutil, warnings
import joblib
import optuna
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Publication plot style ────────────────────────────────────────────────────
_PLT_C: dict = {
    "scatter":    "#2166AC",   # steel blue   — scatter dots / predictions
    "ideal_line": "#B2182B",   # muted red    — 1:1 diagonal reference
    "residual":   "#1B7837",   # forest green — residual scatter & histogram
    "mean_line":  "#1B7837",   # forest green — mean residual vline
    "zero_line":  "#555555",   # dark grey    — y = 0 reference
    "fold":       "#4393C3",   # light blue   — per-fold CV bars
    "fold_mean":  "#D6604D",   # coral        — mean CV bar
    "shap":       "#D4691C",   # burnt orange — SHAP bars
    "shap_bar":   "#2166AC",   # steel blue   — SHAP summary bar plot
    "ig":         "#7D54A4",   # purple       — IG / Expected Gradients bars
    "indirect":   "#D4691C",   # orange       — Fusion indirect path (via M2)
    "direct":     "#2166AC",   # blue         — Fusion direct path (via PPMLP)
    "total":      "#1B7837",   # green        — total combined attribution
    "f_feat":     "#E69F00",   # gold         — M1-merge f encoder features
    "pp_feat":    "#2166AC",   # blue         — M1-merge pp_direct features
    "pressure":   "#2166AC",   # blue         — GradCAM pressure curve
    "gradcam":    "#B2182B",   # red          — GradCAM activation overlay
    "shap_cmap":    "cool",      # SHAP cmap — diverging blue-to-red (low=blue, high=red)
}


def _strip_prefix(name: str) -> str:
    for pfx in ("TCE_", "TCN_"):
        if name.startswith(pfx):
            return name[len(pfx):] + pfx[:-1]   # e.g. TCN_Foo → FooTCN
    for pfx in ("DXP_", "QUA_", "SCA_", "MSS_", "IHR_", "SPE_"):
        if name.startswith(pfx):
            return name[len(pfx):]
    return name


def _strip_prefixes(names) -> list:
    return [_strip_prefix(n) for n in names]


plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10,
    "axes.labelsize": 11, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.titlepad": 10,
    "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": "#E0E0E0", "grid.linestyle": "--",
    "grid.linewidth": 0.5, "axes.axisbelow": True,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3.5, "ytick.major.size": 3.5,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.facecolor": "white", "legend.framealpha": 0.92,
    "legend.edgecolor": "#CCCCCC", "legend.frameon": True,
    "lines.linewidth": 1.8, "patch.linewidth": 0.4,
})
# ─────────────────────────────────────────────────────────────────────────────

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
BASE_OUT = BASE_DIR / "outputs" / "RefModels"
_RUN_TS  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

_CFG_MAP = {
    "PP":  "RefModels_PP_config.json",
    "ABS": "RefModels_ABS_config.json",
    "ALL": "RefModels_AllData_config.json",
}
_MAT_DIR = {"PP": "PP", "ABS": "ABS", "ALL": "FullDataset"}


# ── NN model definitions ──────────────────────────────────────────────────────

def _he_zero_bias(m: nn.Module) -> None:
    """Kaiming-normal weight init + zero bias."""
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(m.bias)


def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _initial_split(n: int, test_frac: float, seed: int, strat_labels=None):
    """One-time deterministic train/test split performed before HPO or CV.
    Returns (dev_idx, test_idx) as np.intp arrays.
    """
    if strat_labels is not None:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
        dev_rel, test_rel = next(sss.split(np.zeros(n), strat_labels))
        return dev_rel.astype(np.intp), test_rel.astype(np.intp)
    rng    = np.random.default_rng(seed)
    perm   = rng.permutation(n)
    n_test = max(1, int(n * test_frac))
    return perm[n_test:].astype(np.intp), perm[:n_test].astype(np.intp)


class EncoderModel(nn.Module):
    """
    Conv1D stack → MaxPool (between layers) → AdaptiveAvgPool1d(1) → MLP head → scalar.
    Input : (B, 1, T)  — normalised pressure curve.
    Output: (B,)       — predicted weight.
    """

    def __init__(self, channels: list, kernels: list, pool_kernels: list,
                 head_hidden: list, dropout: float):
        super().__init__()
        n_conv = len(channels) - 1
        conv_layers = []
        for i in range(n_conv):
            conv_layers += [
                nn.Conv1d(channels[i], channels[i + 1],
                          kernel_size=kernels[i], padding=kernels[i] // 2),
                nn.ReLU(),
            ]
            if i < n_conv - 1 and i < len(pool_kernels) and pool_kernels[i] > 1:
                conv_layers.append(nn.MaxPool1d(pool_kernels[i], stride=pool_kernels[i]))
        self.convs = nn.Sequential(*conv_layers)
        self.pool  = nn.AdaptiveAvgPool1d(1)

        head = []
        prev = channels[-1]
        for h in head_hidden:
            head += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        head.append(nn.Linear(prev, 1))
        self.head = nn.Sequential(*head)
        self._init_weights()

    def _init_weights(self):
        for m in self.convs:
            if isinstance(m, nn.Conv1d):
                _he_zero_bias(m)
        linears = [m for m in self.head if isinstance(m, nn.Linear)]
        for m in linears[:-1]:
            _he_zero_bias(m)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convs(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x).squeeze(-1)


class MLPModel(nn.Module):
    """
    MLP: scalar process parameters → weight.
    Hidden layers: ReLU + Dropout + He init.
    Output: Linear (no activation), PyTorch default init.
    """

    def __init__(self, n_in: int, hidden_dims: list, dropout: float):
        super().__init__()
        layers = []
        prev = n_in
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        linears = [m for m in self.net if isinstance(m, nn.Linear)]
        for m in linears[:-1]:
            _he_zero_bias(m)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ── Input preparation helpers ─────────────────────────────────────────────────

def _enc_prepare(x_np: np.ndarray) -> torch.Tensor:
    """(N, T) → (N, 1, T) float32 tensor."""
    return torch.tensor(x_np[:, None, :], dtype=torch.float32)


def _mlp_prepare(x_np: np.ndarray) -> torch.Tensor:
    """(N, n) → (N, n) float32 tensor."""
    return torch.tensor(x_np, dtype=torch.float32)


# ── Shared utilities ──────────────────────────────────────────────────────────

def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae  = float(mean_absolute_error(y_true, y_pred))
    mse  = float(mean_squared_error(y_true, y_pred))
    rmse = math.sqrt(mse)
    r2   = float(r2_score(y_true, y_pred))
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2, "pred": y_pred}


def load_data(data_cfg: dict, prep_cfg: dict):
    """Join scalar + pressure CSVs; clean; return (part_ids, X_pp, X_pt, y, strat_labels)."""
    df_scalar   = pd.read_csv(BASE_DIR / data_cfg["scalar_csv"],
                               index_col=data_cfg["id_col"])
    df_pressure = pd.read_csv(BASE_DIR / data_cfg["pressure_csv"],
                               index_col=data_cfg["id_col"])

    target_col = data_cfg["target_col"]
    mat_col    = data_cfg.get("material_col")

    if mat_col:
        materials = sorted(df_scalar[mat_col].dropna().unique())
        mat_map   = {m: i for i, m in enumerate(materials)}
        print(f"Material encoding : {mat_map}")
        df_scalar[mat_col] = df_scalar[mat_col].map(mat_map)

    df = df_scalar.join(df_pressure, how="inner")
    print(f"Joined : {df.shape[0]} parts × {df.shape[1]} columns")

    pressure_cols = list(df_pressure.columns)
    pp_cols       = [c for c in df_scalar.columns if c != target_col]

    for col in pp_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # pp_cols are pre-cleaned by clean_data.py (X-outliers already removed).
    # Apply IQR filter to target_col only (y-outlier removal on the dev dataset).
    iqr_k  = prep_cfg["iqr_multiplier"]
    col    = target_col
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr    = q3 - q1
    if iqr > 0:
        y_mask = (df[col] >= q1 - iqr_k * iqr) & (df[col] <= q3 + iqr_k * iqr)
    else:
        y_mask = pd.Series(True, index=df.index)
    n_removed = int((~y_mask).sum())
    df = df[y_mask].copy()
    print(f"y-outliers removed: {n_removed}  →  {len(df)} parts remaining")

    part_ids     = df.index.to_numpy()
    X_pp         = df[pp_cols].to_numpy(dtype=np.float32)
    X_pt         = df[pressure_cols].to_numpy(dtype=np.float32)
    y            = df[target_col].to_numpy(dtype=np.float32)
    strat_labels = df[mat_col].to_numpy() if mat_col else None

    print(f"pp features: {X_pp.shape[1]}  |  time steps: {X_pt.shape[1]}  |  samples: {len(y)}")
    print(f"Target: mean={y.mean():.3f}  std={y.std():.3f}  "
          f"min={y.min():.3f}  max={y.max():.3f}")
    return part_ids, X_pp, X_pt, y, strat_labels, pp_cols


def _split_iter(K: int, seed: int, strat_labels, X: np.ndarray) -> list:
    if strat_labels is not None:
        kf = StratifiedKFold(n_splits=K, shuffle=True, random_state=seed)
        return list(kf.split(X, strat_labels))
    kf = KFold(n_splits=K, shuffle=True, random_state=seed)
    return list(kf.split(X))


def _inner_split(tv_idx: np.ndarray, train_cfg: dict, seed: int, fold: int,
                 strat_labels) -> tuple:
    if strat_labels is not None:
        sss = StratifiedShuffleSplit(n_splits=1,
                                     test_size=train_cfg["val_fraction"],
                                     random_state=seed + fold)
        _tr, _val = next(sss.split(tv_idx, strat_labels[tv_idx]))
        return tv_idx[_tr], tv_idx[_val]
    rng   = np.random.default_rng(seed + fold)
    perm  = rng.permutation(len(tv_idx))
    n_val = max(1, int(len(tv_idx) * train_cfg["val_fraction"]))
    return tv_idx[perm[n_val:]], tv_idx[perm[:n_val]]


def print_summary(metrics_df: pd.DataFrame, best_fold_idx: int,
                  best_fold_mae: float, K: int, label: str = ""):
    print("\n" + "═" * 60)
    print(f"[{label}] [Indicative] {K}-fold CV on dev  "
          f"(best fold: {best_fold_idx + 1}  MAE={best_fold_mae:.4f})")
    print("─" * 60)
    for m in ["MAE", "RMSE", "R2", "MSE"]:
        print(f"  {m:6s}  {metrics_df[m].mean():.4f} ± {metrics_df[m].std():.4f}")
    print("═" * 60)


# ── NN training + evaluation ──────────────────────────────────────────────────

def train_fold_nn(model: nn.Module, prepare_x, X_tr, y_tr, X_val, y_val,
                  tcfg: dict, device: torch.device):
    """Generic NN fold training with MAE + L1 + AdamW + scheduler + early stopping."""
    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=tcfg["lr"],
                                  weight_decay=tcfg["weight_decay"])
    if tcfg["scheduler_factor"] < 1:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=tcfg["scheduler_patience"],
            factor=tcfg["scheduler_factor"])

    t_X_tr  = prepare_x(X_tr).to(device)
    t_y_tr  = torch.tensor(y_tr,  dtype=torch.float32, device=device)
    t_X_val = prepare_x(X_val).to(device)
    t_y_val = torch.tensor(y_val, dtype=torch.float32, device=device)

    loader = DataLoader(TensorDataset(t_X_tr, t_y_tr),
                        batch_size=tcfg["batch_size"], shuffle=True, drop_last=False)

    l1_lambda    = float(tcfg.get("l1_lambda", 0.0))
    best_val_mae = float("inf")
    best_state   = copy.deepcopy(model.state_dict())
    wait         = 0

    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        for b_x, b_y in loader:
            optimizer.zero_grad()
            loss = criterion(model(b_x), b_y)
            if l1_lambda > 0.0:
                l1_reg = sum(p.abs().sum() for p in model.parameters())
                loss   = loss + l1_lambda * l1_reg
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(b_y)
        epoch_loss /= len(y_tr)

        model.eval()
        with torch.no_grad():
            val_mae = criterion(model(t_X_val), t_y_val).item()
        if tcfg["scheduler_factor"] < 1:
            scheduler.step(val_mae)

        if epoch % tcfg["print_every"] == 0:
            print(f"    ep {epoch:4d}  train_loss={epoch_loss:.4f}  val_mae={val_mae:.4f}")

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


def evaluate_nn(model: nn.Module, prepare_x, X: np.ndarray,
                y: np.ndarray, device: torch.device) -> dict:
    t_X = prepare_x(X).to(device)
    model.eval()
    with torch.no_grad():
        pred = model(t_X).cpu().numpy()
    return _metrics(y, pred)


def evaluate_gbdt(model, X: np.ndarray, y: np.ndarray) -> dict:
    pred = model.predict(X).astype(np.float32)
    return _metrics(y, pred)


# ── Plots & saves ─────────────────────────────────────────────────────────────

def save_cv_plots(metrics_df: pd.DataFrame, K: int, material: str,
                  out_dir: Path, label: str):
    """Bar chart of per-fold indicative CV metrics (secondary output)."""
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.5))
    fold_labels = [f"F{i + 1}" for i in range(K)] + ["Mean"]
    bar_colors  = [_PLT_C["fold"]] * K + [_PLT_C["fold_mean"]]
    for ax, metric in zip(axes, ["MAE", "RMSE", "R2", "MSE"]):
        vals = list(metrics_df[metric]) + [metrics_df[metric].mean()]
        bars = ax.bar(fold_labels, vals, color=bar_colors, edgecolor="none", width=0.65)
        ax.set_title(metric)
        ax.set_ylabel(metric)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.3f}", ha="center", va="bottom")
    fig.suptitle(f"[{material} | {label}] Indicative {K}-fold CV on dev set", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "cv_metrics_folds.png")
    plt.close(fig)
    print(f"  CV plots saved → {out_dir}")


def save_final_test_plots(y_test: np.ndarray, preds: np.ndarray,
                          final_metrics: dict, out_dir: Path,
                          material: str, label: str):
    """Scatter + residuals from the final model on the held-out test set (primary output)."""
    # Scatter: measured vs predicted
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, preds, color=_PLT_C["scatter"],
               s=45, alpha=0.8, edgecolors="white", linewidths=0.5, zorder=3)
    lo = min(y_test.min(), preds.min())
    hi = max(y_test.max(), preds.max())
    ax.plot([lo, hi], [lo, hi], color=_PLT_C["ideal_line"], linestyle="--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Measured weight [g]")
    ax.set_ylabel("Predicted weight [g]")
    ax.set_title(
        f"[{material} | {label}] Final Model — Held-out Test\n"
        f"MAE={final_metrics['MAE']:.4f} g   R²={final_metrics['R2']:.4f}",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "scatter_final_test.png")
    plt.close(fig)
    # Residuals histogram
    res = preds - y_test
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hist(res, bins=20, color=_PLT_C["residual"], edgecolor="white", alpha=0.85)
    ax.axvline(0, color=_PLT_C["zero_line"], linestyle="--", linewidth=1.5)
    ax.axvline(float(res.mean()), color=_PLT_C["mean_line"], linestyle="-",
               linewidth=1.5, label=f"Mean={res.mean():.4f}")
    ax.set_xlabel("Residual (pred − real) [g]")
    ax.set_ylabel("Count")
    ax.set_title(f"[{material} | {label}] Residuals  std={res.std():.4f} g")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "residuals_hist_final_test.png")
    plt.close(fig)
    print(f"  Final test plots saved → {out_dir}")


def _check_overall_best(final_mae: float, overall_best_dir: Path):
    p = overall_best_dir / "best_metrics.json"
    prev_mae = float("inf")
    if p.exists():
        prev_mae = json.loads(p.read_text()).get("MAE", float("inf"))
    return prev_mae, final_mae < prev_mae


def save_model_nn(model: nn.Module, final_metrics: dict,
                  run_best_dir: Path, overall_best_dir: Path,
                  arch_info: dict, label: str) -> bool:
    """Save final NN model + test metrics; update overall_best if improved."""
    info = {
        "MAE":    final_metrics["MAE"],
        "RMSE":   final_metrics["RMSE"],
        "R2":     final_metrics["R2"],
        "MSE":    final_metrics["MSE"],
        "source": "final_test",
        **arch_info,
    }
    torch.save(model.state_dict(), run_best_dir / "best_model.pt")
    (run_best_dir / "best_metrics.json").write_text(json.dumps(info, indent=2))
    print(f"\n[{label}] Final model saved  (MAE={final_metrics['MAE']:.4f})  → {run_best_dir}")

    prev_mae, update = _check_overall_best(final_metrics["MAE"], overall_best_dir)
    if update:
        torch.save(model.state_dict(), overall_best_dir / "best_model.pt")
        (overall_best_dir / "best_metrics.json").write_text(json.dumps(info, indent=2))
        print(f"[{label}] Overall best updated  "
              f"({prev_mae:.4f} → {final_metrics['MAE']:.4f})  → {overall_best_dir}")
    else:
        print(f"[{label}] Overall best unchanged  "
              f"(saved={prev_mae:.4f}, this={final_metrics['MAE']:.4f})")
    return update


def save_model_gbdt(model, final_metrics: dict,
                    run_best_dir: Path, overall_best_dir: Path,
                    best_params: dict, label: str) -> bool:
    """Save final GBDT model + test metrics; update overall_best if improved."""
    info = {
        "MAE":        final_metrics["MAE"],
        "RMSE":       final_metrics["RMSE"],
        "R2":         final_metrics["R2"],
        "MSE":        final_metrics["MSE"],
        "source":     "final_test",
        "best_params": best_params,
    }
    joblib.dump(model, run_best_dir / "best_model.joblib")
    (run_best_dir / "best_metrics.json").write_text(json.dumps(info, indent=2))
    print(f"\n[{label}] Final model saved  (MAE={final_metrics['MAE']:.4f})  → {run_best_dir}")

    prev_mae, update = _check_overall_best(final_metrics["MAE"], overall_best_dir)
    if update:
        joblib.dump(model, overall_best_dir / "best_model.joblib")
        (overall_best_dir / "best_metrics.json").write_text(json.dumps(info, indent=2))
        print(f"[{label}] Overall best updated  "
              f"({prev_mae:.4f} → {final_metrics['MAE']:.4f})  → {overall_best_dir}")
    else:
        print(f"[{label}] Overall best unchanged  "
              f"(saved={prev_mae:.4f}, this={final_metrics['MAE']:.4f})")
    return update


def _copy_artefacts(run_best_dir: Path, overall_best_dir: Path):
    for fname in [
        "hpo_best_config.json", "hpo_trials.csv", "cv_fold_metrics.csv",
        "xai_shap_bar.png", "xai_shap_beeswarm.png",
        "xai_ig_bar.png", "xai_gradcam.png", "run_info.json",
    ]:
        src = run_best_dir / fname
        if src.exists():
            shutil.copy2(src, overall_best_dir / fname)


# ── XAI helpers ───────────────────────────────────────────────────────────────

def _save_run_info(rb_dir: Path, ob_dir: Path, mat: str, label: str, updated: bool):
    """Write run_info.json to run_best and (if overall best) to best_overall."""
    info = {
        "run_ts":       _RUN_TS,
        "material":     mat,
        "model":        label,
        "run_best_dir": str(rb_dir.relative_to(BASE_DIR)),
    }
    (rb_dir / "run_info.json").write_text(json.dumps(info, indent=2))
    if updated:
        (ob_dir / "run_info.json").write_text(json.dumps(info, indent=2))
    print(f"  [{label}] run_info saved → {rb_dir}")


def _xai_shap_gbdt(model, X_train: np.ndarray, X_test: np.ndarray,
                   pp_cols: list, out_dir: Path, label: str):
    """SHAP TreeExplainer for GBDT models (LightGBM / XGBoost).
    Produces bar (mean |SHAP|) and beeswarm plots on the held-out test set.
    """
    try:
        import shap
    except ImportError:
        print(f"  [XAI-{label}] shap not installed — skipping SHAP")
        return
    try:
        explainer = shap.TreeExplainer(model, X_train)
        shap_vals = explainer.shap_values(X_test)          # (n_test, n_features)

        feat_names = _strip_prefixes(pp_cols)
        with plt.rc_context(plt.rcParamsDefault):
            shap.summary_plot(shap_vals, X_test, feature_names=feat_names,
                              plot_type="bar", show=False, color=_PLT_C["shap_bar"])
            plt.savefig(out_dir / "xai_shap_bar.png", bbox_inches="tight", dpi=300)
            plt.close()

            shap.summary_plot(shap_vals, X_test, feature_names=feat_names,
                              plot_type="dot", show=False)
            plt.savefig(out_dir / "xai_shap_beeswarm.png", bbox_inches="tight", dpi=300)
            plt.close()
        print(f"  [XAI-{label}] SHAP saved → {out_dir}")
    except Exception as exc:
        print(f"  [XAI-{label}] SHAP failed: {exc}")


def _xai_shap_mlp(model: nn.Module, X_bg_sc: np.ndarray, X_test_sc: np.ndarray,
                  pp_cols: list, out_dir: Path, label: str, device: torch.device,
                  n_bg: int = 100):
    """SHAP GradientExplainer for PyTorch MLP.
    Background = random sample of dev data (scaled). Test = scaled test set.
    """
    try:
        import shap
    except ImportError:
        print(f"  [XAI-{label}] shap not installed — skipping SHAP")
        return
    try:
        model.eval()
        rng    = np.random.default_rng(42)
        idx    = rng.choice(len(X_bg_sc), min(n_bg, len(X_bg_sc)), replace=False)
        bg_t   = torch.tensor(X_bg_sc[idx],  dtype=torch.float32, device=device)
        X_te_t = torch.tensor(X_test_sc,     dtype=torch.float32, device=device)

        # GradientExplainer needs 2-D output (n, 1); MLPModel.forward squeezes to 1-D
        class _Wrap2D(nn.Module):
            def __init__(self, m): super().__init__(); self.m = m
            def forward(self, x): return self.m(x).unsqueeze(-1)

        explainer = shap.GradientExplainer(_Wrap2D(model), bg_t)
        raw       = explainer.shap_values(X_te_t)
        shap_np   = np.array(raw[0] if isinstance(raw, list) else raw)
        if shap_np.ndim == 3 and shap_np.shape[-1] == 1:
            shap_np = shap_np.squeeze(-1)    # (n, feat, 1) → (n, feat) for SHAP ≥0.46

        feat_names = _strip_prefixes(pp_cols)
        mean_abs = np.abs(shap_np).mean(axis=0)
        order = np.argsort(mean_abs)[::-1]
        print(f"  [XAI-{label}] SHAP mean|val| per feature:")
        for i in order:
            print(f"    {feat_names[i]:40s}  {mean_abs[i]:.6f}")

        with plt.rc_context(plt.rcParamsDefault):
            shap.summary_plot(shap_np, X_test_sc, feature_names=feat_names,
                              plot_type="bar", show=False, color=_PLT_C["shap_bar"])
            ax = plt.gca()
            ax.set_xlabel("mean(|SHAP value|)")
            # TODO: here remove manual ocnfig
            ax.set_xlim(0, 3)
            ax.xaxis.grid(True, color="#d0d0d0", linewidth=0.8, zorder=0)
            ax.set_axisbelow(True)
            plt.savefig(out_dir / "xai_shap_bar.png", bbox_inches="tight", dpi=300)
            plt.close()

            shap.summary_plot(shap_np, X_test_sc, feature_names=feat_names,
                              plot_type="dot", show=False)
            plt.savefig(out_dir / "xai_shap_beeswarm.png", bbox_inches="tight", dpi=300)
            plt.close()
        print(f"  [XAI-{label}] SHAP saved → {out_dir}")
    except Exception as exc:
        print(f"  [XAI-{label}] SHAP failed: {exc}")


def _xai_ig_mlp(model: nn.Module, X_test_sc: np.ndarray, X_bg_sc: np.ndarray,
                pp_cols: list, out_dir: Path, label: str, device: torch.device,
                n_steps: int = 50, n_eg_bg: int = 20):
    """Expected Gradients for PyTorch MLP (Riemann midpoint approximation).
    Averages IG over n_eg_bg baselines sampled randomly from X_bg_sc (dev, scaled).
    Plots mean |EG| bar chart.
    """
    try:
        model.eval()
        rng    = np.random.default_rng(42)
        K      = min(n_eg_bg, len(X_bg_sc))
        bl_idx = rng.choice(len(X_bg_sc), size=K, replace=False)
        X_t    = torch.tensor(X_test_sc, dtype=torch.float32, device=device)

        eg_acc = torch.zeros_like(X_t)             # accumulates per-baseline IG
        for k in range(K):
            baseline = torch.tensor(X_bg_sc[bl_idx[k]:bl_idx[k]+1],
                                    dtype=torch.float32, device=device)  # (1, d)
            delta    = X_t - baseline               # (n, d)
            grads    = torch.zeros_like(X_t)
            for step in range(n_steps):
                alpha   = (step + 0.5) / n_steps
                x_alpha = (baseline + alpha * delta).detach().requires_grad_(True)
                model(x_alpha).sum().backward()
                grads  += x_alpha.grad.detach()
            eg_acc += (grads / n_steps) * delta.detach()  # IG for this baseline

        eg      = eg_acc / K                        # (n, d) — Expected Gradients
        eg_mean = eg.abs().mean(dim=0).cpu().numpy()  # mean |EG| per feature

        order = np.argsort(eg_mean)                 # ascending for horizontal bar
        feats = [_strip_prefix(pp_cols[i]) for i in order]
        vals  = eg_mean[order]

        fig, ax = plt.subplots(figsize=(8, max(4, len(pp_cols) * 0.35 + 1)))
        ax.barh(feats, vals, color=_PLT_C["ig"], edgecolor="none")
        ax.set_xlabel("Mean |Expected Gradient|")
        ax.set_title(f"[{label}] Expected Gradients — test set"
                     f" ({K} random baselines from dev)")
        fig.tight_layout()
        fig.savefig(out_dir / "xai_ig_bar.png")
        plt.close(fig)
        print(f"  [XAI-{label}] Expected Gradients saved → {out_dir}")
    except Exception as exc:
        print(f"  [XAI-{label}] EG failed: {exc}")


def _xai_gradcam_encoder(model: EncoderModel, X_test_sc: np.ndarray,
                         out_dir: Path, label: str, device: torch.device):
    """GradCAM on the last Conv1d of EncoderModel.
    Produces (1) mean heatmap overlaid on mean pressure curve and
             (2) per-sample heatmap as an image.
    """
    try:
        model.eval()
        last_conv = None
        for m in model.convs:
            if isinstance(m, nn.Conv1d):
                last_conv = m
        if last_conv is None:
            print(f"  [XAI-{label}] No Conv1d found — skipping GradCAM")
            return

        activations: dict = {}
        gradients:   dict = {}

        def _fwd(module, inp, out):
            activations["val"] = out.detach()

        def _bwd(module, gin, gout):
            gradients["val"] = gout[0].detach()

        fwd_h = last_conv.register_forward_hook(_fwd)
        bwd_h = last_conv.register_full_backward_hook(_bwd)

        X_t = torch.tensor(X_test_sc[:, None, :], dtype=torch.float32, device=device)
        model.zero_grad()
        model(X_t).sum().backward()
        fwd_h.remove()
        bwd_h.remove()

        act = activations["val"]                                   # (B, C, T')
        grd = gradients["val"]                                     # (B, C, T')
        cam = (grd.mean(dim=-1, keepdim=True) * act).sum(dim=1)   # (B, T')
        cam = torch.clamp(cam, min=0).cpu().numpy()                # relu

        cam_max = cam.max(axis=1, keepdims=True)
        cam_max[cam_max == 0] = 1.0
        cam /= cam_max

        T_orig   = X_test_sc.shape[1]
        t_orig   = np.linspace(0, 1, T_orig)
        t_cam    = np.linspace(0, 1, cam.shape[1])
        cam_up   = np.stack([np.interp(t_orig, t_cam, cam[i]) for i in range(len(cam))])
        mean_cam = cam_up.mean(axis=0)                             # (T_orig,)

        fig, axes = plt.subplots(2, 1, figsize=(10, 7))

        ax = axes[0]
        t_ax = np.arange(T_orig)
        ax.plot(t_ax, X_test_sc.mean(axis=0), color=_PLT_C["pressure"],
                linewidth=1.5, label="Mean pressure (test, normalised)")
        ax2 = ax.twinx()
        ax2.spines["right"].set_visible(True)
        ax2.spines["top"].set_visible(False)
        ax2.fill_between(t_ax, mean_cam, alpha=0.35, color=_PLT_C["gradcam"], label="GradCAM (mean)")
        ax2.set_ylabel("GradCAM activation", color=_PLT_C["gradcam"])
        ax2.tick_params(axis="y", labelcolor=_PLT_C["gradcam"])
        ax2.set_ylim(0, 1.3)
        ax.set_xlabel("Time step")
        ax.set_ylabel("Normalised pressure", color=_PLT_C["pressure"])
        ax.tick_params(axis="y", labelcolor=_PLT_C["pressure"])
        ax.set_title(f"[{label}] GradCAM — averaged over test set")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2)

        ax = axes[1]
        im = ax.imshow(cam_up, aspect="auto", cmap="inferno",
                       interpolation="nearest", vmin=0, vmax=1)
        ax.set_xlabel("Time step")
        ax.set_ylabel("Test sample index")
        ax.set_title(f"[{label}] GradCAM per-sample heatmap")
        fig.colorbar(im, ax=ax, label="GradCAM (normalised)")

        fig.tight_layout()
        fig.savefig(out_dir / "xai_gradcam.png")
        plt.close(fig)
        print(f"  [XAI-{label}] GradCAM saved → {out_dir}")
    except Exception as exc:
        print(f"  [XAI-{label}] GradCAM failed: {exc}")


# ── CV runners ────────────────────────────────────────────────────────────────

def run_cv_encoder(X_pt, y, channels, kernels, pool_kernels, head_hidden,
                   dropout, train_cfg, device, seed, strat_labels=None):
    """K-fold CV for Encoder on dev data only (indicative). Scales pressure curves per-fold."""
    K      = train_cfg["k_folds"]
    splits = _split_iter(K, seed, strat_labels, X_pt)

    _dummy = EncoderModel(channels, kernels, pool_kernels, head_hidden, dropout)
    print(f"\n[Encoder] Parameters : {_count_params(_dummy):,}")
    del _dummy

    fold_results  = []
    best_fold_mae = float("inf")
    best_fold_idx = -1

    for fold, (tv_idx, test_idx) in enumerate(splits):
        print(f"\n══ Encoder CV Fold {fold + 1}/{K} ════════════════════════════════")
        tr_idx, val_idx = _inner_split(tv_idx, train_cfg, seed, fold, strat_labels)
        print(f"  Train={len(tr_idx)}  Val={len(val_idx)}  Test={len(test_idx)}")

        pt_min = float(X_pt[tr_idx].min())
        pt_max = float(X_pt[tr_idx].max())
        scale  = lambda x: ((x - pt_min) / (pt_max - pt_min + 1e-8)).astype(np.float32)
        X_tr_s, X_val_s, X_te_s = scale(X_pt[tr_idx]), scale(X_pt[val_idx]), scale(X_pt[test_idx])

        torch.manual_seed(seed + fold)
        model = EncoderModel(channels, kernels, pool_kernels, head_hidden, dropout).to(device)
        model, _ = train_fold_nn(model, _enc_prepare, X_tr_s, y[tr_idx],
                                 X_val_s, y[val_idx], train_cfg, device)

        res = evaluate_nn(model, _enc_prepare, X_te_s, y[test_idx], device)
        fold_results.append({k: v for k, v in res.items() if k != "pred"})
        print(f"  Test  MAE={res['MAE']:.4f}  RMSE={res['RMSE']:.4f}  "
              f"R²={res['R2']:.4f}  MSE={res['MSE']:.4f}")

        if res["MAE"] < best_fold_mae:
            best_fold_mae = res["MAE"]
            best_fold_idx = fold

    return pd.DataFrame(fold_results), best_fold_idx, best_fold_mae


def run_cv_mlp(X_pp, y, n_in, hidden_dims, dropout, train_cfg,
               device, seed, strat_labels=None):
    """K-fold CV for MLP on dev data only (indicative). Applies MinMaxScaler per-fold."""
    K      = train_cfg["k_folds"]
    splits = _split_iter(K, seed, strat_labels, X_pp)

    _dummy = MLPModel(n_in, hidden_dims, dropout)
    print(f"\n[MLP] Parameters : {_count_params(_dummy):,}")
    del _dummy

    fold_results  = []
    best_fold_mae = float("inf")
    best_fold_idx = -1

    for fold, (tv_idx, test_idx) in enumerate(splits):
        print(f"\n══ MLP CV Fold {fold + 1}/{K} ══════════════════════════════════════")
        tr_idx, val_idx = _inner_split(tv_idx, train_cfg, seed, fold, strat_labels)
        print(f"  Train={len(tr_idx)}  Val={len(val_idx)}  Test={len(test_idx)}")

        sc = MinMaxScaler().fit(X_pp[tr_idx])
        X_tr_s  = sc.transform(X_pp[tr_idx]).astype(np.float32)
        X_val_s = sc.transform(X_pp[val_idx]).astype(np.float32)
        X_te_s  = sc.transform(X_pp[test_idx]).astype(np.float32)

        torch.manual_seed(seed + fold)
        model = MLPModel(n_in, hidden_dims, dropout).to(device)
        model, _ = train_fold_nn(model, _mlp_prepare, X_tr_s, y[tr_idx],
                                 X_val_s, y[val_idx], train_cfg, device)

        res = evaluate_nn(model, _mlp_prepare, X_te_s, y[test_idx], device)
        fold_results.append({k: v for k, v in res.items() if k != "pred"})
        print(f"  Test  MAE={res['MAE']:.4f}  RMSE={res['RMSE']:.4f}  "
              f"R²={res['R2']:.4f}  MSE={res['MSE']:.4f}")

        if res["MAE"] < best_fold_mae:
            best_fold_mae = res["MAE"]
            best_fold_idx = fold

    return pd.DataFrame(fold_results), best_fold_idx, best_fold_mae


def run_cv_gbdt(model_class, best_params, X_pp, y, train_cfg,
                seed, strat_labels=None, label="GBDT"):
    """K-fold CV for GBDT on dev data only (indicative). Trains on full tv split."""
    K      = train_cfg["k_folds"]
    splits = _split_iter(K, seed, strat_labels, X_pp)

    fold_results  = []
    best_fold_mae = float("inf")
    best_fold_idx = -1

    for fold, (tv_idx, test_idx) in enumerate(splits):
        print(f"\n══ {label} CV Fold {fold + 1}/{K} ═══════════════════════════════════")
        tr_idx, val_idx = _inner_split(tv_idx, train_cfg, seed, fold, strat_labels)
        train_idx = np.concatenate([tr_idx, val_idx])
        print(f"  Train={len(train_idx)}  Test={len(test_idx)}")

        model = model_class(**best_params)
        model.fit(X_pp[train_idx], y[train_idx])

        res = evaluate_gbdt(model, X_pp[test_idx], y[test_idx])
        fold_results.append({k: v for k, v in res.items() if k != "pred"})
        print(f"  Test  MAE={res['MAE']:.4f}  RMSE={res['RMSE']:.4f}  "
              f"R²={res['R2']:.4f}  MSE={res['MSE']:.4f}")

        if res["MAE"] < best_fold_mae:
            best_fold_mae = res["MAE"]
            best_fold_idx = fold

    return pd.DataFrame(fold_results), best_fold_idx, best_fold_mae


# ── Final-train functions ─────────────────────────────────────────────────────

def final_train_encoder(X_pt, y, dev_idx: np.ndarray,
                        channels, kernels, pool_kernels, head_hidden,
                        dropout, train_cfg: dict, device: torch.device,
                        seed: int, strat_labels=None):
    """Train the final Encoder on all dev data.
    Val split from dev used for early stopping only.
    Returns (model, (pt_min, pt_max)).
    """
    tr_idx, val_idx = _inner_split(dev_idx, train_cfg, seed, 0, strat_labels)
    print(f"\n══ Final Encoder Training ══════════════════════════════════")
    print(f"  Dev Train={len(tr_idx)}  Dev Val={len(val_idx)}  (val for early-stop only)")
    pt_min = float(X_pt[tr_idx].min())
    pt_max = float(X_pt[tr_idx].max())
    scale  = lambda x: ((x - pt_min) / (pt_max - pt_min + 1e-8)).astype(np.float32)
    X_tr_s, X_val_s = scale(X_pt[tr_idx]), scale(X_pt[val_idx])
    final_tcfg = {
        **train_cfg,
        "epochs":   train_cfg.get("final_epochs",   train_cfg["epochs"]),
        "patience": train_cfg.get("final_patience", train_cfg["patience"]),
    }
    torch.manual_seed(seed)
    model = EncoderModel(channels, kernels, pool_kernels, head_hidden, dropout).to(device)
    model, best_val_mae = train_fold_nn(model, _enc_prepare, X_tr_s, y[tr_idx],
                                        X_val_s, y[val_idx], final_tcfg, device)
    print(f"  Best dev-val MAE (early-stop criterion) = {best_val_mae:.4f}")
    return model, (pt_min, pt_max)


def final_train_mlp(X_pp, y, dev_idx: np.ndarray,
                    n_in: int, hidden_dims: list, dropout: float,
                    train_cfg: dict, device: torch.device,
                    seed: int, strat_labels=None):
    """Train the final MLP on all dev data.
    Val split from dev used for early stopping only.
    Returns (model, sc).
    """
    tr_idx, val_idx = _inner_split(dev_idx, train_cfg, seed, 0, strat_labels)
    print(f"\n══ Final MLP Training ══════════════════════════════════════")
    print(f"  Dev Train={len(tr_idx)}  Dev Val={len(val_idx)}  (val for early-stop only)")
    sc = MinMaxScaler().fit(X_pp[tr_idx])
    X_tr_s  = sc.transform(X_pp[tr_idx]).astype(np.float32)
    X_val_s = sc.transform(X_pp[val_idx]).astype(np.float32)
    final_tcfg = {
        **train_cfg,
        "epochs":   train_cfg.get("final_epochs",   train_cfg["epochs"]),
        "patience": train_cfg.get("final_patience", train_cfg["patience"]),
    }
    torch.manual_seed(seed)
    model = MLPModel(n_in, hidden_dims, dropout).to(device)
    model, best_val_mae = train_fold_nn(model, _mlp_prepare, X_tr_s, y[tr_idx],
                                        X_val_s, y[val_idx], final_tcfg, device)
    print(f"  Best dev-val MAE (early-stop criterion) = {best_val_mae:.4f}")
    return model, sc


def final_retrain_gbdt(model_class, best_params: dict,
                       X_pp, y, dev_idx: np.ndarray, label: str = "GBDT"):
    """Retrain GBDT on 100% of dev data (no val/early-stop needed). Returns model."""
    print(f"\n══ Final {label} Retrain ══════════════════════════════════")
    print(f"  Dev Train={len(dev_idx)} (all dev, no val split)")
    model = model_class(**best_params)
    model.fit(X_pp[dev_idx], y[dev_idx])
    return model


# ── Optuna helpers ────────────────────────────────────────────────────────────

def _sof(trial, key, val, log=False):
    return trial.suggest_float(key, float(val[0]), float(val[1]), log=log) \
        if isinstance(val, list) else float(val)


def _soi(trial, key, val, log=False):
    return trial.suggest_int(key, int(val[0]), int(val[1]), log=log) \
        if isinstance(val, list) else int(val)


def suggest_encoder_params(trial, ss: dict):
    mcfg, tcfg = {}, {}
    if "lr" in ss:           tcfg["lr"]           = _sof(trial, "lr",           ss["lr"],           log=True)
    if "weight_decay" in ss: tcfg["weight_decay"] = _sof(trial, "weight_decay", ss["weight_decay"], log=True)
    if "batch_size" in ss:   tcfg["batch_size"]   = _soi(trial, "batch_size",   ss["batch_size"])
    if "dropout" in ss:      mcfg["dropout"]      = _sof(trial, "dropout",      ss["dropout"])
    if "l1_lambda" in ss:    tcfg["l1_lambda"]    = _sof(trial, "l1_lambda",    ss["l1_lambda"],    log=True)

    if "enc_n_conv_layers" in ss:
        enc_n  = _soi(trial, "enc_n_conv_layers", ss["enc_n_conv_layers"])
        ch_cfg = ss.get("enc_channels", 16)
        k_cfg  = ss.get("enc_kernel_size", 5)
        pk_cfg = ss.get("enc_pool_kernel_size", 1)
        channels, kernels, pool_kernels = [1], [], []
        for i in range(enc_n):
            ch = (trial.suggest_int(f"enc_ch_{i}", int(ch_cfg[0]), int(ch_cfg[1]))
                  if isinstance(ch_cfg, list) else int(ch_cfg))
            k  = (trial.suggest_int(f"enc_k_{i}",  int(k_cfg[0]),  int(k_cfg[1]))
                  if isinstance(k_cfg,  list) else int(k_cfg))
            channels.append(ch)
            kernels.append(k)
            if i < enc_n - 1:
                pk = (trial.suggest_int(f"enc_pk_{i}", int(pk_cfg[0]), int(pk_cfg[1]))
                      if isinstance(pk_cfg, list) else int(pk_cfg))
                pool_kernels.append(pk)
        mcfg["channels"]     = channels
        mcfg["kernels"]      = kernels
        mcfg["pool_kernels"] = pool_kernels

    if "head_n_layers" in ss:
        n_h   = _soi(trial, "head_n_layers", ss["head_n_layers"])
        h_cfg = ss.get("head_hidden_size", 64)
        mcfg["head_hidden"] = [
            (trial.suggest_int(f"head_h_{i}", int(h_cfg[0]), int(h_cfg[1]), log=True)
             if isinstance(h_cfg, list) else int(h_cfg))
            for i in range(n_h)
        ]
    return mcfg, tcfg


def suggest_mlp_params(trial, ss: dict):
    mcfg, tcfg = {}, {}
    if "lr" in ss:           tcfg["lr"]           = _sof(trial, "lr",           ss["lr"],           log=True)
    if "weight_decay" in ss: tcfg["weight_decay"] = _sof(trial, "weight_decay", ss["weight_decay"], log=True)
    if "batch_size" in ss:   tcfg["batch_size"]   = _soi(trial, "batch_size",   ss["batch_size"])
    if "dropout" in ss:      mcfg["dropout"]      = _sof(trial, "dropout",      ss["dropout"])
    if "l1_lambda" in ss:    tcfg["l1_lambda"]    = _sof(trial, "l1_lambda",    ss["l1_lambda"],    log=True)
    if "n_layers" in ss:
        n     = _soi(trial, "n_layers", ss["n_layers"])
        h_cfg = ss.get("hidden_size", 64)
        mcfg["hidden_dims"] = [
            (trial.suggest_int(f"h_{i}", int(h_cfg[0]), int(h_cfg[1]), log=True)
             if isinstance(h_cfg, list) else int(h_cfg))
            for i in range(n)
        ]
    return mcfg, tcfg


def suggest_lgbm_params(trial, ss: dict) -> dict:
    p = {"random_state": 42, "n_jobs": -1, "verbose": -1, "objective": "regression_l1"}
    if "n_estimators"     in ss: p["n_estimators"]     = _soi(trial, "n_estimators",     ss["n_estimators"],     log=True)
    if "learning_rate"    in ss: p["learning_rate"]    = _sof(trial, "learning_rate",    ss["learning_rate"],    log=True)
    if "max_depth"        in ss: p["max_depth"]        = _soi(trial, "max_depth",        ss["max_depth"])
    if "num_leaves"       in ss: p["num_leaves"]       = _soi(trial, "num_leaves",       ss["num_leaves"],       log=True)
    if "min_child_samples"in ss: p["min_child_samples"]= _soi(trial, "min_child_samples",ss["min_child_samples"])
    if "subsample"        in ss: p["subsample"]        = _sof(trial, "subsample",        ss["subsample"])
    if "colsample_bytree" in ss: p["colsample_bytree"] = _sof(trial, "colsample_bytree", ss["colsample_bytree"])
    if "reg_alpha"        in ss: p["reg_alpha"]        = _sof(trial, "reg_alpha",        ss["reg_alpha"])
    if "reg_lambda"       in ss: p["reg_lambda"]       = _sof(trial, "reg_lambda",       ss["reg_lambda"])
    return p


def suggest_xgb_params(trial, ss: dict) -> dict:
    p = {"random_state": 42, "n_jobs": -1, "tree_method": "hist",
         "objective": "reg:absoluteerror", "eval_metric": "mae"}
    if "n_estimators"    in ss: p["n_estimators"]    = _soi(trial, "n_estimators",    ss["n_estimators"],    log=True)
    if "learning_rate"   in ss: p["learning_rate"]   = _sof(trial, "learning_rate",   ss["learning_rate"],   log=True)
    if "max_depth"       in ss: p["max_depth"]       = _soi(trial, "max_depth",       ss["max_depth"])
    if "subsample"       in ss: p["subsample"]       = _sof(trial, "subsample",       ss["subsample"])
    if "colsample_bytree"in ss: p["colsample_bytree"]= _sof(trial, "colsample_bytree",ss["colsample_bytree"])
    if "reg_alpha"       in ss: p["reg_alpha"]       = _sof(trial, "reg_alpha",       ss["reg_alpha"])
    if "reg_lambda"      in ss: p["reg_lambda"]      = _sof(trial, "reg_lambda",      ss["reg_lambda"])
    if "min_child_weight"in ss: p["min_child_weight"]= _soi(trial, "min_child_weight",ss["min_child_weight"])
    if "gamma"           in ss: p["gamma"]           = _sof(trial, "gamma",           ss["gamma"])
    return p


# ── Optuna objectives ─────────────────────────────────────────────────────────

def _obj_encoder(trial, X_pt, y, hpo_cfg, base_tcfg, seed, device, strat_labels):
    ss = hpo_cfg["search_space"]
    mcfg_upd, tcfg_upd = suggest_encoder_params(trial, ss)
    tcfg = {**base_tcfg, **tcfg_upd,
            "epochs": hpo_cfg["epochs"], "patience": hpo_cfg["patience"],
            "print_every": 999999, "k_folds": hpo_cfg["n_folds"]}

    channels     = mcfg_upd.get("channels",     [1, 16])
    kernels      = mcfg_upd.get("kernels",       [5])
    pool_kernels = mcfg_upd.get("pool_kernels",  [])
    head_hidden  = mcfg_upd.get("head_hidden",   [64])
    dropout      = mcfg_upd.get("dropout",       base_tcfg.get("dropout", 0.2))

    splits   = _split_iter(hpo_cfg["n_folds"], seed, strat_labels, X_pt)
    mae_vals = []
    for fold, (tv_idx, test_idx) in enumerate(splits):
        tr_idx, val_idx = _inner_split(tv_idx, tcfg, seed, fold, strat_labels)
        pt_min = float(X_pt[tr_idx].min())
        pt_max = float(X_pt[tr_idx].max())
        scale  = lambda x: ((x - pt_min) / (pt_max - pt_min + 1e-8)).astype(np.float32)
        X_tr_s, X_val_s, X_te_s = scale(X_pt[tr_idx]), scale(X_pt[val_idx]), scale(X_pt[test_idx])

        torch.manual_seed(seed + trial.number * 13 + fold)
        model = EncoderModel(channels, kernels, pool_kernels, head_hidden, dropout).to(device)
        model, _ = train_fold_nn(model, _enc_prepare, X_tr_s, y[tr_idx],
                                 X_val_s, y[val_idx], tcfg, device)
        res = evaluate_nn(model, _enc_prepare, X_te_s, y[test_idx], device)
        mae_vals.append(res["MAE"])
        trial.report(float(np.mean(mae_vals)), fold)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    return float(np.mean(mae_vals))


def _obj_mlp(trial, X_pp, y, hpo_cfg, base_tcfg, seed, device, strat_labels):
    ss = hpo_cfg["search_space"]
    mcfg_upd, tcfg_upd = suggest_mlp_params(trial, ss)
    tcfg = {**base_tcfg, **tcfg_upd,
            "epochs": hpo_cfg["epochs"], "patience": hpo_cfg["patience"],
            "print_every": 999999, "k_folds": hpo_cfg["n_folds"]}

    n_in        = X_pp.shape[1]
    hidden_dims = mcfg_upd.get("hidden_dims", [64, 32])
    dropout     = mcfg_upd.get("dropout",     base_tcfg.get("dropout", 0.2))

    splits   = _split_iter(hpo_cfg["n_folds"], seed, strat_labels, X_pp)
    mae_vals = []
    for fold, (tv_idx, test_idx) in enumerate(splits):
        tr_idx, val_idx = _inner_split(tv_idx, tcfg, seed, fold, strat_labels)
        sc = MinMaxScaler().fit(X_pp[tr_idx])
        X_tr_s  = sc.transform(X_pp[tr_idx]).astype(np.float32)
        X_val_s = sc.transform(X_pp[val_idx]).astype(np.float32)
        X_te_s  = sc.transform(X_pp[test_idx]).astype(np.float32)

        torch.manual_seed(seed + trial.number * 13 + fold)
        model = MLPModel(n_in, hidden_dims, dropout).to(device)
        model, _ = train_fold_nn(model, _mlp_prepare, X_tr_s, y[tr_idx],
                                 X_val_s, y[val_idx], tcfg, device)
        res = evaluate_nn(model, _mlp_prepare, X_te_s, y[test_idx], device)
        mae_vals.append(res["MAE"])
        trial.report(float(np.mean(mae_vals)), fold)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    return float(np.mean(mae_vals))


def _obj_gbdt(trial, X_pp, y, hpo_cfg, model_class, suggest_fn, seed, strat_labels):
    ss     = hpo_cfg["search_space"]
    params = suggest_fn(trial, ss)

    splits   = _split_iter(hpo_cfg["n_folds"], seed, strat_labels, X_pp)
    mae_vals = []
    for fold, (tv_idx, test_idx) in enumerate(splits):
        model = model_class(**params)
        model.fit(X_pp[tv_idx], y[tv_idx])
        res = evaluate_gbdt(model, X_pp[test_idx], y[test_idx])
        mae_vals.append(res["MAE"])
        trial.report(float(np.mean(mae_vals)), fold)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    return float(np.mean(mae_vals))


def _run_hpo(study_name, n_trials, n_startup_trials, objective_fn, out_dir, seed,
             pruner_type="hyperband", n_folds=3) -> optuna.Study:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(n_startup_trials=n_startup_trials, seed=seed)
    if pruner_type == "hyperband":
        pruner = optuna.pruners.HyperbandPruner(
            min_resource=1, max_resource=n_folds, reduction_factor=3)
    else:
        pruner = optuna.pruners.MedianPruner(n_startup_trials=n_startup_trials)

    study = optuna.create_study(direction="minimize", sampler=sampler,
                                pruner=pruner, study_name=study_name)
    print(f"\n{'═' * 60}")
    print(f"Optuna HPO  |  study: {study_name}")
    print(f"  Trials: {n_trials}  (startup RS: {n_startup_trials})")
    print(f"  Folds: {n_folds}  |  Sampler: TPE  |  Pruner: {pruner_type}")
    print(f"{'═' * 60}")
    study.optimize(objective_fn, n_trials=n_trials, show_progress_bar=True)

    best = study.best_trial
    print(f"\nBest trial #{best.number}  HPO-MAE={best.value:.6f}")
    for k, v in best.params.items():
        print(f"  {k}: {v}")

    if out_dir is not None:
        try:
            df_t = study.trials_dataframe(attrs=("number", "value", "params", "state"))
            df_t.to_csv(out_dir / "hpo_trials.csv", index=False)
        except Exception as e:
            print(f"[warn] HPO trials CSV: {e}")
    return study


# ── Orchestrators ─────────────────────────────────────────────────────────────

def run_encoder(X_pt, y, cfg, mat, mat_dir, seed, device, strat_labels,
                dev_idx: np.ndarray, test_idx: np.ndarray):
    hpo_cfg   = cfg["encoder_hpo"]
    train_cfg = cfg["training"].copy()
    rb_dir    = BASE_OUT / "Encoder" / mat_dir / "run_best" / _RUN_TS
    ob_dir    = BASE_OUT / "Encoder" / mat_dir / "best_overall"
    for d in [rb_dir, ob_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#' * 60}")
    print("# ENCODER: Conv1D pressure curves → weight")
    print(f"{'#' * 60}")

    dev_strat = strat_labels[dev_idx] if strat_labels is not None else None
    study = _run_hpo(
        study_name=hpo_cfg.get("study_name", "RefEncoder_hpo"),
        n_trials=hpo_cfg["n_trials"],
        n_startup_trials=hpo_cfg["n_startup_trials"],
        objective_fn=lambda t: _obj_encoder(t, X_pt[dev_idx], y[dev_idx], hpo_cfg,
                                            train_cfg, seed, device, dev_strat),
        out_dir=rb_dir, seed=seed,
        pruner_type="hyperband", n_folds=hpo_cfg["n_folds"])

    fixed   = optuna.trial.FixedTrial(study.best_trial.params)
    mcfg_u, tcfg_u = suggest_encoder_params(fixed, hpo_cfg["search_space"])
    train_cfg.update(tcfg_u)

    channels     = mcfg_u.get("channels",     [1, 16])
    kernels      = mcfg_u.get("kernels",       [5])
    pool_kernels = mcfg_u.get("pool_kernels",  [])
    head_hidden  = mcfg_u.get("head_hidden",   [64])
    dropout      = mcfg_u.get("dropout",       train_cfg.get("dropout", 0.2))

    hpo_info = {"trial": study.best_trial.number, "hpo_mae": float(study.best_trial.value),
                "channels": channels, "kernels": kernels, "pool_kernels": pool_kernels,
                "head_hidden": head_hidden, "dropout": dropout, **tcfg_u}
    (rb_dir / "hpo_best_config.json").write_text(json.dumps(hpo_info, indent=2))

    # Indicative CV on dev data
    print(f"\nIndicative CV: {train_cfg['k_folds']} folds, {train_cfg['epochs']} epochs")
    metrics_df, best_fold_idx, best_fold_mae = run_cv_encoder(
        X_pt[dev_idx], y[dev_idx], channels, kernels, pool_kernels, head_hidden,
        dropout, train_cfg, device, seed, dev_strat)

    K = train_cfg["k_folds"]
    metrics_df.index = [f"Fold_{i+1}" for i in range(K)]
    metrics_df.to_csv(rb_dir / "cv_fold_metrics.csv")
    print_summary(metrics_df, best_fold_idx, best_fold_mae, K, "Encoder")
    save_cv_plots(metrics_df, K, mat, rb_dir, "Encoder")

    # Final training on all dev; evaluate on held-out test
    final_model, (pt_min, pt_max) = final_train_encoder(
        X_pt, y, dev_idx, channels, kernels, pool_kernels, head_hidden,
        dropout, train_cfg, device, seed, dev_strat)

    scale = lambda x: ((x - pt_min) / (pt_max - pt_min + 1e-8)).astype(np.float32)
    res = evaluate_nn(final_model, _enc_prepare, scale(X_pt[test_idx]), y[test_idx], device)
    final_metrics = {"MAE": res["MAE"], "RMSE": res["RMSE"],
                     "R2": res["R2"], "MSE": res["MSE"], "source": "final_test"}
    print(f"\n{'═' * 60}")
    print(f"[Encoder] Final Test  MAE={res['MAE']:.4f}  RMSE={res['RMSE']:.4f}  "
          f"R²={res['R2']:.4f}  MSE={res['MSE']:.4f}")
    print(f"{'═' * 60}")
    save_final_test_plots(y[test_idx], res["pred"], final_metrics, rb_dir, mat, "Encoder")

    arch_info = {"channels": channels, "kernels": kernels, "pool_kernels": pool_kernels,
                 "head_hidden": head_hidden, "dropout": dropout}
    updated = save_model_nn(final_model, final_metrics, rb_dir, ob_dir, arch_info, "Encoder")

    # XAI — GradCAM on the last Conv1d layer
    _xai_gradcam_encoder(final_model, scale(X_pt[test_idx]), rb_dir, "Encoder", device)

    _save_run_info(rb_dir, ob_dir, mat, "Encoder", updated)
    if updated:
        save_final_test_plots(y[test_idx], res["pred"], final_metrics, ob_dir, mat, "Encoder")
        save_cv_plots(metrics_df, K, mat, ob_dir, "Encoder")
        _copy_artefacts(rb_dir, ob_dir)
        print(f"[Encoder] All artefacts copied → {ob_dir}")


def run_mlp(X_pp, y, cfg, mat, mat_dir, seed, device, strat_labels,
            dev_idx: np.ndarray, test_idx: np.ndarray, pp_cols: list):
    hpo_cfg   = cfg["mlp_hpo"]
    train_cfg = cfg["training"].copy()
    rb_dir    = BASE_OUT / "MLP" / mat_dir / "run_best" / _RUN_TS
    ob_dir    = BASE_OUT / "MLP" / mat_dir / "best_overall"
    for d in [rb_dir, ob_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#' * 60}")
    print("# MLP: process parameters → weight")
    print(f"{'#' * 60}")

    n_in = X_pp.shape[1]
    dev_strat = strat_labels[dev_idx] if strat_labels is not None else None
    study = _run_hpo(
        study_name=hpo_cfg.get("study_name", "RefMLP_hpo"),
        n_trials=hpo_cfg["n_trials"],
        n_startup_trials=hpo_cfg["n_startup_trials"],
        objective_fn=lambda t: _obj_mlp(t, X_pp[dev_idx], y[dev_idx], hpo_cfg,
                                        train_cfg, seed, device, dev_strat),
        out_dir=rb_dir, seed=seed,
        pruner_type="hyperband", n_folds=hpo_cfg["n_folds"])

    fixed   = optuna.trial.FixedTrial(study.best_trial.params)
    mcfg_u, tcfg_u = suggest_mlp_params(fixed, hpo_cfg["search_space"])
    train_cfg.update(tcfg_u)

    hidden_dims = mcfg_u.get("hidden_dims", [64, 32])
    dropout     = mcfg_u.get("dropout",     train_cfg.get("dropout", 0.2))

    hpo_info = {"trial": study.best_trial.number, "hpo_mae": float(study.best_trial.value),
                "hidden_dims": hidden_dims, "dropout": dropout, **tcfg_u}
    (rb_dir / "hpo_best_config.json").write_text(json.dumps(hpo_info, indent=2))

    # Indicative CV on dev data
    print(f"\nIndicative CV: {train_cfg['k_folds']} folds, {train_cfg['epochs']} epochs")
    metrics_df, best_fold_idx, best_fold_mae = run_cv_mlp(
        X_pp[dev_idx], y[dev_idx], n_in, hidden_dims, dropout,
        train_cfg, device, seed, dev_strat)

    K = train_cfg["k_folds"]
    metrics_df.index = [f"Fold_{i+1}" for i in range(K)]
    metrics_df.to_csv(rb_dir / "cv_fold_metrics.csv")
    print_summary(metrics_df, best_fold_idx, best_fold_mae, K, "MLP")
    save_cv_plots(metrics_df, K, mat, rb_dir, "MLP")

    # Final training on all dev; evaluate on held-out test
    final_model, sc = final_train_mlp(
        X_pp, y, dev_idx, n_in, hidden_dims, dropout,
        train_cfg, device, seed, dev_strat)

    X_te_s = sc.transform(X_pp[test_idx]).astype(np.float32)
    res = evaluate_nn(final_model, _mlp_prepare, X_te_s, y[test_idx], device)
    final_metrics = {"MAE": res["MAE"], "RMSE": res["RMSE"],
                     "R2": res["R2"], "MSE": res["MSE"], "source": "final_test"}
    print(f"\n{'═' * 60}")
    print(f"[MLP] Final Test  MAE={res['MAE']:.4f}  RMSE={res['RMSE']:.4f}  "
          f"R²={res['R2']:.4f}  MSE={res['MSE']:.4f}")
    print(f"{'═' * 60}")
    save_final_test_plots(y[test_idx], res["pred"], final_metrics, rb_dir, mat, "MLP")

    arch_info = {"n_in": n_in, "hidden_dims": hidden_dims, "dropout": dropout}
    updated = save_model_nn(final_model, final_metrics, rb_dir, ob_dir, arch_info, "MLP")

    # XAI — SHAP (GradientExplainer) + Integrated Gradients on scaled data
    X_dev_sc = sc.transform(X_pp[dev_idx]).astype(np.float32)
    _xai_shap_mlp(final_model, X_dev_sc, X_te_s, pp_cols, rb_dir, "MLP", device)
    _xai_ig_mlp(final_model, X_te_s, X_dev_sc, pp_cols, rb_dir, "MLP", device)

    _save_run_info(rb_dir, ob_dir, mat, "MLP", updated)
    if updated:
        save_final_test_plots(y[test_idx], res["pred"], final_metrics, ob_dir, mat, "MLP")
        save_cv_plots(metrics_df, K, mat, ob_dir, "MLP")
        _copy_artefacts(rb_dir, ob_dir)
        print(f"[MLP] All artefacts copied → {ob_dir}")


def run_lgbm(X_pp, y, cfg, mat, mat_dir, seed, strat_labels,
             dev_idx: np.ndarray, test_idx: np.ndarray, pp_cols: list):
    hpo_cfg   = cfg["lgbm_hpo"]
    train_cfg = cfg["training"]
    rb_dir    = BASE_OUT / "LightGBM" / mat_dir / "run_best" / _RUN_TS
    ob_dir    = BASE_OUT / "LightGBM" / mat_dir / "best_overall"
    for d in [rb_dir, ob_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#' * 60}")
    print("# LightGBM: process parameters → weight")
    print(f"{'#' * 60}")

    dev_strat = strat_labels[dev_idx] if strat_labels is not None else None
    study = _run_hpo(
        study_name=hpo_cfg.get("study_name", "RefLGBM_hpo"),
        n_trials=hpo_cfg["n_trials"],
        n_startup_trials=hpo_cfg["n_startup_trials"],
        objective_fn=lambda t: _obj_gbdt(t, X_pp[dev_idx], y[dev_idx], hpo_cfg,
                                         lgb.LGBMRegressor, suggest_lgbm_params,
                                         seed, dev_strat),
        out_dir=rb_dir, seed=seed,
        pruner_type="median", n_folds=hpo_cfg["n_folds"])

    fixed       = optuna.trial.FixedTrial(study.best_trial.params)
    best_params = suggest_lgbm_params(fixed, hpo_cfg["search_space"])
    hpo_info    = {"trial": study.best_trial.number,
                   "hpo_mae": float(study.best_trial.value),
                   "best_params": best_params}
    (rb_dir / "hpo_best_config.json").write_text(json.dumps(hpo_info, indent=2))

    # Indicative CV on dev data
    print(f"\nIndicative CV: {train_cfg['k_folds']} folds")
    metrics_df, best_fold_idx, best_fold_mae = run_cv_gbdt(
        lgb.LGBMRegressor, best_params, X_pp[dev_idx], y[dev_idx],
        train_cfg, seed, dev_strat, label="LightGBM")

    K = train_cfg["k_folds"]
    metrics_df.index = [f"Fold_{i+1}" for i in range(K)]
    metrics_df.to_csv(rb_dir / "cv_fold_metrics.csv")
    print_summary(metrics_df, best_fold_idx, best_fold_mae, K, "LightGBM")
    save_cv_plots(metrics_df, K, mat, rb_dir, "LightGBM")

    # Final retrain on all dev; evaluate on held-out test
    final_model = final_retrain_gbdt(lgb.LGBMRegressor, best_params,
                                     X_pp, y, dev_idx, "LightGBM")
    res = evaluate_gbdt(final_model, X_pp[test_idx], y[test_idx])
    final_metrics = {"MAE": res["MAE"], "RMSE": res["RMSE"],
                     "R2": res["R2"], "MSE": res["MSE"], "source": "final_test"}
    print(f"\n{'═' * 60}")
    print(f"[LightGBM] Final Test  MAE={res['MAE']:.4f}  RMSE={res['RMSE']:.4f}  "
          f"R²={res['R2']:.4f}  MSE={res['MSE']:.4f}")
    print(f"{'═' * 60}")
    save_final_test_plots(y[test_idx], res["pred"], final_metrics, rb_dir, mat, "LightGBM")

    updated = save_model_gbdt(final_model, final_metrics, rb_dir, ob_dir,
                              best_params, "LightGBM")

    # XAI — SHAP TreeExplainer on raw (unscaled) process parameters
    _xai_shap_gbdt(final_model, X_pp[dev_idx], X_pp[test_idx], pp_cols, rb_dir, "LightGBM")

    _save_run_info(rb_dir, ob_dir, mat, "LightGBM", updated)
    if updated:
        save_final_test_plots(y[test_idx], res["pred"], final_metrics, ob_dir, mat, "LightGBM")
        save_cv_plots(metrics_df, K, mat, ob_dir, "LightGBM")
        _copy_artefacts(rb_dir, ob_dir)
        print(f"[LightGBM] All artefacts copied → {ob_dir}")


def run_xgboost(X_pp, y, cfg, mat, mat_dir, seed, strat_labels,
                dev_idx: np.ndarray, test_idx: np.ndarray, pp_cols: list):
    hpo_cfg   = cfg["xgboost_hpo"]
    train_cfg = cfg["training"]
    rb_dir    = BASE_OUT / "XGBoost" / mat_dir / "run_best" / _RUN_TS
    ob_dir    = BASE_OUT / "XGBoost" / mat_dir / "best_overall"
    for d in [rb_dir, ob_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#' * 60}")
    print("# XGBoost: process parameters → weight")
    print(f"{'#' * 60}")

    dev_strat = strat_labels[dev_idx] if strat_labels is not None else None
    study = _run_hpo(
        study_name=hpo_cfg.get("study_name", "RefXGB_hpo"),
        n_trials=hpo_cfg["n_trials"],
        n_startup_trials=hpo_cfg["n_startup_trials"],
        objective_fn=lambda t: _obj_gbdt(t, X_pp[dev_idx], y[dev_idx], hpo_cfg,
                                         xgb.XGBRegressor, suggest_xgb_params,
                                         seed, dev_strat),
        out_dir=rb_dir, seed=seed,
        pruner_type="median", n_folds=hpo_cfg["n_folds"])

    fixed       = optuna.trial.FixedTrial(study.best_trial.params)
    best_params = suggest_xgb_params(fixed, hpo_cfg["search_space"])
    hpo_info    = {"trial": study.best_trial.number,
                   "hpo_mae": float(study.best_trial.value),
                   "best_params": best_params}
    (rb_dir / "hpo_best_config.json").write_text(json.dumps(hpo_info, indent=2))

    # Indicative CV on dev data
    print(f"\nIndicative CV: {train_cfg['k_folds']} folds")
    metrics_df, best_fold_idx, best_fold_mae = run_cv_gbdt(
        xgb.XGBRegressor, best_params, X_pp[dev_idx], y[dev_idx],
        train_cfg, seed, dev_strat, label="XGBoost")

    K = train_cfg["k_folds"]
    metrics_df.index = [f"Fold_{i+1}" for i in range(K)]
    metrics_df.to_csv(rb_dir / "cv_fold_metrics.csv")
    print_summary(metrics_df, best_fold_idx, best_fold_mae, K, "XGBoost")
    save_cv_plots(metrics_df, K, mat, rb_dir, "XGBoost")

    # Final retrain on all dev; evaluate on held-out test
    final_model = final_retrain_gbdt(xgb.XGBRegressor, best_params,
                                     X_pp, y, dev_idx, "XGBoost")
    res = evaluate_gbdt(final_model, X_pp[test_idx], y[test_idx])
    final_metrics = {"MAE": res["MAE"], "RMSE": res["RMSE"],
                     "R2": res["R2"], "MSE": res["MSE"], "source": "final_test"}
    print(f"\n{'═' * 60}")
    print(f"[XGBoost] Final Test  MAE={res['MAE']:.4f}  RMSE={res['RMSE']:.4f}  "
          f"R²={res['R2']:.4f}  MSE={res['MSE']:.4f}")
    print(f"{'═' * 60}")
    save_final_test_plots(y[test_idx], res["pred"], final_metrics, rb_dir, mat, "XGBoost")

    updated = save_model_gbdt(final_model, final_metrics, rb_dir, ob_dir,
                              best_params, "XGBoost")

    # XAI — SHAP TreeExplainer on raw (unscaled) process parameters
    _xai_shap_gbdt(final_model, X_pp[dev_idx], X_pp[test_idx], pp_cols, rb_dir, "XGBoost")

    _save_run_info(rb_dir, ob_dir, mat, "XGBoost", updated)
    if updated:
        save_final_test_plots(y[test_idx], res["pred"], final_metrics, ob_dir, mat, "XGBoost")
        save_cv_plots(metrics_df, K, mat, ob_dir, "XGBoost")
        _copy_artefacts(rb_dir, ob_dir)
        print(f"[XGBoost] All artefacts copied → {ob_dir}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Reference weight-prediction models: Encoder, MLP, LightGBM, XGBoost.")
    parser.add_argument("--material", type=str.upper,
                        choices=["PP", "ABS", "ALL"], required=True,
                        help="Material subset: PP | ABS (plain KFold) or ALL (stratified). "
                             "Case-insensitive.")
    args = parser.parse_args()
    mat  = args.material

    cfg_path = BASE_DIR / "config" / _CFG_MAP[mat]
    cfg      = json.loads(cfg_path.read_text())
    mat_dir  = _MAT_DIR[mat]

    prep_cfg = cfg["preprocessing"]
    seed     = prep_cfg["random_seed"]
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"Device   : {device}")
    print(f"Material : {mat}  →  output subfolder: {mat_dir}")
    print(f"Config   : {cfg_path}")

    _, X_pp, X_pt, y, strat_labels, pp_cols = load_data(cfg["data"], prep_cfg)

    # ── One-time initial split (dev / held-out test) ──────────────────────────
    test_frac = prep_cfg.get("test_fraction", 0.2)
    dev_idx, test_idx = _initial_split(len(y), test_frac, seed, strat_labels)
    print(f"\nDev/test split  →  dev={len(dev_idx)}  test={len(test_idx)}")

    # ── Active-model gates (default: all on) ──────────────────────────────────
    active = cfg.get("active_models", {})
    enc_on  = bool(active.get("encoder", 1))
    mlp_on  = bool(active.get("mlp",     1))
    lgbm_on = bool(active.get("lgbm",    1))
    xgb_on  = bool(active.get("xgboost", 1))
    print(f"\nActive models  →  encoder={enc_on}  mlp={mlp_on}  lgbm={lgbm_on}  xgboost={xgb_on}")

    if enc_on:
        run_encoder( X_pt, y, cfg, mat, mat_dir, seed, device, strat_labels, dev_idx, test_idx)
    else:
        print("\n[encoder]  skipped (active_models.encoder = 0)")

    if mlp_on:
        run_mlp(     X_pp, y, cfg, mat, mat_dir, seed, device, strat_labels, dev_idx, test_idx, pp_cols)
    else:
        print("\n[mlp]      skipped (active_models.mlp = 0)")

    if lgbm_on:
        run_lgbm(    X_pp, y, cfg, mat, mat_dir, seed, strat_labels, dev_idx, test_idx, pp_cols)
    else:
        print("\n[lgbm]     skipped (active_models.lgbm = 0)")

    if xgb_on:
        run_xgboost( X_pp, y, cfg, mat, mat_dir, seed, strat_labels, dev_idx, test_idx, pp_cols)
    else:
        print("\n[xgboost]  skipped (active_models.xgboost = 0)")

    n_active = sum([enc_on, mlp_on, lgbm_on, xgb_on])
    print(f"\n{'═' * 60}")
    print(f"{n_active}/4 reference model(s) complete.")
    print(f"Results saved under: {BASE_OUT / mat_dir}")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
