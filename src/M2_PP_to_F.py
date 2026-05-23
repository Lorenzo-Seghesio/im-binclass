"""
M2_PP_to_F.py
=========================
Multi-output MLP for predicting M1 pressure-encoder features (f_0, f_1, …)
from scalar injection-moulding process parameters.

Usage
-----
  python src/M2_PP_to_F.py --material PP
  python src/M2_PP_to_F.py --material ABS
  python src/M2_PP_to_F.py --material ALL

Parameters
----------
  --material : str, required (case-insensitive)
               PP | ABS  →  single-material subset, plain KFold CV
               ALL       →  full dataset, StratifiedKFold CV (material-balanced splits)
               Config loaded: M2_PP_config.json / M2_ABS_config.json / M2_AllData_config.json

Architecture
------------
  Input : n_pp scalar process parameters (MinMax-scaled per fold)
  Hidden: MLP (ReLU + Dropout, He/Kaiming init + zero bias)
  Output: Linear(n_f)  — no activation, PyTorch default init (regression)

Training
--------
  • MAE loss + explicit L1 weight regularisation
  • AdamW optimiser (weight_decay) + ReduceLROnPlateau scheduler
  • K-fold cross-validation; early stopping per fold (monitored on val MAE, scaled space)
  • MinMaxScaler fitted per fold on training inputs AND training targets
  • Optuna HPO (TPE sampler + Hyperband pruner) or manual config

Config files
------------
  config/M2_PP_config.json        (for --material PP)
  config/M2_ABS_config.json       (for --material ABS)
  config/M2_AllData_config.json   (for --material ALL)

Outputs
-------
  outputs/M2/[material]/run_best/      best model of this run (always written)
  outputs/M2/[material]/best_overall/  overall best across all runs (updated when mean MAE improves)
  Each folder contains:
    best_model.pt         — model weights
    best_metrics.json     — metrics + architecture info
    fold_metrics.csv      — per-fold metrics table
    f_predictions.csv     — full-dataset predictions in original f space (for XAI)
    scatter_best_fold.png — real vs predicted per f target
    metrics_folds.png     — per-fold mean MAE / R² bar chart
    residuals_hist_best_fold.png — residuals histograms per f target
    hpo_best_config.json  — best Optuna config (only in optuna mode)
    hpo_trials.csv        — all Optuna trials   (only in optuna mode)
"""

import argparse
import copy, json, math, shutil, warnings
import optuna
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent

# ── Model definition ──────────────────────────────────────────────────────────

def _he_zero_bias(m: nn.Module) -> None:
    """Kaiming-normal weight init + zero bias."""
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(m.bias)


class M2Model(nn.Module):
    """
    Pure MLP: scalar process parameters → M1 pressure-encoder f features.
    Hidden layers: ReLU + Dropout (He/Kaiming init + zero bias).
    Output layer : Linear (no activation) — multi-output regression.
                   Kept at PyTorch default init.
    """

    def __init__(self, n_in: int, n_out: int, hidden_dims: list, dropout: float):
        super().__init__()
        layers = []
        prev = n_in
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, n_out))    # output: linear, no activation
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        linears = [m for m in self.net if isinstance(m, nn.Linear)]
        for m in linears[:-1]:          # hidden linears: He + zero bias
            _he_zero_bias(m)
        # output linear: PyTorch default (Kaiming uniform, non-zero bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)              # (B, n_out)


def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _initial_split(n: int, test_frac: float, seed: int, strat_labels=None):
    """One-time deterministic train/test split performed before HPO or CV.
    Returns (dev_idx, test_idx) as np.intp arrays.
    Uses StratifiedShuffleSplit when strat_labels is provided (ALL mode)
    so that material proportions are balanced in both splits.
    """
    if strat_labels is not None:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_frac,
                                     random_state=seed)
        dev_rel, test_rel = next(sss.split(np.zeros(n), strat_labels))
        return dev_rel.astype(np.intp), test_rel.astype(np.intp)
    rng    = np.random.default_rng(seed)
    perm   = rng.permutation(n)
    n_test = max(1, int(n * test_frac))
    return perm[n_test:].astype(np.intp), perm[:n_test].astype(np.intp)


# ── Scaling utilities ─────────────────────────────────────────────────────────

def fit_scalers(X_tr: np.ndarray, Y_tr: np.ndarray):
    """Fit MinMaxScalers for inputs and targets on training data only."""
    x_sc = MinMaxScaler()
    y_sc = MinMaxScaler()
    x_sc.fit(X_tr)
    y_sc.fit(Y_tr)
    return x_sc, y_sc


def apply_scalers(X: np.ndarray, Y: np.ndarray,
                  x_sc: MinMaxScaler, y_sc: MinMaxScaler):
    """Apply pre-fitted scalers; return float32 arrays."""
    X_s = x_sc.transform(X).astype(np.float32)
    Y_s = y_sc.transform(Y).astype(np.float32)
    return X_s, Y_s


def to_tensors(X: np.ndarray, Y: np.ndarray, device: torch.device):
    """Convert numpy arrays to PyTorch tensors on the target device."""
    t_x = torch.tensor(X, dtype=torch.float32, device=device)
    t_y = torch.tensor(Y, dtype=torch.float32, device=device)
    return t_x, t_y


# ── Training & evaluation ─────────────────────────────────────────────────────

def train_fold(model: M2Model,
               X_tr, Y_tr, X_val, Y_val,
               tcfg: dict, device: torch.device):
    """Train for one fold with early stopping.

    Training loss = MAE (in scaled space) + l1_lambda * Σ|params|.
    Early stopping monitored on val MAE (scaled space).
    Returns model with best-val-MAE state + the best val MAE value.
    """
    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=tcfg["lr"],
                                  weight_decay=tcfg["weight_decay"])
    if tcfg["scheduler_factor"] < 1:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=tcfg["scheduler_patience"],
            factor=tcfg["scheduler_factor"])

    t_X_tr,  t_Y_tr  = to_tensors(X_tr,  Y_tr,  device)
    t_X_val, t_Y_val = to_tensors(X_val, Y_val, device)

    loader = DataLoader(TensorDataset(t_X_tr, t_Y_tr),
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
            pred = model(b_x)
            loss = criterion(pred, b_y)
            if l1_lambda > 0.0:
                l1_reg = sum(p.abs().sum() for p in model.parameters())
                loss   = loss + l1_lambda * l1_reg
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(b_y)
        epoch_loss /= len(Y_tr)

        model.eval()
        with torch.no_grad():
            val_mae = criterion(model(t_X_val), t_Y_val).item()
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


def evaluate_model(model: M2Model,
                   X_scaled: np.ndarray, Y_orig: np.ndarray,
                   y_sc: MinMaxScaler, f_cols: list,
                   device: torch.device) -> dict:
    """Predict in scaled space, inverse-transform, return per-target and mean metrics.

    All reported metrics (MAE, RMSE, R², MSE) are in the original f-feature space.
    """
    t_x = torch.tensor(X_scaled, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        pred_scaled = model(t_x).cpu().numpy()
    pred = y_sc.inverse_transform(pred_scaled)

    per_target = {}
    for i, fc in enumerate(f_cols):
        mae  = float(mean_absolute_error(Y_orig[:, i], pred[:, i]))
        mse  = float(mean_squared_error(Y_orig[:, i], pred[:, i]))
        rmse = math.sqrt(mse)
        try:
            r2 = float(r2_score(Y_orig[:, i], pred[:, i]))
        except Exception:
            r2 = float("nan")
        per_target[fc] = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}

    mean_mae  = float(np.mean([v["MAE"]  for v in per_target.values()]))
    mean_rmse = float(np.mean([v["RMSE"] for v in per_target.values()]))
    mean_r2   = float(np.nanmean([v["R2"] for v in per_target.values()]))
    mean_mse  = float(np.mean([v["MSE"]  for v in per_target.values()]))

    return {
        "mean_MAE":  mean_mae,
        "mean_RMSE": mean_rmse,
        "mean_R2":   mean_r2,
        "mean_MSE":  mean_mse,
        "per_target": per_target,
        "pred":       pred,
    }


# ── Pipeline functions ────────────────────────────────────────────────────────

def load_data(data_cfg: dict, prep_cfg: dict):
    """Load scalar process-parameter CSV and M1 f-feature CSV; join, clean, drop zero cols.

    Returns
    -------
    part_ids, X, Y, pp_cols, f_cols, strat_labels
        strat_labels is None for single-material configs;
        encoded integer material array when 'material_col' is present in data_cfg.
    """
    df_scalar = pd.read_csv(BASE_DIR / data_cfg["scalar_csv"],
                            index_col=data_cfg["id_col"])
    df_f      = pd.read_csv(BASE_DIR / data_cfg["features_csv"],
                            index_col=data_cfg["id_col"])

    df_scalar = df_scalar.drop(columns=data_cfg.get("drop_cols", []), errors="ignore")
    df        = df_scalar.join(df_f, how="inner")
    print(f"Joined : {df.shape[0]} parts × {df.shape[1]} columns")

    # Encode material column (ALL mode only)
    mat_col = data_cfg.get("material_col")
    if mat_col:
        materials = sorted(df[mat_col].dropna().unique())
        mat_map   = {m: i for i, m in enumerate(materials)}
        print(f"Material encoding : {mat_map}")
        df[mat_col] = df[mat_col].map(mat_map)

    # Identify f columns; drop all-zero ones (invalid M1 run artefacts)
    f_cols_loaded = [c for c in df_f.columns if c in df.columns]
    zero_f_cols   = [c for c in f_cols_loaded if (df[c] == 0.0).all()]
    if zero_f_cols:
        print(f"Dropping all-zero f columns : {zero_f_cols}")
        df.drop(columns=zero_f_cols, inplace=True)
    f_cols = [c for c in f_cols_loaded if c not in zero_f_cols]

    if not f_cols:
        raise ValueError(
            "All f columns are zero — run M1 first to generate valid pressure features.")
    print(f"Active f targets  : {f_cols}")

    pp_cols = [c for c in df.columns if c not in f_cols]

    # Impute missing values in input columns
    for col in pp_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # IQR outlier removal on input columns only
    iqr_k = prep_cfg["iqr_multiplier"]
    mask  = pd.Series(True, index=df.index)
    for col in pp_cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr    = q3 - q1
        if iqr > 0:
            mask &= (df[col] >= q1 - iqr_k * iqr) & (df[col] <= q3 + iqr_k * iqr)
    n_removed = int((~mask).sum())
    df = df[mask].copy()
    print(f"Outliers removed  : {n_removed}  →  {len(df)} parts remaining")

    part_ids     = df.index.to_numpy()
    X            = df[pp_cols].to_numpy(dtype=np.float32)
    Y            = df[f_cols].to_numpy(dtype=np.float32)
    strat_labels = df[mat_col].to_numpy() if mat_col else None

    print(f"Input features : {X.shape[1]}   |   Targets : {Y.shape[1]}   |   "
          f"Samples : {len(part_ids)}")
    for i, fc in enumerate(f_cols):
        print(f"  {fc}  mean={Y[:, i].mean():.4f}  std={Y[:, i].std():.4f}  "
              f"min={Y[:, i].min():.4f}  max={Y[:, i].max():.4f}")

    return part_ids, X, Y, pp_cols, f_cols, strat_labels


def load_m1_train_test_split(splits_path: Path, part_ids: np.ndarray):
    """Load M1 train_test_split.json and map part IDs → integer indices in M2's dataset.

    Parts absent from M2's cleaned dataset (e.g. additional outliers removed by M2)
    are silently excluded from each partition.
    Returns (dev_idx, test_idx) as np.intp arrays.
    """
    data      = json.loads(splits_path.read_text())
    id_to_idx = {pid: i for i, pid in enumerate(part_ids.tolist())}
    dev_idx  = np.array([id_to_idx[p] for p in data["train_part_ids"] if p in id_to_idx],
                        dtype=np.intp)
    test_idx = np.array([id_to_idx[p] for p in data["test_part_ids"]  if p in id_to_idx],
                        dtype=np.intp)
    print(f"M1 train/test split loaded  →  dev={len(dev_idx)}  test={len(test_idx)}")
    return dev_idx, test_idx


def run_cv(X, Y, model_cfg: dict, train_cfg: dict, f_cols: list,
           device: torch.device, seed: int, strat_labels=None,
           precomputed_splits=None):
    """Run K-fold Cross Validation; return metrics DataFrame and best-fold artefacts.

    When precomputed_splits is provided (list of dicts with keys tr_idx, val_idx,
    test_idx as returned by load_m1_splits), those exact partitions are used for
    every fold, ensuring M2 is evaluated on the same train/test boundaries as M1.
    When None, fresh KFold / StratifiedKFold splits are generated from the data.
    """
    # ── Build fold index triples (tr_idx, val_idx, test_idx) ─────────────────
    if precomputed_splits is not None:
        K            = len(precomputed_splits)
        fold_triples = [(fs["tr_idx"], fs["val_idx"], fs["test_idx"])
                        for fs in precomputed_splits]
        print(f"Using {K} precomputed M1 splits for final CV")
    else:
        K = train_cfg["k_folds"]
        if strat_labels is not None:
            kf    = StratifiedKFold(n_splits=K, shuffle=True, random_state=seed)
            outer = list(kf.split(X, strat_labels))
        else:
            kf    = KFold(n_splits=K, shuffle=True, random_state=seed)
            outer = list(kf.split(X))
        fold_triples = []
        for fold_n, (tv_idx, test_idx) in enumerate(outer):
            if strat_labels is not None:
                sss = StratifiedShuffleSplit(n_splits=1,
                                            test_size=train_cfg["val_fraction"],
                                            random_state=seed + fold_n)
                _tr, _val = next(sss.split(tv_idx, strat_labels[tv_idx]))
                tr_idx  = tv_idx[_tr]
                val_idx = tv_idx[_val]
            else:
                rng     = np.random.default_rng(seed + fold_n)
                perm    = rng.permutation(len(tv_idx))
                n_val   = max(1, int(len(tv_idx) * train_cfg["val_fraction"]))
                val_idx = tv_idx[perm[:n_val]]
                tr_idx  = tv_idx[perm[n_val:]]
            fold_triples.append((tr_idx, val_idx, test_idx))

    n_in  = X.shape[1]
    n_out = Y.shape[1]

    _dummy = M2Model(n_in, n_out, model_cfg["hidden_dims"], model_cfg["dropout"])
    print(f"\nModel parameters : {_count_params(_dummy):,}")
    del _dummy

    fold_results  = []
    best_fold_mae = float("inf")
    best_fold_idx = -1
    best_model_cv = None
    best_scalers  = None
    best_Y_test   = None
    best_preds_cv = None

    for fold, (tr_idx, val_idx, test_idx) in enumerate(fold_triples):
        print(f"\n══ Fold {fold + 1}/{K} ══════════════════════════════════════════")
        print(f"  Train={len(tr_idx)}  Val={len(val_idx)}  Test={len(test_idx)}")

        x_sc, y_sc = fit_scalers(X[tr_idx], Y[tr_idx])
        X_tr_s,  Y_tr_s  = apply_scalers(X[tr_idx],   Y[tr_idx],   x_sc, y_sc)
        X_val_s, Y_val_s = apply_scalers(X[val_idx],  Y[val_idx],  x_sc, y_sc)
        X_te_s,  _       = apply_scalers(X[test_idx], Y[test_idx], x_sc, y_sc)

        torch.manual_seed(seed + fold)
        model = M2Model(n_in, n_out, model_cfg["hidden_dims"],
                        model_cfg["dropout"]).to(device)
        model, _ = train_fold(model, X_tr_s, Y_tr_s, X_val_s, Y_val_s,
                              train_cfg, device)

        res = evaluate_model(model, X_te_s, Y[test_idx], y_sc, f_cols, device)
        fold_results.append({
            "mean_MAE":  res["mean_MAE"],
            "mean_RMSE": res["mean_RMSE"],
            "mean_R2":   res["mean_R2"],
            "mean_MSE":  res["mean_MSE"],
        })
        print(f"  Test  mean_MAE={res['mean_MAE']:.4f}  mean_RMSE={res['mean_RMSE']:.4f}  "
              f"mean_R²={res['mean_R2']:.4f}  mean_MSE={res['mean_MSE']:.4f}")
        for fc, m in res["per_target"].items():
            print(f"    {fc}  MAE={m['MAE']:.4f}  R²={m['R2']:.4f}")

        if res["mean_MAE"] < best_fold_mae:
            best_fold_mae = res["mean_MAE"]
            best_fold_idx = fold
            best_model_cv = copy.deepcopy(model)
            best_scalers  = (x_sc, y_sc)
            best_Y_test   = Y[test_idx]
            best_preds_cv = res["pred"]

    return (pd.DataFrame(fold_results),
            best_fold_idx, best_fold_mae,
            best_model_cv, best_scalers,
            best_Y_test, best_preds_cv)


def print_summary(metrics_df: pd.DataFrame, best_fold_idx: int,
                  best_fold_mae: float, K: int):
    print("\n" + "═" * 60)
    print(f"{K}-fold CV summary  (best fold: {best_fold_idx + 1}  "
          f"mean_MAE={best_fold_mae:.4f})")
    print("─" * 60)
    for m in ["mean_MAE", "mean_RMSE", "mean_R2", "mean_MSE"]:
        print(f"  {m:12s}  {metrics_df[m].mean():.4f} ± {metrics_df[m].std():.4f}")
    print("═" * 60)


# ── Plot functions ────────────────────────────────────────────────────────────

def save_cv_plots(metrics_df: pd.DataFrame, K: int, material: str, plots_dir: Path):
    """Bar chart of K-fold CV metrics (indicative — dev data only)."""
    C_BLUE, C_RED = "#0072B2", "#D55E00"
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fold_labels = [f"F{i + 1}" for i in range(K)] + ["Mean"]
    bar_colors  = [C_BLUE] * K + [C_RED]
    for ax, metric in zip(axes, ["mean_MAE", "mean_RMSE", "mean_R2", "mean_MSE"]):
        vals = list(metrics_df[metric]) + [metrics_df[metric].mean()]
        bars = ax.bar(fold_labels, vals, color=bar_colors, edgecolor="k", linewidth=0.4)
        ax.set_title(metric, fontsize=10)
        ax.set_ylabel(metric)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)
    fig.suptitle(f"[{material}] [Indicative] K-fold CV metrics (K={K})", fontsize=13)
    fig.tight_layout()
    fig.savefig(plots_dir / "cv_metrics_folds.png", dpi=150)
    plt.close(fig)
    print(f"CV plots saved → {plots_dir}")


def save_final_test_plots(Y_test: np.ndarray, preds_test: np.ndarray,
                          final_metrics: dict, plots_dir: Path,
                          material: str, f_cols: list):
    """Scatter + residual-hist plots evaluated on the held-out test set (PRIMARY)."""
    C_BLUE, C_RED = "#0072B2", "#D55E00"
    n_f   = len(f_cols)
    ncols = min(n_f, 3)
    nrows = math.ceil(n_f / ncols)

    # Scatter: real vs predicted per f-target
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 5 * nrows), squeeze=False)
    for i, fc in enumerate(f_cols):
        ax  = axes[i // ncols][i % ncols]
        y_r = Y_test[:, i]
        y_p = preds_test[:, i]
        ax.scatter(y_r, y_p, alpha=0.7, edgecolors="k",
                   linewidths=0.3, color=C_BLUE, zorder=3)
        lo = min(y_r.min(), y_p.min())
        hi = max(y_r.max(), y_p.max())
        ax.plot([lo, hi], [lo, hi], color=C_RED, linestyle="--",
                linewidth=1.5, label="Ideal (1:1)")
        ax.set_xlabel(f"Real {fc}", fontsize=10)
        ax.set_ylabel(f"Pred {fc}", fontsize=10)
        ax.set_title(fc, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)
    for i in range(n_f, nrows * ncols):
        axes[i // ncols][i % ncols].set_visible(False)
    fig.suptitle(
        f"[{material}] Final Test — Real vs Predicted\n"
        f"mean_MAE={final_metrics['mean_MAE']:.4f}",
        fontsize=12)
    fig.tight_layout()
    fig.savefig(plots_dir / "scatter_final_test.png", dpi=150)
    plt.close(fig)

    # Residuals histogram per f-target
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 4 * nrows), squeeze=False)
    for i, fc in enumerate(f_cols):
        ax  = axes[i // ncols][i % ncols]
        res = preds_test[:, i] - Y_test[:, i]
        ax.hist(res, bins=20, color=C_BLUE, edgecolor="k", linewidth=0.4, alpha=0.85)
        ax.axvline(0, color=C_RED, linestyle="--", linewidth=1.5)
        ax.axvline(float(res.mean()), color="#009E73", linestyle="-",
                   linewidth=1.5, label=f"Mean={res.mean():.4f}")
        ax.set_xlabel("Residual (pred − real)", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.set_title(f"{fc}  std={res.std():.4f}", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)
    for i in range(n_f, nrows * ncols):
        axes[i // ncols][i % ncols].set_visible(False)
    fig.suptitle(f"[{material}] Residuals by f-target (final test)", fontsize=12)
    fig.tight_layout()
    fig.savefig(plots_dir / "residuals_hist_final_test.png", dpi=150)
    plt.close(fig)
    print(f"Final test plots saved → {plots_dir}")


# ── Save functions ────────────────────────────────────────────────────────────

def save_models(final_model: M2Model, final_metrics: dict,
                run_best_dir: Path, overall_best_dir: Path,
                n_in: int, n_out: int, pp_cols: list, f_cols: list,
                model_cfg: dict, id_col: str) -> bool:
    """Save final model weights + metadata to run_best; update overall_best if improved."""
    run_info = {
        "mean_MAE":   final_metrics["mean_MAE"],
        "mean_RMSE":  final_metrics["mean_RMSE"],
        "mean_R2":    final_metrics["mean_R2"],
        "mean_MSE":   final_metrics["mean_MSE"],
        "per_target": final_metrics.get("per_target", {}),
        "source":     "final_test",
        "n_in":       n_in,
        "n_out":      n_out,
        "pp_cols":    pp_cols,
        "f_cols":     f_cols,
        "model_cfg":  model_cfg,
    }
    torch.save(final_model.state_dict(), run_best_dir / "best_model.pt")
    (run_best_dir / "best_metrics.json").write_text(json.dumps(run_info, indent=2))
    mae = final_metrics["mean_MAE"]
    print(f"\nRun best model saved  (test mean_MAE={mae:.4f})  → {run_best_dir}")

    overall_path = overall_best_dir / "best_metrics.json"
    prev_mae     = float("inf")
    if overall_path.exists():
        prev_mae = json.loads(overall_path.read_text()).get("mean_MAE", float("inf"))

    if mae < prev_mae:
        torch.save(final_model.state_dict(), overall_best_dir / "best_model.pt")
        overall_path.write_text(json.dumps(run_info, indent=2))
        print(f"Overall best updated  ({prev_mae:.4f} → {mae:.4f} mean_MAE)  "
              f"→ {overall_best_dir}")
        return True
    else:
        print(f"Overall best unchanged  (saved={prev_mae:.4f},  this run={mae:.4f})")
        return False


def save_predictions(model: M2Model, X: np.ndarray,
                     x_sc: MinMaxScaler, y_sc: MinMaxScaler,
                     part_ids: np.ndarray, f_cols: list,
                     out_csv: Path, id_col: str,
                     device: torch.device) -> pd.DataFrame:
    """Run the best-fold model over the full cleaned dataset; save predictions CSV."""
    X_s = x_sc.transform(X).astype(np.float32)
    t_x = torch.tensor(X_s, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        pred_scaled = model(t_x).cpu().numpy()
    pred = y_sc.inverse_transform(pred_scaled)
    df_pred = pd.DataFrame(pred, columns=f_cols, index=part_ids)
    df_pred.index.name = id_col
    df_pred.to_csv(out_csv)
    print(f"Predictions saved → {out_csv}  "
          f"({df_pred.shape[0]} parts × {df_pred.shape[1]} targets)")
    return df_pred


def final_train_m2(X, Y, dev_idx: np.ndarray, model_cfg: dict, train_cfg: dict,
                   seed: int, device: torch.device, strat_labels=None):
    """Train the final M2Model on all dev data.
    A small internal validation split (val_fraction × dev) is used solely
    for early stopping — it is never reported.  Final evaluation is always
    performed by the caller on the held-out test set.
    Uses final_epochs / final_patience from train_cfg when present.
    Returns (model, x_sc, y_sc).
    """
    n_in  = X.shape[1]
    n_out = Y.shape[1]
    if strat_labels is not None:
        sss = StratifiedShuffleSplit(n_splits=1,
                                     test_size=train_cfg["val_fraction"],
                                     random_state=seed)
        _tr, _val = next(sss.split(dev_idx, strat_labels[dev_idx]))
        tr_idx  = dev_idx[_tr]
        val_idx = dev_idx[_val]
    else:
        rng     = np.random.default_rng(seed)
        perm    = rng.permutation(len(dev_idx))
        n_val   = max(1, int(len(dev_idx) * train_cfg["val_fraction"]))
        val_idx = dev_idx[perm[:n_val]]
        tr_idx  = dev_idx[perm[n_val:]]

    print(f"\n══ Final Training ═══════════════════════════════════════════")
    print(f"  Dev Train={len(tr_idx)}  Dev Val={len(val_idx)}  "
          f"(val split for early-stop only)")

    x_sc, y_sc = fit_scalers(X[tr_idx], Y[tr_idx])
    X_tr_s,  Y_tr_s  = apply_scalers(X[tr_idx],  Y[tr_idx],  x_sc, y_sc)
    X_val_s, Y_val_s = apply_scalers(X[val_idx], Y[val_idx], x_sc, y_sc)

    final_tcfg = {
        **train_cfg,
        "epochs":   train_cfg.get("final_epochs",   train_cfg["epochs"]),
        "patience": train_cfg.get("final_patience", train_cfg["patience"]),
    }

    torch.manual_seed(seed)
    model = M2Model(n_in, n_out, model_cfg["hidden_dims"], model_cfg["dropout"]).to(device)
    model, best_val_mae = train_fold(model, X_tr_s, Y_tr_s, X_val_s, Y_val_s,
                                     final_tcfg, device)
    print(f"  Best dev-val MAE (early-stop criterion) = {best_val_mae:.4f}")
    return model, x_sc, y_sc


# ── Optuna HPO ────────────────────────────────────────────────────────────────

def _sof(trial, key: str, val, log: bool = False) -> float:
    """suggest_or_fixed float — list [min, max] → suggest; scalar → fixed."""
    return trial.suggest_float(key, float(val[0]), float(val[1]), log=log) \
        if isinstance(val, list) else float(val)


def _soi(trial, key: str, val, log: bool = False) -> int:
    """suggest_or_fixed int — list [min, max] → suggest; scalar → fixed."""
    return trial.suggest_int(key, int(val[0]), int(val[1]), log=log) \
        if isinstance(val, list) else int(val)


def suggest_hyperparams(trial, ss: dict, n_in: int):
    """Sample one trial's hyperparams from search_space.
    Returns (model_updates: dict, train_updates: dict).
    Rule: scalar in search_space → fixed; [min, max] → suggested by Optuna.
    Keys absent from search_space are not returned (caller keeps base config value).
    """
    m_upd = {}   # model cfg overrides
    t_upd = {}   # training cfg overrides

    # ── Training scalars ──────────────────────────────────────────────────────
    if "lr" in ss:
        t_upd["lr"] = _sof(trial, "lr", ss["lr"], log=True)
    if "weight_decay" in ss:
        t_upd["weight_decay"] = _sof(trial, "weight_decay", ss["weight_decay"], log=True)
    if "batch_size" in ss:
        t_upd["batch_size"] = _soi(trial, "batch_size", ss["batch_size"])
    if "dropout" in ss:
        m_upd["dropout"] = _sof(trial, "dropout", ss["dropout"])
    if "l1_lambda" in ss:
        t_upd["l1_lambda"] = _sof(trial, "l1_lambda", ss["l1_lambda"], log=True)

    # ── MLP hidden dims ───────────────────────────────────────────────────────
    if "n_layers" in ss:
        n     = _soi(trial, "n_layers", ss["n_layers"])
        h_cfg = ss.get("hidden_size", 64)
        m_upd["hidden_dims"] = [
            (trial.suggest_int(f"h_{i}", int(h_cfg[0]), int(h_cfg[1]), log=True)
             if isinstance(h_cfg, list) else int(h_cfg))
            for i in range(n)
        ]

    return m_upd, t_upd


def optuna_objective(trial, X, Y, optuna_cfg, base_model_cfg, base_train_cfg,
                     f_cols, seed, device, strat_labels=None):
    """Optuna objective: mini K-fold CV; returns mean test mean_MAE (minimise)."""
    ss      = optuna_cfg["search_space"]
    n_folds = optuna_cfg["n_folds"]
    n_in    = X.shape[1]
    n_out   = Y.shape[1]

    m_upd, t_upd = suggest_hyperparams(trial, ss, n_in)
    mcfg = {**base_model_cfg, **m_upd}
    tcfg = {
        **base_train_cfg,
        **t_upd,
        "epochs":      optuna_cfg["epochs"],
        "patience":    optuna_cfg["patience"],
        "print_every": 999999,          # suppress per-epoch printing during HPO
        "k_folds":     n_folds,
    }

    if strat_labels is not None:
        kf         = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        split_iter = list(kf.split(X, strat_labels))
    else:
        kf         = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        split_iter = list(kf.split(X))

    mae_vals = []
    for fold, (tv_idx, test_idx) in enumerate(split_iter):
        if strat_labels is not None:
            sss = StratifiedShuffleSplit(n_splits=1,
                                        test_size=base_train_cfg["val_fraction"],
                                        random_state=seed + fold)
            _tr, _val = next(sss.split(tv_idx, strat_labels[tv_idx]))
            tr_idx, val_idx = tv_idx[_tr], tv_idx[_val]
        else:
            rng     = np.random.default_rng(seed + fold)
            perm    = rng.permutation(len(tv_idx))
            n_val   = max(1, int(len(tv_idx) * base_train_cfg["val_fraction"]))
            val_idx = tv_idx[perm[:n_val]]
            tr_idx  = tv_idx[perm[n_val:]]

        x_sc, y_sc = fit_scalers(X[tr_idx], Y[tr_idx])
        X_tr_s,  Y_tr_s  = apply_scalers(X[tr_idx],   Y[tr_idx],   x_sc, y_sc)
        X_val_s, Y_val_s = apply_scalers(X[val_idx],  Y[val_idx],  x_sc, y_sc)
        X_te_s,  _       = apply_scalers(X[test_idx], Y[test_idx], x_sc, y_sc)

        torch.manual_seed(seed + trial.number * 13 + fold)
        model = M2Model(n_in, n_out, mcfg["hidden_dims"], mcfg["dropout"]).to(device)
        model, _ = train_fold(model, X_tr_s, Y_tr_s, X_val_s, Y_val_s, tcfg, device)

        res = evaluate_model(model, X_te_s, Y[test_idx], y_sc, f_cols, device)
        mae_vals.append(res["mean_MAE"])

        trial.report(float(np.mean(mae_vals)), fold)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return float(np.mean(mae_vals))


def run_hpo(X, Y, optuna_cfg, base_model_cfg, base_train_cfg,
            f_cols, seed, device, strat_labels=None, out_dir=None):
    """Create Optuna study (TPE sampler + HyperbandPruner), run HPO, return study."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    sampler = optuna.samplers.TPESampler(
        n_startup_trials=optuna_cfg["n_startup_trials"], seed=seed)
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=optuna_cfg.get("hyperband_min_resource", 1),
        max_resource=optuna_cfg["n_folds"],
        reduction_factor=optuna_cfg.get("hyperband_reduction_factor", 3),
    )
    storage    = optuna_cfg.get("storage")
    study_name = optuna_cfg.get("study_name", "M2_hpo")
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name=study_name,
        storage=storage,
        load_if_exists=bool(storage),
    )

    n_trials = optuna_cfg["n_trials"]
    print(f"\n{'═' * 60}")
    print(f"Optuna HPO  |  study: {study_name}")
    print(f"  Trials : {n_trials}  (startup RS: {optuna_cfg['n_startup_trials']})")
    print(f"  Folds  : {optuna_cfg['n_folds']}  |  "
          f"Epochs : {optuna_cfg['epochs']}  |  "
          f"Patience : {optuna_cfg['patience']}")
    print(f"  Sampler: TPE     Pruner: Hyperband")
    print(f"{'═' * 60}")

    study.optimize(
        lambda t: optuna_objective(
            t, X, Y, optuna_cfg, base_model_cfg, base_train_cfg,
            f_cols, seed, device, strat_labels),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best = study.best_trial
    print(f"\nBest trial #{best.number}  HPO-mean-MAE = {best.value:.6f}")
    print("Best sampled params:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")

    if out_dir is not None:
        try:
            df_t = study.trials_dataframe(attrs=("number", "value", "params", "state"))
            df_t.to_csv(out_dir / "hpo_trials.csv", index=False)
            print(f"HPO trials saved → {out_dir / 'hpo_trials.csv'}")
        except Exception as e:
            print(f"[warn] Could not save HPO trials CSV: {e}")

    return study


def best_cfg_from_study(study, optuna_cfg: dict,
                        base_model_cfg: dict, base_train_cfg: dict,
                        n_in: int):
    """Reconstruct best (model_cfg, train_cfg) by replaying best trial params."""
    fixed_trial = optuna.trial.FixedTrial(study.best_trial.params)
    m_upd, t_upd = suggest_hyperparams(fixed_trial, optuna_cfg["search_space"], n_in)
    return {**base_model_cfg, **m_upd}, {**base_train_cfg, **t_upd}


# ── Entry point ───────────────────────────────────────────────────────────────

_CFG_MAP = {
    "PP":  "M2_PP_config.json",
    "ABS": "M2_ABS_config.json",
    "ALL": "M2_AllData_config.json",
}


def main():
    parser = argparse.ArgumentParser(
        description="M2 MLP — multi-output regression: process params → M1 f features.")
    parser.add_argument("--material", type=str.upper,
                        choices=["PP", "ABS", "ALL"], required=True,
                        help=("Material subset: PP or ABS (plain KFold) "
                              "or ALL for full dataset (stratified KFold). "
                              "Case-insensitive."))
    args = parser.parse_args()
    mat  = args.material

    cfg_path = BASE_DIR / "config" / _CFG_MAP[mat]
    cfg      = json.loads(cfg_path.read_text())

    data_cfg  = cfg["data"]
    prep_cfg  = cfg["preprocessing"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    out_cfg   = cfg["output"]

    seed   = prep_cfg["random_seed"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"Device   : {device}")
    print(f"Material : {mat}")
    print(f"Config   : {cfg_path}")

    run_best_dir     = BASE_DIR / out_cfg["run_best_dir"]
    overall_best_dir = BASE_DIR / out_cfg["best_overall_dir"]
    out_dir          = BASE_DIR / out_cfg["output_dir"]
    for d in [out_dir, run_best_dir, overall_best_dir]:
        d.mkdir(parents=True, exist_ok=True)

    part_ids, X, Y, pp_cols, f_cols, strat_labels = load_data(data_cfg, prep_cfg)
    n_in, n_out = X.shape[1], Y.shape[1]

    # ── Initial train/test split ──────────────────────────────────────────────
    # Prefer to reuse M1's exact split so both models see the same test set.
    m1_splits_path = (BASE_DIR / data_cfg["features_csv"]).parent / "train_test_split.json"
    if m1_splits_path.exists():
        dev_idx, test_idx = load_m1_train_test_split(m1_splits_path, part_ids)
    else:
        print(f"[warn] M1 train_test_split.json not found at {m1_splits_path} — "
              f"generating a fresh independent split.")
        test_frac = prep_cfg.get("test_fraction", 0.2)
        dev_idx, test_idx = _initial_split(len(part_ids), test_frac, seed, strat_labels)
    dev_strat = strat_labels[dev_idx] if strat_labels is not None else None
    print(f"Dev/test split  →  dev={len(dev_idx)}  test={len(test_idx)}")

    # ── Mode selection / HPO (dev data only) ─────────────────────────────────
    # mode = "optuna"  → Optuna HPO on dev data, then CV on dev, then final train
    # mode = "manual"  → skip HPO; use model config from the "model" section directly
    mode     = cfg.get("mode", "manual").lower()
    best_hpo = None
    print(f"Mode     : {mode}")

    if mode == "optuna":
        if "optuna" not in cfg:
            raise ValueError("mode='optuna' but no 'optuna' section found in config.")
        study = run_hpo(X[dev_idx], Y[dev_idx], cfg["optuna"], model_cfg, train_cfg,
                        f_cols, seed, device, dev_strat, run_best_dir)
        model_cfg, train_cfg = best_cfg_from_study(
            study, cfg["optuna"], model_cfg, train_cfg, n_in)
        best_hpo = {
            "trial":              study.best_trial.number,
            "hpo_mean_mae":       float(study.best_trial.value),
            "model_cfg":          model_cfg,
            "train_lr":           train_cfg["lr"],
            "train_weight_decay": train_cfg["weight_decay"],
            "train_batch_size":   train_cfg["batch_size"],
            "train_l1_lambda":    train_cfg.get("l1_lambda", 0.0),
        }
        (run_best_dir / "hpo_best_config.json").write_text(json.dumps(best_hpo, indent=2))
        print(f"HPO best config saved → {run_best_dir / 'hpo_best_config.json'}")
        print(f"\nFinal CV: {train_cfg['k_folds']} folds, "
              f"{train_cfg['epochs']} epochs, patience={train_cfg['patience']}")
    else:
        print("Using manual model config — skipping HPO.")

    # ── Indicative K-fold CV (dev only) ──────────────────────────────────────
    metrics_df, best_fold_idx, best_fold_mae = run_cv(
        X[dev_idx], Y[dev_idx], model_cfg, train_cfg,
        f_cols, device, seed, dev_strat)

    K = train_cfg["k_folds"]
    metrics_df.index = [f"Fold_{i+1}" for i in range(K)]
    metrics_df.to_csv(run_best_dir / "cv_fold_metrics.csv")

    print_summary(metrics_df, best_fold_idx, best_fold_mae, K)
    save_cv_plots(metrics_df, K, mat, run_best_dir)

    # ── Final training on all dev; evaluation on held-out test ───────────────
    final_model, x_sc, y_sc = final_train_m2(
        X, Y, dev_idx, model_cfg, train_cfg, seed, device, dev_strat)

    X_te_s = x_sc.transform(X[test_idx]).astype(np.float32)
    final_metrics = evaluate_model(final_model, X_te_s, Y[test_idx], y_sc, f_cols, device)
    print(f"\n{'═' * 60}")
    print(f"Final Test  mean_MAE={final_metrics['mean_MAE']:.4f}  "
          f"mean_RMSE={final_metrics['mean_RMSE']:.4f}  "
          f"mean_R²={final_metrics['mean_R2']:.4f}  "
          f"mean_MSE={final_metrics['mean_MSE']:.4f}")
    for fc, m in final_metrics["per_target"].items():
        print(f"  {fc}  MAE={m['MAE']:.4f}  R²={m['R2']:.4f}")
    print(f"{'═' * 60}")

    save_final_test_plots(Y[test_idx], final_metrics["pred"], final_metrics,
                          run_best_dir, mat, f_cols)

    overall_updated = save_models(
        final_model, final_metrics,
        run_best_dir, overall_best_dir,
        n_in, n_out, pp_cols, f_cols, model_cfg, data_cfg["id_col"])

    df_pred = save_predictions(
        final_model, X, x_sc, y_sc, part_ids, f_cols,
        run_best_dir / "f_predictions.csv", data_cfg["id_col"], device)

    if overall_updated:
        save_final_test_plots(Y[test_idx], final_metrics["pred"], final_metrics,
                              overall_best_dir, mat, f_cols)
        save_cv_plots(metrics_df, K, mat, overall_best_dir)
        metrics_df.to_csv(overall_best_dir / "cv_fold_metrics.csv")
        df_pred.to_csv(overall_best_dir / "f_predictions.csv")
        if mode == "optuna" and best_hpo is not None:
            (overall_best_dir / "hpo_best_config.json").write_text(
                json.dumps(best_hpo, indent=2))
            src_trials = run_best_dir / "hpo_trials.csv"
            if src_trials.exists():
                shutil.copy2(src_trials, overall_best_dir / "hpo_trials.csv")
        print(f"All run-best artefacts also saved to overall best → {overall_best_dir}")


if __name__ == "__main__":
    main()
