"""
M1_PPPT_to_W.py
=========================
Dual-input neural network for predicting injection-moulded part weight (single material only).

Usage
-----
  python src/M1_PPPT_to_W.py --material PP
  python src/M1_PPPT_to_W.py --material ABS
  python src/M1_PPPT_to_W.py --material ALL

Parameters
----------
  --material : str, required (case-insensitive)
               PP | ABS  →  single-material subset, plain KFold CV
               ALL       →  full dataset, StratifiedKFold CV (material-balanced splits)
               Config loaded: M1_PP_config.json / M1_ABS_config.json / M1_AllData_config.json

Architecture
------------
  p(t) branch : Conv1D stack → GlobalAvgPool → Linear → n_f features  (f)
  pp   branch : MLP  (process parameters)    → n_f features
  merge       : concat(f, pp_out) → MLP → 1  (predicted weight)

Training
--------
  • 5-fold cross-validation on the full cleaned dataset
  • AdamW optimiser + MAE loss + early stopping (best fold selected by lowest MAE)
  • MinMax scaling fitted on each fold's training split only

Config files
------------
  config/ProBayes/M1_PP_config.json   (for --material PP)
  config/ProBayes/M1_ABS_config.json  (for --material ABS)
  config/ProBayes/M1_DoE1_config.json  (for --material DoE1)
  config/DoE1/M1_AllData_config.json   (for --material ALL)

Outputs
-------
  outputs/[dataset]/M1/[material]/plots/           scatter & metrics plots
  outputs/[dataset]/M1/[material]/run_best/        best model of this run (always written)
  outputs/[dataset]/M1/[material]/best_overall/    overall best model across all runs (updated when MAE improves)
  outputs/[dataset]/M1/[material]/pressure_features_f.csv   encoder features for every part
"""

import argparse
from datetime import datetime
import copy, json, math, shutil, warnings
import optuna
import numpy as np
import pandas as pd
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
    "ig":         "#7D54A4",   # purple       — IG / Expected Gradients bars
    "indirect":   "#D4691C",   # orange       — Fusion indirect path (via M2)
    "direct":     "#2166AC",   # blue         — Fusion direct path (via PPMLP)
    "total":      "#1B7837",   # green        — total combined attribution
    "f_feat":     "#E69F00",   # gold         — M1-merge f encoder features
    "pp_feat":    "#2166AC",   # blue         — M1-merge pp_direct features
    "pressure":   "#2166AC",   # blue         — GradCAM pressure curve
    "gradcam":    "#B2182B",   # red          — GradCAM activation overlay
}


def _strip_prefix(name: str) -> str:
    for pfx in ("DXP_", "QUA_", "TCE_", "TCN_", "SCA_", "MSS_", "IHR_", "SPE_"):
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
_RUN_TS  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _scaler_to_dict(sc: MinMaxScaler) -> dict:
    """Serialise a fitted MinMaxScaler to a JSON-safe dict."""
    return {
        "feature_range": list(sc.feature_range),
        "data_min_":     sc.data_min_.tolist(),
        "data_max_":     sc.data_max_.tolist(),
        "scale_":        sc.scale_.tolist(),
        "min_":          sc.min_.tolist(),
    }


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

    def __init__(self, channels: list, kernels: list, n_f: int, pool_kernels: list = None):
        super().__init__()
        n_conv = len(channels) - 1           # number of conv layers
        _pk    = pool_kernels or []          # length n_conv-1: one pool per layer except the last
        conv_layers = []
        for i in range(n_conv):
            conv_layers += [
                nn.Conv1d(channels[i], channels[i + 1],
                          kernel_size=kernels[i],
                          padding=kernels[i] // 2),   # "same" padding
                nn.ReLU(),
            ]
            # pool after every conv except the last (AdaptiveAvgPool1d follows the last)
            if i < n_conv - 1 and i < len(_pk) and _pk[i] > 1:
                conv_layers.append(nn.MaxPool1d(kernel_size=_pk[i], stride=_pk[i]))
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
      forward(x_pt, x_pp)  → predicted weight (B,)
      get_f(x_pt)          → encoder features (B, n_f)   [no_grad]
    """

    def __init__(self, n_pp: int, mcfg: dict):
        super().__init__()
        curve_cfgs = mcfg.get("curve_encoders", [])
        if curve_cfgs:
            self.encoders = nn.ModuleList([
                PressureEncoder(
                    channels=ccfg["conv_channels"],
                    kernels=ccfg["conv_kernels"],
                    n_f=ccfg["n_f_features"],
                    pool_kernels=ccfg.get("enc_pool_kernels", []),
                )
                for ccfg in curve_cfgs
            ])
            self.n_f_total = int(sum(int(ccfg["n_f_features"]) for ccfg in curve_cfgs))
        else:
            self.encoders = nn.ModuleList([
                PressureEncoder(
                    channels=mcfg["conv_channels"],
                    kernels=mcfg["conv_kernels"],
                    n_f=mcfg["n_f_features"],
                    pool_kernels=mcfg.get("enc_pool_kernels", []),
                )
            ])
            self.n_f_total = int(mcfg["n_f_features"])
        self.n_curves = len(self.encoders)
        self.pp_mlp  = PPMLP(
            input_dim=n_pp,
            hidden_dims=mcfg["pp_hidden"],
            dropout=mcfg["dropout"],
        )
        merge_in     = self.n_f_total + mcfg["pp_hidden"][-1]
        self.merge   = MergeHead(
            input_dim=merge_in,
            hidden_dims=mcfg["merge_hidden"],
            dropout=mcfg["dropout"],
        )

    def _ensure_curve_list(self, x_pt):
        if isinstance(x_pt, (list, tuple)):
            return list(x_pt)
        return [x_pt]

    def forward(self, x_pt, x_pp: torch.Tensor) -> torch.Tensor:
        x_curves = self._ensure_curve_list(x_pt)
        if len(x_curves) != self.n_curves:
            raise ValueError(
                f"Expected {self.n_curves} pressure curve tensor(s), got {len(x_curves)}")
        f_parts = [enc(x_curves[i]) for i, enc in enumerate(self.encoders)]
        f   = torch.cat(f_parts, dim=1)
        pp  = self.pp_mlp(x_pp)
        return self.merge(torch.cat([f, pp], dim=1))

    @torch.no_grad()
    def get_f(self, x_pt) -> torch.Tensor:
        self.eval()
        x_curves = self._ensure_curve_list(x_pt)
        if len(x_curves) != self.n_curves:
            raise ValueError(
                f"Expected {self.n_curves} pressure curve tensor(s), got {len(x_curves)}")
        return torch.cat([enc(x_curves[i]) for i, enc in enumerate(self.encoders)], dim=1)


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

def fit_scalers(X_pp_tr: np.ndarray, X_pt_tr_list: list[np.ndarray]):
    """Fit MinMax scalers on training data only."""
    pp_sc = MinMaxScaler()
    pp_sc.fit(X_pp_tr)
    pt_scalers = []
    for X_pt_tr in X_pt_tr_list:
        pt_scalers.append({
            "min": float(X_pt_tr.min()),
            "max": float(X_pt_tr.max()),
        })
    return pp_sc, pt_scalers


def apply_scalers(X_pp: np.ndarray, X_pt_list: list[np.ndarray],
                  pp_sc: MinMaxScaler, pt_scalers: list[dict]):
    """Apply pre-fitted scalers."""
    X_pp_s = pp_sc.transform(X_pp).astype(np.float32)
    X_pt_scaled = []
    for X_pt, s in zip(X_pt_list, pt_scalers):
        X_pt_scaled.append(
            ((X_pt - s["min"]) / (s["max"] - s["min"] + 1e-8)).astype(np.float32)
        )
    return X_pp_s, X_pt_scaled


def to_tensors(X_pp: np.ndarray, X_pt_list: list[np.ndarray],
               y_arr: np.ndarray, device: torch.device):
    """Convert numpy arrays to PyTorch tensors on the target device."""
    t_pp = torch.tensor(X_pp,              dtype=torch.float32, device=device)
    t_pt = [
        torch.tensor(X_pt[:, None, :], dtype=torch.float32, device=device)
        for X_pt in X_pt_list
    ]
    t_y  = torch.tensor(y_arr,             dtype=torch.float32, device=device)
    return t_pp, t_pt, t_y


# ── Training & evaluation ─────────────────────────────────────────────────────

def train_fold(model: M1Model,
               Xpp_tr, Xpt_tr_list, y_tr,
               Xpp_val, Xpt_val_list, y_val,
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

    pp_tr,  pt_tr_list,  ty_tr  = to_tensors(Xpp_tr,  Xpt_tr_list,  y_tr,  device)
    pp_val, pt_val_list, ty_val = to_tensors(Xpp_val, Xpt_val_list, y_val, device)

    loader = DataLoader(TensorDataset(pp_tr, *pt_tr_list, ty_tr),
                        batch_size=tcfg["batch_size"], shuffle=True,
                        drop_last=False)

    best_val_mae = float("inf")
    best_state   = copy.deepcopy(model.state_dict())
    wait         = 0

    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        for batch in loader:
            b_pp = batch[0]
            b_pt_list = list(batch[1:-1])
            b_y  = batch[-1]
            optimizer.zero_grad()
            loss = criterion(model(b_pt_list, b_pp), b_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(b_y)
        epoch_loss /= len(y_tr)

        model.eval()
        with torch.no_grad():
            val_mae = criterion(model(pt_val_list, pp_val), ty_val).item()
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
                   Xpp: np.ndarray, Xpt_list: list[np.ndarray],
                   y_arr: np.ndarray, device: torch.device) -> dict:
    """Return MAE / MSE / RMSE / R² and raw predictions."""
    pp_t, pt_t_list, _ = to_tensors(Xpp, Xpt_list, y_arr, device)
    model.eval()
    with torch.no_grad():
        pred = model(pt_t_list, pp_t).cpu().numpy()
    mae  = float(mean_absolute_error(y_arr, pred))
    mse  = float(mean_squared_error(y_arr, pred))
    rmse = math.sqrt(mse)
    r2   = float(r2_score(y_arr, pred))
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2, "pred": pred}


# ── Pipeline functions ────────────────────────────────────────────────────────

def load_data(data_cfg: dict, prep_cfg: dict):
    """Load CSVs, join, impute, remove outliers.

    Returns
    -------
    part_ids, X_pp, X_pt_list, y, strat_labels
        strat_labels is None for single-material configs;
        encoded integer material array when 'material_col' is present in data_cfg.
    """
    df_scalar   = pd.read_csv(BASE_DIR / data_cfg["scalar_csv"],
                               index_col=data_cfg["id_col"])
    pressure_sources = data_cfg.get("pressure_csvs")
    if pressure_sources is None:
        pressure_sources = [{"name": "pressure", "path": data_cfg["pressure_csv"]}]

    pressure_frames = []
    pressure_cols_by_curve = []
    for i, src in enumerate(pressure_sources):
        curve_name = src.get("name", f"curve_{i + 1}")
        curve_path = src.get("path")
        if curve_path is None:
            raise ValueError("Each entry in data.pressure_csvs must define a 'path'.")
        df_curve = pd.read_csv(BASE_DIR / curve_path, index_col=data_cfg["id_col"])
        renamed_cols = [f"{curve_name}__{c}" for c in df_curve.columns]
        df_curve.columns = renamed_cols
        pressure_frames.append(df_curve)
        pressure_cols_by_curve.append((curve_name, renamed_cols))

    df_scalar = df_scalar.drop(columns=data_cfg.get("drop_cols", []), errors="ignore")
    df        = df_scalar.copy()
    for df_curve in pressure_frames:
        df = df.join(df_curve, how="inner")
    print(f"Joined : {df.shape[0]} parts × {df.shape[1]} columns")

    # Encode material column (full-dataset / ALL mode only)
    mat_col = data_cfg.get("material_col")
    if mat_col:
        materials = sorted(df[mat_col].dropna().unique())
        mat_map   = {m: i for i, m in enumerate(materials)}
        print(f"Material encoding : {mat_map}")
        df[mat_col] = df[mat_col].map(mat_map)

    target_col    = data_cfg["target_col"]
    pp_cols       = [c for c in df_scalar.columns if c != target_col]

    for col in pp_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    remove_target_outliers = bool(prep_cfg.get("remove_target_outliers", True))
    y_removed_ids = []
    if remove_target_outliers:
        iqr_k  = prep_cfg["iqr_multiplier"]
        col    = target_col
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr    = q3 - q1
        if iqr > 0:
            y_mask = (df[col] >= q1 - iqr_k * iqr) & (df[col] <= q3 + iqr_k * iqr)
        else:
            y_mask = pd.Series(True, index=df.index)
        y_removed_ids = df.index[~y_mask].tolist()
        n_removed = int((~y_mask).sum())
        df = df[y_mask].copy()
        print(f"y-outliers removed: {n_removed}  →  {len(df)} parts remaining"
              + (f"  (IDs: {y_removed_ids})" if y_removed_ids else ""))
    else:
        print("y-outlier removal disabled by config (preprocessing.remove_target_outliers=false)")

    part_ids     = df.index.to_numpy()
    X_pp         = df[pp_cols].to_numpy(dtype=np.float32)
    X_pt_list    = [df[cols].to_numpy(dtype=np.float32) for _, cols in pressure_cols_by_curve]
    y            = df[target_col].to_numpy(dtype=np.float32)
    strat_labels = df[mat_col].to_numpy() if mat_col else None
    curve_dims = ", ".join([f"{name}={arr.shape[1]}" for (name, _), arr in zip(pressure_cols_by_curve, X_pt_list)])
    print(f"pp features : {X_pp.shape[1]}   |   curves : {len(X_pt_list)} ({curve_dims})   |   samples : {len(y)}")
    print(f"Target      : mean={y.mean():.3f}  std={y.std():.3f}  "
          f"min={y.min():.3f}  max={y.max():.3f}")
    return part_ids, X_pp, X_pt_list, y, strat_labels, pp_cols, y_removed_ids


def run_cv(X_pp, X_pt_list, y, model_cfg: dict, train_cfg: dict,
           device: torch.device, seed: int, strat_labels=None):
    """Run K-fold CV on dev data (indicative); return metrics DataFrame.

    All data passed in must already be the dev subset — test data must
    never be included.  The returned metrics are indicative only; the
    primary evaluation is performed by final_train_m1() on the held-out test set.
    """
    K    = train_cfg["k_folds"]
    n_pp = X_pp.shape[1]

    if strat_labels is not None:
        kf         = StratifiedKFold(n_splits=K, shuffle=True, random_state=seed)
        split_iter = kf.split(X_pp, strat_labels)
    else:
        kf         = KFold(n_splits=K, shuffle=True, random_state=seed)
        split_iter = kf.split(X_pp)

    _dummy = M1Model(n_pp, model_cfg)
    print(f"\nModel parameters : {_count_params(_dummy):,}")
    del _dummy

    fold_results  = []
    best_fold_mae = float("inf")
    best_fold_idx = -1

    for fold, (tv_idx, test_idx) in enumerate(split_iter):
        print(f"\n══ CV Fold {fold + 1}/{K} ════════════════════════════════════════")

        if strat_labels is not None:
            sss = StratifiedShuffleSplit(n_splits=1,
                                        test_size=train_cfg["val_fraction"],
                                        random_state=seed + fold)
            _tr_rel, _val_rel = next(sss.split(tv_idx, strat_labels[tv_idx]))
            tr_idx  = tv_idx[_tr_rel]
            val_idx = tv_idx[_val_rel]
        else:
            rng     = np.random.default_rng(seed + fold)
            perm    = rng.permutation(len(tv_idx))
            n_val   = max(1, int(len(tv_idx) * train_cfg["val_fraction"]))
            val_idx = tv_idx[perm[:n_val]]
            tr_idx  = tv_idx[perm[n_val:]]
        print(f"  Train={len(tr_idx)}  Val={len(val_idx)}  Test={len(test_idx)}")

        X_pt_tr_list = [x[tr_idx] for x in X_pt_list]
        X_pt_val_list = [x[val_idx] for x in X_pt_list]
        X_pt_te_list = [x[test_idx] for x in X_pt_list]

        pp_sc, pt_scalers = fit_scalers(X_pp[tr_idx], X_pt_tr_list)
        Xpp_tr,  Xpt_tr  = apply_scalers(X_pp[tr_idx],   X_pt_tr_list,  pp_sc, pt_scalers)
        Xpp_val, Xpt_val = apply_scalers(X_pp[val_idx],  X_pt_val_list, pp_sc, pt_scalers)
        Xpp_te,  Xpt_te  = apply_scalers(X_pp[test_idx], X_pt_te_list,  pp_sc, pt_scalers)

        torch.manual_seed(seed + fold)
        model = M1Model(n_pp, model_cfg).to(device)
        model, _ = train_fold(model, Xpp_tr, Xpt_tr, y[tr_idx],
                              Xpp_val, Xpt_val, y[val_idx], train_cfg, device)

        res = evaluate_model(model, Xpp_te, Xpt_te, y[test_idx], device)
        fold_results.append({k: v for k, v in res.items() if k != "pred"})
        print(f"  Test  MAE={res['MAE']:.4f}  RMSE={res['RMSE']:.4f}  "
              f"R²={res['R2']:.4f}  MSE={res['MSE']:.4f}")

        if res["MAE"] < best_fold_mae:
            best_fold_mae = res["MAE"]
            best_fold_idx = fold

    return pd.DataFrame(fold_results), best_fold_idx, best_fold_mae


def print_summary(metrics_df: pd.DataFrame, best_fold_idx: int,
                  best_fold_mae: float, K: int):
    print("\n" + "═" * 60)
    print(f"[Indicative] {K}-fold CV on dev  "
          f"(best fold: {best_fold_idx + 1}  MAE={best_fold_mae:.4f})")
    print("─" * 60)
    for m in ["MAE", "RMSE", "R2", "MSE"]:
        print(f"  {m:6s}  {metrics_df[m].mean():.4f} ± {metrics_df[m].std():.4f}")
    print("═" * 60)


def save_cv_plots(metrics_df: pd.DataFrame, K: int, material: str, plots_dir: Path):
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
    fig.suptitle(f"[{material}] Indicative {K}-fold CV on dev set", fontsize=13)
    fig.tight_layout()
    fig.savefig(plots_dir / "cv_metrics_folds.png")
    plt.close(fig)
    print(f"CV plots saved → {plots_dir}")


def save_final_test_plots(y_test: np.ndarray, preds_test: np.ndarray,
                          final_metrics: dict, plots_dir: Path, material: str):
    """Scatter + residual plots from the final model on the held-out test set (primary output)."""
    # Scatter: measured vs predicted
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, preds_test, color=_PLT_C["scatter"],
               s=45, alpha=0.8, edgecolors="white", linewidths=0.5, zorder=3)
    lo = min(y_test.min(), preds_test.min())
    hi = max(y_test.max(), preds_test.max())
    ax.plot([lo, hi], [lo, hi], color=_PLT_C["ideal_line"], linestyle="--",
            linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Measured weight [g]")
    ax.set_ylabel("Predicted weight [g]")
    ax.set_title(
        f"[{material}] Final Model — Held-out Test Set\n"
        f"MAE={final_metrics['MAE']:.4f} g   R²={final_metrics['R2']:.4f}",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "scatter_final_test.png")
    plt.close(fig)

    # Residuals: scatter vs predicted
    residuals = preds_test - y_test
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(preds_test, residuals, color=_PLT_C["residual"],
               s=45, alpha=0.8, edgecolors="white", linewidths=0.5, zorder=3)
    ax.axhline(0, color=_PLT_C["zero_line"], linestyle="--", linewidth=1.5)
    ax.set_xlabel("Predicted weight [g]")
    ax.set_ylabel("Residual (pred − real) [g]")
    ax.set_title(f"[{material}] Residuals vs Predicted (final test)")
    fig.tight_layout()
    fig.savefig(plots_dir / "residuals_scatter_final_test.png")
    plt.close(fig)

    # Residuals: histogram
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hist(residuals, bins=20, color=_PLT_C["residual"], edgecolor="white", alpha=0.85)
    ax.axvline(0, color=_PLT_C["zero_line"], linestyle="--", linewidth=1.5)
    ax.axvline(float(residuals.mean()), color=_PLT_C["mean_line"], linestyle="-",
               linewidth=1.5, label=f"Mean={residuals.mean():.4f}")
    ax.set_xlabel("Residual (pred − real) [g]")
    ax.set_ylabel("Count")
    ax.set_title(f"[{material}] Residuals Distribution (final test)\n"
                 f"std={residuals.std():.4f} g")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "residuals_hist_final_test.png")
    plt.close(fig)
    print(f"Final test plots saved → {plots_dir}")


def save_models(final_model: M1Model, final_metrics: dict,
                n_pp: int, T_list: list[int], model_cfg: dict,
                run_best_dir: Path, overall_best_dir: Path) -> bool:
    """Save final model weights + test metrics; update overall_best if improved.
    best_metrics.json always contains the held-out test set metrics (primary).
    """
    run_info = {
        "MAE":      final_metrics["MAE"],
        "RMSE":     final_metrics["RMSE"],
        "R2":       final_metrics["R2"],
        "MSE":      final_metrics["MSE"],
        "source":   "final_test",
        "n_pp":     n_pp,
        "T_list":   T_list,
        "model_cfg": model_cfg,
    }
    torch.save(final_model.state_dict(), run_best_dir / "best_model.pt")
    (run_best_dir / "best_metrics.json").write_text(json.dumps(run_info, indent=2))
    print(f"\nFinal model saved  (MAE={final_metrics['MAE']:.4f})  → {run_best_dir}")

    overall_path = overall_best_dir / "best_metrics.json"
    prev_mae     = float("inf")
    if overall_path.exists():
        prev_mae = json.loads(overall_path.read_text()).get("MAE", float("inf"))

    if final_metrics["MAE"] < prev_mae:
        torch.save(final_model.state_dict(), overall_best_dir / "best_model.pt")
        overall_path.write_text(json.dumps(run_info, indent=2))
        print(f"Overall best updated  "
              f"({prev_mae:.4f} → {final_metrics['MAE']:.4f} MAE)  → {overall_best_dir}")
        return True
    else:
        print(f"Overall best unchanged  "
              f"(saved MAE={prev_mae:.4f},  this run MAE={final_metrics['MAE']:.4f})")
        return False


def extract_features(best_model_cv, best_scalers, X_pp, X_pt,
                     part_ids, feat_csv: Path, id_col: str,
                     device: torch.device):
    """Run best-fold encoder over the full cleaned dataset; save features CSV."""
    pp_sc, pt_scalers = best_scalers
    _, X_pt_sc_list = apply_scalers(X_pp, X_pt, pp_sc, pt_scalers)
    pt_t_list = [
        torch.tensor(X_pt_sc[:, None, :], dtype=torch.float32, device=device)
        for X_pt_sc in X_pt_sc_list
    ]
    f_features = best_model_cv.get_f(pt_t_list).cpu().numpy()
    f_cols     = [f"f_{i}" for i in range(f_features.shape[1])]
    df_f       = pd.DataFrame(f_features, columns=f_cols, index=part_ids)
    df_f.index.name = id_col
    df_f.to_csv(feat_csv)
    print(f"Pressure features f saved → {feat_csv}  "
          f"({df_f.shape[0]} parts × {df_f.shape[1]} features)")
    return df_f


def final_train_m1(X_pp, X_pt_list, y, dev_idx: np.ndarray,
                   model_cfg: dict, train_cfg: dict,
                   seed: int, device: torch.device, strat_labels=None):
    """Train the final M1Model on all dev data.
    A small internal validation split (val_fraction × dev) is used solely
    for early stopping — it is never reported.  Final evaluation is always
    performed by the caller on the held-out test set.
    Uses final_epochs / final_patience from train_cfg when present.
    Returns (model, (pp_sc, pt_scalers)).
    """
    n_pp = X_pp.shape[1]
    if strat_labels is not None:
        sss = StratifiedShuffleSplit(n_splits=1,
                                     test_size=train_cfg["val_fraction"],
                                     random_state=seed)
        rel = np.arange(len(dev_idx))
        _tr_rel, _val_rel = next(sss.split(rel, strat_labels))
        tr_idx  = dev_idx[_tr_rel]
        val_idx = dev_idx[_val_rel]
    else:
        rng     = np.random.default_rng(seed)
        perm    = rng.permutation(len(dev_idx))
        n_val   = max(1, int(len(dev_idx) * train_cfg["val_fraction"]))
        val_idx = dev_idx[perm[:n_val]]
        tr_idx  = dev_idx[perm[n_val:]]

    print(f"\n══ Final Training ═══════════════════════════════════════════")
    print(f"  Dev Train={len(tr_idx)}  Dev Val={len(val_idx)}  "
          f"(val split for early-stop only)")

    X_pt_tr_list = [x[tr_idx] for x in X_pt_list]
    X_pt_val_list = [x[val_idx] for x in X_pt_list]
    pp_sc, pt_scalers = fit_scalers(X_pp[tr_idx], X_pt_tr_list)
    Xpp_tr,  Xpt_tr  = apply_scalers(X_pp[tr_idx],  X_pt_tr_list,  pp_sc, pt_scalers)
    Xpp_val, Xpt_val = apply_scalers(X_pp[val_idx], X_pt_val_list, pp_sc, pt_scalers)

    final_tcfg = {
        **train_cfg,
        "epochs":   train_cfg.get("final_epochs",   train_cfg["epochs"]),
        "patience": train_cfg.get("final_patience", train_cfg["patience"]),
    }

    torch.manual_seed(seed)
    model = M1Model(n_pp, model_cfg).to(device)
    model, best_val_mae = train_fold(
        model, Xpp_tr, Xpt_tr, y[tr_idx],
        Xpp_val, Xpt_val, y[val_idx], final_tcfg, device)
    print(f"  Best dev-val MAE (early-stop criterion) = {best_val_mae:.4f}")
    return model, (pp_sc, pt_scalers), tr_idx, val_idx


# ── Optuna HPO ────────────────────────────────────────────────────────────────

def _sof(trial, key: str, val, log: bool = False) -> float:
    """suggest_or_fixed float — list [min, max] → suggest; scalar → fixed."""
    return trial.suggest_float(key, float(val[0]), float(val[1]), log=log) \
        if isinstance(val, list) else float(val)


def _soi(trial, key: str, val) -> int:
    """suggest_or_fixed int — list [min, max] → suggest; scalar → fixed."""
    return trial.suggest_int(key, int(val[0]), int(val[1])) \
        if isinstance(val, list) else int(val)


def suggest_hyperparams(trial, ss: dict, n_pp: int,
                        base_model_cfg: dict | None = None):
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

    # ── Conv encoder ──────────────────────────────────────────────────────────
    # Multi-curve models use the same configurable ranges for every branch,
    # but each branch gets its own Optuna parameter names and therefore its
    # own independent samples.  PP/ABS/ALL retain the legacy single-encoder
    # parameter names and behaviour.
    has_curve_encoders = bool(base_model_cfg and base_model_cfg.get("curve_encoders"))
    encoder_keys = {
        "enc_n_conv_layers", "enc_channels", "enc_kernel_size",
        "enc_pool_kernel_size", "n_f_features",
    }
    if has_curve_encoders and encoder_keys.intersection(ss):
        curve_updates = []
        for curve_idx, base_curve in enumerate(base_model_cfg["curve_encoders"]):
            curve_cfg = copy.deepcopy(base_curve)
            enc_prefix = f"enc_c{curve_idx}"
            base_channels = curve_cfg.get("conv_channels", [1, 16])
            base_kernels = curve_cfg.get("conv_kernels", [5])
            base_pools = curve_cfg.get("enc_pool_kernels", [])

            enc_n_key = f"{enc_prefix}_n_conv_layers"
            enc_n = (_soi(trial, enc_n_key, ss["enc_n_conv_layers"])
                     if "enc_n_conv_layers" in ss
                     else len(base_channels) - 1)
            conv_channels = [1]
            conv_kernels = []
            pool_kernels = []
            for i in range(enc_n):
                ch_key = f"{enc_prefix}_ch_{i}"
                k_key = f"{enc_prefix}_k_{i}"
                ch = (_soi(trial, ch_key, ss["enc_channels"])
                      if "enc_channels" in ss
                      else (base_channels[i + 1] if i + 1 < len(base_channels) else 16))
                kernel = (_soi(trial, k_key, ss["enc_kernel_size"])
                          if "enc_kernel_size" in ss
                          else (base_kernels[i] if i < len(base_kernels) else 5))
                conv_channels.append(ch)
                conv_kernels.append(kernel)

                if i < enc_n - 1:
                    pool_key = f"{enc_prefix}_pk_{i}"
                    pool = (_soi(trial, pool_key, ss["enc_pool_kernel_size"])
                            if "enc_pool_kernel_size" in ss
                            else (base_pools[i] if i < len(base_pools) else 1))
                    pool_kernels.append(pool)

            curve_cfg["conv_channels"] = conv_channels
            curve_cfg["conv_kernels"] = conv_kernels
            curve_cfg["enc_pool_kernels"] = pool_kernels
            if "n_f_features" in ss:
                curve_cfg["n_f_features"] = _soi(
                    trial, f"{enc_prefix}_n_f_features", ss["n_f_features"])
            curve_updates.append(curve_cfg)
        m_upd["curve_encoders"] = curve_updates

    elif "enc_n_conv_layers" in ss:
        enc_n  = _soi(trial, "enc_n_conv_layers", ss["enc_n_conv_layers"])
        ch_cfg = ss.get("enc_channels", 16)
        k_cfg  = ss.get("enc_kernel_size", 5)
        pk_cfg = ss.get("enc_pool_kernel_size", 1)
        conv_channels = [1]
        conv_kernels  = []
        pool_kernels  = []
        for i in range(enc_n):
            ch = (trial.suggest_int(f"enc_ch_{i}", int(ch_cfg[0]), int(ch_cfg[1]))
                  if isinstance(ch_cfg, list) else int(ch_cfg))
            k  = (trial.suggest_int(f"enc_k_{i}",  int(k_cfg[0]),  int(k_cfg[1]))
                  if isinstance(k_cfg,  list) else int(k_cfg))
            conv_channels.append(ch)
            conv_kernels.append(k)
            # pool after every conv except the last
            if i < enc_n - 1:
                pk = (trial.suggest_int(f"enc_pk_{i}", int(pk_cfg[0]), int(pk_cfg[1]))
                      if isinstance(pk_cfg, list) else int(pk_cfg))
                pool_kernels.append(pk)
        m_upd["conv_channels"]    = conv_channels
        m_upd["conv_kernels"]     = conv_kernels
        m_upd["enc_pool_kernels"] = pool_kernels

    if not has_curve_encoders and "n_f_features" in ss:
        m_upd["n_f_features"] = _soi(trial, "n_f_features", ss["n_f_features"])

    # ── PP MLP ────────────────────────────────────────────────────────────────
    if "pp_n_layers" in ss:
        pp_n  = _soi(trial, "pp_n_layers", ss["pp_n_layers"])
        h_cfg = ss.get("pp_hidden_size", 16)
        m_upd["pp_hidden"] = [
            (trial.suggest_int(f"pp_h_{i}", int(h_cfg[0]), int(h_cfg[1]), log=True)
             if isinstance(h_cfg, list) else int(h_cfg))
            for i in range(pp_n)
        ]

    # ── Merge MLP ─────────────────────────────────────────────────────────────
    if "merge_n_layers" in ss:
        m_n   = _soi(trial, "merge_n_layers", ss["merge_n_layers"])
        h_cfg = ss.get("merge_hidden_size", 16)
        m_upd["merge_hidden"] = [
            (trial.suggest_int(f"merge_h_{i}", int(h_cfg[0]), int(h_cfg[1]), log=True)
             if isinstance(h_cfg, list) else int(h_cfg))
            for i in range(m_n)
        ]

    return m_upd, t_upd


def optuna_objective(trial, X_pp, X_pt_list, y,
                     optuna_cfg, base_model_cfg, base_train_cfg,
                     seed, device, strat_labels=None):
    """Optuna objective: mini K-fold CV, returns mean test MAE (minimize)."""
    ss      = optuna_cfg["search_space"]
    n_folds = optuna_cfg["n_folds"]
    n_pp    = X_pp.shape[1]

    m_upd, t_upd = suggest_hyperparams(trial, ss, n_pp, base_model_cfg)
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
        split_iter = list(kf.split(X_pp, strat_labels))
    else:
        kf         = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        split_iter = list(kf.split(X_pp))

    mae_vals = []
    for fold, (tv_idx, test_idx) in enumerate(split_iter):
        # Inner validation split (mirrors run_cv logic)
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

        X_pt_tr_list = [x[tr_idx] for x in X_pt_list]
        X_pt_val_list = [x[val_idx] for x in X_pt_list]
        X_pt_te_list = [x[test_idx] for x in X_pt_list]

        pp_sc, pt_scalers = fit_scalers(X_pp[tr_idx], X_pt_tr_list)
        Xpp_tr,  Xpt_tr  = apply_scalers(X_pp[tr_idx],   X_pt_tr_list,  pp_sc, pt_scalers)
        Xpp_val, Xpt_val = apply_scalers(X_pp[val_idx],  X_pt_val_list, pp_sc, pt_scalers)
        Xpp_te,  Xpt_te  = apply_scalers(X_pp[test_idx], X_pt_te_list,  pp_sc, pt_scalers)

        torch.manual_seed(seed + trial.number * 13 + fold)
        model = M1Model(n_pp, mcfg).to(device)
        model, _ = train_fold(model, Xpp_tr, Xpt_tr, y[tr_idx],
                              Xpp_val, Xpt_val, y[val_idx], tcfg, device)

        res = evaluate_model(model, Xpp_te, Xpt_te, y[test_idx], device)
        mae_vals.append(res["MAE"])

        trial.report(float(np.mean(mae_vals)), fold)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return float(np.mean(mae_vals))


def run_hpo(X_pp, X_pt_list, y, optuna_cfg, base_model_cfg, base_train_cfg,
            seed, device, strat_labels=None, out_dir=None):
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
    study_name = optuna_cfg.get("study_name", "M1_hpo")
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
            t, X_pp, X_pt_list, y, optuna_cfg, base_model_cfg, base_train_cfg,
            seed, device, strat_labels),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best = study.best_trial
    print(f"\nBest trial #{best.number}  HPO-MAE = {best.value:.6f}")
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
                        n_pp: int):
    """Reconstruct best (model_cfg, train_cfg) by replaying best trial params."""
    fixed_trial = optuna.trial.FixedTrial(study.best_trial.params)
    m_upd, t_upd = suggest_hyperparams(
        fixed_trial, optuna_cfg["search_space"], n_pp, base_model_cfg)
    return {**base_model_cfg, **m_upd}, {**base_train_cfg, **t_upd}


# ── Entry point ───────────────────────────────────────────────────────────────

_CFG_MAP = {
    "PP":  "ProBayes/M1_PP_config.json",
    "ABS": "ProBayes/M1_ABS_config.json",
    "ALL": "ProBayes/M1_AllData_config.json",
    "DOE1": "DoE1/M1_DoE1_config.json",
}


def main():
    parser = argparse.ArgumentParser(
        description="M1 dual-input NN — injection-moulded part weight prediction.")
    parser.add_argument("--material", type=str.upper,
                        choices=["PP", "ABS", "ALL", "DOE1"], required=True,
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

    out_dir          = BASE_DIR / out_cfg["output_dir"]
    run_best_dir     = BASE_DIR / out_cfg["run_best_dir"] / _RUN_TS
    overall_best_dir = BASE_DIR / out_cfg["best_overall_dir"]
    for d in [out_dir, run_best_dir, overall_best_dir]:
        d.mkdir(parents=True, exist_ok=True)

    part_ids, X_pp, X_pt_list, y, strat_labels, pp_cols, y_removed_ids = load_data(data_cfg, prep_cfg)
    n_pp = X_pp.shape[1]
    T_list = [arr.shape[1] for arr in X_pt_list]

    # ── Initial train/test split (done once; test data NEVER used in HPO or CV) ──
    test_frac = prep_cfg.get("test_fraction", 0.2)
    dev_idx, test_idx = _initial_split(len(part_ids), test_frac, seed, strat_labels)
    dev_strat = strat_labels[dev_idx] if strat_labels is not None else None
    print(f"Initial split  →  dev={len(dev_idx)}  test={len(test_idx)}  "
          f"(test_fraction={test_frac})")

    # ── Mode selection / HPO (dev data only) ─────────────────────────────────
    mode     = cfg.get("mode", "manual").lower()
    best_hpo = None
    print(f"Mode     : {mode}")

    if mode == "optuna":
        if "optuna" not in cfg:
            raise ValueError("mode='optuna' but no 'optuna' section found in config.")
        study = run_hpo(X_pp[dev_idx], [x[dev_idx] for x in X_pt_list], y[dev_idx],
                        cfg["optuna"], model_cfg, train_cfg,
                        seed, device, dev_strat, run_best_dir)
        model_cfg, train_cfg = best_cfg_from_study(
            study, cfg["optuna"], model_cfg, train_cfg, n_pp)
        best_hpo = {
            "trial":              study.best_trial.number,
            "hpo_mean_mae":       float(study.best_trial.value),
            "model_cfg":          model_cfg,
            "train_lr":           train_cfg["lr"],
            "train_weight_decay": train_cfg["weight_decay"],
            "train_batch_size":   train_cfg["batch_size"],
        }
        (run_best_dir / "hpo_best_config.json").write_text(json.dumps(best_hpo, indent=2))
        print(f"HPO best config saved → {run_best_dir / 'hpo_best_config.json'}")
        print(f"\nFinal CV: {train_cfg['k_folds']} folds, "
              f"{train_cfg['epochs']} epochs, patience={train_cfg['patience']}")
    else:
        print("Using manual model config — skipping HPO.")

    # ── Indicative K-fold CV (dev only) ──────────────────────────────────────
    metrics_df, best_fold_idx, best_fold_mae = run_cv(
        X_pp[dev_idx], [x[dev_idx] for x in X_pt_list], y[dev_idx],
        model_cfg, train_cfg, device, seed, dev_strat)

    K = train_cfg["k_folds"]
    metrics_df.index = [f"Fold_{i+1}" for i in range(K)]
    metrics_df.to_csv(run_best_dir / "cv_fold_metrics.csv")
    print_summary(metrics_df, best_fold_idx, best_fold_mae, K)
    save_cv_plots(metrics_df, K, mat, run_best_dir)

    # ── Final training on all dev; evaluation on held-out test ───────────────
    final_model, final_scalers, tr_idx_final, val_idx_final = final_train_m1(
        X_pp, X_pt_list, y, dev_idx, model_cfg, train_cfg, seed, device, dev_strat)
    pp_sc, pt_scalers = final_scalers
    Xpp_te, Xpt_te = apply_scalers(X_pp[test_idx], [x[test_idx] for x in X_pt_list], pp_sc, pt_scalers)
    final_metrics = evaluate_model(final_model, Xpp_te, Xpt_te, y[test_idx], device)

    print(f"\n{'═' * 60}")
    print(f"Final Test  MAE={final_metrics['MAE']:.4f}  "
          f"RMSE={final_metrics['RMSE']:.4f}  "
          f"R²={final_metrics['R2']:.4f}  MSE={final_metrics['MSE']:.4f}")
    print(f"{'═' * 60}")

    save_final_test_plots(y[test_idx], final_metrics["pred"], final_metrics,
                          run_best_dir, mat)

    overall_updated = save_models(final_model, final_metrics, n_pp, T_list, model_cfg,
                                  run_best_dir, overall_best_dir)

    # Extract f-features for the FULL cleaned dataset (all parts, incl. test)
    # so that M2 has features for every part when it evaluates its own test set.
    df_f = extract_features(final_model, final_scalers, X_pp, X_pt_list,
                            part_ids, run_best_dir / "pressure_features_f.csv",
                            data_cfg["id_col"], device)

    # Save train/test split so M2 can reuse the exact same data division
    split_out = {
        "id_col":         data_cfg["id_col"],
        "train_part_ids": part_ids[dev_idx].tolist(),
        "test_part_ids":  part_ids[test_idx].tolist(),
    }
    (run_best_dir / "train_test_split.json").write_text(json.dumps(split_out, indent=2))
    print(f"Train/test split saved → {run_best_dir / 'train_test_split.json'}")

    # Save data_processing.json — shared contract for M2 and Fusion
    dp = {
        "run_ts":         _RUN_TS,
        "id_col":         data_cfg["id_col"],
        "random_seed":    seed,
        "pp_cols":        pp_cols,
        "train_part_ids": part_ids[tr_idx_final].tolist(),
        "val_part_ids":   part_ids[val_idx_final].tolist(),
        "test_part_ids":  part_ids[test_idx].tolist(),
        "pp_sc":          _scaler_to_dict(pp_sc),
        "pt_scalers":     pt_scalers,
        "pt_min":         float(pt_scalers[0]["min"]),
        "pt_max":         float(pt_scalers[0]["max"]),
        "y_filter": {
            "target_col":    data_cfg["target_col"],
            "iqr_multiplier": prep_cfg["iqr_multiplier"],
            "enabled":       bool(prep_cfg.get("remove_target_outliers", True)),
            "removed_ids":   y_removed_ids,
        },
    }
    (run_best_dir / "data_processing.json").write_text(json.dumps(dp, indent=2))
    print(f"Data processing saved → {run_best_dir / 'data_processing.json'}")

    # Save run_info.json — lightweight run identifier for users
    run_info = {
        "run_ts":       _RUN_TS,
        "material":     mat,
        "run_best_dir": str(run_best_dir.relative_to(BASE_DIR)),
    }
    (run_best_dir / "run_info.json").write_text(json.dumps(run_info, indent=2))
    print(f"Run info saved → {run_best_dir / 'run_info.json'}")

    if overall_updated:
        save_final_test_plots(y[test_idx], final_metrics["pred"], final_metrics,
                              overall_best_dir, mat)
        save_cv_plots(metrics_df, K, mat, overall_best_dir)
        metrics_df.to_csv(overall_best_dir / "cv_fold_metrics.csv")
        shutil.copy2(run_best_dir / "train_test_split.json",
                     overall_best_dir / "train_test_split.json")
        shutil.copy2(run_best_dir / "data_processing.json",
                     overall_best_dir / "data_processing.json")
        (overall_best_dir / "run_info.json").write_text(json.dumps(run_info, indent=2))
        df_f.index.name = data_cfg["id_col"]
        df_f.to_csv(overall_best_dir / "pressure_features_f.csv")
        if mode == "optuna" and best_hpo is not None:
            (overall_best_dir / "hpo_best_config.json").write_text(
                json.dumps(best_hpo, indent=2))
            src_trials = run_best_dir / "hpo_trials.csv"
            if src_trials.exists():
                shutil.copy2(src_trials, overall_best_dir / "hpo_trials.csv")
        print(f"All run-best artefacts also saved to overall best → {overall_best_dir}")


if __name__ == "__main__":
    main()

