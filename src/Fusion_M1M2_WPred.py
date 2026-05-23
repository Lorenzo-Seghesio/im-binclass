"""
Fusion_M1M2_WPred.py
====================
Fusion of M1's PPMLP + MergeHead with M2 acting as the encoder substitute.

Architecture
------------
  W = Fusion(pp_indirect, pp_direct)

  Indirect path (pressure-feature estimation via M2):
      pp_indirect → [M2.x_sc] → M2 → [M2.y_sc⁻¹] → f_active
                  → zero-padded to full n_f vector at M2's f_positions
                  → f_full  (n_f dims, zeros at inactive positions)

  Direct path (process parameters → M1's pp MLP):
      pp_direct → [M1.pp_sc] → M1.pp_mlp → pp_out  (pp_hidden[-1] dims)

  Merge (M1's frozen merge head):
      concat([f_full, pp_out]) → M1.merge → W

  Note: the zero-padding is correct by construction because M1.merge was
  trained with the same all-zero pattern at inactive f positions.

Usage
-----
  # Use models from standard output paths (after running M1/M2):
  python src/Fusion_M1M2_WPred.py --material PP

  # Override model paths (e.g. when using z_Old models for testing):
  python src/Fusion_M1M2_WPred.py --material PP \\
      --m1-dir outputs/z_Old/M1/PP/best_overall \\
      --m2-dir outputs/z_Old/M2/PP/best_overall

Arguments
---------
  --material       : PP | ABS  (single-material subset, case-insensitive)
  --m1-dir         : M1 best_overall directory
                     (default: outputs/M1/{material}/best_overall)
  --m2-dir         : M2 best_overall directory
                     (default: outputs/M2/{material}/best_overall)
  --seed           : random seed used to reconstruct scalers (default: 42)
  --val-fraction   : val fraction used by M1/M2 final_train (default: 0.1)
  --n-ig-steps     : integration steps for IG and IH (default: 50)
  --shap-bg        : max background samples for SHAP GradientExplainer (default: 100)
  --ih-samples     : max test samples used for Integrated Hessians (default: 20)

Outputs (under outputs/Fusion/{YYYY-MM-DD}_{HH-MM-SS}/)
---------
  test_metrics.json              Fusion model metrics on held-out test set
  test_predictions.csv           ID_Part, y_true, y_pred
  scatter_test.png               Predicted vs. real weight scatter plot
  residuals_scatter_test.png     Residuals vs. predicted scatter
  residuals_hist_test.png        Residuals histogram
  feature_attributions.csv       Per-feature mean |attribution| for all XAI methods
  shap_values.csv                Per-sample SHAP values  (N_test × 20 cols)
  ig_values.csv                  Per-sample IG values    (N_test × 20 cols)
  jacobian_values.csv            Per-sample Jacobian values (N_test × 20 cols)
  ih_matrix.csv                  Mean Integrated-Hessians matrix (20 × 20)
  xai_bar_{method}.png           Stacked bar: mean |attr| direct + indirect
  xai_total_effect.png           Total (direct + indirect) per feature, all methods
  ih_heatmap.png                 IH interaction matrix heat-map
  fusion_model_info.json         Architecture + scaler info + which model files used
  m1_best_model.pt  /  m2_best_model.pt   (copies of used model weights)
  m1_best_metrics.json / m2_best_metrics.json
"""

import argparse
import copy
import json
import math
import shutil
import sys
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent

# ── Check optional XAI dependencies ──────────────────────────────────────────
try:
    import shap as shap_lib
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False
    print("[WARN] shap not installed — SHAP analysis will be skipped. "
          "Install with: pip install shap")

try:
    from captum.attr import IntegratedGradients
    _HAS_CAPTUM = True
except ImportError:
    _HAS_CAPTUM = False
    print("[WARN] captum not installed — IG analysis will be skipped. "
          "Install with: pip install captum")


# ══════════════════════════════════════════════════════════════════════════════
# Model class definitions  (mirrored from M1/M2 — avoids import-collision)
# ══════════════════════════════════════════════════════════════════════════════

def _he_zero_bias(m: nn.Module) -> None:
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(m.bias)


class PressureEncoder(nn.Module):
    def __init__(self, channels, kernels, n_f, pool_kernels=None):
        super().__init__()
        n_conv = len(channels) - 1
        _pk    = pool_kernels or []
        conv_layers = []
        for i in range(n_conv):
            conv_layers += [
                nn.Conv1d(channels[i], channels[i + 1],
                          kernel_size=kernels[i], padding=kernels[i] // 2),
                nn.ReLU(),
            ]
            if i < n_conv - 1 and i < len(_pk) and _pk[i] > 1:
                conv_layers.append(nn.MaxPool1d(kernel_size=_pk[i], stride=_pk[i]))
        self.convs = nn.Sequential(*conv_layers)
        self.pool  = nn.AdaptiveAvgPool1d(1)
        self.fc    = nn.Linear(channels[-1], n_f)
        self.act   = nn.ReLU()
        for m in self.convs:
            if isinstance(m, nn.Conv1d):
                _he_zero_bias(m)
        _he_zero_bias(self.fc)

    def forward(self, x):
        x = self.convs(x)
        x = self.pool(x).squeeze(-1)
        return self.act(self.fc(x))


class PPMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout):
        super().__init__()
        layers = []
        dims   = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            if i < len(dims) - 2:
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                _he_zero_bias(m)

    def forward(self, x):
        return self.net(x)


class MergeHead(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout):
        super().__init__()
        layers = []
        dims   = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)
        linears = [m for m in self.net if isinstance(m, nn.Linear)]
        for m in linears[:-1]:
            _he_zero_bias(m)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class M1Model(nn.Module):
    def __init__(self, n_pp, mcfg):
        super().__init__()
        self.encoder = PressureEncoder(
            channels=mcfg["conv_channels"],
            kernels=mcfg["conv_kernels"],
            n_f=mcfg["n_f_features"],
            pool_kernels=mcfg.get("enc_pool_kernels", []),
        )
        self.pp_mlp  = PPMLP(n_pp, mcfg["pp_hidden"], mcfg["dropout"])
        merge_in     = mcfg["n_f_features"] + mcfg["pp_hidden"][-1]
        self.merge   = MergeHead(merge_in, mcfg["merge_hidden"], mcfg["dropout"])

    def forward(self, x_pt, x_pp):
        f  = self.encoder(x_pt)
        pp = self.pp_mlp(x_pp)
        return self.merge(torch.cat([f, pp], dim=1))


class M2Model(nn.Module):
    def __init__(self, n_in, n_out, hidden_dims, dropout):
        super().__init__()
        layers = []
        prev = n_in
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, n_out))
        self.net = nn.Sequential(*layers)
        linears = [m for m in self.net if isinstance(m, nn.Linear)]
        for m in linears[:-1]:
            _he_zero_bias(m)

    def forward(self, x):
        return self.net(x)


# ══════════════════════════════════════════════════════════════════════════════
# Differentiable MinMax scaler (for gradient-tracked inference)
# ══════════════════════════════════════════════════════════════════════════════

class TorchMinMaxScaler(nn.Module):
    """Wraps a fitted sklearn MinMaxScaler; exposes differentiable
    transform / inverse_transform as PyTorch operations."""

    def __init__(self, sklearn_scaler: MinMaxScaler):
        super().__init__()
        # sklearn stores: X_scaled = X * scale_ + min_
        self.register_buffer("scale_",
            torch.tensor(sklearn_scaler.scale_, dtype=torch.float32))
        self.register_buffer("min_",
            torch.tensor(sklearn_scaler.min_,   dtype=torch.float32))

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale_ + self.min_

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.min_) / self.scale_


# ══════════════════════════════════════════════════════════════════════════════
# FusionModel
# ══════════════════════════════════════════════════════════════════════════════

class FusionModel(nn.Module):
    """
    Frozen fusion of M2 (encoder substitute) + M1's PPMLP + M1's MergeHead.

    forward(pp_indirect, pp_direct) → W  (raw unscaled pp inputs → weight)

    pp_indirect : (B, n_pp)  — raw process parameters → indirect path via M2
    pp_direct   : (B, n_pp)  — raw process parameters → direct path via PPMLP
    """

    def __init__(
        self,
        m2_model:      M2Model,
        m1_pp_mlp:     PPMLP,
        m1_merge:      MergeHead,
        m2_x_scaler:   TorchMinMaxScaler,
        m2_y_scaler:   TorchMinMaxScaler,
        m1_x_scaler:   TorchMinMaxScaler,
        n_f:           int,
        f_positions:   list,          # ints — M2's output cols → position in n_f vector
    ):
        super().__init__()
        self.m2        = m2_model
        self.pp_mlp    = m1_pp_mlp
        self.merge     = m1_merge
        self.m2_x_sc   = m2_x_scaler
        self.m2_y_sc   = m2_y_scaler
        self.m1_x_sc   = m1_x_scaler
        self.n_f       = n_f
        self.f_positions = f_positions  # e.g. [3] when only f_3 is active

        # freeze all sub-modules
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, pp_indirect: torch.Tensor,
                pp_direct: torch.Tensor) -> torch.Tensor:
        B = pp_indirect.shape[0]

        # ── Indirect path ──────────────────────────────────────────────────
        pp_ind_s = self.m2_x_sc.transform(pp_indirect)    # MinMax scale
        f_scaled = self.m2(pp_ind_s)                       # (B, n_m2_out)
        f_orig   = self.m2_y_sc.inverse_transform(f_scaled)  # back to encoder space

        # embed M2 predictions into the full n_f-dimensional f vector
        # inactive positions remain zero (differentiable construction via cat)
        parts = []
        m2_col = 0
        for i in range(self.n_f):
            if i in self.f_positions:
                parts.append(f_orig[:, m2_col : m2_col + 1])
                m2_col += 1
            else:
                parts.append(torch.zeros(B, 1,
                                         device=pp_indirect.device,
                                         dtype=pp_indirect.dtype))
        f_full = torch.cat(parts, dim=1)                   # (B, n_f)

        # ── Direct path ────────────────────────────────────────────────────
        pp_dir_s = self.m1_x_sc.transform(pp_direct)      # MinMax scale
        pp_out   = self.pp_mlp(pp_dir_s)                   # (B, pp_hidden[-1])

        # ── Merge ──────────────────────────────────────────────────────────
        merged = torch.cat([f_full, pp_out], dim=1)        # (B, n_f + pp_hidden[-1])
        return self.merge(merged)                           # (B,)


class FlatFusionWrapper(nn.Module):
    """Accepts a single (B, 2*n_pp) tensor for XAI tools.
    First n_pp columns → pp_indirect; last n_pp → pp_direct."""

    def __init__(self, fusion: FusionModel, n_pp: int):
        super().__init__()
        self.fusion = fusion
        self.n_pp   = n_pp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pp_ind = x[:, :self.n_pp]
        pp_dir = x[:, self.n_pp:]
        return self.fusion(pp_ind, pp_dir)


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_scalar_data(material: str, iqr_multiplier: float = 1.5,
                     filter_target: bool = True):
    """Load scalar_features_{material}.csv; impute + IQR-filter; return DataFrame."""
    csv_path = (BASE_DIR / "data" / "Fraunhofer_ProBayes_Dataset"
                / "extracted" / f"scalar_features_{material}.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"Scalar CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, index_col="ID_Part")
    target_col = "SCA_PartWeight"
    pp_cols = [c for c in df.columns if c != target_col]

    # impute pp columns
    for col in pp_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # IQR outlier removal
    cols_to_filter = pp_cols + ([target_col] if filter_target else [])
    mask = pd.Series(True, index=df.index)
    for col in cols_to_filter:
        if col not in df.columns:
            continue
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            mask &= (df[col] >= q1 - iqr_multiplier * iqr) & \
                    (df[col] <= q3 + iqr_multiplier * iqr)

    n_removed = int((~mask).sum())
    df = df[mask].copy()
    print(f"  Loaded {len(df)} parts  (removed {n_removed} outliers, "
          f"filter_target={filter_target})")
    return df, pp_cols, target_col


def load_train_test_split(m1_dir: Path):
    """Load train_test_split.json from m1_dir (best_overall); fall back to run_best."""
    for candidate in [
        m1_dir / "train_test_split.json",
        m1_dir.parent / "run_best" / "train_test_split.json",
    ]:
        if candidate.exists():
            print(f"  train_test_split.json  ← {candidate}")
            return json.loads(candidate.read_text())
    raise FileNotFoundError(
        f"train_test_split.json not found in {m1_dir} or {m1_dir.parent / 'run_best'}")


def map_split(split_data: dict, part_ids: np.ndarray):
    """Map part-ID lists from JSON → integer indices into part_ids array."""
    id_to_idx = {pid: i for i, pid in enumerate(part_ids.tolist())}
    dev_idx  = np.array([id_to_idx[p] for p in split_data["train_part_ids"]
                         if p in id_to_idx], dtype=np.intp)
    test_idx = np.array([id_to_idx[p] for p in split_data["test_part_ids"]
                         if p in id_to_idx], dtype=np.intp)
    print(f"  Split mapped  → dev={len(dev_idx)}  test={len(test_idx)}")
    return dev_idx, test_idx


# ══════════════════════════════════════════════════════════════════════════════
# Scaler reconstruction
# ══════════════════════════════════════════════════════════════════════════════

def reconstruct_m1_pp_scaler(X_pp: np.ndarray, dev_idx: np.ndarray,
                              seed: int = 42, val_fraction: float = 0.1):
    """Replicate M1 final_train_m1 scaler fitting exactly (seed, val_fraction)."""
    rng   = np.random.default_rng(seed)
    perm  = rng.permutation(len(dev_idx))
    n_val = max(1, int(len(dev_idx) * val_fraction))
    tr_idx = dev_idx[perm[n_val:]]          # same as M1 final_train_m1

    sc = MinMaxScaler()
    sc.fit(X_pp[tr_idx])
    print(f"  M1 pp_sc reconstructed on {len(tr_idx)} dev-train samples")
    return sc


def reconstruct_m2_scalers(X_pp: np.ndarray, Y_f: np.ndarray,
                            dev_idx: np.ndarray,
                            seed: int = 42, val_fraction: float = 0.1):
    """Replicate M2 final_train_m2 scaler fitting (x_sc + y_sc)."""
    rng   = np.random.default_rng(seed)
    perm  = rng.permutation(len(dev_idx))
    n_val = max(1, int(len(dev_idx) * val_fraction))
    tr_idx = dev_idx[perm[n_val:]]          # same as M2 final_train_m2

    x_sc = MinMaxScaler(); x_sc.fit(X_pp[tr_idx])
    y_sc = MinMaxScaler(); y_sc.fit(Y_f[tr_idx])
    print(f"  M2 x_sc/y_sc reconstructed on {len(tr_idx)} dev-train samples")
    return x_sc, y_sc


# ══════════════════════════════════════════════════════════════════════════════
# Model loading
# ══════════════════════════════════════════════════════════════════════════════

def load_m1(m1_dir: Path, n_pp: int, device: torch.device):
    """Load M1 best_overall model; return (M1Model, model_cfg)."""
    meta_path = m1_dir / "best_metrics.json"
    wt_path   = m1_dir / "best_model.pt"
    if not meta_path.exists():
        raise FileNotFoundError(f"M1 best_metrics.json not found: {meta_path}")
    if not wt_path.exists():
        raise FileNotFoundError(f"M1 best_model.pt not found: {wt_path}")

    meta = json.loads(meta_path.read_text())
    mcfg = meta["model_cfg"]
    model = M1Model(n_pp, mcfg).to(device)
    model.load_state_dict(torch.load(wt_path, map_location=device))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"  M1 loaded  (MAE={meta.get('MAE', '?'):.4f}  "
          f"R²={meta.get('R2', '?'):.4f})  ← {wt_path}")
    return model, mcfg, meta


def load_m2(m2_dir: Path, device: torch.device):
    """Load M2 best_overall model; return (M2Model, model_cfg, meta)."""
    meta_path = m2_dir / "best_metrics.json"
    wt_path   = m2_dir / "best_model.pt"
    if not meta_path.exists():
        raise FileNotFoundError(f"M2 best_metrics.json not found: {meta_path}")
    if not wt_path.exists():
        raise FileNotFoundError(f"M2 best_model.pt not found: {wt_path}")

    meta  = json.loads(meta_path.read_text())
    mcfg  = meta["model_cfg"]
    n_in  = meta["n_in"]
    n_out = meta["n_out"]
    model = M2Model(n_in, n_out, mcfg["hidden_dims"], mcfg["dropout"]).to(device)
    model.load_state_dict(torch.load(wt_path, map_location=device))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    mean_mae = meta.get("mean_MAE", "?")
    print(f"  M2 loaded  (mean_MAE={mean_mae:.4f}  f_cols={meta['f_cols']})  ← {wt_path}")
    return model, mcfg, meta


def get_f_positions(m1_dir: Path, m2_f_cols: list) -> list:
    """
    Return the integer positions in M1's full n_f vector where M2's predicted
    f features belong.  Derived directly from the column names (f_3 → 3).
    """
    positions = [int(fc.split("_")[1]) for fc in m2_f_cols]
    feat_csv  = m1_dir / "pressure_features_f.csv"
    if feat_csv.exists():
        df_f   = pd.read_csv(feat_csv, index_col=0, nrows=5)
        f_cols = [c for c in df_f.columns]
        zero_f = [c for c in f_cols if (df_f[c].abs() < 1e-6).all()]
        nonzero_f = [c for c in f_cols if c not in zero_f]
        print(f"  pressure_features_f.csv: {len(f_cols)} f cols, "
              f"non-zero = {nonzero_f}, zero = {zero_f}")
    print(f"  f_positions for M2→M1 embedding: {positions}")
    return positions


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_fusion(fusion: FusionModel, X_pp: np.ndarray, y: np.ndarray,
                    device: torch.device) -> dict:
    """Forward test data through FusionModel; return metrics + predictions."""
    t_pp = torch.tensor(X_pp, dtype=torch.float32, device=device)
    fusion.eval()
    with torch.no_grad():
        preds = fusion(t_pp, t_pp).cpu().numpy()
    mae  = float(mean_absolute_error(y, preds))
    mse  = float(mean_squared_error(y, preds))
    rmse = math.sqrt(mse)
    r2   = float(r2_score(y, preds))
    print(f"\n{'═'*55}")
    print(f"  Fusion test metrics:")
    print(f"    MAE  = {mae:.4f} g")
    print(f"    RMSE = {rmse:.4f} g")
    print(f"    R²   = {r2:.4f}")
    print(f"    MSE  = {mse:.4f}")
    print(f"{'═'*55}")
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2, "pred": preds}


# ══════════════════════════════════════════════════════════════════════════════
# XAI — helpers
# ══════════════════════════════════════════════════════════════════════════════

def _flat_col_names(pp_cols: list) -> list:
    """Column names for the 20-D flattened input: [pp_0_indirect, ..., pp_9_direct]."""
    return [f"{c}_indirect" for c in pp_cols] + [f"{c}_direct" for c in pp_cols]


def _split_indirect_direct(arr: np.ndarray, n_pp: int):
    """Split (N, 2*n_pp) attribution array into indirect and direct halves."""
    return arr[:, :n_pp], arr[:, n_pp:]


# ──────────────────────────────────────────────────────────────────────────────
# SHAP (GradientExplainer)
# ──────────────────────────────────────────────────────────────────────────────

class _ShapWrapper(nn.Module):
    """Thin wrapper that converts scalar output (B,) → (B,1) for SHAP."""
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).unsqueeze(-1)   # (B,) → (B, 1)


def run_shap(flat_model: FlatFusionWrapper,
             X_dev_flat:  np.ndarray,
             X_test_flat: np.ndarray,
             n_bg:        int,
             device:      torch.device) -> np.ndarray:
    """Run SHAP GradientExplainer; return shap_values (N_test, 2*n_pp)."""
    if not _HAS_SHAP:
        print("  [SKIP] SHAP not available")
        return None

    print(f"  Running SHAP GradientExplainer  "
          f"(background: {min(n_bg, len(X_dev_flat))} samples) ...")
    rng = np.random.default_rng(0)
    bg_idx = rng.choice(len(X_dev_flat), size=min(n_bg, len(X_dev_flat)), replace=False)
    bg_np  = X_dev_flat[bg_idx].astype(np.float32)
    bg_t   = torch.tensor(bg_np, dtype=torch.float32, device=device)
    x_t    = torch.tensor(X_test_flat.astype(np.float32), dtype=torch.float32, device=device)

    # SHAP GradientExplainer requires 2-D output (B, n_outputs)
    shap_model = _ShapWrapper(flat_model).to(device)
    shap_model.eval()
    e  = shap_lib.GradientExplainer(shap_model, bg_t)
    sv = e.shap_values(x_t)
    # GradientExplainer returns list-of-arrays for multi-output; take output 0
    if isinstance(sv, list):
        sv = sv[0]
    sv = np.array(sv, dtype=np.float32)
    if sv.ndim == 3:           # (N, d, 1) → (N, d)
        sv = sv[..., 0]
    print(f"  SHAP done  shape={sv.shape}")
    return sv


# ──────────────────────────────────────────────────────────────────────────────
# Vanilla Integrated Gradients (captum)
# ──────────────────────────────────────────────────────────────────────────────

def run_ig(flat_model: FlatFusionWrapper,
           X_test_flat: np.ndarray,
           baseline_flat: np.ndarray,
           n_steps: int,
           device: torch.device) -> np.ndarray:
    """Run captum IntegratedGradients; return ig_values (N_test, 2*n_pp)."""
    if not _HAS_CAPTUM:
        print("  [SKIP] captum not available")
        return None

    print(f"  Running Integrated Gradients  (steps={n_steps}) ...")
    flat_model.eval()
    ig = IntegratedGradients(flat_model)
    x_t  = torch.tensor(X_test_flat.astype(np.float32), dtype=torch.float32, device=device)
    bl_t = torch.tensor(baseline_flat.astype(np.float32)[None, :], dtype=torch.float32,
                        device=device).expand_as(x_t)
    attr = ig.attribute(x_t, baselines=bl_t, n_steps=n_steps, internal_batch_size=64)
    ig_vals = attr.detach().cpu().numpy()
    print(f"  IG done  shape={ig_vals.shape}")
    return ig_vals


# ──────────────────────────────────────────────────────────────────────────────
# Integrated Hessians  (custom, single-integral formula — Janizek et al. 2021)
# ──────────────────────────────────────────────────────────────────────────────

def _hessian_single_sample(flat_model: nn.Module,
                            x: torch.Tensor,          # (1, d)
                            baseline: torch.Tensor,   # (1, d)
                            n_steps: int) -> torch.Tensor:
    """
    Compute the Integrated Hessians for one sample.
    IH(i,j) = delta_i * delta_j * mean_alpha{ d²f/dx_i dx_j (x' + alpha*delta) }

    Returns IH matrix (d, d).
    Note: For ReLU networks the Hessian is zero almost everywhere (piecewise
    linear), so IH values may be numerically small.
    """
    d     = x.shape[1]
    delta = (x - baseline).detach()          # (1, d)

    h_acc = torch.zeros(d, d, dtype=torch.float64, device=x.device)

    for k in range(n_steps):
        alpha   = (k + 0.5) / n_steps        # midpoint rule
        x_interp = (baseline + alpha * delta).to(x.device).detach()
        x_interp.requires_grad_(True)

        y    = flat_model(x_interp)           # (1,)
        grad = torch.autograd.grad(y.sum(), x_interp,
                                   create_graph=True)[0]  # (1, d)
        for i in range(d):
            g2 = torch.autograd.grad(
                grad[0, i], x_interp,
                retain_graph=(i < d - 1),
            )[0]  # (1, d)
            h_acc[i] += g2[0].detach().double()

    h_mean = h_acc / n_steps                  # (d, d)
    delta0 = delta[0].double()               # (d,)
    ih = delta0.unsqueeze(1) * delta0.unsqueeze(0) * h_mean  # (d, d)
    return ih.float()


def run_integrated_hessians(flat_model: nn.Module,
                             X_test_flat:  np.ndarray,
                             baseline_flat: np.ndarray,
                             n_steps:      int,
                             max_samples:  int,
                             device:       torch.device) -> np.ndarray:
    """
    Compute mean IH matrix over (at most) max_samples test samples.
    Returns ih_mean (d, d) averaged over the chosen samples.
    """
    n_samp = min(max_samples, len(X_test_flat))
    print(f"  Running Integrated Hessians  "
          f"({n_samp} samples × {n_steps} steps, d={X_test_flat.shape[1]}) ...")
    bl_t = torch.tensor(baseline_flat.astype(np.float32)[None, :],
                        dtype=torch.float32, device=device)
    d    = X_test_flat.shape[1]
    ih_acc = torch.zeros(d, d, device=device)
    flat_model.eval()

    for i in range(n_samp):
        x_t = torch.tensor(X_test_flat[i:i+1].astype(np.float32),
                            dtype=torch.float32, device=device)
        ih_i = _hessian_single_sample(flat_model, x_t, bl_t, n_steps)
        ih_acc += ih_i.to(device)
        if (i + 1) % max(1, n_samp // 5) == 0:
            print(f"    IH progress: {i+1}/{n_samp}")

    ih_mean = (ih_acc / n_samp).cpu().numpy()
    print(f"  IH done  shape={ih_mean.shape}  "
          f"mean|IH|={np.abs(ih_mean).mean():.2e}")
    return ih_mean


# ──────────────────────────────────────────────────────────────────────────────
# Jacobian pathway analysis
# ──────────────────────────────────────────────────────────────────────────────

def run_jacobian(flat_model: FlatFusionWrapper,
                 X_test_flat: np.ndarray,
                 device: torch.device) -> np.ndarray:
    """
    Compute per-sample local Jacobians dW/dx where x = [pp_indirect; pp_direct].
    Returns jac (N_test, 2*n_pp).

    Because the model has no cross-sample mixing, the vectorised gradient of
    sum(W) w.r.t. the (N, 2n_pp) input matrix gives per-sample Jacobians.
    """
    print("  Running Jacobian pathway analysis ...")
    flat_model.eval()
    x_t = torch.tensor(X_test_flat.astype(np.float32),
                        dtype=torch.float32, device=device)
    x_t.requires_grad_(True)
    W   = flat_model(x_t)                   # (N_test,)
    jac = torch.autograd.grad(W.sum(), x_t)[0]  # (N_test, 2*n_pp)
    jac_np = jac.detach().cpu().numpy()
    print(f"  Jacobian done  shape={jac_np.shape}")
    return jac_np


# ══════════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════════

C_IND  = "#D55E00"   # indirect (orange)
C_DIR  = "#0072B2"   # direct   (blue)
C_TOT  = "#009E73"   # total    (green)
C_RED  = "#CC3311"
C_BLUE = "#0072B2"


def _save_scatter(y_true, y_pred, metrics, out_path: Path, material: str):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.7, edgecolors="k",
               linewidths=0.3, color=C_BLUE, zorder=3)
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], color=C_RED, linestyle="--", linewidth=1.5)
    ax.set_xlabel("Real weight [g]"); ax.set_ylabel("Predicted weight [g]")
    ax.set_title(f"[{material}] Fusion Model — Test Set\n"
                 f"MAE={metrics['MAE']:.4f} g   R²={metrics['R2']:.4f}")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def _save_residuals_scatter(y_true, y_pred, out_path: Path, material: str):
    res = y_pred - y_true
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_pred, res, alpha=0.7, edgecolors="k",
               linewidths=0.3, color=C_BLUE, zorder=3)
    ax.axhline(0, color=C_RED, linestyle="--", linewidth=1.5)
    ax.set_xlabel("Predicted weight [g]"); ax.set_ylabel("Residual (pred − real) [g]")
    ax.set_title(f"[{material}] Residuals vs Predicted")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def _save_residuals_hist(y_true, y_pred, out_path: Path, material: str):
    res = y_pred - y_true
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hist(res, bins=20, color=C_BLUE, edgecolor="k", linewidth=0.4, alpha=0.85)
    ax.axvline(0, color=C_RED, linestyle="--", linewidth=1.5)
    ax.axvline(float(res.mean()), color=C_TOT, linestyle="-", linewidth=1.5,
               label=f"Mean={res.mean():.4f}")
    ax.set_xlabel("Residual [g]"); ax.set_ylabel("Count")
    ax.set_title(f"[{material}] Residuals Distribution\nstd={res.std():.4f} g")
    ax.legend(); ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def _bar_direct_indirect(mean_abs_ind: np.ndarray, mean_abs_dir: np.ndarray,
                          pp_cols: list, title: str, out_path: Path):
    """Stacked horizontal bar: indirect (left) + direct (right)."""
    n  = len(pp_cols)
    y  = np.arange(n)
    fig, ax = plt.subplots(figsize=(9, max(4, n * 0.45)))
    ax.barh(y, mean_abs_ind, label="Indirect (via M2)", color=C_IND, alpha=0.85)
    ax.barh(y, mean_abs_dir, left=mean_abs_ind, label="Direct (via PPMLP)",
            color=C_DIR, alpha=0.85)
    ax.set_yticks(y); ax.set_yticklabels(pp_cols, fontsize=9)
    ax.set_xlabel("Mean |attribution|"); ax.set_title(title)
    ax.legend(fontsize=9); ax.grid(True, axis="x", linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def save_xai_bars(summary_df: pd.DataFrame, pp_cols: list, out_dir: Path,
                  material: str):
    """One stacked bar plot per XAI method."""
    methods = {
        "shap":     ("SHAP",                "shap_indirect_mean_abs", "shap_direct_mean_abs"),
        "ig":       ("Integrated Gradients", "ig_indirect_mean_abs",   "ig_direct_mean_abs"),
        "jacobian": ("Jacobian Pathway",     "jac_indirect_mean_abs",  "jac_direct_mean_abs"),
    }
    for tag, (label, col_ind, col_dir) in methods.items():
        if col_ind not in summary_df.columns:
            continue
        _bar_direct_indirect(
            summary_df[col_ind].values,
            summary_df[col_dir].values,
            pp_cols,
            f"[{material}] {label} — Direct vs Indirect feature influence",
            out_dir / f"xai_bar_{tag}.png",
        )
    print(f"  XAI bar plots saved → {out_dir}")


def save_total_effect_plot(summary_df: pd.DataFrame, pp_cols: list,
                           out_dir: Path, material: str):
    """Grouped bar: total effect per feature per method."""
    method_cols = {
        "SHAP":     "shap_total_mean_abs",
        "IG":       "ig_total_mean_abs",
        "Jacobian": "jac_total_mean_abs",
    }
    available = {k: v for k, v in method_cols.items() if v in summary_df.columns}
    if not available:
        return
    n = len(pp_cols)
    x = np.arange(n)
    w = 0.8 / len(available)
    colours = [C_IND, C_DIR, C_TOT]

    fig, ax = plt.subplots(figsize=(12, max(4, n * 0.4)))
    for i, (label, col) in enumerate(available.items()):
        offset = (i - (len(available) - 1) / 2) * w
        ax.bar(x + offset, summary_df[col].values, w, label=label,
               color=colours[i % len(colours)], edgecolor="k", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(pp_cols, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Mean |total attribution|")
    ax.set_title(f"[{material}] Total Feature Effect (direct + indirect) — All Methods")
    ax.legend(); ax.grid(True, axis="y", linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_dir / "xai_total_effect.png", dpi=150)
    plt.close(fig)
    print(f"  Total-effect plot saved → {out_dir / 'xai_total_effect.png'}")


def save_ih_heatmap(ih_matrix: np.ndarray, flat_cols: list, out_path: Path,
                    material: str):
    """Heat-map of mean |IH| interaction matrix."""
    d = ih_matrix.shape[0]
    fig, ax = plt.subplots(figsize=(max(8, d * 0.45), max(7, d * 0.45)))
    vmax = np.abs(ih_matrix).max()
    im = ax.imshow(ih_matrix, cmap="RdBu_r", aspect="auto",
                   vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(d)); ax.set_yticks(range(d))
    ax.set_xticklabels(flat_cols, rotation=90, fontsize=7)
    ax.set_yticklabels(flat_cols, fontsize=7)
    ax.set_title(f"[{material}] Integrated Hessians — mean interaction matrix")
    plt.colorbar(im, ax=ax, shrink=0.7, label="IH value")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  IH heatmap saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Build attribution summary
# ══════════════════════════════════════════════════════════════════════════════

def build_summary(pp_cols: list,
                  shap_vals:    np.ndarray | None,
                  ig_vals:      np.ndarray | None,
                  jac_vals:     np.ndarray | None,
                  n_pp:         int) -> pd.DataFrame:
    """Aggregate per-sample attributions into per-feature mean |attr|."""
    rows = []
    for i, col in enumerate(pp_cols):
        row = {"feature": col}
        for tag, vals in [("shap", shap_vals), ("ig", ig_vals), ("jac", jac_vals)]:
            if vals is None:
                continue
            ind = vals[:, i]          # indirect (first n_pp cols)
            dir_ = vals[:, n_pp + i]  # direct   (second n_pp cols)
            row[f"{tag}_indirect_mean"]     = float(ind.mean())
            row[f"{tag}_direct_mean"]       = float(dir_.mean())
            row[f"{tag}_total_mean"]        = float((ind + dir_).mean())
            row[f"{tag}_indirect_mean_abs"] = float(np.abs(ind).mean())
            row[f"{tag}_direct_mean_abs"]   = float(np.abs(dir_).mean())
            row[f"{tag}_total_mean_abs"]    = float(np.abs(ind + dir_).mean())
        rows.append(row)
    return pd.DataFrame(rows).set_index("feature")


def build_per_sample_df(vals: np.ndarray, part_ids: np.ndarray,
                        flat_cols: list) -> pd.DataFrame:
    df = pd.DataFrame(vals, columns=flat_cols)
    df.insert(0, "ID_Part", part_ids)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Fusion M1+M2 weight prediction + XAI")
    p.add_argument("--material", required=True, choices=["PP", "ABS", "pp", "abs"],
                   help="Material (PP or ABS)")
    p.add_argument("--m1-dir", default=None,
                   help="Path to M1 best_overall dir "
                        "(default: outputs/M1/{material}/best_overall)")
    p.add_argument("--m2-dir", default=None,
                   help="Path to M2 best_overall dir "
                        "(default: outputs/M2/{material}/best_overall)")
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--n-ig-steps",   type=int,   default=50)
    p.add_argument("--shap-bg",      type=int,   default=100)
    p.add_argument("--ih-samples",   type=int,   default=20)
    return p.parse_args()


def main():
    args    = parse_args()
    MAT     = args.material.upper()
    SEED    = args.seed
    VAL_FR  = args.val_fraction
    RUN_TS  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'═'*60}")
    print(f"  Fusion M1+M2  |  material={MAT}  |  device={device}")
    print(f"{'═'*60}")

    # ── Resolve model directories ──────────────────────────────────────────
    m1_dir = Path(args.m1_dir) if args.m1_dir else \
             BASE_DIR / "outputs" / "M1" / MAT / "best_overall"
    m2_dir = Path(args.m2_dir) if args.m2_dir else \
             BASE_DIR / "outputs" / "M2" / MAT / "best_overall"

    for label, d in [("M1 dir", m1_dir), ("M2 dir", m2_dir)]:
        if not d.is_dir():
            sys.exit(f"[ERROR] {label} not found: {d}\n"
                     f"  Run M1/M2 training first, or pass --m1-dir / --m2-dir.")
    print(f"\n  M1 dir : {m1_dir}")
    print(f"  M2 dir : {m2_dir}")

    # ── Output directory ───────────────────────────────────────────────────
    out_dir = BASE_DIR / "outputs" / "Fusion" / f"{RUN_TS}_{MAT}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Output : {out_dir}\n")

    # ══════════════════════════════════════════════════════════════════════
    # Step 1 — Load and preprocess data
    # ══════════════════════════════════════════════════════════════════════
    print("── Step 1: Loading data ────────────────────────────────────")
    # M1-style preprocessing (IQR on pp + target)
    df_m1, pp_cols, target_col = load_scalar_data(MAT, iqr_multiplier=1.5,
                                                   filter_target=True)
    part_ids_m1 = df_m1.index.to_numpy()
    X_pp_m1     = df_m1[pp_cols].to_numpy(dtype=np.float32)
    y_m1        = df_m1[target_col].to_numpy(dtype=np.float32)
    n_pp        = X_pp_m1.shape[1]
    print(f"  M1-style dataset: {len(df_m1)} parts × {n_pp} pp features")

    # M2-style preprocessing (IQR on pp only, no target)
    df_m2, _, _ = load_scalar_data(MAT, iqr_multiplier=1.5, filter_target=False)
    part_ids_m2 = df_m2.index.to_numpy()
    X_pp_m2     = df_m2[pp_cols].to_numpy(dtype=np.float32)
    print(f"  M2-style dataset: {len(df_m2)} parts × {n_pp} pp features")

    # M1's f-features (target for M2 scaler reconstruction)
    feat_csv = m1_dir / "pressure_features_f.csv"
    if not feat_csv.exists():
        sys.exit(f"[ERROR] pressure_features_f.csv not found: {feat_csv}")
    df_f_all  = pd.read_csv(feat_csv, index_col=0)
    print(f"  pressure_features_f.csv: {df_f_all.shape}")

    # ══════════════════════════════════════════════════════════════════════
    # Step 2 — Load train/test split; map to indices
    # ══════════════════════════════════════════════════════════════════════
    print("\n── Step 2: Loading train/test split ────────────────────────")
    split_data  = load_train_test_split(m1_dir)
    dev_idx_m1, test_idx_m1 = map_split(split_data, part_ids_m1)
    dev_idx_m2, test_idx_m2 = map_split(split_data, part_ids_m2)

    # ══════════════════════════════════════════════════════════════════════
    # Step 3 — Load models
    # ══════════════════════════════════════════════════════════════════════
    print("\n── Step 3: Loading M1 and M2 models ───────────────────────")
    m1_model, m1_cfg, m1_meta = load_m1(m1_dir, n_pp, device)
    m2_model, m2_cfg, m2_meta = load_m2(m2_dir, device)

    m2_f_cols   = m2_meta["f_cols"]        # e.g. ["f_3"]
    n_f         = m1_cfg["n_f_features"]   # e.g. 5
    f_positions = get_f_positions(m1_dir, m2_f_cols)

    # ══════════════════════════════════════════════════════════════════════
    # Step 4 — Reconstruct scalers
    # ══════════════════════════════════════════════════════════════════════
    print("\n── Step 4: Reconstructing scalers ─────────────────────────")
    m1_pp_sc = reconstruct_m1_pp_scaler(X_pp_m1, dev_idx_m1, SEED, VAL_FR)

    # M2's y targets = f columns that M2 predicts (in M1's feature space)
    # Join f features to M2's preprocessed dataset by part ID
    df_f_m2 = df_m2.join(df_f_all[m2_f_cols], how="inner")
    Y_f_m2  = df_f_m2[m2_f_cols].to_numpy(dtype=np.float32)
    X_pp_m2_joined = df_f_m2[pp_cols].to_numpy(dtype=np.float32)
    part_ids_m2_joined = df_f_m2.index.to_numpy()

    # re-map M1 split to the joined M2 dataset (intersection)
    dev_idx_m2j, _ = map_split(split_data, part_ids_m2_joined)

    m2_x_sc, m2_y_sc = reconstruct_m2_scalers(
        X_pp_m2_joined, Y_f_m2, dev_idx_m2j, SEED, VAL_FR)

    # ══════════════════════════════════════════════════════════════════════
    # Step 5 — Build FusionModel
    # ══════════════════════════════════════════════════════════════════════
    print("\n── Step 5: Building FusionModel ────────────────────────────")
    t_m1_x_sc = TorchMinMaxScaler(m1_pp_sc).to(device)
    t_m2_x_sc = TorchMinMaxScaler(m2_x_sc).to(device)
    t_m2_y_sc = TorchMinMaxScaler(m2_y_sc).to(device)

    fusion = FusionModel(
        m2_model   = m2_model,
        m1_pp_mlp  = m1_model.pp_mlp,
        m1_merge   = m1_model.merge,
        m2_x_scaler = t_m2_x_sc,
        m2_y_scaler = t_m2_y_sc,
        m1_x_scaler = t_m1_x_sc,
        n_f          = n_f,
        f_positions  = f_positions,
    ).to(device)
    fusion.eval()

    flat_fusion = FlatFusionWrapper(fusion, n_pp).to(device)
    flat_fusion.eval()

    print(f"  FusionModel ready  "
          f"[n_pp={n_pp}  n_f={n_f}  f_positions={f_positions}]")
    print(f"  MergeHead input = n_f + pp_hidden[-1] = "
          f"{n_f} + {m1_cfg['pp_hidden'][-1]} = {n_f + m1_cfg['pp_hidden'][-1]}")

    # ══════════════════════════════════════════════════════════════════════
    # Step 6 — Evaluate on test set
    # ══════════════════════════════════════════════════════════════════════
    print("\n── Step 6: Test evaluation ─────────────────────────────────")
    X_test   = X_pp_m1[test_idx_m1]
    y_test   = y_m1[test_idx_m1]
    ids_test = part_ids_m1[test_idx_m1]

    X_dev    = X_pp_m1[dev_idx_m1]
    y_dev    = y_m1[dev_idx_m1]

    metrics = evaluate_fusion(fusion, X_test, y_test, device)
    preds   = metrics.pop("pred")

    # Save test predictions + metrics
    pd.DataFrame({
        "ID_Part": ids_test,
        "y_true":  y_test,
        "y_pred":  preds,
    }).to_csv(out_dir / "test_predictions.csv", index=False)
    (out_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2))

    # Prediction plots
    _save_scatter(y_test, preds, metrics,
                  out_dir / "scatter_test.png", MAT)
    _save_residuals_scatter(y_test, preds,
                            out_dir / "residuals_scatter_test.png", MAT)
    _save_residuals_hist(y_test, preds,
                         out_dir / "residuals_hist_test.png", MAT)
    print("  Test plots saved.")

    # ══════════════════════════════════════════════════════════════════════
    # Step 7 — XAI setup
    # ══════════════════════════════════════════════════════════════════════
    print("\n── Step 7: XAI setup ───────────────────────────────────────")
    # Flattened inputs: [pp_indirect | pp_direct] = [X_pp | X_pp]
    X_test_flat = np.concatenate([X_test, X_test], axis=1).astype(np.float32)
    X_dev_flat  = np.concatenate([X_dev,  X_dev],  axis=1).astype(np.float32)
    flat_cols   = _flat_col_names(pp_cols)

    # IG/IH baseline = mean of dev data (more physically meaningful than zeros)
    baseline_flat = X_dev_flat.mean(axis=0).astype(np.float32)
    print(f"  Test flat shape : {X_test_flat.shape}")
    print(f"  Dev  flat shape : {X_dev_flat.shape}")
    print(f"  IG/IH baseline  : mean of dev set")

    # ══════════════════════════════════════════════════════════════════════
    # Step 8 — SHAP
    # ══════════════════════════════════════════════════════════════════════
    print("\n── Step 8: SHAP ────────────────────────────────────────────")
    shap_vals = run_shap(flat_fusion, X_dev_flat, X_test_flat,
                         args.shap_bg, device)

    # ══════════════════════════════════════════════════════════════════════
    # Step 9 — Integrated Gradients
    # ══════════════════════════════════════════════════════════════════════
    print("\n── Step 9: Integrated Gradients ────────────────────────────")
    ig_vals = run_ig(flat_fusion, X_test_flat, baseline_flat,
                     args.n_ig_steps, device)

    # ══════════════════════════════════════════════════════════════════════
    # Step 10 — Integrated Hessians
    # ══════════════════════════════════════════════════════════════════════
    print("\n── Step 10: Integrated Hessians ────────────────────────────")
    ih_matrix = run_integrated_hessians(
        flat_fusion, X_test_flat, baseline_flat,
        args.n_ig_steps, args.ih_samples, device)

    # ══════════════════════════════════════════════════════════════════════
    # Step 11 — Jacobian pathway analysis
    # ══════════════════════════════════════════════════════════════════════
    print("\n── Step 11: Jacobian pathway analysis ──────────────────────")
    jac_vals = run_jacobian(flat_fusion, X_test_flat, device)

    # ══════════════════════════════════════════════════════════════════════
    # Step 12 — Save XAI results
    # ══════════════════════════════════════════════════════════════════════
    print("\n── Step 12: Saving XAI results ─────────────────────────────")

    # per-sample CSVs
    if shap_vals is not None:
        build_per_sample_df(shap_vals, ids_test, flat_cols).to_csv(
            out_dir / "shap_values.csv", index=False)
    if ig_vals is not None:
        build_per_sample_df(ig_vals, ids_test, flat_cols).to_csv(
            out_dir / "ig_values.csv", index=False)
    build_per_sample_df(jac_vals, ids_test, flat_cols).to_csv(
        out_dir / "jacobian_values.csv", index=False)

    # IH matrix
    ih_df = pd.DataFrame(ih_matrix, index=flat_cols, columns=flat_cols)
    ih_df.to_csv(out_dir / "ih_matrix.csv")

    # summary (mean |attr| per feature)
    summary = build_summary(pp_cols, shap_vals, ig_vals, jac_vals, n_pp)
    summary.to_csv(out_dir / "feature_attributions.csv")
    print(f"  feature_attributions.csv saved")
    print(summary.to_string())

    # ══════════════════════════════════════════════════════════════════════
    # Step 13 — XAI plots
    # ══════════════════════════════════════════════════════════════════════
    print("\n── Step 13: XAI plots ──────────────────────────────────────")
    save_xai_bars(summary, pp_cols, out_dir, MAT)
    save_total_effect_plot(summary, pp_cols, out_dir, MAT)
    save_ih_heatmap(ih_matrix, flat_cols, out_dir / "ih_heatmap.png", MAT)

    # ══════════════════════════════════════════════════════════════════════
    # Step 14 — Copy model files; save fusion info
    # ══════════════════════════════════════════════════════════════════════
    print("\n── Step 14: Saving model copies & fusion info ──────────────")
    for fname in ["best_model.pt", "best_metrics.json", "hpo_best_config.json"]:
        for src_dir, prefix in [(m1_dir, "m1"), (m2_dir, "m2")]:
            src = src_dir / fname
            if src.exists():
                shutil.copy2(src, out_dir / f"{prefix}_{fname}")

    # Also save pressure_features_f.csv for reference
    src_feat = m1_dir / "pressure_features_f.csv"
    if src_feat.exists():
        shutil.copy2(src_feat, out_dir / "pressure_features_f.csv")

    # Save test data
    df_test_data = df_m1.iloc[test_idx_m1].copy()
    df_test_data["y_pred"] = preds
    df_test_data.to_csv(out_dir / "test_data_with_pred.csv")

    # Fusion model info JSON
    info = {
        "material":     MAT,
        "run_ts":       RUN_TS,
        "device":       str(device),
        "m1_dir":       str(m1_dir),
        "m2_dir":       str(m2_dir),
        "n_pp":         n_pp,
        "n_f":          n_f,
        "f_positions":  f_positions,
        "m2_f_cols":    m2_f_cols,
        "pp_cols":      pp_cols,
        "test_metrics": metrics,
        "n_test":       int(len(test_idx_m1)),
        "n_dev":        int(len(dev_idx_m1)),
        "seed":         SEED,
        "val_fraction": VAL_FR,
        "n_ig_steps":   args.n_ig_steps,
        "ih_samples":   args.ih_samples,
        "shap_bg":      args.shap_bg,
        "m1_model_cfg": m1_cfg,
        "m2_model_cfg": m2_cfg,
        "m1_metrics":   {k: v for k, v in m1_meta.items()
                         if k not in ("model_cfg",)},
        "m2_metrics":   {k: v for k, v in m2_meta.items()
                         if k not in ("model_cfg",)},
        "scaler_info": {
            "m1_pp_sc_data_min":   m1_pp_sc.data_min_.tolist(),
            "m1_pp_sc_data_max":   m1_pp_sc.data_max_.tolist(),
            "m2_x_sc_data_min":    m2_x_sc.data_min_.tolist(),
            "m2_x_sc_data_max":    m2_x_sc.data_max_.tolist(),
            "m2_y_sc_data_min":    m2_y_sc.data_min_.tolist(),
            "m2_y_sc_data_max":    m2_y_sc.data_max_.tolist(),
        },
    }
    (out_dir / "fusion_model_info.json").write_text(json.dumps(info, indent=2))

    print(f"\n{'═'*60}")
    print(f"  All outputs saved → {out_dir}")
    print(f"  Fusion MAE  = {metrics['MAE']:.4f} g")
    print(f"  Fusion R²   = {metrics['R2']:.4f}")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
