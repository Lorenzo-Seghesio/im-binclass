"""
Graphic_NNModel.py
==================
Visualize the architecture of a saved IM-ML model using visualtorch.

CONFIGURATION
-------------
Set the three macro variables below, then run:

    python src/Utility/Graphic_NNModel.py

MODEL_DIR   : path (relative to project root) to the model directory.
              Examples:
                "outputs/ProBayes/RefModels/MLP/PP/best_overall"
                "outputs/ProBayes/RefModels/Encoder/PP/best_overall"
                "outputs/ProBayes/M1/PP/best_overall"
                "outputs/ProBayes/M2/PP/best_overall"
                "outputs/ProBayes/Fusion/2026-05-25_10-49-58_PP"
                "outputs/ProBayes/Reg/MLP/PP/models/best_model_overall"
                "outputs/ProBayes/BC/MLP/PP/models/best_model_overall"

VIEW_TYPE   : "layered"  — colored box per layer type (great for MLPs)
              "graph"    — computation graph (requires Graphviz)

OUTPUT_NAME : filename for the saved image (PNG).
"""

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
MODEL_DIR   = "outputs/ProBayes/Fusion/2026-05-25_10-49-58_PP"
VIEW_TYPE   = "graph"   # "layered" | "graph"
OUTPUT_NAME = "model_architecture.png"
# ─────────────────────────────────────────────────────────────────────────────

import sys
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import visualtorch

# ── Project root on sys.path ──────────────────────────────────────────────────
# This file lives at  src/Utility/Graphic_NNModel.py
# → go up 3 levels (Utility → src → project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Visualization dummy temporal length for Conv1D models ─────────────────────
T_VIZ = 128   # used only for graph tracing; actual T does not affect structure


# ══════════════════════════════════════════════════════════════════════════════
# Wrappers — expose multi-input models via a single tensor input
# ══════════════════════════════════════════════════════════════════════════════

class M1Wrapper(nn.Module):
    """Single-tensor wrapper around M1Model for visualtorch compatibility.

    Concatenates the flattened pressure curve (T_viz timesteps) and the
    process-parameter vector into one flat input:
        x : (B, T_viz + n_pp)
    """

    def __init__(self, m1_model: nn.Module, n_pp: int, T_viz: int):
        super().__init__()
        self.m1    = m1_model
        self.n_pp  = n_pp
        self.T_viz = T_viz

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_pt = x[:, : self.T_viz].unsqueeze(1)   # (B, 1, T_viz)
        x_pp = x[:, self.T_viz :]                 # (B, n_pp)
        return self.m1(x_pt, x_pp)


# ══════════════════════════════════════════════════════════════════════════════
# Helper: dummy identity MinMaxScaler
# ══════════════════════════════════════════════════════════════════════════════

def _make_dummy_scaler(n_features: int):
    """Return a sklearn MinMaxScaler configured as the identity transform.

    Used only to satisfy TorchMinMaxScaler's constructor when loading the
    Fusion model for visualization.  Scaler values do not affect the graph
    structure, only the arithmetic values — irrelevant for architecture plots.
    """
    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler()
    sc.scale_         = np.ones(n_features,  dtype=np.float64)
    sc.min_           = np.zeros(n_features, dtype=np.float64)  # sklearn: X_sc = X*scale_ + min_
    sc.data_min_      = np.zeros(n_features, dtype=np.float64)
    sc.data_max_      = np.ones(n_features,  dtype=np.float64)
    sc.data_range_    = np.ones(n_features,  dtype=np.float64)
    sc.n_features_in_ = n_features
    return sc


# ══════════════════════════════════════════════════════════════════════════════
# Model type detection
# ══════════════════════════════════════════════════════════════════════════════

def detect_model_type(model_dir: Path) -> str:
    """Infer model type from directory path and its contents."""
    # Fusion directory always contains fusion_model_info.json
    if (model_dir / "fusion_model_info.json").exists():
        return "fusion"

    p = str(model_dir).replace("\\", "/")   # normalise on Windows too

    # Non-NN models → not supported
    if any(tag in p for tag in ("BC/GBT", "Reg/GBT", "LightGBM", "XGBoost")):
        raise ValueError(
            f"Path '{p}' points to a tree-based model (GBT/LightGBM/XGBoost).\n"
            "visualtorch only supports PyTorch nn.Module models."
        )

    if "RefModels/Encoder" in p:
        return "ref_encoder"
    if "RefModels/MLP" in p:
        return "ref_mlp"
    if "/M1/" in p or p.endswith("/M1"):
        return "m1"
    if "/M2/" in p or p.endswith("/M2"):
        return "m2"
    if "Reg/MLP" in p:
        return "reg_mlp"
    if "BC/" in p:
        return "bc_mlp"

    raise ValueError(
        f"Cannot detect model type from path:\n  {model_dir}\n"
        "Expected a path containing one of:\n"
        "  RefModels/Encoder, RefModels/MLP, /M1/, /M2/,\n"
        "  Reg/MLP, BC/, or a Fusion directory containing fusion_model_info.json."
    )


# ══════════════════════════════════════════════════════════════════════════════
# Model loaders
# ══════════════════════════════════════════════════════════════════════════════

def load_ref_mlp(model_dir: Path):
    """Reference MLP: process params → weight."""
    from src.Reference_Models_W_Pred import MLPModel

    meta    = json.loads((model_dir / "best_metrics.json").read_text())
    n_in    = meta["n_in"]
    hidden  = meta["hidden_dims"]
    dropout = meta["dropout"]

    model = MLPModel(n_in, hidden, dropout)
    state = torch.load(model_dir / "best_model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model, (1, n_in)


def load_ref_encoder(model_dir: Path):
    """Reference Encoder: Conv1D pressure curve → weight."""
    from src.Reference_Models_W_Pred import EncoderModel

    meta         = json.loads((model_dir / "best_metrics.json").read_text())
    channels     = meta["channels"]
    kernels      = meta["kernels"]
    pool_kernels = meta["pool_kernels"]
    head_hidden  = meta["head_hidden"]
    dropout      = meta["dropout"]

    model = EncoderModel(channels, kernels, pool_kernels, head_hidden, dropout)
    state = torch.load(model_dir / "best_model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model, (1, 1, T_VIZ)


def load_m1(model_dir: Path):
    """M1: (pressure curve, process params) → weight."""
    from src.M1_PPPT_to_W import M1Model

    meta  = json.loads((model_dir / "best_metrics.json").read_text())
    n_pp  = meta["n_pp"]
    mcfg  = meta["model_cfg"]

    model = M1Model(n_pp, mcfg)
    state = torch.load(model_dir / "best_model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)

    wrapper = M1Wrapper(model, n_pp, T_VIZ)
    return wrapper, (1, T_VIZ + n_pp)


def load_m2(model_dir: Path):
    """M2: process params → encoder features."""
    from src.M2_PP_to_F import M2Model

    meta   = json.loads((model_dir / "best_metrics.json").read_text())
    n_in   = meta["n_in"]
    n_out  = meta["n_out"]
    mcfg   = meta["model_cfg"]

    model = M2Model(n_in, n_out, mcfg["hidden_dims"], mcfg["dropout"])
    state = torch.load(model_dir / "best_model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model, (1, n_in)


def load_fusion(model_dir: Path):
    """Full Fusion model: (pp_indirect ‖ pp_direct) → weight."""
    from src.Fusion_M1M2_WPred import (
        M1Model as FM1Model,
        M2Model as FM2Model,
        FusionModel,
        TorchMinMaxScaler,
        FlatFusionWrapper,
    )

    info   = json.loads((model_dir / "fusion_model_info.json").read_text())
    m2meta = json.loads((model_dir / "m2_best_metrics.json").read_text())

    n_pp        = info["n_pp"]
    n_f         = info["n_f"]
    f_positions = info["f_positions"]
    m1_cfg      = info["m1_model_cfg"]
    m2_cfg      = info["m2_model_cfg"]
    n_in        = m2meta["n_in"]
    n_out       = m2meta["n_out"]

    # ── Sub-models ────────────────────────────────────────────────────────────
    m1 = FM1Model(n_pp, m1_cfg)
    m1.load_state_dict(
        torch.load(model_dir / "m1_best_model.pt", map_location="cpu", weights_only=True)
    )

    m2 = FM2Model(n_in, n_out, m2_cfg["hidden_dims"], m2_cfg["dropout"])
    m2.load_state_dict(
        torch.load(model_dir / "m2_best_model.pt", map_location="cpu", weights_only=True)
    )

    # ── Dummy identity scalers (structure unchanged; values irrelevant for viz)
    m2_x_sc = TorchMinMaxScaler(_make_dummy_scaler(n_pp))
    m2_y_sc = TorchMinMaxScaler(_make_dummy_scaler(n_out))
    m1_x_sc = TorchMinMaxScaler(_make_dummy_scaler(n_pp))

    fusion  = FusionModel(m2, m1.pp_mlp, m1.merge,
                          m2_x_sc, m2_y_sc, m1_x_sc,
                          n_f, f_positions)
    wrapper = FlatFusionWrapper(fusion, n_pp)
    return wrapper, (1, 2 * n_pp)


def load_reg_mlp(model_dir: Path):
    """Regression MLP: process params → weight."""
    from src.Reg_MLP_IM import MLPRegression

    meta     = json.loads((model_dir / "metadata.json").read_text())
    n_feat   = meta["n_features"]
    hp       = meta["hyperparameters"]
    dropout  = hp["dropout"]
    n_layers = hp["n_layers"]
    layers_dim = [hp[f"size_layer{i}"] for i in range(n_layers)]

    model = MLPRegression(input_size=n_feat, layers_dim=layers_dim, dropout=dropout)
    pt_files = sorted(model_dir.glob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt model file found in {model_dir}")
    state = torch.load(pt_files[0], map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model, (1, n_feat)


def load_bc_mlp(model_dir: Path):
    """Binary Classification MLP: process params → logit."""
    from src.BC_MLP_IM import BinaryClassifier

    meta     = json.loads((model_dir / "metadata.json").read_text())
    n_feat   = meta["n_features"]
    hp       = meta["hyperparameters"]
    dropout  = hp["dropout"]
    n_layers = hp["n_layers"]
    layers_dim = [hp[f"size_layer{i}"] for i in range(n_layers)]

    model = BinaryClassifier(input_size=n_feat, layers_dim=layers_dim, dropout=dropout)
    pt_files = sorted(model_dir.glob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt model file found in {model_dir}")
    state = torch.load(pt_files[0], map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model, (1, n_feat)


# ══════════════════════════════════════════════════════════════════════════════
# Dispatch table
# ══════════════════════════════════════════════════════════════════════════════

_LOADERS = {
    "ref_mlp":     load_ref_mlp,
    "ref_encoder": load_ref_encoder,
    "m1":          load_m1,
    "m2":          load_m2,
    "fusion":      load_fusion,
    "reg_mlp":     load_reg_mlp,
    "bc_mlp":      load_bc_mlp,
}


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    model_dir = (PROJECT_ROOT / MODEL_DIR).resolve()

    if not model_dir.exists():
        raise FileNotFoundError(f"MODEL_DIR not found:\n  {model_dir}")

    model_type = detect_model_type(model_dir)
    print(f"Model type  : {model_type}")
    print(f"Model dir   : {model_dir}")

    model, input_shape = _LOADERS[model_type](model_dir)
    model.eval()

    print(f"Input shape : {input_shape}")
    print(f"View type   : {VIEW_TYPE}")
    print("Rendering …")

    if VIEW_TYPE == "layered":
        img = visualtorch.layered_view(model, input_shape=input_shape)
    elif VIEW_TYPE == "graph":
        img = visualtorch.graph_view(model, input_shape=input_shape)
    else:
        raise ValueError(
            f"Unknown VIEW_TYPE {VIEW_TYPE!r}.  Choose 'layered' or 'graph'."
        )

    out_path = model_dir / OUTPUT_NAME
    img.save(str(out_path))
    print(f"Saved → {out_path}")
