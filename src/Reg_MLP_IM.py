# Reg_MLP_IM.py
#
# Regression using MLPs for weight prediction for Injection Molding (IM) data.
#
# This code is part of a machine learning project for regression using MLPs with hyperparameter optimization (HPO) and pruning techniques.
# It includes model definition, training, evaluation, and visualization of results.

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import argparse
import time
import matplotlib.pyplot as plt
import optuna
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # project root (works from any subfolder)

# Output root — overridden at runtime in __main__ based on dataset choice
# e.g. outputs/ProBayes/Reg/ABS, outputs/ProBayes/Reg/PP_1, etc.
OUT_DIR = BASE_DIR / 'outputs/ProBayes/Reg'

# Colour palette (Tableau 10) — consistent across all plot types
CAVITY_COLORS      = {'P1': '#4E79A7', 'P2': '#F28E2B'}   # steel blue / orange
MODEL_COLORS       = {'TPE': '#59A14F', 'RS': '#B07AA1'}  # green / purple
PERFECT_PRED_COLOR = '#333333'                             # dark gray

# Global variables
best_metric_global = float('inf')
best_model_global = None
best_metric_RS_global = float('inf')
best_model_RS_global = None
best_params_RS_global = None

test_csv_path = str(BASE_DIR / 'data/IM_Data_Test.csv')  # Path to the test dataset
train_csv_path = str(BASE_DIR / 'data/IM_Data_Train.csv')  # Path to the training dataset


# === Model Definition ===
class MLPRegression(nn.Module):
    def __init__(self, input_size=18, layers_dim=1, dropout=0.2):
        super().__init__()

        layers = []

        for layer_dim in layers_dim:
            layers.append(nn.Linear(input_size, layer_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_size = layer_dim

        layers.append(nn.Linear(input_size, 1))

        self.net = nn.Sequential(*layers) 

    def forward(self, x):
        return self.net(x)


# === EarlyStopping Callback ===
class EarlyStopping:
    def __init__(self, patience=5, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, score, model):
        if self.best_score is None or score > self.best_score + self.delta:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# === Data Loader ===
def load_dataset(csv_path, return_groups=False, return_cavity=False):
    df = pd.read_csv(csv_path)
    
    target_col = 'Product weight g'
    drop_cols = [c for c in ['shot', 'cavity', target_col] if c in df.columns]

    # Check if 'shot' column exists for grouping
    if 'shot' in df.columns and return_groups:
        groups = df['shot'].values
    else:
        groups = None

    # Extract cavity labels (from shot_position) BEFORE building X.
    # shot_position stays as a feature in X so the model can learn the per-cavity offset.
    if return_cavity:
        cavity_labels = df['shot_position'].values if 'shot_position' in df.columns else None
    
    y = df[target_col].values
    X = df.drop(columns=drop_cols).values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    if return_groups and return_cavity:
        return X, y, groups, cavity_labels
    if return_groups:
        return X, y, groups
    if return_cavity:
        return X, y, cavity_labels
    return X, y


# === Per-cavity target normalisation helpers ===
def _compute_cavity_stats(y, cavity_labels):
    """Compute per-cavity (mean, std) from y.
    Returns a dict {cavity_id: (mean, std)}.
    Falls back to {None: (mean, std)} when cavity_labels is None or single-valued.
    """
    if cavity_labels is None or len(np.unique(cavity_labels)) <= 1:
        return {None: (float(y.mean()), max(float(y.std()), 1e-8))}
    stats = {}
    for cav in np.unique(cavity_labels):
        ys = y[cavity_labels == cav]
        stats[int(cav)] = (float(ys.mean()), max(float(ys.std()), 1e-8))
    return stats


def _normalize_y(y, cavity_labels, cavity_stats):
    """Normalise y using per-cavity stats (works on any train/val/test subset)."""
    y_norm = np.empty(len(y), dtype=float)
    if None in cavity_stats:
        m, s = cavity_stats[None]
        y_norm[:] = (y - m) / s
    else:
        for cav, (m, s) in cavity_stats.items():
            mask = cavity_labels == cav
            y_norm[mask] = (y[mask] - m) / s
    return y_norm


def _build_inv_arrays(cavity_labels, cavity_stats, n=None):
    """Build per-sample y_mean and y_std arrays for inverse-transform:
    y_orig = y_norm * y_std_arr + y_mean_arr  (element-wise numpy broadcast).
    cavity_labels may be None for single-cavity datasets; pass n=len(y) in that case."""
    if cavity_labels is not None:
        n = len(cavity_labels)
    y_mean_arr = np.empty(n, dtype=float)
    y_std_arr  = np.empty(n, dtype=float)
    if None in cavity_stats:
        m, s = cavity_stats[None]
        y_mean_arr[:] = m
        y_std_arr[:]  = s
    else:
        for cav, (m, s) in cavity_stats.items():
            mask = cavity_labels == cav
            y_mean_arr[mask] = m
            y_std_arr[mask]  = s
    return y_mean_arr, y_std_arr


# === Weight Initialization for Regression ===
def init_weights_with_prior(model: nn.Module, pos_prior: float | None = None, method: str = "kaiming"):
    """
    Initialize Linear layers for regression:
      - Hidden: Kaiming (He) for ReLU
      - Output: Xavier for weights; bias set to target mean if provided
    
    Parameters:
    - pos_prior: For regression, this should be the mean of the target values
    """
    # Find last Linear layer (output)
    last_linear = None
    if hasattr(model, "net"):
        for m in reversed(model.net):
            if isinstance(m, nn.Linear):
                last_linear = m
                break

    def _init(m):
        if isinstance(m, nn.Linear):
            if m is last_linear:
                # Output layer: Xavier initialization
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    if pos_prior is not None:
                        # For regression: initialize bias to mean of target values
                        m.bias.data.fill_(float(pos_prior))
                    else:
                        nn.init.zeros_(m.bias)
            else:
                # Hidden layers: Kaiming initialization for ReLU
                if method == "kaiming":
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                else:
                    nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    model.apply(_init)
    return model

# === Outlier Detection using IQR ===
def detect_outliers_iqr(series, multiplier=1.5):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    return (series < lower_bound) | (series > upper_bound)


# === Training Function ===
def train_one_fold_test(model, train_loader, val_loader, device, criterion, optimizer, patience=5, max_epochs=100, plot_metrics=False, print_early_stopping=False, fold=0, sampler="", opt_metric='mae'):
    
    early_stopping = EarlyStopping(patience)

    train_losses = []
    val_losses = []
    val_r2_scores = []

    for epoch in range(max_epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        epoch_val_loss = 0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                outputs = model(xb)
                epoch_val_loss += criterion(outputs, yb).item()
                all_preds.extend(outputs.cpu().numpy().flatten())
                all_targets.extend(yb.cpu().numpy().flatten())

        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        r2 = r2_score(all_targets, all_preds)
        val_r2_scores.append(r2)

        early_stopping(-compute_optuna_metric(all_targets, all_preds, opt_metric), model)
        if early_stopping.early_stop:
            if print_early_stopping:
                print(f"Early stopping at epoch {epoch + 1}")
            break

    if print_early_stopping and not early_stopping.early_stop:
        print(f"Training completed after {max_epochs} epochs without early stopping.")

    model.load_state_dict(early_stopping.best_model_state)

    # Plot training curves if requested
    if plot_metrics:

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss plot
        ax1.plot(train_losses, label='Train MAE Loss', color='blue')
        ax1.plot(val_losses, label='Val MAE Loss', color='red')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('MAE Loss')
        ax1.set_title('{} - Training and Validation Loss Fold {}'.format(sampler, fold + 1))
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # R² plot
        ax2.plot(val_r2_scores, label='Val R² Score', color='green')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('R² Score')
        ax2.set_title('{} - Validation R² Score Fold {}'.format(sampler, fold + 1))
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(str(OUT_DIR / 'images/{}_training_curves_fold_{}.png'.format(sampler, fold + 1)))
        plt.close()

    return model


# === Training Function ===
def train_one_fold_hpo(model, train_loader, val_loader, device, criterion, optimizer, patience=5, max_epochs=100, opt_metric='mae'):
    
    early_stopping = EarlyStopping(patience)

    for epoch in range(max_epochs):
        # Training phase
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                outputs = model(xb)
                all_preds.extend(outputs.cpu().numpy().flatten())
                all_targets.extend(yb.cpu().numpy().flatten())

        # Early stopping: negate optuna metric so higher score = better model
        early_stopping(-compute_optuna_metric(all_targets, all_preds, opt_metric), model)
        if early_stopping.early_stop:
            break

    model.load_state_dict(early_stopping.best_model_state)

    return model


# === Optimization metric helper ===
# All metrics are returned as a value to MINIMIZE (lower = better).
# R2 is stored as 1-R2 so it also follows "lower is better".
VALID_OPT_METRICS = ['mae', 'rmse', 'r2', 'mape', 'max_error']

def compute_optuna_metric(y_true, y_pred, metric):
    """Returns a value to MINIMIZE. Lower = better for all. R2 → 1-R2."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if metric == 'mae':
        return float(mean_absolute_error(y_true, y_pred))
    elif metric == 'rmse':
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))
    elif metric == 'r2':
        return float(1.0 - r2_score(y_true, y_pred))
    elif metric == 'mape':
        return float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))) * 100)
    elif metric == 'max_error':
        return float(np.max(np.abs(y_true - y_pred)))
    else:
        raise ValueError(f"Unknown opt_metric '{metric}'. Choose from {VALID_OPT_METRICS}")


# === Evaluate Model ===
def evaluate_model(model, loader, device, metric='mae'):
    """Run inference on a DataLoader and return the Optuna-minimize metric (lower = better; R2 → 1-R2)."""
    if metric not in VALID_OPT_METRICS:
        raise ValueError(f"metric must be one of {VALID_OPT_METRICS}")
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            all_preds.extend(outputs.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
    return compute_optuna_metric(all_labels, all_preds, metric)


# === Save Best Overall Model ===
def save_best_overall_model(model, model_name, mae, rmse, r2, mape, max_error, X_train, y_train, X_test, y_test, params, opt_metric='mae'):
    """
    Save the best overall model with its metadata, data, and hyperparameters.
    Only saves if the current model is better than the previously saved one.
    
    Parameters:
    - model: trained PyTorch model
    - model_name: 'TPE' or 'RS'
    - mae: Mean Absolute Error
    - rmse: Root Mean Squared Error
    - r2: R² score
    - mape: Mean Absolute Percentage Error
    - max_error: Maximum error
    - X_train, y_train: training data
    - X_test, y_test: test data
    - params: hyperparameters dictionary
    
    Returns:
    - saved: True if model was saved (better than previous), False otherwise
    """
    best_model_dir = str(OUT_DIR / 'models/best_model_overall')
    metadata_file = os.path.join(best_model_dir, 'metadata.json')
    
    # Map all metrics to an optuna score (always lower = better; R2 → 1-R2)
    _scores = {'mae': mae, 'rmse': rmse, 'r2': 1.0 - r2, 'mape': mape, 'max_error': max_error}
    curr_score = _scores[opt_metric]

    # Check if there's a previous best model
    should_save = True
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            prev_metadata = json.load(f)
        
        prev_score = prev_metadata.get('opt_score', prev_metadata.get('mae'))
        
        print(f"\n=== Comparing with previous best model ({opt_metric.upper()}) ===")
        print(f"Previous best: {prev_metadata['model_name']} - {opt_metric.upper()}: {prev_score:.4f}")
        print(f"Current model: {model_name} - {opt_metric.upper()}: {curr_score:.4f}")
        
        if curr_score >= prev_score:
            print(f"Current model is not better than previous best. Not saving.")
            should_save = False
        else:
            print(f"Current model is BETTER! Overwriting previous best model.")
    else:
        print(f"\n=== No previous best model found. Saving current model as best. ===")
        print(f"Current model: {model_name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
    
    if should_save:
        # Create directory if it doesn't exist (or clear it)
        if os.path.exists(best_model_dir):
            shutil.rmtree(best_model_dir)
        os.makedirs(best_model_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(best_model_dir, f'best_model_{model_name}.pt')
        torch.save(model.state_dict(), model_path)
        
        # Save training data
        train_data = pd.DataFrame(X_train)
        train_data['Product Weight g'] = y_train
        train_data.to_csv(os.path.join(best_model_dir, 'train_data.csv'), index=False)
        
        # Save test data
        test_data = pd.DataFrame(X_test)
        test_data['Product Weight g'] = y_test
        test_data.to_csv(os.path.join(best_model_dir, 'test_data.csv'), index=False)
        
        # Save metadata
        metadata = {
            'model_name': model_name,
            'opt_metric': opt_metric,
            'opt_score': float(curr_score),
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'mape': float(mape),
            'max_error': float(max_error),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'hyperparameters': params,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'n_features': X_train.shape[1]
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        # Copy visualization files for the best model
        scatter_plot_src = str(OUT_DIR / f'images/scatter_plot_{model_name}.png')
        scatter_plot_dst = os.path.join(best_model_dir, 'scatter_plot.png')
        if os.path.exists(scatter_plot_src):
            shutil.copy2(scatter_plot_src, scatter_plot_dst)
        
        residual_plot_src = str(OUT_DIR / f'images/residual_plot_{model_name}.png')
        residual_plot_dst = os.path.join(best_model_dir, 'residual_plot.png')
        if os.path.exists(residual_plot_src):
            shutil.copy2(residual_plot_src, residual_plot_dst)
        
        # Copy comparison plots
        comparison_src = str(OUT_DIR / 'images/metrics_comparison.png')
        comparison_dst = os.path.join(best_model_dir, 'metrics_comparison.png')
        if os.path.exists(comparison_src):
            shutil.copy2(comparison_src, comparison_dst)
        
        print(f"\n✅ Best overall model saved to {best_model_dir}/")
        print(f"   - Model: best_model_{model_name}.pt")
        print(f"   - Train data: train_data.csv ({len(X_train)} samples)")
        print(f"   - Test data: test_data.csv ({len(X_test)} samples)")
        print(f"   - Metadata: metadata.json")
        print(f"   - Scatter plot: scatter_plot.png")
        print(f"   - Residual plot: residual_plot.png")
        print(f"   - Metrics comparison: metrics_comparison.png")
        
    return should_save

# === Evaluate Model and plot results ===
def evaluate_and_plot_results(model_tp, model_rs, X_test, y_test, device, save_path=None,
                               y_mean=0.0, y_std=1.0):
    """
    Evaluate regression models and create visualizations.

    Parameters:
    - model_tp: TPE optimized model
    - model_rs: Random search optimized model
    - X_test: test features
    - y_test: normalised test targets (pass y_mean/y_std to recover original scale)
    - device: torch device
    - y_mean, y_std: target normalisation parameters; predictions are inverse-transformed before metrics
    - save_path: path to save comparison plot

    Returns:
    - Dictionary with metrics for both models (all in original scale)
    """
    # Evaluate model_tp
    model_tp.eval()
    with torch.no_grad():
        y_pred_tp = model_tp(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy().flatten()
    # Inverse-transform to original scale
    y_pred_tp = y_pred_tp * y_std + y_mean
    y_test    = y_test * y_std + y_mean  # rebind once — all subsequent code uses original-scale values

    mae_tp = mean_absolute_error(y_test, y_pred_tp)
    rmse_tp = np.sqrt(mean_squared_error(y_test, y_pred_tp))
    r2_tp = r2_score(y_test, y_pred_tp)
    mape_tp = np.mean(np.abs((y_test - y_pred_tp) / y_test)) * 100
    max_error_tp = np.max(np.abs(y_test - y_pred_tp))

    print(f"\n=== TPE Model Test Results ===")
    print(f"MAE: {mae_tp:.4f}")
    print(f"RMSE: {rmse_tp:.4f}")
    print(f"R²: {r2_tp:.4f}")
    print(f"MAPE: {mape_tp:.2f}%")
    print(f"Max Error: {max_error_tp:.4f}")

    # Evaluate model_rs
    model_rs.eval()
    with torch.no_grad():
        y_pred_rs = model_rs(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy().flatten()
    y_pred_rs = y_pred_rs * y_std + y_mean  # inverse-transform

    mae_rs = mean_absolute_error(y_test, y_pred_rs)
    rmse_rs = np.sqrt(mean_squared_error(y_test, y_pred_rs))
    r2_rs = r2_score(y_test, y_pred_rs)
    mape_rs = np.mean(np.abs((y_test - y_pred_rs) / y_test)) * 100
    max_error_rs = np.max(np.abs(y_test - y_pred_rs))

    print(f"\n=== RS Model Test Results ===")
    print(f"MAE: {mae_rs:.4f}")
    print(f"RMSE: {rmse_rs:.4f}")
    print(f"R²: {r2_rs:.4f}")
    print(f"MAPE: {mape_rs:.2f}%")
    print(f"Max Error: {max_error_rs:.4f}")

    # Determine plot colours: single-cavity → cavity colour; double-cavity → model colours
    _name = OUT_DIR.name
    if _name.endswith('_1'):
        _color_tp = _color_rs = CAVITY_COLORS['P1']
    elif _name.endswith('_2'):
        _color_tp = _color_rs = CAVITY_COLORS['P2']
    else:
        _color_tp, _color_rs = MODEL_COLORS['TPE'], MODEL_COLORS['RS']

    # Create scatter plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # TPE scatter plot
    ax1.scatter(y_test, y_pred_tp, alpha=0.5, s=30, color=_color_tp)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             '--', color=PERFECT_PRED_COLOR, lw=2, label='Perfect prediction')
    ax1.set_xlabel('True Values [g]')
    ax1.set_ylabel('Predictions [g]')
    ax1.set_title(f'TPE Model: Predicted vs True\nMAE={mae_tp:.4f}, R²={r2_tp:.4f}')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # RS scatter plot
    ax2.scatter(y_test, y_pred_rs, alpha=0.5, s=30, color=_color_rs)
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             '--', color=PERFECT_PRED_COLOR, lw=2, label='Perfect prediction')
    ax2.set_xlabel('True Values [g]')
    ax2.set_ylabel('Predictions [g]')
    ax2.set_title(f'RS Model: Predicted vs True\nMAE={mae_rs:.4f}, R²={r2_rs:.4f}')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(str(OUT_DIR / 'images/scatter_plots_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save individual scatter plots
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred_tp, alpha=0.5, s=30, color=_color_tp)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             '--', color=PERFECT_PRED_COLOR, lw=2, label='Perfect prediction')
    plt.xlabel('True Values [g]')
    plt.ylabel('Predictions [g]')
    plt.title(f'TPE Model: Predicted vs True\nMAE={mae_tp:.4f}, R²={r2_tp:.4f}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(str(OUT_DIR / 'images/scatter_plot_TPE.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred_rs, alpha=0.5, s=30, color=_color_rs)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             '--', color=PERFECT_PRED_COLOR, lw=2, label='Perfect prediction')
    plt.xlabel('True Values [g]')
    plt.ylabel('Predictions [g]')
    plt.title(f'RS Model: Predicted vs True\nMAE={mae_rs:.4f}, R²={r2_rs:.4f}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(str(OUT_DIR / 'images/scatter_plot_RS.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Create residual plots
    residuals_tp = y_test - y_pred_tp
    residuals_rs = y_test - y_pred_rs
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # TPE residual plot
    ax1.scatter(y_pred_tp, residuals_tp, alpha=0.5, s=30, color=_color_tp)
    ax1.axhline(y=0, color=PERFECT_PRED_COLOR, linestyle='--', lw=2)
    ax1.set_xlabel('Predicted Values [g]')
    ax1.set_ylabel('Residuals [g]')
    ax1.set_title(f'TPE Model: Residual Plot\nMAE={mae_tp:.4f}')
    ax1.grid(alpha=0.3)
    
    # RS residual plot
    ax2.scatter(y_pred_rs, residuals_rs, alpha=0.5, s=30, color=_color_rs)
    ax2.axhline(y=0, color=PERFECT_PRED_COLOR, linestyle='--', lw=2)
    ax2.set_xlabel('Predicted Values [g]')
    ax2.set_ylabel('Residuals [g]')
    ax2.set_title(f'RS Model: Residual Plot\nMAE={mae_rs:.4f}')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(str(OUT_DIR / 'images/residual_plots_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save individual residual plots
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred_tp, residuals_tp, alpha=0.5, s=30, color=_color_tp)
    plt.axhline(y=0, color=PERFECT_PRED_COLOR, linestyle='--', lw=2)
    plt.xlabel('Predicted Values [g]')
    plt.ylabel('Residuals [g]')
    plt.title(f'TPE Model: Residual Plot\nMAE={mae_tp:.4f}')
    plt.grid(alpha=0.3)
    plt.savefig(str(OUT_DIR / 'images/residual_plot_TPE.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred_rs, residuals_rs, alpha=0.5, s=30, color=_color_rs)
    plt.axhline(y=0, color=PERFECT_PRED_COLOR, linestyle='--', lw=2)
    plt.xlabel('Predicted Values [g]')
    plt.ylabel('Residuals [g]')
    plt.title(f'RS Model: Residual Plot\nMAE={mae_rs:.4f}')
    plt.grid(alpha=0.3)
    plt.savefig(str(OUT_DIR / 'images/residual_plot_RS.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Create metrics comparison bar chart
    metrics_names = ['MAE', 'RMSE', 'R²', 'MAPE (%)', 'Max Error']
    tp_values = [mae_tp, rmse_tp, r2_tp, mape_tp, max_error_tp]
    rs_values = [mae_rs, rmse_rs, r2_rs, mape_rs, max_error_rs]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, tp_values, width, label='TPE Model', alpha=0.8)
    bars2 = ax.bar(x + width/2, rs_values, width, label='RS Model', alpha=0.8)
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Values')
    ax.set_title('Regression Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(str(OUT_DIR / 'images/metrics_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlots saved in {OUT_DIR / 'images'}:")
    print(f"  - scatter_plot_TPE.png")
    print(f"  - scatter_plot_RS.png")
    print(f"  - scatter_plots_comparison.png")
    print(f"  - residual_plot_TPE.png")
    print(f"  - residual_plot_RS.png")
    print(f"  - residual_plots_comparison.png")
    print(f"  - metrics_comparison.png")
    
    # Return metrics for both models
    return {
        'tp': {
            'mae': mae_tp,
            'rmse': rmse_tp,
            'r2': r2_tp,
            'mape': mape_tp,
            'max_error': max_error_tp
        },
        'rs': {
            'mae': mae_rs,
            'rmse': rmse_rs,
            'r2': r2_rs,
            'mape': mape_rs,
            'max_error': max_error_rs
        }
    }


# === Objective Function ===
def objective(trial, csv_path, n_startup_trials=10, sampler="RandomSampler", hparam_cfg=None):
    global best_metric_global
    global best_model_global
    global best_metric_RS_global
    global best_model_RS_global
    global best_params_RS_global

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, y, groups, cavity_labels = load_dataset(csv_path, return_groups=True, return_cavity=True)
    # y is normalised per fold inside the loop (per-cavity, train-fold stats only — no leakage).

    # Use GroupKFold if groups are available, otherwise use KFold
    if groups is not None:
        skf = GroupKFold(n_splits=5)
    else:
        skf = KFold(n_splits=5, shuffle=True, random_state=42)

    hp = (hparam_cfg or {}).get('hyperparameters', {})
    opt_metric = (hparam_cfg or {}).get('opt_metric', 'mae')
    # Hyperparameters — scalar in config → fixed; [min, max] → optimized by Optuna
    lr_cfg = hp.get('lr', [1e-5, 1e-2])
    lr = trial.suggest_float("lr", lr_cfg[0], lr_cfg[1], log=True) if isinstance(lr_cfg, list) else float(lr_cfg)

    bs_cfg = hp.get('batch_size', 32)
    batch_size = trial.suggest_int("batch_size", bs_cfg[0], bs_cfg[1]) if isinstance(bs_cfg, list) else int(bs_cfg)

    do_cfg = hp.get('dropout', [0.0, 0.4])
    dropout = trial.suggest_float("dropout", do_cfg[0], do_cfg[1]) if isinstance(do_cfg, list) else float(do_cfg)

    wd_cfg = hp.get('weight_decay', [0, 1e-2])
    weight_decay = trial.suggest_float("weight_decay", wd_cfg[0], wd_cfg[1], log=bool(wd_cfg[0] > 0)) if isinstance(wd_cfg, list) else float(wd_cfg)

    nl_cfg = hp.get('n_layers', [2, 4])
    n_layers = trial.suggest_int("n_layers", nl_cfg[0], nl_cfg[1]) if isinstance(nl_cfg, list) else int(nl_cfg)

    sl_cfg = hp.get('size_1st_hidden_layer', None)
    layers_dim = []
    if isinstance(sl_cfg, list) or sl_cfg is None:
        if isinstance(sl_cfg, list):
            neuron_max_limit = sl_cfg[1]
            neuron_min_limit = sl_cfg[0]
        else:
            # Fallback: dynamic bounds from feature count
            neuron_min_limit = X.shape[1]
            neuron_max_limit = 20 * X.shape[1]
        for i in range(n_layers):
            size_layer = trial.suggest_int("size_layer{}".format(i), neuron_min_limit, neuron_max_limit, log=True)
            neuron_max_limit = size_layer
            neuron_min_limit = 1
            layers_dim.append(size_layer)
    else:
        # scalar: first layer fixed; record in trial.params for consistency
        fixed = int(sl_cfg)
        trial.suggest_int("size_layer0", fixed, fixed)
        layers_dim.append(fixed)
        neuron_max_limit = fixed
        for i in range(1, n_layers):
            size_layer = trial.suggest_int("size_layer{}".format(i), 1, neuron_max_limit)
            neuron_max_limit = size_layer
            layers_dim.append(size_layer)

    metric_values = []
    best_metric = float('inf')
    best_model = None

    # Split with groups if available
    if groups is not None:
        fold_iterator = skf.split(X, y, groups=groups)
    else:
        fold_iterator = skf.split(X, y)
    
    for fold, (train_idx, val_idx) in enumerate(fold_iterator):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train_raw, y_val_raw = y[train_idx], y[val_idx]
        # Per-cavity normalisation using train-fold statistics only (no leakage from val).
        # For single-cavity datasets cavity_labels is None → falls back to global stats.
        cav_train = cavity_labels[train_idx] if cavity_labels is not None else None
        cav_val   = cavity_labels[val_idx]   if cavity_labels is not None else None
        fold_stats = _compute_cavity_stats(y_train_raw, cav_train)
        y_train = _normalize_y(y_train_raw, cav_train, fold_stats)
        y_val   = _normalize_y(y_val_raw,   cav_val,   fold_stats)

        train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                 torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
        val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                               torch.tensor(y_val, dtype=torch.float32).unsqueeze(1))

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        model = MLPRegression(input_size=X.shape[1], layers_dim=layers_dim, dropout=dropout).to(device)
        init_weights_with_prior(model, pos_prior=0.0)  # y is normalised → output bias starts at 0
        criterion = nn.MSELoss()  # MSE penalises deviations from the trend; L1 predicts the median # TODO: Check if using MSE is better than using L1
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Train
        model = train_one_fold_hpo(model, train_loader, val_loader, device, criterion, optimizer, opt_metric=opt_metric)

        # Evaluate
        metric_value = evaluate_model(model, val_loader, device, opt_metric)
        metric_values.append(metric_value)

        if metric_value < best_metric:
            best_metric = metric_value
            best_model = model
        
        trial.report(np.mean(metric_values), fold)

        if trial.should_prune():
            raise optuna.TrialPruned()

    metric_mean = np.mean(metric_values)

    if best_model is not None:
        if best_model_global is None or metric_mean < best_metric_global:
            best_metric_global = metric_mean
            best_model_global = best_model
            torch.save(best_model_global.state_dict(), str(OUT_DIR / f"models/best_model_{opt_metric.upper()}_global.pt"))
            if (sampler == "TPESampler") and (trial.number < n_startup_trials):
                    best_metric_RS_global = best_metric_global
                    best_model_RS_global = best_model_global
                    best_params_RS_global = trial.params  # Store the TRIAL parameters (hyperparameters), not model weights
                    torch.save(best_model_RS_global.state_dict(), str(OUT_DIR / f"models/best_model_{opt_metric.upper()}_RS.pt"))

    return metric_mean


# === Retrain Final Model ===
def train_and_save_best_model(params_tpe, params_rs, epochs=100, csv_path_train=str(BASE_DIR / 'data/IM_Data_Train.csv'), csv_path_test=str(BASE_DIR / 'data/IM_Data_Test.csv'), hparam_cfg=None):
    hp = (hparam_cfg or {}).get('hyperparameters', {})
    opt_metric = (hparam_cfg or {}).get('opt_metric', 'mae')
    # hyperparameters: use trial value if optimized (present in params), else config/ProBayes/default
    lr_tp = params_tpe.get('lr', 1e-3) if isinstance(hp.get('lr'), list) else hp.get('lr', 1e-3)
    dropout_tp = params_tpe.get('dropout', 0.2) if isinstance(hp.get('dropout'), list) else hp.get('dropout', 0.2)
    weight_decay_tp = params_tpe.get('weight_decay', 1e-4) if isinstance(hp.get('weight_decay'), list) else hp.get('weight_decay', 1e-4)
    batch_size_tp = params_tpe.get('batch_size', 32) if isinstance(hp.get('batch_size'), list) else hp.get('batch_size', 32)
    lr_rs = params_rs.get('lr', 1e-3) if isinstance(hp.get('lr'), list) else hp.get('lr', 1e-3)
    dropout_rs = params_rs.get('dropout', 0.2) if isinstance(hp.get('dropout'), list) else hp.get('dropout', 0.2)
    weight_decay_rs = params_rs.get('weight_decay', 1e-4) if isinstance(hp.get('weight_decay'), list) else hp.get('weight_decay', 1e-4)
    batch_size_rs = params_rs.get('batch_size', 32) if isinstance(hp.get('batch_size'), list) else hp.get('batch_size', 32)


    print(f"\nTraining the best model TPE and RS...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train data
    X_train, y_train, groups, cavity_labels_train = load_dataset(csv_path_train, return_groups=True, return_cavity=True)
    print(f"The Data has {X_train.shape[0]} samples and {X_train.shape[1]} features.")
    # Per-cavity normalisation stats from the full training set.
    # y_mean_arr / y_std_arr are per-sample arrays used for inverse-transform on the test set.
    # Inside the CV loop, per-fold-per-cavity stats are recomputed from train indices only.
    cavity_stats_train = _compute_cavity_stats(y_train, cavity_labels_train)
    if None in cavity_stats_train:
        m, s = cavity_stats_train[None]
        cv_pct = s / abs(m) * 100
        print(f"Target — mean: {m:.4f} g, std: {s:.4f} g, CV: {cv_pct:.2f}%")
        if cv_pct < 1.0:
            print("  [!] Low variance target (CV < 1%). Setting opt_metric='r2' in config is strongly recommended.")
    else:
        for cav, (m, s) in sorted(cavity_stats_train.items()):
            cv_pct = s / abs(m) * 100
            print(f"  Cavity {cav} — mean: {m:.4f} g, std: {s:.4f} g, CV: {cv_pct:.2f}%")
            if cv_pct < 1.0:
                print(f"    [!] Cavity {cav}: low variance target (CV < 1%). Setting opt_metric='r2' is recommended.")

    # Use GroupKFold if groups are available, otherwise use KFold
    if groups is not None:
        skf = GroupKFold(n_splits=5)
        print(f"Using GroupKFold to keep shots together in folds")
    else:
        skf = KFold(n_splits=5, shuffle=True, random_state=42)
        print(f"Using KFold (no shot grouping available)")

    metric_values_tp = []
    metric_values_rs = []

    # Increased patience and epochs for the final training phase
    early_stopping_patience = 20
    num_epochs = 200

    # Split with groups if available
    if groups is not None:
        fold_iterator = skf.split(X_train, y_train, groups=groups)
    else:
        fold_iterator = skf.split(X_train, y_train)

    for fold, (train_idx, val_idx) in enumerate(fold_iterator):

        # Per-fold per-cavity normalisation — train-fold statistics only, no leakage from val fold.
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold_raw, y_val_fold_raw = y_train[train_idx], y_train[val_idx]
        cav_train_fold = cavity_labels_train[train_idx] if cavity_labels_train is not None else None
        cav_val_fold   = cavity_labels_train[val_idx]   if cavity_labels_train is not None else None
        fold_stats = _compute_cavity_stats(y_train_fold_raw, cav_train_fold)
        y_train_fold = _normalize_y(y_train_fold_raw, cav_train_fold, fold_stats)
        y_val_fold   = _normalize_y(y_val_fold_raw,   cav_val_fold,   fold_stats)

        # Initialize models for each fold
        # TPE
        model_tp = MLPRegression(input_size=X_train.shape[1], layers_dim=[params_tpe["size_layer{}".format(i)] for i in range(params_tpe["n_layers"])], dropout=dropout_tp).to(device)
        criterion_tp = nn.MSELoss()
        optimizer_tp = torch.optim.AdamW(model_tp.parameters(), lr=lr_tp, weight_decay=weight_decay_tp)
        init_weights_with_prior(model_tp, pos_prior=0.0)  # y is normalised
        # RS
        model_rs = MLPRegression(input_size=X_train.shape[1], layers_dim=[params_rs["size_layer{}".format(i)] for i in range(params_rs["n_layers"])], dropout=dropout_rs).to(device)
        criterion_rs = nn.MSELoss()
        optimizer_rs = torch.optim.AdamW(model_rs.parameters(), lr=lr_rs, weight_decay=weight_decay_rs)
        init_weights_with_prior(model_rs, pos_prior=0.0)

        train_ds = TensorDataset(torch.tensor(X_train_fold, dtype=torch.float32),
                                 torch.tensor(y_train_fold, dtype=torch.float32).unsqueeze(1))
        val_ds = TensorDataset(torch.tensor(X_val_fold, dtype=torch.float32),
                               torch.tensor(y_val_fold, dtype=torch.float32).unsqueeze(1))

        train_loader_tpe = DataLoader(train_ds, batch_size=batch_size_tp, shuffle=True)
        val_loader_tpe = DataLoader(val_ds, batch_size=batch_size_tp)

        train_loader_rs = DataLoader(train_ds, batch_size=batch_size_rs, shuffle=True)
        val_loader_rs = DataLoader(val_ds, batch_size=batch_size_rs)

        # Train TP and RS
        model_tp = train_one_fold_test(model_tp, train_loader_tpe, val_loader_tpe, device, criterion_tp, optimizer_tp, early_stopping_patience, num_epochs, plot_metrics=True, print_early_stopping=True, fold=fold, sampler="TPE", opt_metric=opt_metric)
        model_rs = train_one_fold_test(model_rs, train_loader_rs, val_loader_rs, device, criterion_rs, optimizer_rs, early_stopping_patience, num_epochs, plot_metrics=True, print_early_stopping=True, fold=fold, sampler="RS", opt_metric=opt_metric)

        # Collect CV metrics (fold scores only — fold models are discarded afterwards)
        metric_values_tp.append(evaluate_model(model_tp, val_loader_tpe, device, opt_metric))
        metric_values_rs.append(evaluate_model(model_rs, val_loader_rs, device, opt_metric))

    metric_label = opt_metric.upper()
    print(f"\nCV {metric_label} — TPE: {np.mean(metric_values_tp):.4f} ± {np.std(metric_values_tp):.4f}")
    print(f"CV {metric_label} — RS:  {np.mean(metric_values_rs):.4f} ± {np.std(metric_values_rs):.4f}")
    if opt_metric in ('mae', 'rmse', 'max_error'):
        print(f"  (fold {metric_label} is in normalised units; multiply by the cavity std to convert to grams)")

    # Retrain final models on ALL training data.
    # cavity_stats_train comes from the full training CSV — correct, no test leakage.
    print(f"\nRetraining final TPE and RS models on all training data...")
    y_train_n = _normalize_y(y_train, cavity_labels_train, cavity_stats_train)

    final_model_tp = MLPRegression(input_size=X_train.shape[1], layers_dim=[params_tpe["size_layer{}".format(i)] for i in range(params_tpe["n_layers"])], dropout=dropout_tp).to(device)
    init_weights_with_prior(final_model_tp, pos_prior=0.0)
    optimizer_final_tp = torch.optim.AdamW(final_model_tp.parameters(), lr=lr_tp, weight_decay=weight_decay_tp)

    final_model_rs = MLPRegression(input_size=X_train.shape[1], layers_dim=[params_rs["size_layer{}".format(i)] for i in range(params_rs["n_layers"])], dropout=dropout_rs).to(device)
    init_weights_with_prior(final_model_rs, pos_prior=0.0)
    optimizer_final_rs = torch.optim.AdamW(final_model_rs.parameters(), lr=lr_rs, weight_decay=weight_decay_rs)

    all_train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                 torch.tensor(y_train_n, dtype=torch.float32).unsqueeze(1))
    criterion_final = nn.MSELoss()

    for name_f, model_f, optimizer_f, bs_f in [
        ("TPE", final_model_tp, optimizer_final_tp, batch_size_tp),
        ("RS",  final_model_rs, optimizer_final_rs, batch_size_rs),
    ]:
        loader_f = DataLoader(all_train_ds, batch_size=bs_f, shuffle=True)
        print(f"  {name_f}: training for {num_epochs} epochs on {len(y_train_n)} samples...")
        for epoch in range(num_epochs):
            model_f.train()
            for xb, yb in loader_f:
                xb, yb = xb.to(device), yb.to(device)
                optimizer_f.zero_grad()
                criterion_final(model_f(xb), yb).backward()
                optimizer_f.step()

    torch.save(final_model_tp.state_dict(), str(OUT_DIR / f"models/best_model_{metric_label}_TP.pt"))
    torch.save(final_model_rs.state_dict(), str(OUT_DIR / f"models/best_model_{metric_label}_RS.pt"))

    # Model evaluation
    print(f"\n=== Final Test Set Evaluation ===")
    # Load test data — use cavity labels so per-sample inverse-transform works correctly
    X_test, y_test, _, cavity_labels_test = load_dataset(csv_path_test, return_groups=True, return_cavity=True)
    # Normalise test targets with TRAINING cavity statistics (no data leakage)
    y_test_n = _normalize_y(y_test, cavity_labels_test, cavity_stats_train)
    # Build per-sample arrays for element-wise inverse-transform in evaluate/plot
    y_mean_test_arr, y_std_test_arr = _build_inv_arrays(cavity_labels_test, cavity_stats_train, n=len(y_test))
    metrics = evaluate_and_plot_results(final_model_tp, final_model_rs, X_test, y_test_n, device=device,
                                        y_mean=y_mean_test_arr, y_std=y_std_test_arr)

    tp_opt_val = (1 - metrics['tp']['r2']) if opt_metric == 'r2' else metrics['tp'][opt_metric]
    rs_opt_val = (1 - metrics['rs']['r2']) if opt_metric == 'r2' else metrics['rs'][opt_metric]

    print(f"\n=== Final Model Comparison (opt_metric={opt_metric}) ===")
    print(f"TPE Model - MAE: {metrics['tp']['mae']:.4f}, RMSE: {metrics['tp']['rmse']:.4f}, R²: {metrics['tp']['r2']:.4f}")
    print(f"RS Model  - MAE: {metrics['rs']['mae']:.4f}, RMSE: {metrics['rs']['rmse']:.4f}, R²: {metrics['rs']['r2']:.4f}")

    # Save the better model as best overall
    if tp_opt_val <= rs_opt_val:
        print(f"\nTPE model performs better ({opt_metric.upper()}: {tp_opt_val:.4f}). Checking if it should be saved as best overall...")
        save_best_overall_model(
            model=final_model_tp,
            model_name='TPE',
            mae=metrics['tp']['mae'],
            rmse=metrics['tp']['rmse'],
            r2=metrics['tp']['r2'],
            mape=metrics['tp']['mape'],
            max_error=metrics['tp']['max_error'],
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            params=params_tpe,
            opt_metric=opt_metric
        )
    else:
        print(f"\nRS model performs better ({opt_metric.upper()}: {rs_opt_val:.4f}). Checking if it should be saved as best overall...")
        save_best_overall_model(
            model=final_model_rs,
            model_name='RS',
            mae=metrics['rs']['mae'],
            rmse=metrics['rs']['rmse'],
            r2=metrics['rs']['r2'],
            mape=metrics['rs']['mape'],
            max_error=metrics['rs']['max_error'],
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            params=params_rs,
            opt_metric=opt_metric
        )

    # Per-cavity evaluation on the best (winning) model only
    best_winner = final_model_tp if tp_opt_val <= rs_opt_val else final_model_rs
    winner_name = 'TPE' if tp_opt_val <= rs_opt_val else 'RS'
    _report_per_cavity_metrics(best_winner, winner_name, csv_path_test, device,
                               cavity_stats=cavity_stats_train)


# === Run Optuna Optimization ===
def run_optimization(sampler, pruner, csv_path, n_trials=100, n_startup_trials=10, hparam_cfg=None):
    global best_metric_RS_global
    global best_model_RS_global
    global best_params_RS_global
    
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    study.optimize(lambda trial: objective(trial, csv_path=csv_path, n_startup_trials=n_startup_trials, sampler=sampler.__class__.__name__, hparam_cfg=hparam_cfg), n_trials=n_trials, timeout=3600)

    opt_metric = (hparam_cfg or {}).get('opt_metric', 'mae')
    # Best trial overall (hoping is TPE)
    if sampler.__class__.__name__ == "TPESampler":
        print("\n=== Best model TPE - after initial {} RS and {} TPE trials ===".format(n_startup_trials, (n_trials - n_startup_trials)))
    elif sampler.__class__.__name__ == "RandomSampler":
        print("\n=== RandomSampler ===")
    print("Best trial:")
    trial = study.best_trial
    if opt_metric.upper() == 'R2':
        print(f"  {opt_metric.upper()}: {1 - trial.value:.4f}")
    else:
        print(f"  {opt_metric.upper()}: {trial.value:.4f}")
    for key, value in trial.params.items():
        print(f"  {key}: {value}")

    # If TPE optimizer, Best trial RS
    if (sampler.__class__.__name__ == "TPESampler") and (n_startup_trials > 0) and (best_params_RS_global is not None):
            print("\n=== Best model RS - found with initial {} RS trials ===".format(n_startup_trials))
            print(f"  {opt_metric.upper()}: {best_metric_RS_global:.4f}")
            for key, value in best_params_RS_global.items():
                print(f"  {key}: {value}")

    # Visualizations
    # fig = plot_optimization_history(study)
    # fig = plot_intermediate_values(study)
    # fig = plot_parallel_coordinate(study)
    # fig = plot_contour(study)
    # fig = plot_slice(study)
    # fig = plot_param_importances(study)
    # fig = plot_edf(study)
    # fig = plot_rank(study)
    # fig = plot_timeline(study)
    # plt.title("Parameters importances")
    # plt.show()

    return trial

# === Per-Cavity Metrics (double-cavity datasets only) ===
def _report_per_cavity_metrics(best_model, model_name, test_csv_path, device, cavity_stats=None):
    """Print per-cavity regression metrics and scatter plot for the best model only.
    No-op when test CSV has no 'cavity' column (single-cavity datasets).
    cavity_stats: dict returned by _compute_cavity_stats() from the training set."""
    df_raw = pd.read_csv(test_csv_path)
    if 'cavity' not in df_raw.columns:
        return
    X_test, y_test, _, cavity_labels_test = load_dataset(test_csv_path, return_groups=True, return_cavity=True)
    # Build per-sample inverse-transform arrays using training cavity stats
    y_mean_arr, y_std_arr = _build_inv_arrays(cavity_labels_test, cavity_stats or {None: (0.0, 1.0)}, n=len(y_test))
    cavities = sorted(df_raw['cavity'].unique())
    print("\n" + "="*55)
    print(f"=== Per-Cavity Test Set Evaluation ({model_name}) ===")
    fig, axes = plt.subplots(1, len(cavities), figsize=(6 * len(cavities), 5))
    if len(cavities) == 1:
        axes = [axes]
    for ax, cav in zip(axes, cavities):
        mask = (df_raw['cavity'] == cav).values
        X_cav, y_cav = X_test[mask], y_test[mask] # y_cav_n
        y_mean_cav, y_std_cav = y_mean_arr[mask], y_std_arr[mask]
        best_model.eval()
        with torch.no_grad():
            y_pred = best_model(torch.tensor(X_cav, dtype=torch.float32).to(device)).cpu().numpy().flatten()
        # Inverse-transform both prediction and true values to original scale
        y_pred = y_pred * y_std_cav + y_mean_cav
        #y_cav  = y_cav_n * y_std_cav + y_mean_cav
        mae  = float(mean_absolute_error(y_cav, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_cav, y_pred)))
        r2   = float(r2_score(y_cav, y_pred))
        mape = float(np.mean(np.abs((y_cav - y_pred) / np.clip(np.abs(y_cav), 1e-8, None))) * 100)
        print(f"\n--- Cavity {cav} ({mask.sum()} samples) ---")
        print(f"  MAE: {mae:.4f}  RMSE: {rmse:.4f}  R\u00b2: {r2:.4f}  MAPE: {mape:.2f}%")
        lims = [min(y_cav.min(), y_pred.min()), max(y_cav.max(), y_pred.max())]
        ax.scatter(y_cav, y_pred, alpha=0.5, s=30, color=CAVITY_COLORS.get(str(cav), CAVITY_COLORS['P1']))
        ax.plot(lims, lims, '--', color=PERFECT_PRED_COLOR, lw=2, label='Perfect prediction')
        ax.set_xlabel('True Values [g]'); ax.set_ylabel('Predictions [g]')
        ax.set_title(f'{model_name} \u2014 Cavity {cav}\nMAE={mae:.4f}, R\u00b2={r2:.4f}')
        ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(OUT_DIR / f'images/per_cavity_scatter_{model_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Per-cavity scatter plot saved: per_cavity_scatter_{model_name}.png")


# === Process double cavity dataset ===
def process_double_cavity_dataset(csv_path_1, csv_path_2, train_csv_path, test_csv_path):
    # Read data
    df_1 = pd.read_csv(csv_path_1)
    df_2 = pd.read_csv(csv_path_2)
    
    # Check if both datasets have 'shot' column
    if 'shot' not in df_1.columns or 'shot' not in df_2.columns:
        raise ValueError("Both datasets must have 'shot' column for synchronized splitting")
    
    print(f"Dataset P1: {len(df_1)} samples")
    print(f"Dataset P2: {len(df_2)} samples")

    # Detect outliers in both datasets
    outliers_p1 = detect_outliers_iqr(df_1['Product weight g'])
    outliers_p2 = detect_outliers_iqr(df_2['Product weight g'])

    print(f"Dataset P1 - Outliers detected: {outliers_p1.sum()}")
    print(f"Dataset P2 - Outliers detected: {outliers_p2.sum()}")

    # Remove outliers from both datasets
    df_1 = df_1[~outliers_p1].reset_index(drop=True)
    df_2 = df_2[~outliers_p2].reset_index(drop=True)

    
    # Split shot numbers into train and test (80/20)
    unique_shots = df_1['shot'].unique()
    np.random.seed(41)
    shuffled_shots = np.random.permutation(unique_shots)
    split_idx = int(len(shuffled_shots) * 0.8)
    train_shots = shuffled_shots[:split_idx]
    test_shots = shuffled_shots[split_idx:]
    print(f"Train shots: {len(train_shots)}, Test shots: {len(test_shots)}")

    # Split df_1 based on shot numbers
    Data_train_df_1 = df_1[df_1['shot'].isin(train_shots)].copy()
    Data_test_df_1 = df_1[df_1['shot'].isin(test_shots)].copy()

    print(f"Dataset P1 - Train: {len(Data_train_df_1)} samples, Test: {len(Data_test_df_1)} samples")
    
    # Split df_2 based on shot numbers
    Data_train_df_2 = df_2[df_2['shot'].isin(train_shots)].copy()
    Data_test_df_2 = df_2[df_2['shot'].isin(test_shots)].copy()
    
    print(f"Dataset P2 - Train: {len(Data_train_df_2)} samples, Test: {len(Data_test_df_2)} samples")
    
    # Add cavity labels before combining
    Data_train_df_1['cavity'] = 'P1'; Data_test_df_1['cavity'] = 'P1'
    Data_train_df_2['cavity'] = 'P2'; Data_test_df_2['cavity'] = 'P2'

    # Combine train datasets from P1 and P2
    Data_train_combined = pd.concat([Data_train_df_1, Data_train_df_2], axis=0, ignore_index=True)
    
    # Combine test datasets from P1 and P2
    Data_test_combined = pd.concat([Data_test_df_1, Data_test_df_2], axis=0, ignore_index=True)
    
    print(f"Combined - Train: {len(Data_train_combined)} samples, Test: {len(Data_test_combined)} samples")
    
    # Shuffle but keep shot groups together (shuffle at shot level, not row level)
    train_shots_order = Data_train_combined['shot'].unique()
    np.random.seed(42)
    np.random.shuffle(train_shots_order)
    Data_train_combined = Data_train_combined.set_index('shot').loc[train_shots_order].reset_index()
    
    test_shots_order = Data_test_combined['shot'].unique()
    np.random.seed(42)
    np.random.shuffle(test_shots_order)
    Data_test_combined = Data_test_combined.set_index('shot').loc[test_shots_order].reset_index()

    print(f"Shuffled datasets (preserving shot groups) - Train: {len(Data_train_combined)} samples, Test: {len(Data_test_combined)} samples")

    # print(f"Shuffled datasets - Train: {len(Data_train_df_1)} samples, Test: {len(Data_test_df_1)} samples")
    
    # Save combined datasets
    Data_train_combined.to_csv(train_csv_path, index=False)
    Data_test_combined.to_csv(test_csv_path, index=False)
    print(f"Saved combined train and test datasets to {train_csv_path} and {test_csv_path}")
    # print(Data_train_combined, Data_test_combined)


# === process single cavity dataset ===
def process_single_cavity_dataset(csv_path, train_csv_path, test_csv_path):
    # Read data
    df = pd.read_csv(csv_path)

    # Drop shot-related columns — not needed for single cavity
    cols_to_drop = [c for c in ['shot', 'shot_position'] if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"Dropped columns: {cols_to_drop}")

    print(f"Dataset: {len(df)} samples, {df.shape[1]} columns")

    # Detect and remove outliers
    outliers = detect_outliers_iqr(df['Product weight g'])
    print(f"Outliers detected: {outliers.sum()}")
    df = df[~outliers].reset_index(drop=True)

    # Simple random 80/20 train/test split (row-level, no shot grouping)
    np.random.seed(41)
    idx = np.random.permutation(len(df))
    split_idx = int(len(idx) * 0.8)
    train_idx = idx[:split_idx]
    test_idx = idx[split_idx:]

    Data_train_df = df.iloc[train_idx].reset_index(drop=True)
    Data_test_df = df.iloc[test_idx].reset_index(drop=True)

    print(f"Train: {len(Data_train_df)} samples, Test: {len(Data_test_df)} samples")

    # Save datasets
    Data_train_df.to_csv(train_csv_path, index=False)
    Data_test_df.to_csv(test_csv_path, index=False)
    print(f"Saved train and test datasets to {train_csv_path} and {test_csv_path}")


# === Main ===
if __name__ == "__main__":
    # Load config
    parser = argparse.ArgumentParser(description="Train a regression model.")
    parser.add_argument('--config', type=str, default=str(BASE_DIR / 'config/ProBayes/Reg_MLP_config.json'),
                        help="Path to the JSON config file (default: config/ProBayes/Reg_MLP_config.json)")
    parser.add_argument('--dataset', type=str, choices=['pp', 'abs', 'PP', 'ABS', 'PP_1', 'PP_2', 'ABS_1', 'ABS_2', 'pp_1', 'pp_2', 'abs_1', 'abs_2'],
                        help="Dataset to use: pp (both 1 and 2 cavities), abs (both 1 and 2 cavities), pp_1, pp_2, abs_1, abs_2. Overrides the value in the config file.")
    parser.add_argument('--opt_metric', type=str, choices=['mae', 'rmse', 'r2', 'mape', 'max_error'],
                        help="Metric to optimize: mae, rmse, r2, mape, or max_error. Overrides the value in the config file.")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = json.load(f)

    # CLI --dataset overrides config value
    if args.dataset:
        cfg['dataset'] = args.dataset
        print(f"\n[CLI override] dataset")

    # CLI --opt_metric overrides config value
    if args.opt_metric:
        cfg['opt_metric'] = args.opt_metric
        print(f"\n[CLI override] optimization metric")
    print(f"\nOptimization metric: {cfg['opt_metric'].upper()}")

    dataset = cfg.get('dataset', 'ABS').upper()
    if dataset in ['PP', 'ABS']:
        double_cavity = True
        if dataset == 'PP':
            csv_path_1 = str(BASE_DIR / 'data/DATA_PP_P1_W.csv')
            csv_path_2 = str(BASE_DIR / 'data/DATA_PP_P2_W.csv')
            print(f"\nUsing PP dataset (DATA_PP_P1_W.csv + DATA_PP_P2_W.csv)\n")
        elif dataset == 'ABS':
            csv_path_1 = str(BASE_DIR / 'data/DATA_ABS_P1_W.csv')
            csv_path_2 = str(BASE_DIR / 'data/DATA_ABS_P2_W.csv')
            print(f"\nUsing ABS dataset (DATA_ABS_P1_W.csv + DATA_ABS_P2_W.csv)\n")
    elif dataset in ['PP_1', 'PP_2', 'ABS_1', 'ABS_2']:
        double_cavity = False
        if dataset == 'PP_1':
            csv_path_1 = str(BASE_DIR / 'data/DATA_PP_P1_W.csv')
            csv_path_2 = None
        elif dataset == 'PP_2':
            csv_path_1 = str(BASE_DIR / 'data/DATA_PP_P2_W.csv')
            csv_path_2 = None
        elif dataset == 'ABS_1':
            csv_path_1 = str(BASE_DIR / 'data/DATA_ABS_P1_W.csv')
            csv_path_2 = None
        elif dataset == 'ABS_2':
            csv_path_1 = str(BASE_DIR / 'data/DATA_ABS_P2_W.csv')
            csv_path_2 = None
    else:
        raise ValueError(f"Unknown dataset '{dataset}' in config. "
                         f"Choose 'PP', 'ABS', 'PP_1', 'PP_2', 'ABS_1', or 'ABS_2'.")

    # Set dataset-specific output directory and create subdirectories
    OUT_DIR = BASE_DIR / f'outputs/ProBayes/Reg/MLP/{dataset}'
    (OUT_DIR / 'models').mkdir(parents=True, exist_ok=True)
    (OUT_DIR / 'images').mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUT_DIR}")

    start_time = time.time()

    if double_cavity:
        print(f"Processing double cavity dataset: {dataset}")
        process_double_cavity_dataset(csv_path_1, csv_path_2, train_csv_path=train_csv_path, test_csv_path=test_csv_path)
    else:
        print(f"Processing single cavity dataset: {dataset}")
        process_single_cavity_dataset(csv_path_1, train_csv_path=train_csv_path, test_csv_path=test_csv_path)
    

    # # Compute Mutual Information (MI) Scores for each feature in train data only
    # X_train_mi = Data_train_df.iloc[:, :-1].values
    # y_train_mi = Data_train_df.iloc[:, -1].values
    # MI_scores = mutual_info_classif(X_train_mi, y_train_mi, random_state=42)
    # for i, score in enumerate(MI_scores):
    #     print(f"Feature {i}: MI Score = {score:.4f}")
    # plt.figure(figsize=(10, 6))
    # plt.bar(range(len(MI_scores)), MI_scores, color='skyblue')
    # plt.xlabel('Feature Index')
    # plt.ylabel('Mutual Information Score')
    # plt.title('Mutual Information Scores for Features')
    # plt.xticks(range(len(MI_scores)), df.columns[:-1], rotation=45, ha='right')
    # plt.tight_layout()
    # plt.savefig("outputs/ProBayes/Reg/images/mi_scores.png")
    # plt.show()
    
    # # Run HPO otpimization with RS sampler and MedianPruner
    # print(f"\nStarting RS optimization...\n")
    # sampler = optuna.samplers.RandomSampler(seed=42)  # Use RandomSampler for simplicity
    # # pruner = optuna.pruners.MedianPruner(n_warmup_steps=1, n_startup_trials=10)
    # pruner = optuna.pruners.HyperbandPruner(min_resource=1, max_resource=80, reduction_factor=3)
    # best_trial_rs = run_optimization(sampler, pruner, train_csv_path, n_trials=40)

    # Run HPO otpimization with TPE sampler and HyperbandPruner
    print(f"\nStarting TPE optimization...\n")
    optuna_trials = cfg.get('optuna_trials', {})
    n_startup_trials = optuna_trials.get('startup_trials', 10)
    n_trials = optuna_trials.get('tot_trials', 100)
    print(f"Total Optuna trials: {n_trials} (with {n_startup_trials} startup trials for RS)")
    sampler = optuna.samplers.TPESampler(n_startup_trials=n_startup_trials, seed=42) #(n_startup_trials=10, seed=31) # Here tried to add some startup trials
    pruner = optuna.pruners.HyperbandPruner(min_resource=1, max_resource=80, reduction_factor=3)
    best_trial_tpe = run_optimization(sampler, pruner, train_csv_path, n_trials=n_trials, n_startup_trials=n_startup_trials, hparam_cfg=cfg)

    # Retrain the best models
    train_and_save_best_model(params_tpe=best_trial_tpe.params, params_rs=best_params_RS_global, epochs=200, csv_path_train=train_csv_path, csv_path_test=test_csv_path, hparam_cfg=cfg)

    # Print total time taken
    end_time = time.time()
    print(f"\nTotal time taken: {end_time - start_time:.2f} seconds")
