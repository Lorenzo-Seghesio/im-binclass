# Reg_GBT_IM.py
#
# Regression using Gradient Boosted Trees for weight prediction (Injection Moulding).
# Same data pipeline, Optuna HPO structure, config convention, and output layout as Reg_MLP_IM.py.
# Models are saved with joblib instead of torch.save.

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import argparse
import time
import matplotlib.pyplot as plt
import optuna
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
try:
    from Utility.optuna_seeding import resolve_optuna_seed
except ModuleNotFoundError:
    from src.Utility.optuna_seeding import resolve_optuna_seed

BASE_DIR = Path(__file__).resolve().parent.parent  # project root

# Output root — overridden at runtime in __main__ based on dataset choice
OUT_DIR = BASE_DIR / 'outputs/ProBayes/Reg_GBT'

# Colour palette (Wong, colorblind-friendly) — consistent across all plot types
CAVITY_COLORS    = {'P1': '#0072B2', 'P2': '#D55E00'}   # blue / vermilion
MODEL_COLORS     = {'TPE': '#009E73', 'RS': '#CC79A7'}  # teal / mauve
PERFECT_PRED_COLOR = '#333333'                           # dark gray

# Globals for tracking best models across Optuna trials
best_metric_global = float('inf')
best_model_global = None
best_metric_RS_global = float('inf')
best_model_RS_global = None
best_params_RS_global = None

test_csv_path = str(BASE_DIR / 'data/IM_Data_Test.csv')
train_csv_path = str(BASE_DIR / 'data/IM_Data_Train.csv')


# === Outlier Detection using IQR ===
def detect_outliers_iqr(series, multiplier=1.5):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    return (series < Q1 - multiplier * IQR) | (series > Q3 + multiplier * IQR)


# === Data Loader ===
def load_dataset(csv_path, return_groups=False):
    df = pd.read_csv(csv_path)
    target_col = 'Product weight g'
    drop_cols = [c for c in ['shot', 'cavity', target_col] if c in df.columns]

    if 'shot' in df.columns and return_groups:
        groups = df['shot'].values
    else:
        groups = None

    y = df[target_col].values
    X = df.drop(columns=drop_cols).values
    X = StandardScaler().fit_transform(X)

    if return_groups:
        return X, y, groups
    return X, y


# === Optimization metric helper ===
# All values are to MINIMIZE (lower = better). R2 → 1-R2.
VALID_OPT_METRICS = ['mae', 'rmse', 'r2', 'mape', 'max_error']

def compute_optuna_metric(y_true, y_pred, metric):
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


# === Build GBT Regressor from params + config ===
def _build_regressor(params, hp):
    """Construct GradientBoostingRegressor. Scalar in config → fixed; list → from trial params."""
    def get(key, default):
        cfg = hp.get(key)
        if isinstance(cfg, list):
            return params.get(key, default)
        return cfg if cfg is not None else default

    return GradientBoostingRegressor(
        n_estimators=int(get('n_estimators', 100)),
        max_depth=int(get('max_depth', 3)),
        learning_rate=float(get('learning_rate', 0.1)),
        subsample=float(get('subsample', 1.0)),
        min_samples_split=int(get('min_samples_split', 2)),
        min_samples_leaf=int(get('min_samples_leaf', 1)),
        max_features=float(get('max_features', 1.0)),
        random_state=42,
    )


# === Save Best Overall Model ===
def save_best_overall_model(model, model_name, mae, rmse, r2, mape, max_error,
                            X_train, y_train, X_test, y_test, params, opt_metric='mae'):
    best_model_dir = str(OUT_DIR / 'models/best_model_overall')
    metadata_file = os.path.join(best_model_dir, 'metadata.json')

    _scores = {'mae': mae, 'rmse': rmse, 'r2': 1.0 - r2, 'mape': mape, 'max_error': max_error}
    curr_score = _scores[opt_metric]

    should_save = True
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            prev_metadata = json.load(f)
        prev_score = prev_metadata.get('opt_score', prev_metadata.get('mae'))
        print(f"\n=== Comparing with previous best model ({opt_metric.upper()}) ===")
        print(f"Previous best: {prev_metadata['model_name']} - {opt_metric.upper()}: {prev_score:.4f}")
        print(f"Current model: {model_name} - {opt_metric.upper()}: {curr_score:.4f}")
        if curr_score >= prev_score:
            print("Current model is not better than previous best. Not saving.")
            should_save = False
        else:
            print("Current model is BETTER! Overwriting previous best model.")
    else:
        print(f"\n=== No previous best model found. Saving current model as best. ===")
        print(f"Current model: {model_name} — MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

    if should_save:
        if os.path.exists(best_model_dir):
            shutil.rmtree(best_model_dir)
        os.makedirs(best_model_dir, exist_ok=True)

        joblib.dump(model, os.path.join(best_model_dir, f'best_model_{model_name}.joblib'))

        train_data = pd.DataFrame(X_train)
        train_data['Product weight g'] = y_train
        train_data.to_csv(os.path.join(best_model_dir, 'train_data.csv'), index=False)

        test_data = pd.DataFrame(X_test)
        test_data['Product weight g'] = y_test
        test_data.to_csv(os.path.join(best_model_dir, 'test_data.csv'), index=False)

        metadata = {
            'model_name': model_name,
            'opt_metric': opt_metric,
            'opt_score': float(curr_score),
            'mae': float(mae), 'rmse': float(rmse), 'r2': float(r2),
            'mape': float(mape), 'max_error': float(max_error),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'hyperparameters': params,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'n_features': X_train.shape[1],
        }
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)

        for name in [f'scatter_plot_{model_name}', f'residual_plot_{model_name}', 'metrics_comparison']:
            src = str(OUT_DIR / f'images/{name}.png')
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(best_model_dir, f'{name}.png'))

        print(f"\n✅ Best overall model saved to {best_model_dir}/")

    return should_save


# === Evaluate and Plot Results ===
def evaluate_and_plot_results(model_tp, model_rs, X_test, y_test):
    """Run both models on the test set and produce all comparison plots."""
    y_pred_tp = model_tp.predict(X_test)
    y_pred_rs = model_rs.predict(X_test)

    # Single-cavity run → use the cavity's designated colour for both models.
    # Double-cavity run → keep the standard TPE/RS model colours.
    _name = OUT_DIR.name
    if _name.endswith('_1'):
        _color_tp = _color_rs = CAVITY_COLORS['P1']
    elif _name.endswith('_2'):
        _color_tp = _color_rs = CAVITY_COLORS['P2']
    else:
        _color_tp, _color_rs = MODEL_COLORS['TPE'], MODEL_COLORS['RS']

    def _metrics(y_true, y_pred):
        mae  = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2   = float(r2_score(y_true, y_pred))
        mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))) * 100)
        mxe  = float(np.max(np.abs(y_true - y_pred)))
        return mae, rmse, r2, mape, mxe

    mae_tp, rmse_tp, r2_tp, mape_tp, mxe_tp = _metrics(y_test, y_pred_tp)
    mae_rs, rmse_rs, r2_rs, mape_rs, mxe_rs = _metrics(y_test, y_pred_rs)

    for label, mae, rmse, r2, mape, mxe in [
        ('TPE', mae_tp, rmse_tp, r2_tp, mape_tp, mxe_tp),
        ('RS',  mae_rs, rmse_rs, r2_rs, mape_rs, mxe_rs),
    ]:
        print(f"\n=== {label} Model Test Results ===")
        print(f"MAE: {mae:.4f}  RMSE: {rmse:.4f}  R²: {r2:.4f}  MAPE: {mape:.2f}%  Max Error: {mxe:.4f}")

    # ---- scatter plots ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    for ax, y_pred, label, color, mae, r2 in [
        (ax1, y_pred_tp, 'TPE', _color_tp, mae_tp, r2_tp),
        (ax2, y_pred_rs, 'RS',  _color_rs, mae_rs, r2_rs),
    ]:
        ax.scatter(y_test, y_pred, alpha=0.5, s=30, color=color)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                '--', color=PERFECT_PRED_COLOR, lw=2, label='Perfect prediction')
        ax.set_xlabel('True Values [g]'); ax.set_ylabel('Predictions [g]')
        ax.set_title(f'{label} Model: Predicted vs True\nMAE={mae:.4f}, R²={r2:.4f}')
        ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(OUT_DIR / 'images/scatter_plots_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    for y_pred, label, color, mae, r2 in [
        (y_pred_tp, 'TPE', _color_tp, mae_tp, r2_tp),
        (y_pred_rs, 'RS',  _color_rs, mae_rs, r2_rs),
    ]:
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.5, s=30, color=color)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                 '--', color=PERFECT_PRED_COLOR, lw=2, label='Perfect prediction')
        plt.xlabel('True Values [g]'); plt.ylabel('Predictions [g]')
        plt.title(f'{label} Model: Predicted vs True\nMAE={mae:.4f}, R²={r2:.4f}')
        plt.legend(); plt.grid(alpha=0.3)
        plt.savefig(str(OUT_DIR / f'images/scatter_plot_{label}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # ---- residual plots ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    for ax, y_pred, label, color, mae in [
        (ax1, y_pred_tp, 'TPE', _color_tp, mae_tp),
        (ax2, y_pred_rs, 'RS',  _color_rs, mae_rs),
    ]:
        residuals = y_test - y_pred
        ax.scatter(y_pred, residuals, alpha=0.5, s=30, color=color)
        ax.axhline(y=0, color='r', linestyle='--', lw=2)
        ax.set_xlabel('Predicted Values [g]'); ax.set_ylabel('Residuals [g]')
        ax.set_title(f'{label} Model: Residual Plot\nMAE={mae:.4f}'); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(OUT_DIR / 'images/residual_plots_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    for y_pred, label, color, mae in [
        (y_pred_tp, 'TPE', _color_tp, mae_tp),
        (y_pred_rs, 'RS',  _color_rs, mae_rs),
    ]:
        residuals = y_test - y_pred
        plt.figure(figsize=(8, 6))
        plt.scatter(y_pred, residuals, alpha=0.5, s=30, color=color)
        plt.axhline(y=0, color='r', linestyle='--', lw=2)
        plt.xlabel('Predicted Values [g]'); plt.ylabel('Residuals [g]')
        plt.title(f'{label} Model: Residual Plot\nMAE={mae:.4f}'); plt.grid(alpha=0.3)
        plt.savefig(str(OUT_DIR / f'images/residual_plot_{label}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # ---- metrics bar chart ----
    metrics_names = ['MAE', 'RMSE', 'R²', 'MAPE (%)', 'Max Error']
    tp_values = [mae_tp, rmse_tp, r2_tp, mape_tp, mxe_tp]
    rs_values = [mae_rs, rmse_rs, r2_rs, mape_rs, mxe_rs]
    x = np.arange(len(metrics_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width / 2, tp_values, width, label='TPE Model', alpha=0.8)
    bars2 = ax.bar(x + width / 2, rs_values, width, label='RS Model', alpha=0.8)
    ax.set_xlabel('Metrics'); ax.set_ylabel('Values')
    ax.set_title('GBT Regression Metrics Comparison')
    ax.set_xticks(x); ax.set_xticklabels(metrics_names)
    ax.legend(); ax.grid(alpha=0.3, axis='y')
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h, f'{h:.3f}',
                    ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(str(OUT_DIR / 'images/metrics_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nPlots saved in {OUT_DIR / 'images'}")
    return {
        'tp': {'mae': mae_tp, 'rmse': rmse_tp, 'r2': r2_tp, 'mape': mape_tp, 'max_error': mxe_tp},
        'rs': {'mae': mae_rs, 'rmse': rmse_rs, 'r2': r2_rs, 'mape': mape_rs, 'max_error': mxe_rs},
    }


# === Per-Cavity Metrics (double-cavity datasets only) ===
def _report_per_cavity_metrics(best_model, model_name, test_csv_path):
    """Print per-cavity regression metrics and scatter plot for the best model only.
    No-op when test CSV has no 'cavity' column (single-cavity datasets)."""
    df_raw = pd.read_csv(test_csv_path)
    if 'cavity' not in df_raw.columns:
        return
    X_test, y_test, _ = load_dataset(test_csv_path, return_groups=True)
    cavities = sorted(df_raw['cavity'].unique())
    print("\n" + "="*55)
    print(f"=== Per-Cavity Test Set Evaluation ({model_name}) ===")
    fig, axes = plt.subplots(1, len(cavities), figsize=(6 * len(cavities), 5))
    if len(cavities) == 1:
        axes = [axes]
    for ax, cav in zip(axes, cavities):
        mask = (df_raw['cavity'] == cav).values
        X_cav, y_cav = X_test[mask], y_test[mask]
        y_pred = best_model.predict(X_cav)
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


# === Objective Function ===
def objective(trial, csv_path, n_startup_trials=10, sampler="RandomSampler", hparam_cfg=None):
    global best_metric_global, best_model_global
    global best_metric_RS_global, best_model_RS_global, best_params_RS_global

    X, y, groups = load_dataset(csv_path, return_groups=True)
    skf = GroupKFold(n_splits=5) if groups is not None else KFold(n_splits=5, shuffle=True, random_state=42)

    hp = (hparam_cfg or {}).get('hyperparameters', {})
    opt_metric = (hparam_cfg or {}).get('opt_metric', 'mae')

    # Hyperparameters — scalar in config → fixed; [min, max] → optimized by Optuna
    ne_cfg  = hp.get('n_estimators',    [50, 500])
    md_cfg  = hp.get('max_depth',       [2, 8])
    lr_cfg  = hp.get('learning_rate',   [0.01, 0.3])
    ss_cfg  = hp.get('subsample',       [0.5, 1.0])
    mss_cfg = hp.get('min_samples_split', [2, 20])
    msl_cfg = hp.get('min_samples_leaf',  [1, 10])
    mf_cfg  = hp.get('max_features',    1.0)

    n_estimators     = trial.suggest_int("n_estimators",     ne_cfg[0],  ne_cfg[1])  if isinstance(ne_cfg, list)  else int(ne_cfg)
    max_depth        = trial.suggest_int("max_depth",        md_cfg[0],  md_cfg[1])  if isinstance(md_cfg, list)  else int(md_cfg)
    learning_rate    = trial.suggest_float("learning_rate",  lr_cfg[0],  lr_cfg[1],  log=True) if isinstance(lr_cfg, list)  else float(lr_cfg)
    subsample        = trial.suggest_float("subsample",      ss_cfg[0],  ss_cfg[1])  if isinstance(ss_cfg, list)  else float(ss_cfg)
    min_samples_split = trial.suggest_int("min_samples_split", mss_cfg[0], mss_cfg[1]) if isinstance(mss_cfg, list) else int(mss_cfg)
    min_samples_leaf  = trial.suggest_int("min_samples_leaf",  msl_cfg[0], msl_cfg[1]) if isinstance(msl_cfg, list) else int(msl_cfg)
    max_features      = trial.suggest_float("max_features",   mf_cfg[0],  mf_cfg[1])  if isinstance(mf_cfg, list)  else float(mf_cfg)

    metric_values = []
    best_metric   = float('inf')
    best_model    = None

    fold_iter = skf.split(X, y, groups=groups) if groups is not None else skf.split(X, y)
    for fold, (train_idx, val_idx) in enumerate(fold_iter):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = GradientBoostingRegressor(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
            subsample=subsample, min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf, max_features=max_features,
            random_state=42,
        )
        model.fit(X_train, y_train)

        metric_value = compute_optuna_metric(y_val, model.predict(X_val), opt_metric)
        metric_values.append(metric_value)

        if metric_value < best_metric:
            best_metric = metric_value
            best_model  = model

        trial.report(np.mean(metric_values), fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

    metric_mean = float(np.mean(metric_values))

    if best_model is not None and (best_model_global is None or metric_mean < best_metric_global):
        best_metric_global = metric_mean
        best_model_global  = best_model
        joblib.dump(best_model_global, str(OUT_DIR / f"models/best_model_{opt_metric.upper()}_global.joblib"))
        if sampler == "TPESampler" and trial.number < n_startup_trials:
            best_metric_RS_global = best_metric_global
            best_model_RS_global  = best_model_global
            best_params_RS_global = trial.params
            joblib.dump(best_model_RS_global, str(OUT_DIR / f"models/best_model_{opt_metric.upper()}_RS.joblib"))

    return metric_mean


# === Train and Evaluate Final Models ===
def train_and_save_best_model(params_tpe, params_rs,
                              csv_path_train=str(BASE_DIR / 'data/IM_Data_Train.csv'),
                              csv_path_test=str(BASE_DIR / 'data/IM_Data_Test.csv'),
                              hparam_cfg=None):
    hp = (hparam_cfg or {}).get('hyperparameters', {})
    opt_metric = (hparam_cfg or {}).get('opt_metric', 'mae')
    metric_label = opt_metric.upper()

    print(f"\nTraining final TPE and RS models on full training set...")
    X_train, y_train, groups = load_dataset(csv_path_train, return_groups=True)
    X_test,  y_test,  _      = load_dataset(csv_path_test,  return_groups=True)
    print(f"Train: {X_train.shape[0]} samples, {X_train.shape[1]} features | Test: {X_test.shape[0]} samples")

    # 5-fold CV for metric estimates
    if groups is not None:
        skf = GroupKFold(n_splits=5)
        fold_iter = list(skf.split(X_train, y_train, groups=groups))
        print("Using GroupKFold to keep shots together in folds")
    else:
        skf = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_iter = list(skf.split(X_train, y_train))
        print("Using KFold (no shot grouping available)")

    metric_values_tp, metric_values_rs = [], []
    for train_idx, val_idx in fold_iter:
        X_f, X_v = X_train[train_idx], X_train[val_idx]
        y_f, y_v = y_train[train_idx], y_train[val_idx]

        m_tp = _build_regressor(params_tpe, hp)
        m_tp.fit(X_f, y_f)
        metric_values_tp.append(compute_optuna_metric(y_v, m_tp.predict(X_v), opt_metric))

        m_rs = _build_regressor(params_rs, hp)
        m_rs.fit(X_f, y_f)
        metric_values_rs.append(compute_optuna_metric(y_v, m_rs.predict(X_v), opt_metric))

    print(f"\nCV {metric_label} — TPE: {np.mean(metric_values_tp):.4f} ± {np.std(metric_values_tp):.4f}")
    print(f"CV {metric_label} — RS:  {np.mean(metric_values_rs):.4f} ± {np.std(metric_values_rs):.4f}")

    # Retrain on full training set
    final_model_tp = _build_regressor(params_tpe, hp)
    final_model_tp.fit(X_train, y_train)
    final_model_rs = _build_regressor(params_rs, hp)
    final_model_rs.fit(X_train, y_train)

    joblib.dump(final_model_tp, str(OUT_DIR / f"models/best_model_{metric_label}_TP.joblib"))
    joblib.dump(final_model_rs, str(OUT_DIR / f"models/best_model_{metric_label}_RS.joblib"))

    # Evaluate on test set
    print(f"\n=== Final Test Set Evaluation ===")
    metrics = evaluate_and_plot_results(final_model_tp, final_model_rs, X_test, y_test)

    tp_opt = (1 - metrics['tp']['r2']) if opt_metric == 'r2' else metrics['tp'][opt_metric]
    rs_opt = (1 - metrics['rs']['r2']) if opt_metric == 'r2' else metrics['rs'][opt_metric]

    print(f"\n=== Final Model Comparison (opt_metric={opt_metric}) ===")
    print(f"TPE — MAE: {metrics['tp']['mae']:.4f}, RMSE: {metrics['tp']['rmse']:.4f}, R²: {metrics['tp']['r2']:.4f}")
    print(f"RS  — MAE: {metrics['rs']['mae']:.4f}, RMSE: {metrics['rs']['rmse']:.4f}, R²: {metrics['rs']['r2']:.4f}")

    if tp_opt <= rs_opt:
        print(f"\nTPE model performs better ({metric_label}: {tp_opt:.4f}). Saving as best overall...")
        save_best_overall_model(
            model=final_model_tp, model_name='TPE',
            mae=metrics['tp']['mae'], rmse=metrics['tp']['rmse'], r2=metrics['tp']['r2'],
            mape=metrics['tp']['mape'], max_error=metrics['tp']['max_error'],
            X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
            params=params_tpe, opt_metric=opt_metric,
        )
    else:
        print(f"\nRS model performs better ({metric_label}: {rs_opt:.4f}). Saving as best overall...")
        save_best_overall_model(
            model=final_model_rs, model_name='RS',
            mae=metrics['rs']['mae'], rmse=metrics['rs']['rmse'], r2=metrics['rs']['r2'],
            mape=metrics['rs']['mape'], max_error=metrics['rs']['max_error'],
            X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
            params=params_rs, opt_metric=opt_metric,
        )

    # Per-cavity evaluation on the best (winning) model only
    best_winner = final_model_tp if tp_opt <= rs_opt else final_model_rs
    winner_name = 'TPE' if tp_opt <= rs_opt else 'RS'
    _report_per_cavity_metrics(best_winner, winner_name, csv_path_test)


# === Run Optuna Optimization ===
def run_optimization(sampler, pruner, csv_path, n_trials=50, n_startup_trials=10, hparam_cfg=None):
    global best_metric_RS_global, best_model_RS_global, best_params_RS_global

    opt_metric = (hparam_cfg or {}).get('opt_metric', 'mae')
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    study.optimize(
        lambda trial: objective(trial, csv_path=csv_path, n_startup_trials=n_startup_trials,
                                sampler=sampler.__class__.__name__, hparam_cfg=hparam_cfg),
        n_trials=n_trials, timeout=3600,
    )

    if sampler.__class__.__name__ == "TPESampler":
        print(f"\n=== Best model TPE — after {n_startup_trials} RS and {n_trials - n_startup_trials} TPE trials ===")
    trial = study.best_trial
    print(f"  {opt_metric.upper()}: {trial.value:.4f}")
    for key, value in trial.params.items():
        print(f"  {key}: {value}")

    if sampler.__class__.__name__ == "TPESampler" and n_startup_trials > 0 and best_params_RS_global is not None:
        print(f"\n=== Best model RS — found with {n_startup_trials} startup trials ===")
        print(f"  {opt_metric.upper()}: {best_metric_RS_global:.4f}")
        for key, value in best_params_RS_global.items():
            print(f"  {key}: {value}")

    return trial


# === Data Analysis Plots (weight distribution + feature correlations) ===
def _plot_data_analysis(df, label):
    """Save weight distribution and feature-weight correlation plots after outlier removal."""
    analysis_dir = OUT_DIR / 'data_analysis'
    analysis_dir.mkdir(parents=True, exist_ok=True)

    weight_col   = 'Product weight g'
    skip_cols    = {'shot', 'cavity', weight_col}
    feature_cols = [c for c in df.columns if c not in skip_cols]

    # --- Weight distribution: histogram + time series ---
    w = df[weight_col]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(w, bins=40, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Weight [g]')
    axes[0].set_ylabel('Count')
    axes[0].set_title(
        f'{label} — Weight Distribution\n'
        f'n={len(w)},  μ={w.mean():.4f} g,  σ={w.std():.4f} g'
    )
    axes[0].grid(alpha=0.3)

    axes[1].plot(w.values, alpha=0.6, linewidth=0.8, color='steelblue')
    axes[1].set_xlabel('Sample Index')
    axes[1].set_ylabel('Weight [g]')
    axes[1].set_title(f'{label} — Weight over Samples')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(analysis_dir / f'weight_distribution_{label}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [data_analysis] Weight plot saved: weight_distribution_{label}.png")

    # --- Feature–weight Pearson correlation ---
    corr = (
        df[feature_cols + [weight_col]]
        .corr()[weight_col]
        .drop(weight_col)
        .sort_values(key=abs, ascending=False)
    )

    n_feat = len(corr)
    fig_w  = max(10, n_feat * 0.45)
    fig, ax = plt.subplots(figsize=(fig_w, 6))
    colors = ['steelblue' if v >= 0 else 'salmon' for v in corr.values]
    ax.bar(range(n_feat), corr.values, color=colors)
    ax.set_xticks(range(n_feat))
    ax.set_xticklabels(corr.index, rotation=90, fontsize=8)
    ax.set_ylabel('Pearson Correlation with Weight')
    ax.set_title(f'{label} — Feature Correlation with Product Weight')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(str(analysis_dir / f'feature_correlation_{label}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    corr_df = corr.rename('correlation_with_weight').reset_index()
    corr_df.columns = ['feature', 'correlation_with_weight']
    corr_df.to_csv(str(analysis_dir / f'feature_correlation_{label}.csv'), index=False)
    print(f"  [data_analysis] Correlation plot + CSV saved: feature_correlation_{label}.*")

    # --- Scatter: every feature vs weight (sorted by |correlation|) ---
    ncols = min(4, n_feat)
    nrows = int(np.ceil(n_feat / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes_flat = np.array(axes).reshape(-1)
    for i, feat in enumerate(corr.index):          # corr.index is sorted by |r|
        ax = axes_flat[i]
        ax.scatter(df[feat], df[weight_col], alpha=0.3, s=12, color='steelblue')
        ax.set_xlabel(feat, fontsize=8)
        ax.set_ylabel('Weight [g]', fontsize=8)
        ax.set_title(f'r = {corr[feat]:.3f}', fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
    for j in range(n_feat, len(axes_flat)):
        axes_flat[j].set_visible(False)
    plt.suptitle(f'{label} — Feature vs Weight (per part)', fontsize=12)
    plt.tight_layout()
    plt.savefig(str(analysis_dir / f'feature_vs_weight_{label}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [data_analysis] Feature vs weight scatter saved: feature_vs_weight_{label}.png")


# === Process Double Cavity Dataset ===
def process_double_cavity_dataset(csv_path_1, csv_path_2, train_csv_path, test_csv_path,
                                  data_analysis=False):
    df_1 = pd.read_csv(csv_path_1)
    df_2 = pd.read_csv(csv_path_2)
    if 'shot' not in df_1.columns or 'shot' not in df_2.columns:
        raise ValueError("Both datasets must have 'shot' column for synchronized splitting")

    print(f"Dataset P1: {len(df_1)} samples | Dataset P2: {len(df_2)} samples")
    outliers_p1 = detect_outliers_iqr(df_1['Product weight g'])
    outliers_p2 = detect_outliers_iqr(df_2['Product weight g'])
    print(f"Outliers — P1: {outliers_p1.sum()}, P2: {outliers_p2.sum()}")
    df_1 = df_1[~outliers_p1].reset_index(drop=True)
    df_2 = df_2[~outliers_p2].reset_index(drop=True)

    if data_analysis:
        _plot_data_analysis(df_1, f'{OUT_DIR.name}_P1')
        _plot_data_analysis(df_2, f'{OUT_DIR.name}_P2')

    unique_shots = df_1['shot'].unique()
    np.random.seed(41)
    shuffled_shots = np.random.permutation(unique_shots)
    split_idx    = int(len(shuffled_shots) * 0.8)
    train_shots  = shuffled_shots[:split_idx]
    test_shots   = shuffled_shots[split_idx:]
    print(f"Train shots: {len(train_shots)}, Test shots: {len(test_shots)}")

    def split_by_shots(df):
        return df[df['shot'].isin(train_shots)].copy(), df[df['shot'].isin(test_shots)].copy()

    tr1, te1 = split_by_shots(df_1)
    tr2, te2 = split_by_shots(df_2)
    print(f"P1 — Train: {len(tr1)}, Test: {len(te1)} | P2 — Train: {len(tr2)}, Test: {len(te2)}")

    tr1['cavity'] = 'P1'; te1['cavity'] = 'P1'
    tr2['cavity'] = 'P2'; te2['cavity'] = 'P2'

    Data_train = pd.concat([tr1, tr2], ignore_index=True)
    Data_test  = pd.concat([te1, te2], ignore_index=True)

    # Shuffle preserving shot groups
    for df, seed in [(Data_train, 42), (Data_test, 42)]:
        shots = df['shot'].unique()
        np.random.seed(seed); np.random.shuffle(shots)
        df[:] = df.set_index('shot').loc[shots].reset_index()

    print(f"Combined — Train: {len(Data_train)}, Test: {len(Data_test)}")
    Data_train.to_csv(train_csv_path, index=False)
    Data_test.to_csv(test_csv_path, index=False)
    print(f"Saved to {train_csv_path} and {test_csv_path}")


# === Process Single Cavity Dataset ===
def process_single_cavity_dataset(csv_path, train_csv_path, test_csv_path,
                                  data_analysis=False):
    df = pd.read_csv(csv_path)
    cols_to_drop = [c for c in ['shot', 'shot_position'] if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"Dropped columns: {cols_to_drop}")

    print(f"Dataset: {len(df)} samples, {df.shape[1]} columns")
    outliers = detect_outliers_iqr(df['Product weight g'])
    print(f"Outliers detected: {outliers.sum()}")
    df = df[~outliers].reset_index(drop=True)

    if data_analysis:
        _plot_data_analysis(df, OUT_DIR.name)

    np.random.seed(41)
    idx = np.random.permutation(len(df))
    split_idx = int(len(idx) * 0.8)
    df.iloc[idx[:split_idx]].reset_index(drop=True).to_csv(train_csv_path, index=False)
    df.iloc[idx[split_idx:]].reset_index(drop=True).to_csv(test_csv_path,  index=False)
    print(f"Train: {split_idx} samples, Test: {len(idx) - split_idx} samples")
    print(f"Saved to {train_csv_path} and {test_csv_path}")


# === Main ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a GBT regression model.")
    parser.add_argument('--config', type=str, default=str(BASE_DIR / 'config/ProBayes/Reg_GBT_config.json'))
    parser.add_argument('--dataset', type=str,
                        choices=['pp', 'abs', 'PP', 'ABS', 'PP_1', 'PP_2', 'ABS_1', 'ABS_2',
                                 'pp_1', 'pp_2', 'abs_1', 'abs_2'])
    parser.add_argument('--opt_metric', type=str, choices=VALID_OPT_METRICS)
    parser.add_argument('--optuna-seed', type=int, default=None,
                        help='Optuna sampler seed; omit to generate a fresh seed.')
    parser.add_argument('--data_analysis', action='store_true',
                        help='Generate and save data analysis plots (weight distribution, '
                             'feature correlations, feature-vs-weight scatters).')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = json.load(f)

    if args.dataset:
        cfg['dataset'] = args.dataset
        print(f"\n[CLI override] dataset set to '{args.dataset.upper()}'")
    if args.opt_metric:
        cfg['opt_metric'] = args.opt_metric
        print(f"\n[CLI override] opt_metric set to '{args.opt_metric}'")
    if args.optuna_seed is not None:
        cfg.setdefault('optuna_trials', {})['sampler_seed'] = args.optuna_seed
        print(f"\n[CLI override] Optuna sampler seed set to {args.optuna_seed}")

    print(f"Optimization metric: {cfg.get('opt_metric', 'mae').upper()}")

    dataset = cfg.get('dataset', 'ABS').upper()
    if dataset in ['PP', 'ABS']:
        double_cavity = True
        csv_path_1 = str(BASE_DIR / f'data/DATA_{dataset}_P1_W.csv')
        csv_path_2 = str(BASE_DIR / f'data/DATA_{dataset}_P2_W.csv')
        print(f"\nUsing {dataset} dataset (P1 + P2)\n")
    elif dataset in ['PP_1', 'PP_2', 'ABS_1', 'ABS_2']:
        double_cavity = False
        mat, cav = dataset.split('_')
        csv_path_1 = str(BASE_DIR / f'data/DATA_{mat}_P{cav}_W.csv')
        csv_path_2 = None
        print(f"\nUsing {dataset} dataset\n")
    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose PP, ABS, PP_1, PP_2, ABS_1, or ABS_2.")

    OUT_DIR = BASE_DIR / f'outputs/ProBayes/Reg/GBT/{dataset}'
    (OUT_DIR / 'models').mkdir(parents=True, exist_ok=True)
    (OUT_DIR / 'images').mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUT_DIR}")

    start_time = time.time()

    if double_cavity:
        process_double_cavity_dataset(csv_path_1, csv_path_2, train_csv_path, test_csv_path,
                                      data_analysis=args.data_analysis)
    else:
        process_single_cavity_dataset(csv_path_1, train_csv_path, test_csv_path,
                                      data_analysis=args.data_analysis)

    optuna_trials    = cfg.get('optuna_trials', {})
    sampler_seed     = resolve_optuna_seed(optuna_trials)
    n_startup_trials = optuna_trials.get('startup_trials', 10)
    n_trials         = optuna_trials.get('tot_trials', 50)
    print(f"\nStarting TPE optimization ({n_trials} trials, {n_startup_trials} startup RS)...\n")

    #optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(n_startup_trials=n_startup_trials, seed=sampler_seed)
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    best_trial_tpe = run_optimization(sampler, pruner, train_csv_path,
                                      n_trials=n_trials, n_startup_trials=n_startup_trials,
                                      hparam_cfg=cfg)

    train_and_save_best_model(
        params_tpe=best_trial_tpe.params,
        params_rs=best_params_RS_global,
        csv_path_train=train_csv_path,
        csv_path_test=test_csv_path,
        hparam_cfg=cfg,
    )

    run_info = {
        'run_ts': datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
        'dataset': dataset,
        'split_seed': 42,
        'sampler_seed': sampler_seed,
        'config': str(Path(args.config).resolve()),
    }
    (OUT_DIR / 'run_info.json').write_text(json.dumps(run_info, indent=4))
    print(f"Optuna run info saved to {OUT_DIR / 'run_info.json'}")

    print(f"\nTotal time: {time.time() - start_time:.2f} seconds")
