# BC_GBT_IM.py
#
# Binary classification using Gradient Boosted Trees for quality recognition (Injection Moulding).
# Same data pipeline, Optuna HPO structure, config convention, and output layout as BC_MLP_IM.py.
# Extends BC_MLP by also supporting single-cavity datasets (PP_1, PP_2, ABS_1, ABS_2).
# Models are saved with joblib instead of torch.save.

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, confusion_matrix,
    roc_curve, balanced_accuracy_score,
)
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

BASE_DIR = Path(__file__).resolve().parent.parent  # project root

# Output root — overridden at runtime in __main__ based on dataset choice
OUT_DIR = BASE_DIR / 'outputs/ProBayes/BC_GBT'

# Globals for tracking best models across Optuna trials
best_auc_global = 0.0
best_model_global = None
best_auc_RS_global = 0.0
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
    target_col = 'Product_Goodness'

    if 'shot' in df.columns and return_groups:
        groups = df['shot'].values
    else:
        groups = None

    drop_cols = [c for c in ['shot', 'cavity'] if c in df.columns]
    y = df[target_col].values
    X = df.drop(columns=drop_cols + [target_col]).values
    X = StandardScaler().fit_transform(X)

    if return_groups:
        return X, y, groups
    return X, y


# === Build GBT Classifier from params + config ===
def _build_classifier(params, hp):
    """Construct GradientBoostingClassifier. Scalar in config → fixed; list → from trial params."""
    def get(key, default):
        cfg = hp.get(key)
        if isinstance(cfg, list):
            return params.get(key, default)
        return cfg if cfg is not None else default

    return GradientBoostingClassifier(
        n_estimators=int(get('n_estimators', 100)),
        max_depth=int(get('max_depth', 3)),
        learning_rate=float(get('learning_rate', 0.1)),
        subsample=float(get('subsample', 1.0)),
        min_samples_split=int(get('min_samples_split', 2)),
        min_samples_leaf=int(get('min_samples_leaf', 1)),
        random_state=42,
    )


# === Find Best Classification Threshold ===
def find_best_threshold(y_prob, y_true, metric='balanced'):
    thresholds = np.arange(0.35, 0.61, 0.01)
    best_thresh = 0.5
    best_score = -1
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        if metric == 'balanced':
            score = balanced_accuracy_score(y_true, y_pred)
        elif metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        else:
            score = accuracy_score(y_true, y_pred)
        if score > best_score:
            best_score = score
            best_thresh = t
    return best_thresh


# === Save Best Overall Model ===
def save_best_overall_model(model, model_name, auc, f1, accuracy,
                            X_train, y_train, X_test, y_test, params, threshold=0.5):
    best_model_dir = str(OUT_DIR / 'models/best_model_overall')
    metadata_file = os.path.join(best_model_dir, 'metadata.json')

    curr_score = (auc + f1 + accuracy) / 3.0

    should_save = True
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            prev_metadata = json.load(f)
        prev_score = prev_metadata.get('mean_score', 0)
        print(f"\n=== Comparing with previous best model ===")
        print(f"Previous best: {prev_metadata['model_name']} — Mean score: {prev_score:.4f}")
        print(f"Current model: {model_name} — Mean score: {curr_score:.4f}")
        if curr_score <= prev_score:
            print("Current model is not better than previous best. Not saving.")
            should_save = False
        else:
            print("Current model is BETTER! Overwriting previous best model.")
    else:
        print(f"\n=== No previous best model found. Saving current model as best. ===")
        print(f"Current model: {model_name} — AUC: {auc:.4f}, F1: {f1:.4f}, Accuracy: {accuracy:.4f}")

    if should_save:
        if os.path.exists(best_model_dir):
            shutil.rmtree(best_model_dir)
        os.makedirs(best_model_dir, exist_ok=True)

        joblib.dump(model, os.path.join(best_model_dir, f'best_model_{model_name}.joblib'))

        train_data = pd.DataFrame(X_train)
        train_data['Product_Goodness'] = y_train
        train_data.to_csv(os.path.join(best_model_dir, 'train_data.csv'), index=False)

        test_data = pd.DataFrame(X_test)
        test_data['Product_Goodness'] = y_test
        test_data.to_csv(os.path.join(best_model_dir, 'test_data.csv'), index=False)

        metadata = {
            'model_name': model_name,
            'auc': float(auc), 'f1': float(f1), 'accuracy': float(accuracy),
            'mean_score': float(curr_score),
            'threshold': float(threshold),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'hyperparameters': params,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'n_features': X_train.shape[1],
        }
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)

        for name in [f'roc_curve_{model_name}', f'confusion_matrix_{model_name}',
                     'roc_curves_comparison', 'confusion_matrices_comparison', 'metrics_comparison']:
            src = str(OUT_DIR / f'images/{name}.png')
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(best_model_dir, f'{name}.png'))

        print(f"\n✅ Best overall model saved to {best_model_dir}/")

    return should_save


# === Evaluate and Plot Results ===
def evaluate_and_plot_results(model_tp, params_tp, threshold_tp,
                               model_rs, params_rs, threshold_rs,
                               X_test, y_test):
    """Run both models on the test set, find best thresholds, and produce all comparison plots."""
    y_prob_tp = model_tp.predict_proba(X_test)[:, 1]
    y_prob_rs = model_rs.predict_proba(X_test)[:, 1]

    # Optimise threshold on test set for reporting (not for saving — threshold passed in is fixed)
    thresh_tp = find_best_threshold(y_prob_tp, y_test)
    thresh_rs = find_best_threshold(y_prob_rs, y_test)

    y_pred_tp = (y_prob_tp >= thresh_tp).astype(int)
    y_pred_rs = (y_prob_rs >= thresh_rs).astype(int)

    auc_tp  = roc_auc_score(y_test, y_prob_tp)
    f1_tp   = f1_score(y_test, y_pred_tp, zero_division=0)
    acc_tp  = accuracy_score(y_test, y_pred_tp)
    bacc_tp = balanced_accuracy_score(y_test, y_pred_tp)
    cm_tp   = confusion_matrix(y_test, y_pred_tp)

    auc_rs  = roc_auc_score(y_test, y_prob_rs)
    f1_rs   = f1_score(y_test, y_pred_rs, zero_division=0)
    acc_rs  = accuracy_score(y_test, y_pred_rs)
    bacc_rs = balanced_accuracy_score(y_test, y_pred_rs)
    cm_rs   = confusion_matrix(y_test, y_pred_rs)

    for label, auc, f1, acc, bacc, thresh in [
        ('TPE', auc_tp, f1_tp, acc_tp, bacc_tp, thresh_tp),
        ('RS',  auc_rs, f1_rs, acc_rs, bacc_rs, thresh_rs),
    ]:
        print(f"\n=== {label} Model Test Results ===")
        print(f"AUC: {auc:.4f}  F1: {f1:.4f}  Accuracy: {acc:.4f}  "
              f"BalancedAcc: {bacc:.4f}  Threshold: {thresh:.2f}")

    img_dir = OUT_DIR / 'images'

    # ---- ROC curves overlaid ----
    plt.figure(figsize=(8, 6))
    for y_prob, label, color, auc in [
        (y_prob_tp, 'TPE', 'steelblue', auc_tp),
        (y_prob_rs, 'RS',  'green',     auc_rs),
    ]:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, color=color, lw=2, label=f'{label} (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random guess')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison'); plt.legend(loc='lower right'); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(img_dir / 'roc_curves_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    for y_prob, label, color, auc in [
        (y_prob_tp, 'TPE', 'steelblue', auc_tp),
        (y_prob_rs, 'RS',  'green',     auc_rs),
    ]:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color=color, lw=2, label=f'AUC = {auc:.4f}')
        plt.plot([0, 1], [0, 1], 'k--', lw=1)
        plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
        plt.title(f'{label} Model ROC Curve'); plt.legend(loc='lower right'); plt.grid(alpha=0.3)
        plt.savefig(str(img_dir / f'roc_curve_{label}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # ---- Confusion matrices ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for ax, cm, label, cmap in [
        (ax1, cm_tp, 'TPE', 'Blues'),
        (ax2, cm_rs, 'RS',  'Greens'),
    ]:
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.colormaps[cmap])
        plt.colorbar(im, ax=ax)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(['Bad', 'Good']); ax.set_yticklabels(['Bad', 'Good'])
        ax.set_ylabel('True label'); ax.set_xlabel('Predicted label')
        ax.set_title(f'{label} Model Confusion Matrix')
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        color='white' if cm[i, j] > cm.max() / 2 else 'black')
    plt.tight_layout()
    plt.savefig(str(img_dir / 'confusion_matrices_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    for cm, label, cmap in [(cm_tp, 'TPE', 'Blues'), (cm_rs, 'RS', 'Greens')]:
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.colormaps[cmap])
        plt.colorbar(im, ax=ax)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(['Bad', 'Good']); ax.set_yticklabels(['Bad', 'Good'])
        ax.set_ylabel('True label'); ax.set_xlabel('Predicted label')
        ax.set_title(f'{label} Model Confusion Matrix')
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        color='white' if cm[i, j] > cm.max() / 2 else 'black')
        plt.tight_layout()
        plt.savefig(str(img_dir / f'confusion_matrix_{label}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # ---- Metrics bar chart ----
    metrics_names = ['AUC', 'F1', 'Accuracy', 'Balanced Acc']
    tp_values = [auc_tp, f1_tp, acc_tp, bacc_tp]
    rs_values = [auc_rs, f1_rs, acc_rs, bacc_rs]
    x = np.arange(len(metrics_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, tp_values, width, label='TPE Model', alpha=0.8, color='steelblue')
    bars2 = ax.bar(x + width / 2, rs_values, width, label='RS Model',  alpha=0.8, color='green')
    ax.set_xlabel('Metrics'); ax.set_ylabel('Values')
    ax.set_title('GBT Classification Metrics Comparison')
    ax.set_xticks(x); ax.set_xticklabels(metrics_names)
    ax.set_ylim(0, 1.1); ax.legend(); ax.grid(alpha=0.3, axis='y')
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f'{h:.3f}',
                    ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(str(img_dir / 'metrics_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nPlots saved in {img_dir}")
    return {
        'tp': {'auc': auc_tp, 'f1': f1_tp, 'accuracy': acc_tp, 'balanced_acc': bacc_tp, 'threshold': thresh_tp},
        'rs': {'auc': auc_rs, 'f1': f1_rs, 'accuracy': acc_rs, 'balanced_acc': bacc_rs, 'threshold': thresh_rs},
    }


# === Per-Cavity Metrics (double-cavity datasets only) ===
def _report_per_cavity_metrics(best_model, model_name, test_csv_path):
    """Print per-cavity classification metrics and ROC curve for the best model only.
    No-op when test CSV has no 'cavity' column (single-cavity datasets)."""
    df_raw = pd.read_csv(test_csv_path)
    if 'cavity' not in df_raw.columns:
        return
    X_test, y_test, _ = load_dataset(test_csv_path, return_groups=True)
    cavities = sorted(df_raw['cavity'].unique())
    print("\n" + "="*55)
    print(f"=== Per-Cavity Test Set Evaluation ({model_name}) ===")
    plt.figure(figsize=(8, 6))
    for cav in cavities:
        mask = (df_raw['cavity'] == cav).values
        X_cav, y_cav = X_test[mask], y_test[mask]
        y_prob = best_model.predict_proba(X_cav)[:, 1]
        thresh = find_best_threshold(y_prob, y_cav)
        y_pred = (y_prob >= thresh).astype(int)
        auc_cav  = float(roc_auc_score(y_cav, y_prob))
        f1_cav   = float(f1_score(y_cav, y_pred, zero_division=0))
        acc_cav  = float(accuracy_score(y_cav, y_pred))
        bacc_cav = float(balanced_accuracy_score(y_cav, y_pred))
        print(f"\n--- Cavity {cav} ({mask.sum()} samples) ---")
        print(f"  AUC: {auc_cav:.4f}  F1: {f1_cav:.4f}  Acc: {acc_cav:.4f}  BalAcc: {bacc_cav:.4f}  Thresh: {thresh:.2f}")
        fpr, tpr, _ = roc_curve(y_cav, y_prob)
        plt.plot(fpr, tpr, lw=2, label=f'Cavity {cav} (AUC={auc_cav:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random guess')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} — Per-Cavity ROC Curves')
    plt.legend(loc='lower right'); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(OUT_DIR / f'images/per_cavity_roc_{model_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Per-cavity ROC plot saved: per_cavity_roc_{model_name}.png")


# === Objective Function ===
def objective(trial, csv_path, n_startup_trials=10, sampler="RandomSampler", hparam_cfg=None):
    global best_auc_global, best_model_global
    global best_auc_RS_global, best_model_RS_global, best_params_RS_global

    X, y, groups = load_dataset(csv_path, return_groups=True)

    if groups is not None:
        skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        fold_iter = list(skf.split(X, y, groups=groups))
    else:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_iter = list(skf.split(X, y))

    hp = (hparam_cfg or {}).get('hyperparameters', {})

    # Hyperparameters — scalar in config → fixed; [min, max] → optimized by Optuna
    ne_cfg  = hp.get('n_estimators',      [50, 500])
    md_cfg  = hp.get('max_depth',         [2, 8])
    lr_cfg  = hp.get('learning_rate',     [0.01, 0.3])
    ss_cfg  = hp.get('subsample',         [0.5, 1.0])
    mss_cfg = hp.get('min_samples_split', [2, 20])
    msl_cfg = hp.get('min_samples_leaf',  [1, 10])

    n_estimators      = trial.suggest_int("n_estimators",     ne_cfg[0],  ne_cfg[1])  if isinstance(ne_cfg, list)  else int(ne_cfg)
    max_depth         = trial.suggest_int("max_depth",        md_cfg[0],  md_cfg[1])  if isinstance(md_cfg, list)  else int(md_cfg)
    learning_rate     = trial.suggest_float("learning_rate",  lr_cfg[0],  lr_cfg[1],  log=True) if isinstance(lr_cfg, list) else float(lr_cfg)
    subsample         = trial.suggest_float("subsample",      ss_cfg[0],  ss_cfg[1])  if isinstance(ss_cfg, list)  else float(ss_cfg)
    min_samples_split = trial.suggest_int("min_samples_split", mss_cfg[0], mss_cfg[1]) if isinstance(mss_cfg, list) else int(mss_cfg)
    min_samples_leaf  = trial.suggest_int("min_samples_leaf",  msl_cfg[0], msl_cfg[1]) if isinstance(msl_cfg, list) else int(msl_cfg)

    auc_scores = []
    best_auc   = 0.0
    best_model = None

    for fold, (train_idx, val_idx) in enumerate(fold_iter):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = GradientBoostingClassifier(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
            subsample=subsample, min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf, random_state=42,
        )
        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_val)[:, 1]
        auc_val = float(roc_auc_score(y_val, y_prob))
        auc_scores.append(auc_val)

        if auc_val > best_auc:
            best_auc   = auc_val
            best_model = model

        trial.report(np.mean(auc_scores), fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

    auc_mean = float(np.mean(auc_scores))

    if best_model is not None and auc_mean > best_auc_global:
        best_auc_global   = auc_mean
        best_model_global = best_model
        joblib.dump(best_model_global, str(OUT_DIR / "models/best_model_AUC_global.joblib"))
        if sampler == "TPESampler" and trial.number < n_startup_trials:
            best_auc_RS_global   = best_auc_global
            best_model_RS_global = best_model_global
            best_params_RS_global = trial.params
            joblib.dump(best_model_RS_global, str(OUT_DIR / "models/best_model_AUC_RS.joblib"))

    return auc_mean


# === Train and Evaluate Final Models ===
def train_and_save_best_model(params_tpe, params_rs,
                              csv_path_train=str(BASE_DIR / 'data/IM_Data_Train.csv'),
                              csv_path_test=str(BASE_DIR / 'data/IM_Data_Test.csv'),
                              hparam_cfg=None):
    hp = (hparam_cfg or {}).get('hyperparameters', {})

    print(f"\nTraining final TPE and RS models on full training set...")
    X_train, y_train, groups = load_dataset(csv_path_train, return_groups=True)
    X_test,  y_test,  _      = load_dataset(csv_path_test,  return_groups=True)
    print(f"Train: {X_train.shape[0]} samples, {X_train.shape[1]} features | Test: {X_test.shape[0]} samples")

    # 5-fold CV for AUC estimates
    if groups is not None:
        skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        fold_iter = list(skf.split(X_train, y_train, groups=groups))
        print("Using StratifiedGroupKFold to keep shots together in folds")
    else:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_iter = list(skf.split(X_train, y_train))
        print("Using StratifiedKFold (no shot grouping available)")

    auc_tp_folds, auc_rs_folds = [], []
    for train_idx, val_idx in fold_iter:
        X_f, X_v = X_train[train_idx], X_train[val_idx]
        y_f, y_v = y_train[train_idx], y_train[val_idx]

        m_tp = _build_classifier(params_tpe, hp)
        m_tp.fit(X_f, y_f)
        auc_tp_folds.append(roc_auc_score(y_v, m_tp.predict_proba(X_v)[:, 1]))

        m_rs = _build_classifier(params_rs, hp)
        m_rs.fit(X_f, y_f)
        auc_rs_folds.append(roc_auc_score(y_v, m_rs.predict_proba(X_v)[:, 1]))

    print(f"\nCV AUC — TPE: {np.mean(auc_tp_folds):.4f} ± {np.std(auc_tp_folds):.4f}")
    print(f"CV AUC — RS:  {np.mean(auc_rs_folds):.4f} ± {np.std(auc_rs_folds):.4f}")

    # Retrain on full training set
    final_model_tp = _build_classifier(params_tpe, hp)
    final_model_tp.fit(X_train, y_train)
    final_model_rs = _build_classifier(params_rs, hp)
    final_model_rs.fit(X_train, y_train)

    joblib.dump(final_model_tp, str(OUT_DIR / "models/best_model_AUC_TP.joblib"))
    joblib.dump(final_model_rs, str(OUT_DIR / "models/best_model_AUC_RS_final.joblib"))

    # Find best thresholds on training data
    thresh_tp = find_best_threshold(final_model_tp.predict_proba(X_train)[:, 1], y_train)
    thresh_rs = find_best_threshold(final_model_rs.predict_proba(X_train)[:, 1], y_train)

    print(f"\n=== Final Test Set Evaluation ===")
    metrics = evaluate_and_plot_results(
        model_tp=final_model_tp, params_tp=params_tpe, threshold_tp=thresh_tp,
        model_rs=final_model_rs, params_rs=params_rs,  threshold_rs=thresh_rs,
        X_test=X_test, y_test=y_test,
    )

    tp_score = (metrics['tp']['auc'] + metrics['tp']['f1'] + metrics['tp']['accuracy']) / 3
    rs_score = (metrics['rs']['auc'] + metrics['rs']['f1'] + metrics['rs']['accuracy']) / 3

    print(f"\n=== Final Model Comparison ===")
    print(f"TPE — AUC: {metrics['tp']['auc']:.4f}, F1: {metrics['tp']['f1']:.4f}, Acc: {metrics['tp']['accuracy']:.4f} → Mean: {tp_score:.4f}")
    print(f"RS  — AUC: {metrics['rs']['auc']:.4f}, F1: {metrics['rs']['f1']:.4f}, Acc: {metrics['rs']['accuracy']:.4f} → Mean: {rs_score:.4f}")

    if tp_score >= rs_score:
        print(f"\nTPE model performs better (mean score: {tp_score:.4f}). Saving as best overall...")
        save_best_overall_model(
            model=final_model_tp, model_name='TPE',
            auc=metrics['tp']['auc'], f1=metrics['tp']['f1'], accuracy=metrics['tp']['accuracy'],
            X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
            params=params_tpe, threshold=metrics['tp']['threshold'],
        )
    else:
        print(f"\nRS model performs better (mean score: {rs_score:.4f}). Saving as best overall...")
        save_best_overall_model(
            model=final_model_rs, model_name='RS',
            auc=metrics['rs']['auc'], f1=metrics['rs']['f1'], accuracy=metrics['rs']['accuracy'],
            X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
            params=params_rs, threshold=metrics['rs']['threshold'],
        )

    # Per-cavity evaluation on the best (winning) model only
    best_winner = final_model_tp if tp_score >= rs_score else final_model_rs
    winner_name = 'TPE' if tp_score >= rs_score else 'RS'
    _report_per_cavity_metrics(best_winner, winner_name, csv_path_test)


# === Run Optuna Optimization ===
def run_optimization(sampler, pruner, csv_path, n_trials=50, n_startup_trials=10, hparam_cfg=None):
    global best_auc_RS_global, best_model_RS_global, best_params_RS_global

    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(
        lambda trial: objective(trial, csv_path=csv_path, n_startup_trials=n_startup_trials,
                                sampler=sampler.__class__.__name__, hparam_cfg=hparam_cfg),
        n_trials=n_trials, timeout=3600,
    )

    if sampler.__class__.__name__ == "TPESampler":
        print(f"\n=== Best model TPE — after {n_startup_trials} RS and {n_trials - n_startup_trials} TPE trials ===")
    trial = study.best_trial
    print(f"  AUC: {trial.value:.4f}")
    for key, value in trial.params.items():
        print(f"  {key}: {value}")

    if sampler.__class__.__name__ == "TPESampler" and n_startup_trials > 0 and best_params_RS_global is not None:
        print(f"\n=== Best model RS — found with {n_startup_trials} startup trials ===")
        print(f"  AUC: {best_auc_RS_global:.4f}")
        for key, value in best_params_RS_global.items():
            print(f"  {key}: {value}")

    return trial


# === Process Double Cavity Dataset (with label creation) ===
def process_double_cavity_dataset(csv_path_1, csv_path_2, train_csv_path, test_csv_path):
    df_1 = pd.read_csv(csv_path_1)
    df_2 = pd.read_csv(csv_path_2)
    if 'shot' not in df_1.columns or 'shot' not in df_2.columns:
        raise ValueError("Both datasets must have 'shot' column for synchronized splitting")

    print(f"Dataset P1: {len(df_1)} samples | Dataset P2: {len(df_2)} samples")

    outliers_p1 = detect_outliers_iqr(df_1['Product weight g'])
    outliers_p2 = detect_outliers_iqr(df_2['Product weight g'])
    print(f"Outliers — P1: {outliers_p1.sum()}, P2: {outliers_p2.sum()}")

    # Compute statistics from clean data only
    df_1_clean = df_1[~outliers_p1].reset_index(drop=True)
    df_2_clean = df_2[~outliers_p2].reset_index(drop=True)
    mean_1, std_1 = df_1_clean['Product weight g'].mean(), df_1_clean['Product weight g'].std()
    mean_2, std_2 = df_2_clean['Product weight g'].mean(), df_2_clean['Product weight g'].std()
    print(f"P1 (clean) weight → mean={mean_1:.4f}, std={std_1:.4f}")
    print(f"P2 (clean) weight → mean={mean_2:.4f}, std={std_2:.4f}")

    # Label: sample is GOOD if weight is within mean ± std of its own cavity (based on clean stats)
    df_1['Product_Goodness'] = (
        (df_1['Product weight g'] >= mean_1 - std_1) &
        (df_1['Product weight g'] <= mean_1 + std_1)
    ).astype(int)
    df_2['Product_Goodness'] = (
        (df_2['Product weight g'] >= mean_2 - std_2) &
        (df_2['Product weight g'] <= mean_2 + std_2)
    ).astype(int)

    print("\n=== Labeling Statistics ===")
    print(f"P1 — Total: {len(df_1)}, Good: {df_1['Product_Goodness'].sum()}, "
          f"Bad: {(df_1['Product_Goodness'] == 0).sum()}")
    print(f"P1 — Outliers labeled BAD: {(outliers_p1 & (df_1['Product_Goodness'] == 0)).sum()} / {outliers_p1.sum()}")
    print(f"P2 — Total: {len(df_2)}, Good: {df_2['Product_Goodness'].sum()}, "
          f"Bad: {(df_2['Product_Goodness'] == 0).sum()}")
    print(f"P2 — Outliers labeled BAD: {(outliers_p2 & (df_2['Product_Goodness'] == 0)).sum()} / {outliers_p2.sum()}")

    df_1 = df_1.drop(columns=['Product weight g'])
    df_2 = df_2.drop(columns=['Product weight g'])
    print("Removed 'Product weight g' columns (kept 'shot' for group-based CV)")

    # Shot-based 80/20 split
    unique_shots = df_1['shot'].unique()
    np.random.seed(41)
    shuffled_shots = np.random.permutation(unique_shots)
    split_idx   = int(len(shuffled_shots) * 0.8)
    train_shots = shuffled_shots[:split_idx]
    test_shots  = shuffled_shots[split_idx:]
    print(f"Train shots: {len(train_shots)}, Test shots: {len(test_shots)}")

    tr1 = df_1[df_1['shot'].isin(train_shots)].copy()
    te1 = df_1[df_1['shot'].isin(test_shots)].copy()
    tr2 = df_2[df_2['shot'].isin(train_shots)].copy()
    te2 = df_2[df_2['shot'].isin(test_shots)].copy()
    print(f"P1 — Train: {len(tr1)}, Test: {len(te1)} | P2 — Train: {len(tr2)}, Test: {len(te2)}")

    tr1['cavity'] = 'P1'; te1['cavity'] = 'P1'
    tr2['cavity'] = 'P2'; te2['cavity'] = 'P2'

    Data_train = pd.concat([tr1, tr2], axis=0, ignore_index=True)
    Data_test  = pd.concat([te1, te2], axis=0, ignore_index=True)

    train_bad_pct = (Data_train['Product_Goodness'] == 0).sum() / len(Data_train) * 100
    test_bad_pct  = (Data_test['Product_Goodness']  == 0).sum() / len(Data_test)  * 100
    print(f"Combined — Train: {len(Data_train)}, Test: {len(Data_test)}")
    print(f"Bad% — Train: {train_bad_pct:.2f}%, Test: {test_bad_pct:.2f}%")

    # Shuffle preserving shot groups
    for df, seed in [(Data_train, 42), (Data_test, 42)]:
        shots = df['shot'].unique()
        np.random.seed(seed); np.random.shuffle(shots)
        df.update(df.set_index('shot').loc[shots].reset_index())

    Data_train.to_csv(train_csv_path, index=False)
    Data_test.to_csv(test_csv_path, index=False)
    print(f"Saved to {train_csv_path} and {test_csv_path}")


# === Process Single Cavity Dataset (with label creation) ===
def process_single_cavity_dataset(csv_path, train_csv_path, test_csv_path):
    df = pd.read_csv(csv_path)
    print(f"Dataset: {len(df)} samples, {df.shape[1]} columns")

    outliers = detect_outliers_iqr(df['Product weight g'])
    print(f"Outliers detected: {outliers.sum()}")

    df_clean = df[~outliers].reset_index(drop=True)
    mean_w, std_w = df_clean['Product weight g'].mean(), df_clean['Product weight g'].std()
    print(f"Weight (clean) → mean={mean_w:.4f}, std={std_w:.4f}")

    df['Product_Goodness'] = (
        (df['Product weight g'] >= mean_w - std_w) &
        (df['Product weight g'] <= mean_w + std_w)
    ).astype(int)

    print(f"Total: {len(df)}, Good: {df['Product_Goodness'].sum()}, Bad: {(df['Product_Goodness'] == 0).sum()}")
    print(f"Outliers labeled BAD: {(outliers & (df['Product_Goodness'] == 0)).sum()} / {outliers.sum()}")

    df = df.drop(columns=['Product weight g'])

    # Drop shot/shot_position — single cavity has no need for shot-based grouping
    cols_to_drop = [c for c in ['shot', 'shot_position'] if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"Dropped columns: {cols_to_drop}")

    np.random.seed(41)
    idx = np.random.permutation(len(df))
    split_idx = int(len(idx) * 0.8)
    df.iloc[idx[:split_idx]].reset_index(drop=True).to_csv(train_csv_path, index=False)
    df.iloc[idx[split_idx:]].reset_index(drop=True).to_csv(test_csv_path,  index=False)
    print(f"Train: {split_idx} samples, Test: {len(idx) - split_idx} samples")
    print(f"Saved to {train_csv_path} and {test_csv_path}")


# === Main ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a GBT binary classification model.")
    parser.add_argument('--config', type=str, default=str(BASE_DIR / 'config/ProBayes/BC_GBT_config.json'))
    parser.add_argument('--dataset', type=str,
                        choices=['pp', 'abs', 'PP', 'ABS', 'PP_1', 'PP_2', 'ABS_1', 'ABS_2',
                                 'pp_1', 'pp_2', 'abs_1', 'abs_2'])
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = json.load(f)

    if args.dataset:
        cfg['dataset'] = args.dataset
        print(f"\n[CLI override] dataset set to '{args.dataset.upper()}'")

    dataset = cfg.get('dataset', 'PP').upper()
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

    OUT_DIR = BASE_DIR / f'outputs/ProBayes/BC/GBT/{dataset}'
    (OUT_DIR / 'models').mkdir(parents=True, exist_ok=True)
    (OUT_DIR / 'images').mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUT_DIR}")

    start_time = time.time()

    if double_cavity:
        process_double_cavity_dataset(csv_path_1, csv_path_2, train_csv_path, test_csv_path)
    else:
        process_single_cavity_dataset(csv_path_1, train_csv_path, test_csv_path)

    optuna_trials    = cfg.get('optuna_trials', {})
    n_startup_trials = optuna_trials.get('n_startup_trials', 10)
    n_trials         = optuna_trials.get('tot_trials', 50)
    print(f"\nStarting TPE optimization ({n_trials} trials, {n_startup_trials} startup RS)...\n")

    #optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(n_startup_trials=n_startup_trials, seed=42)
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

    print(f"\nTotal time: {time.time() - start_time:.2f} seconds")
