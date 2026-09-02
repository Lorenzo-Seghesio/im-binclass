# BC_MLP_IM.py
# === Binary Classification MLP Model Definition and Utilities for Injection Moulding data ===
#
# This code is part of a machine learning project for binary classification using MLPs with hyperparameter optimization (HPO) and pruning techniques.
# It includes model definition, training, evaluation, and visualization of results.

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold, train_test_split, GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, auc, roc_curve, RocCurveDisplay, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_selection import mutual_info_classif
import time
import argparse
from plotly.io import show
import matplotlib.pyplot as plt
import optuna
from optuna.visualization.matplotlib import plot_contour
from optuna.visualization.matplotlib import plot_edf
from optuna.visualization.matplotlib import plot_intermediate_values
from optuna.visualization.matplotlib import plot_optimization_history
from optuna.visualization.matplotlib import plot_parallel_coordinate
from optuna.visualization.matplotlib import plot_param_importances
from optuna.visualization.matplotlib import plot_rank
from optuna.visualization.matplotlib import plot_slice
from optuna.visualization.matplotlib import plot_timeline
import math
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
try:
    from Utility.optuna_seeding import resolve_optuna_seed
except ModuleNotFoundError:
    from src.Utility.optuna_seeding import resolve_optuna_seed

BASE_DIR = Path(__file__).resolve().parent.parent  # project root (works from any subfolder)

# Output root — overridden at runtime in __main__ based on dataset choice
OUT_DIR = BASE_DIR / 'outputs/ProBayes/BC'

# Global variables
best_auc_global = 0
best_model_global = None
best_auc_RS_global = 0
best_model_RS_global = None
best_params_RS_global = None

test_csv_path = str(BASE_DIR / 'data/IM_Data_Test.csv')  # Path to the test dataset
train_csv_path = str(BASE_DIR / 'data/IM_Data_Train.csv')  # Path to the training dataset


# === Model Definition ===
class BinaryClassifier(nn.Module):
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


# === Binary Focal Loss ===
class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits).clamp(min=1e-7, max=1 - 1e-7)
        targets = targets.float()
        loss_pos = -self.alpha * (1 - probs) ** self.gamma * targets * torch.log(probs)
        loss_neg = -(1 - self.alpha) * probs ** self.gamma * (1 - targets) * torch.log(1 - probs)
        loss = loss_pos + loss_neg
        return loss.mean() if self.reduction == 'mean' else loss.sum()


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


# === Weight Initialization with Prior ===
def init_weights_with_prior(model: nn.Module, pos_prior: float | None = None, method: str = "kaiming"):
    """
    Initialize Linear layers:
      - Hidden: Kaiming (He) for ReLU
      - Output: Xavier; bias set to logit(pos_prior) if provided
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
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    if pos_prior is not None:
                        p = max(min(float(pos_prior), 1 - 1e-4), 1e-4)
                        m.bias.data.fill_(math.log(p / (1 - p)))
                    else:
                        nn.init.zeros_(m.bias)
            else:
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
def train_one_fold_test(model, train_loader, val_loader, device, criterion, optimizer, patience=5, max_epochs=100, plot_metrics=False, print_early_stopping=False, fold=0, sampler=""):
    
    early_stopping = EarlyStopping(patience)

    train_losses = []
    val_losses = []
    val_aucs = []

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
        probs, targets = [], []
        epoch_val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                outputs = model(xb)
                prob = torch.sigmoid(outputs).float()
                probs.extend(prob.cpu().numpy())
                targets.extend(yb.cpu().numpy())
                epoch_val_loss += criterion(outputs, yb).item()

        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        auc_score = roc_auc_score(targets, probs)
        val_aucs.append(auc_score)

        # early_stopping(auc_score, model)
        early_stopping(-avg_val_loss, model)
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
        ax1.plot(train_losses, label='Train Loss', color='blue')
        ax1.plot(val_losses, label='Val Loss', color='red')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('{} - Training and Validation Loss Fold {}'.format(sampler, fold + 1))
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # AUC plot
        ax2.plot(val_aucs, label='Val AUC', color='green')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('AUC')
        ax2.set_title('{} - Validation AUC Fold {}'.format(sampler, fold + 1))
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(str(OUT_DIR / 'images/{}_training_curves_fold_{}.png'.format(sampler, fold + 1)))
        plt.close()

    return model


# === Training Function ===
def train_one_fold_hpo(model, train_loader, val_loader, device, criterion, optimizer, patience=5, max_epochs=100):
    
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
        probs, targets = [], []
        epoch_val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                outputs = model(xb)
                prob = torch.sigmoid(outputs).float()
                probs.extend(prob.cpu().numpy())
                targets.extend(yb.cpu().numpy())
                epoch_val_loss += criterion(outputs, yb).item()

        avg_val_loss = epoch_val_loss / len(val_loader)

        # auc_score = roc_auc_score(targets, probs)

        # early_stopping(auc_score, model)
        early_stopping(-avg_val_loss, model)
        if early_stopping.early_stop:
            break

    model.load_state_dict(early_stopping.best_model_state)

    return model


# === Evaluate Model ===
def evaluate_model(model, loader, device, metric='f1'):
    if metric not in ['f1', 'accuracy', 'auc']:
        raise ValueError("Metric must be 'f1' or 'accuracy' or 'auc'")
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).float()
            preds = (probs > 0.5).float()
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    if metric == 'f1':
        f1 = f1_score(all_labels, all_preds)
        return f1
    elif metric == 'accuracy':
        accuracy = accuracy_score(all_labels, all_preds)
        return accuracy
    elif metric == 'auc':
        auc_score = roc_auc_score(all_labels, all_probs)
        return auc_score

# === Find Best Threshold on Validation Data ===
def find_best_threshold(model, loader, device, metric='balanced'):
    """
    Find optimal classification threshold using validation data.
    
    Parameters:
    - model: trained model
    - loader: validation data loader
    - device: torch device
    - metric: 'f1', 'accuracy', or 'balanced' (mean of F1 and accuracy)
    
    Returns:
    - best_threshold: optimal threshold value
    - best_score: score at best threshold
    """
    model.eval()
    all_probs, all_labels = [], []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).float()
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_probs = np.array(all_probs).flatten()
    all_labels = np.array(all_labels).flatten()
    
    # Try different thresholds
    best_threshold = 0.5
    best_score = 0
    
    for threshold in np.arange(0.35, 0.65, 0.05):
        preds = (all_probs > threshold).astype(int)
        
        if metric == 'f1':
            score = f1_score(all_labels, preds)
        elif metric == 'accuracy':
            score = accuracy_score(all_labels, preds)
        elif metric == 'balanced':
            score = (f1_score(all_labels, preds) + accuracy_score(all_labels, preds)) / 2
        else:
            raise ValueError("Metric must be 'f1', 'accuracy', or 'balanced'")
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold, best_score

# === Save Best Overall Model ===
def save_best_overall_model(model, model_name, threshold, auc_roc, f1, accuracy, X_train, y_train, X_test, y_test, params):
    """
    Save the best overall model with its metadata, data, and hyperparameters.
    Only saves if the current model is better than the previously saved one.
    
    Parameters:
    - model: trained PyTorch model
    - model_name: 'TPE' or 'RS'
    - threshold: optimal threshold
    - auc_roc: ROC AUC score
    - f1: F1 score at optimal threshold
    - accuracy: Accuracy at optimal threshold
    - X_train, y_train: training data
    - X_test, y_test: test data
    - params: hyperparameters dictionary
    
    Returns:
    - saved: True if model was saved (better than previous), False otherwise
    """
    best_model_dir = str(OUT_DIR / 'models/best_model_overall')
    metadata_file = os.path.join(best_model_dir, 'metadata.json')
    
    # Calculate mean score
    mean_score = (auc_roc + f1 + accuracy) / 3
    
    # Check if there's a previous best model
    should_save = True
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            prev_metadata = json.load(f)
        
        prev_mean_score = (prev_metadata['auc_roc'] + prev_metadata['f1'] + prev_metadata['accuracy']) / 3
        
        print(f"\n=== Comparing with previous best model ===")
        print(f"Previous best: {prev_metadata['model_name']} - Mean score: {prev_mean_score:.4f} (AUC: {prev_metadata['auc_roc']:.4f}, F1: {prev_metadata['f1']:.4f}, Acc: {prev_metadata['accuracy']:.4f})")
        print(f"Current model: {model_name} - Mean score: {mean_score:.4f} (AUC: {auc_roc:.4f}, F1: {f1:.4f}, Acc: {accuracy:.4f})")
        
        if mean_score <= prev_mean_score:
            print(f"Current model is not better than previous best. Not saving.")
            should_save = False
        else:
            print(f"Current model is BETTER! Overwriting previous best model.")
    else:
        print(f"\n=== No previous best model found. Saving current model as best. ===")
        print(f"Current model: {model_name} - Mean score: {mean_score:.4f} (AUC: {auc_roc:.4f}, F1: {f1:.4f}, Acc: {accuracy:.4f})")
    
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
        train_data['Product_Goodness'] = y_train
        train_data.to_csv(os.path.join(best_model_dir, 'train_data.csv'), index=False)
        
        # Save test data
        test_data = pd.DataFrame(X_test)
        test_data['Product_Goodness'] = y_test
        test_data.to_csv(os.path.join(best_model_dir, 'test_data.csv'), index=False)
        
        # Save metadata
        metadata = {
            'model_name': model_name,
            'threshold': float(threshold),
            'auc_roc': float(auc_roc),
            'f1': float(f1),
            'accuracy': float(accuracy),
            'mean_score': float(mean_score),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'hyperparameters': params,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'n_features': X_train.shape[1]
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        # Copy confusion matrix for the best model
        confusion_matrix_src = str(OUT_DIR / f'images/confusion_matrix_{model_name}.png')
        confusion_matrix_dst = os.path.join(best_model_dir, 'confusion_matrix.png')
        if os.path.exists(confusion_matrix_src):
            shutil.copy2(confusion_matrix_src, confusion_matrix_dst)
        
        # Copy ROC curve comparison (with both TPE and RS)
        roc_curve_src = str(OUT_DIR / 'images/auc_opt_roc_curve.png')
        roc_curve_dst = os.path.join(best_model_dir, 'roc_curve_comparison.png')
        if os.path.exists(roc_curve_src):
            shutil.copy2(roc_curve_src, roc_curve_dst)
        
        print(f"\n✅ Best overall model saved to {best_model_dir}/")
        print(f"   - Model: best_model_{model_name}.pt")
        print(f"   - Train data: train_data.csv ({len(X_train)} samples)")
        print(f"   - Test data: test_data.csv ({len(X_test)} samples)")
        print(f"   - Metadata: metadata.json")
        print(f"   - Confusion matrix: confusion_matrix.png")
        print(f"   - ROC curve: roc_curve_comparison.png")
        
    return should_save

# === Evaluate Model and plot results ===
def evaluate_and_plot_results(model_tp, model_rs, X_test, y_test, device, threshold_tp=0.5, threshold_rs=0.5, save_path=None, roc_curve_path=None):
        
        # Evaluate model_tp
        model_tp.eval()
        with torch.no_grad():
            test_outputs_presigmoid_tp = model_tp(torch.tensor(X_test, dtype=torch.float32).to(device))
            test_outputs_prob_tp = torch.sigmoid(test_outputs_presigmoid_tp).float().cpu().numpy()
            thresholds_tp = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
            test_preds_tp = {threshold: (test_outputs_prob_tp > threshold).astype(float) for threshold in thresholds_tp}

        fpr_tp, tpr_tp, _ = roc_curve(y_test, test_outputs_prob_tp)
        roc_auc_tp = auc(fpr_tp, tpr_tp)

        results_tp = {}
        for threshold, preds in test_preds_tp.items():
            f1 = f1_score(y_test, preds)
            acc = accuracy_score(y_test, preds)
            results_tp[threshold] = {"f1": f1, "accuracy": acc}

        print(f"\n=== TPE Model Test Results ===")
        print(f"ROC AUC on Test (TP): {roc_auc_tp:.4f}")
        print(f"Validation-selected threshold: {threshold_tp:.2f}")
        for threshold, metrics in results_tp.items():
            marker = " <-- VALIDATION SELECTED" if abs(threshold - threshold_tp) < 0.01 else ""
            print(f"Threshold {threshold} - F1: {metrics['f1']:.4f}, Accuracy: {metrics['accuracy']:.4f}{marker}")

        # Evaluate model_rs
        model_rs.eval()
        with torch.no_grad():
            test_outputs_presigmoid_rs = model_rs(torch.tensor(X_test, dtype=torch.float32).to(device))
            test_outputs_prob_rs = torch.sigmoid(test_outputs_presigmoid_rs).float().cpu().numpy()
            thresholds_rs = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
            test_preds_rs = {threshold: (test_outputs_prob_rs > threshold).astype(float) for threshold in thresholds_rs}

        fpr_rs, tpr_rs, _ = roc_curve(y_test, test_outputs_prob_rs)
        roc_auc_rs = auc(fpr_rs, tpr_rs)

        results_rs = {}
        for threshold, preds in test_preds_rs.items():
            f1 = f1_score(y_test, preds)
            acc = accuracy_score(y_test, preds)
            results_rs[threshold] = {"f1": f1, "accuracy": acc}

        print(f"\n=== RS Model Test Results ===")
        print(f"ROC AUC on Test (RS): {roc_auc_rs:.4f}")
        print(f"Validation-selected threshold: {threshold_rs:.2f}")
        for threshold, metrics in results_rs.items():
            marker = " <-- VALIDATION SELECTED" if abs(threshold - threshold_rs) < 0.01 else ""
            print(f"Threshold {threshold} - F1: {metrics['f1']:.4f}, Accuracy: {metrics['accuracy']:.4f}{marker}")

        # Compute confusion matrices using validation-selected thresholds
        y_pred_tp = (test_outputs_prob_tp > threshold_tp).astype(int).flatten()
        y_pred_rs = (test_outputs_prob_rs > threshold_rs).astype(int).flatten()
        
        cm_tp = confusion_matrix(y_test, y_pred_tp)
        cm_rs = confusion_matrix(y_test, y_pred_rs)
        
        print(f"\n=== Confusion Matrices (Validation-Selected Thresholds) ===")
        print(f"\nTPE Model (Threshold = {threshold_tp:.2f}):")
        print(f"True Negatives: {cm_tp[0, 0]}, False Positives: {cm_tp[0, 1]}")
        print(f"False Negatives: {cm_tp[1, 0]}, True Positives: {cm_tp[1, 1]}")
        
        print(f"\nRS Model (Threshold = {threshold_rs:.2f}):")
        print(f"True Negatives: {cm_rs[0, 0]}, False Positives: {cm_rs[0, 1]}")
        print(f"False Negatives: {cm_rs[1, 0]}, True Positives: {cm_rs[1, 1]}")
        
        # Plot confusion matrices separately
        # TPE confusion matrix
        fig_tp = plt.figure(figsize=(8, 6))
        disp_tp = ConfusionMatrixDisplay(confusion_matrix=cm_tp, display_labels=['Bad (0)', 'Good (1)'])
        disp_tp.plot(cmap='Blues', values_format='d')
        plt.title(f'TPE Model - Confusion Matrix\n(Threshold = {threshold_tp:.2f}, AUC = {roc_auc_tp:.4f})')
        plt.tight_layout()
        plt.savefig(str(OUT_DIR / 'images/confusion_matrix_TPE.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # RS confusion matrix
        fig_rs = plt.figure(figsize=(8, 6))
        disp_rs = ConfusionMatrixDisplay(confusion_matrix=cm_rs, display_labels=['Bad (0)', 'Good (1)'])
        disp_rs.plot(cmap='Greens', values_format='d')
        plt.title(f'RS Model - Confusion Matrix\n(Threshold = {threshold_rs:.2f}, AUC = {roc_auc_rs:.4f})')
        plt.tight_layout()
        plt.savefig(str(OUT_DIR / 'images/confusion_matrix_RS.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot both confusion matrices together
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        disp_tp_combined = ConfusionMatrixDisplay(confusion_matrix=cm_tp, display_labels=['Bad (0)', 'Good (1)'])
        disp_tp_combined.plot(ax=ax1, cmap='Blues', values_format='d')
        ax1.set_title(f'TPE Model - Confusion Matrix\n(Threshold = {threshold_tp:.2f}, AUC = {roc_auc_tp:.4f})')
        
        disp_rs_combined = ConfusionMatrixDisplay(confusion_matrix=cm_rs, display_labels=['Bad (0)', 'Good (1)'])
        disp_rs_combined.plot(ax=ax2, cmap='Greens', values_format='d')
        ax2.set_title(f'RS Model - Confusion Matrix\n(Threshold = {threshold_rs:.2f}, AUC = {roc_auc_rs:.4f})')
        
        plt.tight_layout()
        plt.savefig(str(OUT_DIR / 'images/confusion_matrices_combined.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nConfusion matrices saved:")
        print(f"  - outputs/ProBayes/{OUT_DIR.relative_to(BASE_DIR)}/images/confusion_matrix_TPE.png")
        print(f"  - outputs/ProBayes/{OUT_DIR.relative_to(BASE_DIR)}/images/confusion_matrix_RS.png")
        print(f"  - outputs/ProBayes/{OUT_DIR.relative_to(BASE_DIR)}/images/confusion_matrices_combined.png")

        # Plot ROC curves together
        plt.figure(figsize=(10, 6))
        plt.plot(fpr_tp, tpr_tp, label=f'TPE Model (AUC = {roc_auc_tp:.4f})', color='red', linestyle='-', linewidth=2)
        plt.plot(fpr_rs, tpr_rs, label=f'Random Searching Model (AUC = {roc_auc_rs:.4f})', color='blue', linestyle='--', linewidth=2)
        plt.plot([0, 1], [0, 1], color='gray', linestyle=':', label='Random Guess', linewidth=1)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        plt.savefig(roc_curve_path or str(OUT_DIR / 'images/auc_opt_roc_curve.png'))
        plt.close()
        
        # Get metrics at validation-selected thresholds for return
        y_pred_tp_selected = (test_outputs_prob_tp > threshold_tp).astype(int).flatten()
        y_pred_rs_selected = (test_outputs_prob_rs > threshold_rs).astype(int).flatten()
        
        f1_tp_selected = f1_score(y_test, y_pred_tp_selected)
        acc_tp_selected = accuracy_score(y_test, y_pred_tp_selected)
        
        f1_rs_selected = f1_score(y_test, y_pred_rs_selected)
        acc_rs_selected = accuracy_score(y_test, y_pred_rs_selected)
        
        # Return metrics for both models
        return {
            'tp': {'auc': roc_auc_tp, 'f1': f1_tp_selected, 'accuracy': acc_tp_selected, 'threshold': threshold_tp},
            'rs': {'auc': roc_auc_rs, 'f1': f1_rs_selected, 'accuracy': acc_rs_selected, 'threshold': threshold_rs}
        }


# === Per-Cavity Metrics (double-cavity datasets only) ===
def _report_per_cavity_metrics(best_model, model_name, test_csv_path, device):
    """Print per-cavity classification metrics and ROC curve for the best model only.
    No-op when test CSV has no 'cavity' column (single-cavity datasets)."""
    df_raw = pd.read_csv(test_csv_path)
    if 'cavity' not in df_raw.columns:
        return
    X_test, y_test, _ = load_dataset(test_csv_path, return_groups=True)
    cavities = sorted(df_raw['cavity'].unique())
    print("\n" + "="*55)
    print(f"=== Per-Cavity Test Set Evaluation ({model_name}) ===")
    best_model.eval()
    plt.figure(figsize=(8, 6))
    for cav in cavities:
        mask = (df_raw['cavity'] == cav).values
        X_cav, y_cav = X_test[mask], y_test[mask]
        with torch.no_grad():
            logits = best_model(torch.tensor(X_cav, dtype=torch.float32).to(device))
            y_prob = torch.sigmoid(logits).cpu().numpy().flatten()
        thresh, _ = find_best_threshold(best_model,
                                        DataLoader(TensorDataset(
                                            torch.tensor(X_cav, dtype=torch.float32),
                                            torch.tensor(y_cav, dtype=torch.float32).unsqueeze(1)),
                                            batch_size=64), device)
        y_pred = (y_prob >= thresh).astype(int)
        auc_cav  = float(roc_auc_score(y_cav, y_prob))
        f1_cav   = float(f1_score(y_cav, y_pred, zero_division=0))
        acc_cav  = float(accuracy_score(y_cav, y_pred))
        print(f"\n--- Cavity {cav} ({mask.sum()} samples) ---")
        print(f"  AUC: {auc_cav:.4f}  F1: {f1_cav:.4f}  Acc: {acc_cav:.4f}  Thresh: {thresh:.2f}")
        fpr, tpr, _ = roc_curve(y_cav, y_prob)
        plt.plot(fpr, tpr, lw=2, label=f'Cavity {cav} (AUC={auc_cav:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random guess')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} \u2014 Per-Cavity ROC Curves')
    plt.legend(loc='lower right'); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(OUT_DIR / f'images/per_cavity_roc_{model_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Per-cavity ROC plot saved: per_cavity_roc_{model_name}.png")


# === Objective Function ===
def objective(trial, csv_path=str(BASE_DIR / 'data/DATA_ABS_&_PP_Binary.csv'), n_startup_trials=10, sampler="RandomSampler", hparam_cfg=None):
    global best_auc_global
    global best_model_global
    global best_auc_RS_global
    global best_model_RS_global
    global best_params_RS_global

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, y, groups = load_dataset(csv_path, return_groups=True)
    
    # Use StratifiedGroupKFold if groups are available, otherwise use StratifiedKFold
    if groups is not None:
        skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    else:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    hp = (hparam_cfg or {}).get('hyperparameters', {})

    # Hyperparameters — scalar in config → fixed; [min, max] → optimized by Optuna
    lr_cfg = hp.get('lr', [1e-5, 1e-2])
    lr = trial.suggest_float("lr", lr_cfg[0], lr_cfg[1], log=True) if isinstance(lr_cfg, list) else float(lr_cfg)

    alpha_cfg = hp.get('alpha', [0.1, 0.9])
    alpha = trial.suggest_float("alpha", alpha_cfg[0], alpha_cfg[1]) if isinstance(alpha_cfg, list) else float(alpha_cfg)

    gamma_cfg = hp.get('gamma', [0.5, 5.0])
    gamma = trial.suggest_float("gamma", gamma_cfg[0], gamma_cfg[1]) if isinstance(gamma_cfg, list) else float(gamma_cfg)

    bs_cfg = hp.get('batch_size', 32)
    batch_size = trial.suggest_int("batch_size", bs_cfg[0], bs_cfg[1]) if isinstance(bs_cfg, list) else int(bs_cfg)

    do_cfg = hp.get('dropout', 0.2)
    dropout = trial.suggest_float("dropout", do_cfg[0], do_cfg[1]) if isinstance(do_cfg, list) else float(do_cfg)

    wd_cfg = hp.get('weight_decay', [1e-6, 1e-2])
    weight_decay = trial.suggest_float("weight_decay", wd_cfg[0], wd_cfg[1], log=True) if isinstance(wd_cfg, list) else float(wd_cfg)

    nl_cfg = hp.get('n_layers', [2, 4])
    n_layers = trial.suggest_int("n_layers", nl_cfg[0], nl_cfg[1]) if isinstance(nl_cfg, list) else int(nl_cfg)

    sl_cfg = hp.get('size_1st_hidden_layer', None)
    layers_dim = []
    if isinstance(sl_cfg, list) or sl_cfg is None:
        if isinstance(sl_cfg, list):
            # list: first layer size is optimized between these bounds; subsequent layers are optimized with dynamic bounds based on previous layer
            neuron_max_limit = sl_cfg[1]
            neuron_min_limit = sl_cfg[0]
        else:
            # None: Fallback - dynamic bounds from feature count
            neuron_min_limit = int(1/6 * X.shape[1])
            neuron_max_limit = 5 * X.shape[1]
        for i in range(n_layers):
            size_layer = trial.suggest_int("size_layer{}".format(i), neuron_min_limit, neuron_max_limit, log=True)
            neuron_max_limit = size_layer
            neuron_min_limit = 1
            layers_dim.append(size_layer)
    else:
        # scalar: first layer is fixed at this exact size; subsequent layers are optimized with dynamic bounds based on this fixed size
        fixed = int(sl_cfg)
        trial.suggest_int("size_layer0", fixed, fixed)
        layers_dim.append(fixed)
        neuron_max_limit = fixed
        for i in range(1, n_layers):
            size_layer = trial.suggest_int("size_layer{}".format(i), 1, neuron_max_limit)
            neuron_max_limit = size_layer
            layers_dim.append(size_layer)

    auc_scores = []
    best_auc = 0
    best_model = None

    # Split with groups if available
    if groups is not None:
        fold_iterator = skf.split(X, y, groups=groups)
    else:
        fold_iterator = skf.split(X, y)
    
    for fold, (train_idx, val_idx) in enumerate(fold_iterator):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                 torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
        val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                               torch.tensor(y_val, dtype=torch.float32).unsqueeze(1))

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        model = BinaryClassifier(input_size=X.shape[1], layers_dim=layers_dim, dropout=dropout).to(device)
        # Initialize using training fold positive prior
        init_weights_with_prior(model, pos_prior=float(y_train.mean()))
        criterion = BinaryFocalLoss(alpha=alpha, gamma=gamma)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Train
        model = train_one_fold_hpo(model, train_loader, val_loader, device, criterion, optimizer)

        # Evaluate
        auc_score = evaluate_model(model, val_loader, device, 'auc')
        auc_scores.append(auc_score)

        if auc_score > best_auc:
            best_auc = auc_score
            best_model = model
        
        trial.report(np.mean(auc_score), fold)


        if trial.should_prune():
            raise optuna.TrialPruned()

    if best_model is not None:
        if best_model_global is None or np.mean(auc_scores) > best_auc_global:
            best_auc_global = np.mean(auc_scores)
            best_model_global = best_model
            torch.save(best_model_global.state_dict(), str(OUT_DIR / "models/best_model_AUC_global.pt"))
            if (sampler == "TPESampler") and (trial.number < n_startup_trials):
                    best_auc_RS_global = best_auc_global
                    best_model_RS_global = best_model_global
                    best_params_RS_global = trial.params  # Store the TRIAL parameters (hyperparameters), not model weights
                    torch.save(best_model_RS_global.state_dict(), str(OUT_DIR / "models/best_model_AUC_RS.pt"))

    return np.mean(auc_scores)


# === Retrain Final Model ===
def train_and_save_best_model(params_tpe, params_rs, epochs=100, csv_path_train=str(BASE_DIR / 'data/IM_Data_Train.csv'), csv_path_test=str(BASE_DIR / 'data/IM_Data_Test.csv'), hparam_cfg=None):
    hp = (hparam_cfg or {}).get('hyperparameters', {})
    # hyperparameters: use trial value if optimized (present in params), else config/ProBayes/default
    lr_tp = params_tpe.get('lr', 1e-3) if isinstance(hp.get('lr'), list) else hp.get('lr', 1e-3)
    alpha_tp = params_tpe.get('alpha', 0.25) if isinstance(hp.get('alpha'), list) else hp.get('alpha', 0.25)
    gamma_tp = params_tpe.get('gamma', 2.0) if isinstance(hp.get('gamma'), list) else hp.get('gamma', 2.0)
    dropout_tp = params_tpe.get('dropout', 0.2) if isinstance(hp.get('dropout'), list) else hp.get('dropout', 0.2)
    weight_decay_tp = params_tpe.get('weight_decay', 1e-4) if isinstance(hp.get('weight_decay'), list) else hp.get('weight_decay', 1e-4)
    batch_size_tp = params_tpe.get('batch_size', 32) if isinstance(hp.get('batch_size'), list) else hp.get('batch_size', 32)
    lr_rs = params_rs.get('lr', 1e-3) if isinstance(hp.get('lr'), list) else hp.get('lr', 1e-3)
    alpha_rs = params_rs.get('alpha', 0.25) if isinstance(hp.get('alpha'), list) else hp.get('alpha', 0.25)
    gamma_rs = params_rs.get('gamma', 2.0) if isinstance(hp.get('gamma'), list) else hp.get('gamma', 2.0)
    dropout_rs = params_rs.get('dropout', 0.2) if isinstance(hp.get('dropout'), list) else hp.get('dropout', 0.2)
    weight_decay_rs = params_rs.get('weight_decay', 1e-4) if isinstance(hp.get('weight_decay'), list) else hp.get('weight_decay', 1e-4)
    batch_size_rs = params_rs.get('batch_size', 32) if isinstance(hp.get('batch_size'), list) else hp.get('batch_size', 32)

    print(f"\nTraining the best model TPE and RS...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train data
    X_train, y_train, groups = load_dataset(csv_path_train, return_groups=True)
    print(f"The Data has {X_train.shape[0]} samples and {X_train.shape[1]} features.")

    # Use StratifiedGroupKFold if groups are available, otherwise use StratifiedKFold
    if groups is not None:
        skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        print(f"Using StratifiedGroupKFold to keep shots together in folds")
    else:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        print(f"Using StratifiedKFold (no shot grouping available)")

    auc_scores_tp = []
    best_auc_tp = 0
    best_model_tp = None
    best_threshold_tp = 0.5
    best_val_loader_tp = None

    auc_scores_rs = []
    best_auc_rs = 0
    best_model_rs = None
    best_threshold_rs = 0.5
    best_val_loader_rs = None
    
    # Store all thresholds from each fold
    thresholds_tp = []
    thresholds_rs = []

    # In final training I increase the patience of the early stopping to 20 and the numkber of epochs to 200
    early_stopping_patience = 20
    num_epochs = 200

    # Split with groups if available
    if groups is not None:
        fold_iterator = skf.split(X_train, y_train, groups=groups)
    else:
        fold_iterator = skf.split(X_train, y_train)

    for fold, (train_idx, val_idx) in enumerate(fold_iterator):
        
        # Initialize models for each fold
        # TPE
        model_tp = BinaryClassifier(input_size=X_train.shape[1], layers_dim=[params_tpe["size_layer{}".format(i)] for i in range(params_tpe["n_layers"])], dropout=dropout_tp).to(device)
        criterion_tp = BinaryFocalLoss(alpha=alpha_tp, gamma=gamma_tp)
        optimizer_tp = torch.optim.AdamW(model_tp.parameters(), lr=lr_tp, weight_decay=weight_decay_tp)
        init_weights_with_prior(model_tp, pos_prior=float(y_train.mean()))
        #RS
        model_rs = BinaryClassifier(input_size=X_train.shape[1], layers_dim=[params_rs["size_layer{}".format(i)] for i in range(params_rs["n_layers"])], dropout=dropout_rs).to(device)
        criterion_rs = BinaryFocalLoss(alpha=alpha_rs, gamma=gamma_rs)
        optimizer_rs = torch.optim.AdamW(model_rs.parameters(), lr=lr_rs, weight_decay=weight_decay_rs)
        init_weights_with_prior(model_rs, pos_prior=float(y_train.mean()))

        # Prepare data loaders
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

        train_ds = TensorDataset(torch.tensor(X_train_fold, dtype=torch.float32),
                                 torch.tensor(y_train_fold, dtype=torch.float32).unsqueeze(1))
        val_ds = TensorDataset(torch.tensor(X_val_fold, dtype=torch.float32),
                               torch.tensor(y_val_fold, dtype=torch.float32).unsqueeze(1))
        
        train_loader_tpe = DataLoader(train_ds, batch_size=batch_size_tp, shuffle=True)
        val_loader_tpe = DataLoader(val_ds, batch_size=batch_size_tp)

        train_loader_rs = DataLoader(train_ds, batch_size=batch_size_rs, shuffle=True)
        val_loader_rs = DataLoader(val_ds, batch_size=batch_size_rs)

        # Train TP and RS
        model_tp = train_one_fold_test(model_tp, train_loader_tpe, val_loader_tpe, device, criterion_tp, optimizer_tp, early_stopping_patience, num_epochs, plot_metrics=True, print_early_stopping=True, fold=fold, sampler="TPE")
        model_rs = train_one_fold_test(model_rs, train_loader_rs, val_loader_rs, device, criterion_rs, optimizer_rs, early_stopping_patience, num_epochs, plot_metrics=True, print_early_stopping=True, fold=fold, sampler="RS")

        # Find best threshold on validation set for each fold
        threshold_tp_fold, score_tp_fold = find_best_threshold(model_tp, val_loader_tpe, device, metric='balanced')
        threshold_rs_fold, score_rs_fold = find_best_threshold(model_rs, val_loader_rs, device, metric='balanced')
        
        thresholds_tp.append(threshold_tp_fold)
        thresholds_rs.append(threshold_rs_fold)
        
        print(f"Fold {fold+1} - TPE best threshold: {threshold_tp_fold:.2f} (balanced score: {score_tp_fold:.4f})")
        print(f"Fold {fold+1} - RS best threshold: {threshold_rs_fold:.2f} (balanced score: {score_rs_fold:.4f})")

        # Evaluate TP
        auc_score_tp = evaluate_model(model_tp, val_loader_tpe, device, 'auc')
        auc_scores_tp.append(auc_score_tp)
        if auc_score_tp > best_auc_tp:
            best_auc_tp = auc_score_tp
            best_model_tp = model_tp
            best_threshold_tp = threshold_tp_fold
            best_val_loader_tp = val_loader_tpe

        # Evaluate RS
        auc_score_rs = evaluate_model(model_rs, val_loader_rs, device, 'auc')
        auc_scores_rs.append(auc_score_rs)
        if auc_score_rs > best_auc_rs:
            best_auc_rs = auc_score_rs
            best_model_rs = model_rs
            best_threshold_rs = threshold_rs_fold
            best_val_loader_rs = val_loader_rs

    print(f"\nTP: Best AUC Score across folds: {best_auc_tp:.4f} and mean AUC Score: {np.mean(auc_scores_tp):.4f}, after {len(auc_scores_tp)} folds.")
    print(f"RS: Best AUC Score across folds: {best_auc_rs:.4f} and mean AUC Score: {np.mean(auc_scores_rs):.4f}, after {len(auc_scores_rs)} folds.")
    
    # Report threshold statistics
    print(f"\nThreshold statistics across folds:")
    print(f"TPE - Mean threshold: {np.mean(thresholds_tp):.2f} ± {np.std(thresholds_tp):.2f}, Best fold threshold: {best_threshold_tp:.2f}")
    print(f"RS  - Mean threshold: {np.mean(thresholds_rs):.2f} ± {np.std(thresholds_rs):.2f}, Best fold threshold: {best_threshold_rs:.2f}")

    torch.save(best_model_tp.state_dict(), str(OUT_DIR / "models/best_model_AUC_TP.pt"))
    torch.save(best_model_rs.state_dict(), str(OUT_DIR / "models/best_model_AUC_RS.pt"))

    # Model evaluation
    print(f"\n=== Final Test Set Evaluation ===")
    # Load test data (with return_groups=True to remove 'shot' column)
    X_test, y_test, _ = load_dataset(csv_path_test, return_groups=True)
    metrics = evaluate_and_plot_results(best_model_tp, best_model_rs, X_test, y_test, device=device, 
                            threshold_tp=best_threshold_tp, threshold_rs=best_threshold_rs,
                            save_path=str(OUT_DIR / "images/test_results_AUC.png"), roc_curve_path=str(OUT_DIR / "images/auc_opt_roc_curve.png"))
    
    # Determine which model is better
    mean_score_tp = (metrics['tp']['auc'] + metrics['tp']['f1'] + metrics['tp']['accuracy']) / 3
    mean_score_rs = (metrics['rs']['auc'] + metrics['rs']['f1'] + metrics['rs']['accuracy']) / 3
    
    print(f"\n=== Final Model Comparison ===")
    print(f"TPE Model - Mean score: {mean_score_tp:.4f} (AUC: {metrics['tp']['auc']:.4f}, F1: {metrics['tp']['f1']:.4f}, Acc: {metrics['tp']['accuracy']:.4f})")
    print(f"RS Model  - Mean score: {mean_score_rs:.4f} (AUC: {metrics['rs']['auc']:.4f}, F1: {metrics['rs']['f1']:.4f}, Acc: {metrics['rs']['accuracy']:.4f})")
    
    # Save the better model as best overall
    if mean_score_tp >= mean_score_rs:
        print(f"\nTPE model performs better. Checking if it should be saved as best overall...")
        save_best_overall_model(
            model=best_model_tp,
            model_name='TPE',
            threshold=metrics['tp']['threshold'],
            auc_roc=metrics['tp']['auc'],
            f1=metrics['tp']['f1'],
            accuracy=metrics['tp']['accuracy'],
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            params=params_tpe
        )
    else:
        print(f"\nRS model performs better. Checking if it should be saved as best overall...")
        save_best_overall_model(
            model=best_model_rs,
            model_name='RS',
            threshold=metrics['rs']['threshold'],
            auc_roc=metrics['rs']['auc'],
            f1=metrics['rs']['f1'],
            accuracy=metrics['rs']['accuracy'],
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            params=params_rs
        )

    # Per-cavity evaluation on the best (winning) model only
    best_winner = best_model_tp if mean_score_tp >= mean_score_rs else best_model_rs
    winner_name = 'TPE' if mean_score_tp >= mean_score_rs else 'RS'
    _report_per_cavity_metrics(best_winner, winner_name, csv_path_test, device)


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

    df_1_clean = df_1[~outliers_p1].reset_index(drop=True)
    df_2_clean = df_2[~outliers_p2].reset_index(drop=True)
    mean_1, std_1 = df_1_clean['Product weight g'].mean(), df_1_clean['Product weight g'].std()
    mean_2, std_2 = df_2_clean['Product weight g'].mean(), df_2_clean['Product weight g'].std()
    print(f"P1 (clean) weight → mean={mean_1:.4f}, std={std_1:.4f}")
    print(f"P2 (clean) weight → mean={mean_2:.4f}, std={std_2:.4f}")

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
    train_shot_order = Data_train['shot'].unique()
    np.random.seed(42); np.random.shuffle(train_shot_order)
    Data_train = Data_train.set_index('shot').loc[train_shot_order].reset_index()
    test_shot_order = Data_test['shot'].unique()
    np.random.seed(42); np.random.shuffle(test_shot_order)
    Data_test = Data_test.set_index('shot').loc[test_shot_order].reset_index()

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


# === Run Optuna Optimization ===
def run_optimization(sampler, pruner, csv_path=str(BASE_DIR / 'data/DATA_ABS_&_PP_Binary.csv'), n_trials=100, n_startup_trials=10, hparam_cfg=None):
    global best_auc_RS_global
    global best_model_RS_global
    global best_params_RS_global
    
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(lambda trial: objective(trial, csv_path=csv_path, n_startup_trials=n_startup_trials, sampler=sampler.__class__.__name__, hparam_cfg=hparam_cfg), n_trials=n_trials, timeout=3600)

    # Best trial overall (hoping is TPE)
    if sampler.__class__.__name__ == "TPESampler":
        print("\n=== Best model TPE - after initial {} RS and {} TPE trials ===".format(n_startup_trials, (n_trials - n_startup_trials)))
    elif sampler.__class__.__name__ == "RandomSampler":
        print("\n=== RandomSampler ===")
    print("Best trial:")
    trial = study.best_trial
    print(f"  AUC Score: {trial.value:.4f}")
    for key, value in trial.params.items():
        print(f"  {key}: {value}")

    # If TPE optimizer, Best trial RS
    if (sampler.__class__.__name__ == "TPESampler") and (n_startup_trials > 0) and (best_params_RS_global is not None):
            print("\n=== Best model RS - found with initial {} RS trials ===".format(n_startup_trials))
            print(f"  AUC Score: {best_auc_RS_global:.4f}")
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


if __name__ == "__main__":
    # Load config
    parser = argparse.ArgumentParser(description="Train a binary classification model.")
    parser.add_argument('--config', type=str, default=str(BASE_DIR / 'config/ProBayes/BC_MLP_config.json'),
                        help="Path to the JSON config file (default: config/ProBayes/BC_MLP_config.json)")
    parser.add_argument('--dataset', type=str,
                        choices=['pp', 'abs', 'PP', 'ABS', 'PP_1', 'PP_2', 'ABS_1', 'ABS_2',
                                 'pp_1', 'pp_2', 'abs_1', 'abs_2'],
                        help="Dataset to use. Overrides the value in the config file.")
    parser.add_argument('--optuna-seed', type=int, default=None,
                        help='Optuna sampler seed; omit to generate a fresh seed.')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = json.load(f)

    # CLI --dataset overrides config value
    if args.dataset:
        cfg['dataset'] = args.dataset
        print(f"\n[CLI override] dataset set to '{args.dataset.upper()}'")
    if args.optuna_seed is not None:
        cfg.setdefault('optuna_trials', {})['sampler_seed'] = args.optuna_seed
        print(f"\n[CLI override] Optuna sampler seed set to {args.optuna_seed}")

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

    OUT_DIR = BASE_DIR / f'outputs/ProBayes/BC/MLP/{dataset}'
    (OUT_DIR / 'models').mkdir(parents=True, exist_ok=True)
    (OUT_DIR / 'images').mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUT_DIR}")

    train_csv_path = str(OUT_DIR / 'train_data.csv')
    test_csv_path  = str(OUT_DIR / 'test_data.csv')

    start_time = time.time()

    if double_cavity:
        process_double_cavity_dataset(csv_path_1, csv_path_2, train_csv_path, test_csv_path)
    else:
        process_single_cavity_dataset(csv_path_1, train_csv_path, test_csv_path)

    # Run HPO optimisation with TPE sampler and HyperbandPruner
    print(f"\nStarting TPE optimization...\n")
    optuna_trials    = cfg.get('optuna_trials', {})
    sampler_seed     = resolve_optuna_seed(optuna_trials)
    n_startup_trials = optuna_trials.get('n_startup_trials', 10)
    n_trials         = optuna_trials.get('tot_trials', 100)
    print(f"Total Optuna trials: {n_trials} (with {n_startup_trials} startup trials for RS)")
    sampler = optuna.samplers.TPESampler(n_startup_trials=n_startup_trials, seed=sampler_seed)
    pruner  = optuna.pruners.HyperbandPruner(min_resource=1, max_resource=80, reduction_factor=3)
    best_trial_tpe = run_optimization(sampler, pruner, train_csv_path, n_trials=n_trials, n_startup_trials=n_startup_trials, hparam_cfg=cfg)

    # Retrain the best models
    train_and_save_best_model(params_tpe=best_trial_tpe.params, params_rs=best_params_RS_global, epochs=200, csv_path_train=train_csv_path, csv_path_test=test_csv_path, hparam_cfg=cfg)

    run_info = {
        'run_ts': datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
        'dataset': dataset,
        'split_seed': 42,
        'sampler_seed': sampler_seed,
        'config': str(Path(args.config).resolve()),
    }
    (OUT_DIR / 'run_info.json').write_text(json.dumps(run_info, indent=4))
    print(f"Optuna run info saved to {OUT_DIR / 'run_info.json'}")

    # Print total time taken
    end_time = time.time()
    print(f"\nTotal time taken: {end_time - start_time:.2f} seconds")
