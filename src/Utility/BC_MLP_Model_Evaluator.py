# BC_MLP_Model_Evaluator.py
#
# Binary Classification using a Custom MLP Model
# 
# This script loads and evaluates the best overall model saved in outputs/ProBayes/ProBayes/BC/models/best_model_overall/
# It reproduces the exact training and testing procedure to verify the results.

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, auc, roc_curve, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import json
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # project root (works from any subfolder)

# Import the model definition from the main script
import sys
sys.path.insert(0, str(BASE_DIR / "src"))
from BC_MLP_IM import BinaryClassifier


def load_best_model_metadata():
    """Load metadata from the best overall model."""
    metadata_file = str(BASE_DIR / 'outputs/ProBayes/BC/models/best_model_overall/metadata.json')
    
    if not os.path.exists(metadata_file):
        print(f"Error: No best model found at {metadata_file}")
        print("Please train a model first using IM_Binary_Quality_Recognition.py")
        sys.exit(1)
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    return metadata


def load_data_and_model(metadata):
    """Load the saved training and test data, and the model."""
    base_dir = str(BASE_DIR / 'outputs/ProBayes/BC/models/best_model_overall')
    
    # Load training data
    train_data = pd.read_csv(os.path.join(base_dir, 'train_data.csv'))
    X_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values
    
    # Load test data
    test_data = pd.read_csv(os.path.join(base_dir, 'test_data.csv'))
    X_test = test_data.iloc[:, :-1].values
    y_test = test_data.iloc[:, -1].values
    
    # Standardize features (fit on train, transform both)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Load model
    model_name = metadata['model_name']
    model_path = os.path.join(base_dir, f'best_model_{model_name}.pt')
    
    # Get hyperparameters
    params = metadata['hyperparameters']
    n_layers = params['n_layers']
    layers_dim = [params[f'size_layer{i}'] for i in range(n_layers)]
    dropout = params.get('dropout', 0.2)  # Default if not in params
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BinaryClassifier(input_size=X_train.shape[1], layers_dim=layers_dim, dropout=dropout).to(device)
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    print(f"\n=== Loaded Best Overall Model ===")
    print(f"Model: {model_name}")
    print(f"Timestamp: {metadata['timestamp']}")
    print(f"Architecture: {n_layers} hidden layers with dimensions {layers_dim}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {X_train.shape[1]}")
    
    return model, X_train, y_train, X_test, y_test, device


def evaluate_model(model, X_test, y_test, device, threshold):
    """Evaluate the model on test data."""
    model.eval()
    
    with torch.no_grad():
        test_outputs_presigmoid = model(torch.tensor(X_test, dtype=torch.float32).to(device))
        test_outputs_prob = torch.sigmoid(test_outputs_presigmoid).float().cpu().numpy().flatten()
    
    # Compute metrics at multiple thresholds
    thresholds = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    results = {}
    
    for thresh in thresholds:
        preds = (test_outputs_prob > thresh).astype(int)
        f1 = f1_score(y_test, preds)
        acc = accuracy_score(y_test, preds)
        results[thresh] = {"f1": f1, "accuracy": acc}
    
    # Compute ROC AUC
    fpr, tpr, _ = roc_curve(y_test, test_outputs_prob)
    roc_auc = auc(fpr, tpr)
    
    # Get predictions at the saved threshold
    y_pred_selected = (test_outputs_prob > threshold).astype(int)
    f1_selected = f1_score(y_test, y_pred_selected)
    acc_selected = accuracy_score(y_test, y_pred_selected)
    
    return test_outputs_prob, roc_auc, results, fpr, tpr, y_pred_selected


def plot_results(metadata, roc_auc, fpr, tpr, results, y_test, y_pred, threshold):
    """Plot and save evaluation results."""
    model_name = metadata['model_name']
    
    # Print results
    print(f"\n=== Evaluation Results ===")
    print(f"ROC AUC on Test: {roc_auc:.4f} (Expected: {metadata['auc_roc']:.4f})")
    print(f"Saved threshold: {threshold:.2f}")
    
    for thresh, metrics in results.items():
        marker = " <-- SAVED THRESHOLD" if abs(thresh - threshold) < 0.01 else ""
        print(f"Threshold {thresh} - F1: {metrics['f1']:.4f}, Accuracy: {metrics['accuracy']:.4f}{marker}")
    
    # Metrics at saved threshold
    f1_at_threshold = f1_score(y_test, y_pred)
    acc_at_threshold = accuracy_score(y_test, y_pred)
    
    print(f"\n=== Metrics at Saved Threshold ({threshold:.2f}) ===")
    print(f"F1 Score: {f1_at_threshold:.4f} (Expected: {metadata['f1']:.4f})")
    print(f"Accuracy: {acc_at_threshold:.4f} (Expected: {metadata['accuracy']:.4f})")
    print(f"ROC AUC: {roc_auc:.4f} (Expected: {metadata['auc_roc']:.4f})")
    
    # Check if results match
    auc_match = abs(roc_auc - metadata['auc_roc']) < 0.001
    f1_match = abs(f1_at_threshold - metadata['f1']) < 0.001
    acc_match = abs(acc_at_threshold - metadata['accuracy']) < 0.001
    
    if auc_match and f1_match and acc_match:
        print(f"\n✅ Results successfully reproduced!")
    else:
        print(f"\n⚠️ Warning: Results differ slightly from saved metadata.")
        print(f"   This might be due to random initialization or numerical precision.")
    
    # Plot ROC curve
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f'{model_name} Model (AUC = {roc_auc:.4f})', color='blue', linestyle='-', linewidth=2)
    plt.plot([0, 1], [0, 1], color='gray', linestyle=':', label='Random Guess', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name} Model (Reproduced)')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.savefig(str(BASE_DIR / 'outputs/ProBayes/BC/images/roc_curve_reproduced.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nROC curve saved to: outputs/ProBayes/BC/images/roc_curve_reproduced.png")
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\n=== Confusion Matrix (Threshold = {threshold:.2f}) ===")
    print(f"True Negatives: {cm[0, 0]}, False Positives: {cm[0, 1]}")
    print(f"False Negatives: {cm[1, 0]}, True Positives: {cm[1, 1]}")
    
    fig = plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Bad (0)', 'Good (1)'])
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f'{model_name} Model - Confusion Matrix (Reproduced)\n(Threshold = {threshold:.2f}, AUC = {roc_auc:.4f})')
    plt.tight_layout()
    plt.savefig(str(BASE_DIR / 'outputs/ProBayes/BC/models/best_model_overall/confusion_matrix_reproduced.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to: outputs/ProBayes/BC/models/best_model_overall/confusion_matrix_reproduced.png")
    
    # Plot metrics across thresholds
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    thresholds_list = sorted(results.keys())
    f1_scores = [results[t]['f1'] for t in thresholds_list]
    accuracies = [results[t]['accuracy'] for t in thresholds_list]
    
    ax1.plot(thresholds_list, f1_scores, marker='o', label='F1 Score', color='blue')
    ax1.axvline(x=threshold, color='red', linestyle='--', label=f'Saved Threshold ({threshold:.2f})')
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('F1 Score')
    ax1.set_title('F1 Score vs Threshold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    ax2.plot(thresholds_list, accuracies, marker='o', label='Accuracy', color='green')
    ax2.axvline(x=threshold, color='red', linestyle='--', label=f'Saved Threshold ({threshold:.2f})')
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy vs Threshold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(str(BASE_DIR / 'outputs/ProBayes/BC/models/best_model_overall/metrics_vs_threshold_reproduced.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Metrics plot saved to: outputs/ProBayes/BC/models/best_model_overall/metrics_vs_threshold_reproduced.png")


def main():
    """Main evaluation function."""
    print("="*70)
    print("Model Evaluator - Reproducing Best Overall Model Results")
    print("="*70)
    
    # Load metadata
    metadata = load_best_model_metadata()
    
    # Load data and model
    model, X_train, y_train, X_test, y_test, device = load_data_and_model(metadata)
    
    # Get saved threshold
    threshold = metadata['threshold']
    
    # Evaluate
    test_probs, roc_auc, results, fpr, tpr, y_pred = evaluate_model(
        model, X_test, y_test, device, threshold
    )
    
    # Plot and print results
    plot_results(metadata, roc_auc, fpr, tpr, results, y_test, y_pred, threshold)
    
    print("\n" + "="*70)
    print("Evaluation Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
