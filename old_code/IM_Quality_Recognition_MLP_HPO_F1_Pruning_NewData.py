import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, auc, roc_curve, RocCurveDisplay
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
def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y


# === Training Function ===
def train_one_fold(model, train_loader, val_loader, device, criterion, optimizer, patience=5, max_epochs=100):
    early_stopping = EarlyStopping(patience)
    for epoch in range(max_epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        preds, targets = [], []
        # val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                outputs = model(xb)
                probs = torch.sigmoid(outputs)
                pred = (probs > 0.5).float()
                preds.extend(pred.cpu().numpy())
                targets.extend(yb.cpu().numpy())
                # val_loss += criterion(outputs, yb).item()

        f1 = f1_score(targets, preds)
        early_stopping(f1, model)
        # early_stopping(val_loss, model)
        if early_stopping.early_stop:
            break

    model.load_state_dict(early_stopping.best_model_state)
    return model


# === Evaluate Model ===
def evaluate_model(model, loader, device, metric='f1'):
    if metric not in ['f1', 'accuracy']:
        raise ValueError("Metric must be 'f1' or 'accuracy'")
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    if metric == 'f1':
        f1 = f1_score(all_labels, all_preds)
        return f1
    elif metric == 'accuracy':
        accuracy = accuracy_score(all_labels, all_preds)
        return accuracy


# === Objective Function ===
def objective(trial, csv_path='data/DATA_ABS_&_PP_Binary.csv'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, y = load_dataset(csv_path)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    alpha = trial.suggest_float("alpha", 0.1, 0.9)
    gamma = trial.suggest_float("gamma", 0.5, 5.0)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    n_layers = trial.suggest_int("n_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.2, 0.5)
    layers_dim = [
        trial.suggest_int("size_layer{}".format(i), 4, 128, log=True) for i in range(n_layers)  # TODO Here I can start from 4 or mmaybe from X.shape[1] ??
    ]

    f1_scores = []
    best_f1 = 0

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                 torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
        val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                               torch.tensor(y_val, dtype=torch.float32).unsqueeze(1))

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        model = BinaryClassifier(input_size=X.shape[1], layers_dim=layers_dim, dropout=dropout).to(device)
        criterion = BinaryFocalLoss(alpha=alpha, gamma=gamma)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Train
        model = train_one_fold(model, train_loader, val_loader, device, criterion, optimizer)

        # Evaluate
        f1 = evaluate_model(model, val_loader, device, 'f1')
        f1_scores.append(f1)

        if f1 > best_f1:
            best_f1 = f1
        
        trial.report(np.mean(f1), fold)

        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(f1_scores)


# === Run Optuna Optimization ===
def run_optimization(csv_path='data/DATA_ABS_&_PP_Binary.csv'):
    n_trials = 20 # TODO here retunr to 100
    sampler = optuna.samplers.TPESampler(n_startup_trials=10, seed=31) # Here tried to add some startup trials
    pruner = optuna.pruners.HyperbandPruner(min_resource=1, max_resource=100, reduction_factor=3)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(lambda trial: objective(trial, csv_path=csv_path), n_trials=n_trials, timeout=3600)

    # Best trial
    print("\nBest trial:")
    trial = study.best_trial
    print(f"  F1 Score: {trial.value:.4f}")
    for key, value in trial.params.items():
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


# === Retrain Final Model ===
def train_and_save_best_model(params, epochs=100, csv_path='data/DATA_ABS_&_PP_Binary.csv', save_path="models/best_model_F1.pt"):
    print(f"\nTraining the best model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, y = load_dataset(csv_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    print(f"The Data has {X.shape[0]} samples and {X.shape[1]} features.")

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                             torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
    train_loader = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=True)

    model = BinaryClassifier(input_size=X.shape[1], layers_dim=[params["size_layer{}".format(i)] for i in range(params["n_layers"])]).to(device)
    criterion = BinaryFocalLoss(alpha=params["alpha"], gamma=params["gamma"])
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])

    early_stopping = EarlyStopping(patience=5)
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            outputs = model(torch.tensor(X_test, dtype=torch.float32).to(device))
            preds = (torch.sigmoid(outputs) > 0.5).float().cpu().numpy()
        f1 = f1_score(y_test, preds)
        early_stopping(f1, model)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch + 1} with F1 Score: {f1:.4f}")
            break

    model.load_state_dict(early_stopping.best_model_state)
    torch.save(model.state_dict(), save_path)

    model.eval()
    with torch.no_grad():
        test_outputs = model(torch.tensor(X_test, dtype=torch.float32).to(device))
        test_outputs_prob = torch.sigmoid(test_outputs).float().cpu().numpy()
        test_preds = (test_outputs_prob > 0.5).astype(float)
    final_f1 = f1_score(y_test, test_preds)
    final_acc = accuracy_score(y_test, test_preds)
    fpr, tpr, _ = roc_curve(y_test, test_outputs_prob)
    roc_auc = auc(fpr, tpr)
    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, name='MLP')
    display.plot()
    plt.title('ROC Curve')
    plt.savefig('images/roc_curve.png')
    plt.show()

        # Plot test results
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label='True Labels', color='red', alpha=0.6)
    plt.plot(test_preds, label='Predicted Probabilities (Threshold 0.5)', color='blue', alpha=0.6)
    plt.legend()
    plt.savefig('images/test_results_F1.png')
    plt.show()

    print(f"Final best model saved to {save_path}\nF1 Score on Test: {final_f1:.4f}, Accuracy on Test: {final_acc:.4f}, ROC AUC: {roc_auc:.4f}")


if __name__ == "__main__":
    start_time = time.time()
    # Argument parser
    parser = argparse.ArgumentParser(description="Train a binary classification model.")
    parser.add_argument('--first_data', action='store_true', help="Decice whether to use first dataset, if selected only the first dataset will be used")
    parser.add_argument('--first_data_full', action='store_true', help="Decice whether to use first dataset with also measurments, if selected only the first dataset with measurment will be used")
    parser.add_argument('--pp_data', action='store_true', help="Decice whether to use PP dataset, can be used together with ABS dataset")
    parser.add_argument('--abs_data', action='store_true', help="Decice whether to use the ABS dataset, can be used together with PP dataset")
    args = parser.parse_args()
    # Load data path
    if args.first_data:
        print("\nUsing the first dataset.\n")
        csv_path = 'data/IM_Data.csv'
    elif args.first_data_full:
        print("\nUsing the first dataset with all measurments.\n")
        csv_path = 'data/IM_Data_Full.csv'
    else:
        if args.pp_data:
            if args.abs_data:
                print("\nUsing the full dataset, both PP and ABS data.\n")
                csv_path = 'data/DATA_ABS_&_PP_Binary.csv'
            else:
                print("\nUsing only the PP dataset.\n")
                csv_path = 'data/DATA_PP_Binary.csv'
        elif args.abs_data:
            print("\nUsing only the ABS dataset.\n")
            csv_path = 'data/DATA_ABS_Binary.csv'
        else:
            print("\nUsing by default the full dataset, both PP and ABS data.\n")
            csv_path = 'data/DATA_ABS_&_PP_Binary.csv'
    # Run HPO otpimization
    best_trial = run_optimization(csv_path)
    # Retrain the best model
    train_and_save_best_model(params=best_trial.params, epochs=1000, csv_path=csv_path)
    # Print total time taken
    end_time = time.time()
    print(f"\nTotal time taken: {end_time - start_time:.2f} seconds")
