import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import optuna
import time
import argparse


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
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation loss for early stopping
        model.eval()
        preds, targets = [], []
        # val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                probs = torch.sigmoid(outputs)
                pred = (probs > 0.5).float()
                preds.extend(pred.cpu().numpy())
                targets.extend(labels.cpu().numpy())
                # val_loss += criterion(outputs, labels).item()

        acc = accuracy_score(targets, preds)
        early_stopping(acc, model)
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


# === Optuna Objective Function ===
def objective(trial, csv_path='data/IM_Data.csv'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    alpha = trial.suggest_float("alpha", 0.1, 0.9)
    gamma = trial.suggest_float("gamma", 0.5, 5.0)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    n_layers = trial.suggest_int("n_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.2, 0.5)
    layers_dim = [
        trial.suggest_int("size_layer{}".format(i), 4, 128, log=True) for i in range(n_layers)
    ]

    # Prepare data
    X, y = load_dataset(csv_path)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    dataset = TensorDataset(X_tensor, y_tensor)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    accuracies = []
    best_accuracy = 0

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size)

        model = BinaryClassifier(input_size=X.shape[1], layers_dim=layers_dim, dropout=dropout).to(device)
        criterion = BinaryFocalLoss(alpha=alpha, gamma=gamma)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # Train
        model = train_one_fold(model, train_loader, val_loader, device, criterion, optimizer)
        
        # Evaluate
        accuracy = evaluate_model(model, val_loader, device, 'accuracy')
        accuracies.append(accuracy)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
        
        trial.report(np.mean(accuracies), fold)

        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(accuracies)


# === Run Optuna Optimization ===
def run_optimization(csv_path='data/IM_Data.csv'):
    n_trials = 100
    sampler = optuna.samplers.TPESampler(seed=1)
    pruner = optuna.pruners.HyperbandPruner(min_resource=1, max_resource=100, reduction_factor=3)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(lambda trial: objective(trial, csv_path=csv_path), n_trials=n_trials, timeout=3600)

    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Accuracy: {trial.value:.4f}")
    for key, value in trial.params.items():
        print(f"  {key}: {value}")

    return trial


# === Retrain Final Model ===
def train_and_save_best_model(params, epochs=100, csv_path='data/IM_Data.csv', save_path="models/best_model_accuracy.pt"):
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
        acc = accuracy_score(y_test, preds)
        early_stopping(acc, model)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch + 1} with accuracy: {acc:.4f}")
            break

    model.load_state_dict(early_stopping.best_model_state)
    torch.save(model.state_dict(), save_path)

    model.eval()
    with torch.no_grad():
        test_outputs = model(torch.tensor(X_test, dtype=torch.float32).to(device))
        test_preds = (torch.sigmoid(test_outputs) > 0.5).float().cpu().numpy()
    final_f1 = f1_score(y_test, test_preds)
    final_acc = accuracy_score(y_test, test_preds)

    print(f"Final best model saved to {save_path}\nF1 Score on Test: {final_f1:.4f}, Accuracy on Test: {final_acc:.4f}")


if __name__ == "__main__":
    start_time = time.time()
    # Argument parser
    parser = argparse.ArgumentParser(description="Train a binary classification model.")
    parser.add_argument('--full_data', action='store_true', help="Decice whether to use the full dataset or the one without measurements with NaN values.")
    args = parser.parse_args()
    # Load data path
    if args.full_data:
        print("\nUsing the full dataset.\n")
        csv_path = 'data/IM_Data_Full.csv'
    else:
        print("\nUsing the dataset without measurements.\n")
        csv_path = 'data/IM_Data.csv'
    # Run HPO otpimization
    best_trial = run_optimization(csv_path)
    # Retrain the best model
    train_and_save_best_model(params=best_trial.params, epochs=1000, csv_path=csv_path)
    # Print total time taken
    end_time = time.time()
    print(f"\nTotal time taken: {end_time - start_time:.2f} seconds")
