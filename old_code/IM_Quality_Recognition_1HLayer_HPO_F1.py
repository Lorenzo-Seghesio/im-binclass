import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score
import optuna
import os
import time

# === Binary Focal Loss ===
class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits).clamp(1e-7, 1 - 1e-7)
        targets = targets.float()
        loss_pos = -self.alpha * (1 - probs) ** self.gamma * targets * torch.log(probs)
        loss_neg = -(1 - self.alpha) * probs ** self.gamma * (1 - targets) * torch.log(1 - probs)
        return (loss_pos + loss_neg).mean()


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


# === Model ===
class SimpleBinaryClassifier(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x):
        return self.net(x)


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
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                outputs = model(xb)
                probs = torch.sigmoid(outputs)
                pred = (probs > 0.5).float()
                preds.extend(pred.cpu().numpy())
                targets.extend(yb.cpu().numpy())

        f1 = f1_score(targets, preds)
        early_stopping(f1, model)
        if early_stopping.early_stop:
            break

    model.load_state_dict(early_stopping.best_model_state)
    return model


# === Objective Function ===
def objective(trial):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    alpha = trial.suggest_float("alpha", 0.1, 0.9)
    gamma = trial.suggest_float("gamma", 0.5, 5.0)
    hidden_size = trial.suggest_int("hidden_size", 16, 128)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    X, y = load_dataset("data/IM_Data.csv")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    f1_scores = []
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                 torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
        val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                               torch.tensor(y_val, dtype=torch.float32).unsqueeze(1))

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        model = SimpleBinaryClassifier(input_dim=X.shape[1], hidden_size=hidden_size).to(device)
        criterion = BinaryFocalLoss(alpha=alpha, gamma=gamma)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        model = train_one_fold(model, train_loader, val_loader, device, criterion, optimizer)

        # Evaluate
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                probs = torch.sigmoid(model(xb))
                pred = (probs > 0.5).float()
                preds.extend(pred.cpu().numpy())
                targets.extend(yb.cpu().numpy())

        f1_scores.append(f1_score(targets, preds))

    return np.mean(f1_scores)


# === Main Optuna ===
def run_optimization():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)

    print("Best trial:")
    trial = study.best_trial
    print(f"  F1 Score: {trial.value:.4f}")
    for key, value in trial.params.items():
        print(f"  {key}: {value}")

    return trial


# === Retrain Final Model ===
def train_and_save_best_model(params, save_path="best_model_F1.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, y = load_dataset("data/IM_Data.csv")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                             torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
    train_loader = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=True)

    model = SimpleBinaryClassifier(input_size=X.shape[1], hidden_size=params["hidden_size"]).to(device)
    criterion = BinaryFocalLoss(alpha=params["alpha"], gamma=params["gamma"])
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])

    early_stopping = EarlyStopping(patience=5)
    for epoch in range(100):
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
            break

    model.load_state_dict(early_stopping.best_model_state)
    torch.save(model.state_dict(), save_path)

    model.eval()
    with torch.no_grad():
        test_outputs = model(torch.tensor(X_test, dtype=torch.float32).to(device))
        test_preds = (torch.sigmoid(test_outputs) > 0.5).float().cpu().numpy()
    final_f1 = f1_score(y_test, test_preds)
    final_acc = accuracy_score(y_test, test_preds)

    print(f"\nFinal best model saved to {save_path}\nF1 Score on Test: {final_f1:.4f}, Accuracy on Test: {final_acc:.4f}")


if __name__ == "__main__":
    start_time = time.time()
    best_trial = run_optimization()
    train_and_save_best_model(best_trial.params)
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")