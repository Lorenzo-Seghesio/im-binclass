import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, auc, roc_curve, RocCurveDisplay, roc_auc_score
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

# Global variables
best_auc_global = 0
best_model_global = None

batch_size = 32
dropout = 0.2


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
        probs, targets = [], []
        # val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                outputs = model(xb)
                prob = torch.sigmoid(outputs).float()
                probs.extend(prob.cpu().numpy())
                targets.extend(yb.cpu().numpy())
                # val_loss += criterion(outputs, yb).item()

        auc_score = roc_auc_score(targets, probs)
        early_stopping(auc_score, model)
        # early_stopping(val_loss, model)
        if early_stopping.early_stop:
            break

    model.load_state_dict(early_stopping.best_model_state)
    return model


# === Evaluate Model ===
def evaluate_model(model, loader, device, metric='f1'):
    if metric not in ['f1', 'accuracy', 'auc']:
        raise ValueError("Metric must be 'f1' or 'accuracy'")
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

# === Evaluate Model and plot results ===
def evaluate_and_plot_results(model_tp, X_test_tp, y_test_tp, model_rs, X_test_rs, y_test_rs, device, save_path="images/test_results_AUC.png", roc_curve_path="images/auc_opt_roc_curve.png"):
        
        # Evaluate model_tp
        model_tp.eval()
        with torch.no_grad():
            test_outputs_presigmoid_tp = model_tp(torch.tensor(X_test_tp, dtype=torch.float32).to(device))
            test_outputs_prob_tp = torch.sigmoid(test_outputs_presigmoid_tp).float().cpu().numpy()
            thresholds_tp = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6]
            test_preds_tp = {threshold: (test_outputs_prob_tp > threshold).astype(float) for threshold in thresholds_tp}

        fpr_tp, tpr_tp, _ = roc_curve(y_test_tp, test_outputs_prob_tp)
        roc_auc_tp = auc(fpr_tp, tpr_tp)

        results_tp = {}
        for threshold, preds in test_preds_tp.items():
            f1 = f1_score(y_test_tp, preds)
            acc = accuracy_score(y_test_tp, preds)
            results_tp[threshold] = {"f1": f1, "accuracy": acc}

        print(f"ROC AUC on Test (TP): {roc_auc_tp:.4f}")
        for threshold, metrics in results_tp.items():
            print(f"Threshold {threshold} - F1 Score on Test (TP): {metrics['f1']:.4f}, Accuracy on Test (TP): {metrics['accuracy']:.4f}")

        # Evaluate model_rs
        model_rs.eval()
        with torch.no_grad():
            test_outputs_presigmoid_rs = model_rs(torch.tensor(X_test_rs, dtype=torch.float32).to(device))
            test_outputs_prob_rs = torch.sigmoid(test_outputs_presigmoid_rs).float().cpu().numpy()
            thresholds_rs = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6]
            test_preds_rs = {threshold: (test_outputs_prob_rs > threshold).astype(float) for threshold in thresholds_rs}

        fpr_rs, tpr_rs, _ = roc_curve(y_test_rs, test_outputs_prob_rs)
        roc_auc_rs = auc(fpr_rs, tpr_rs)

        results_rs = {}
        for threshold, preds in test_preds_rs.items():
            f1 = f1_score(y_test_rs, preds)
            acc = accuracy_score(y_test_rs, preds)
            results_rs[threshold] = {"f1": f1, "accuracy": acc}

        print(f"ROC AUC on Test (RS): {roc_auc_rs:.4f}")
        for threshold, metrics in results_rs.items():
            print(f"Threshold {threshold} - F1 Score on Test (RS): {metrics['f1']:.4f}, Accuracy on Test (RS): {metrics['accuracy']:.4f}")

        # Plot ROC curves together
        plt.figure(figsize=(10, 6))
        plt.plot(fpr_rs, tpr_rs, label=f'Random Searching Model (AUC = {roc_auc_rs:.4f})', color='blue', linestyle='--', linewidth=2)
        plt.plot(fpr_tp, tpr_tp, label=f'TPE Model (AUC = {roc_auc_tp:.4f})', color='red', linestyle='-', linewidth=2)
        plt.plot([0, 1], [0, 1], color='gray', linestyle=':', label='Random Guess', linewidth=1)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        plt.savefig(roc_curve_path)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.show()

        # # Plot test results
        # plt.figure(figsize=(10, 6))
        # for i, threshold in enumerate([0.3, 0.5, 0.6], start=1):
        #     plt.subplot(3, 1, i)
        #     plt.plot(y_test_tp, label='True Labels', color='green', alpha=0.6)
        #     plt.plot(test_preds[threshold], label=f'Predicted Probabilities (Threshold {threshold})', alpha=0.6)
        #     plt.legend()
        #     plt.title(f'Threshold {threshold}')
        # plt.tight_layout()
        # plt.savefig(save_path)
        # plt.show()


# === Objective Function ===
def objective(trial, csv_path='data/DATA_ABS_&_PP_Binary.csv'):
    global best_auc_global
    global best_model_global
    global batch_size
    global dropout

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, y = load_dataset(csv_path)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    alpha = trial.suggest_float("alpha", 0.1, 0.9)
    gamma = trial.suggest_float("gamma", 0.5, 5.0)
    # batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    n_layers = trial.suggest_int("n_layers", 1, 8)
    # dropout = trial.suggest_float("dropout", 0.0, 0.4)
    layers_dim = [
        trial.suggest_int("size_layer{}".format(i), 4, 128, log=True) for i in range(n_layers)  # TODO Here I can start from 4 or mmaybe from X.shape[1] ??
    ]

    auc_scores = []
    best_auc = 0
    best_model = None

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
            torch.save(best_model_global.state_dict(), "models/best_model_AUC_optuna.pt")

    return np.mean(auc_scores)


# === Retrain Final Model ===
def train_and_save_best_model(params_tpe, params_rs, epochs=100, csv_path='data/DATA_ABS_&_PP_Binary.csv'):
    global best_model_global
    global batch_size
    global dropout

    # batch_size = params["batch_size"]
    # dropout = params["dropout"]

    print(f"\nTraining the best model TPE and RS...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, y = load_dataset(csv_path)
    print(f"The Data has {X.shape[0]} samples and {X.shape[1]} features.")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    batch_size = 32
    dropout = 0.0
    # TPE
    model_tp = BinaryClassifier(input_size=X.shape[1], layers_dim=[params_tpe["size_layer{}".format(i)] for i in range(params_tpe["n_layers"])], dropout=dropout).to(device)
    criterion_tp = BinaryFocalLoss(alpha=params_tpe["alpha"], gamma=params_tpe["gamma"])
    optimizer_tp = torch.optim.Adam(model_tp.parameters(), lr=params_tpe["lr"])
    #RS
    model_rs = BinaryClassifier(input_size=X.shape[1], layers_dim=[params_tpe["size_layer{}".format(i)] for i in range(params_tpe["n_layers"])], dropout=dropout).to(device)
    criterion_rs = BinaryFocalLoss(alpha=params_tpe["alpha"], gamma=params_tpe["gamma"])
    optimizer_rs = torch.optim.Adam(model_rs.parameters(), lr=params_tpe["lr"])

    auc_scores_tp = []
    best_auc_tp = 0
    best_model_tp = None

    auc_scores_rs = []
    best_auc_rs = 0
    best_model_rs = None

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                 torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
        val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                               torch.tensor(y_val, dtype=torch.float32).unsqueeze(1))

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        # Train TP and RS
        model_tp = train_one_fold(model_tp, train_loader, val_loader, device, criterion_tp, optimizer_tp)
        model_rs = train_one_fold(model_rs, train_loader, val_loader, device, criterion_rs, optimizer_rs)

        # Evaluate TP
        auc_score_tp = evaluate_model(model_tp, val_loader, device, 'auc')
        auc_scores_tp.append(auc_score_tp)
        if auc_score_tp > best_auc_tp:
            best_auc_tp = auc_score_tp
            best_model_tp = model_tp
            X_test_tp, y_test_tp = X_val, y_val  # Save the test set for final evaluation
        # Evaluate RS
        auc_score_rs = evaluate_model(model_rs, val_loader, device, 'auc')
        auc_scores_rs.append(auc_score_rs)
        if auc_score_rs > best_auc_rs:
            best_auc_rs = auc_score_rs
            best_model_rs = model_rs
            X_test_rs, y_test_rs = X_val, y_val  # Save the test set for final evaluation
    
    print(f"TP: Best AUC Score across folds: {best_auc_tp:.4f} and mean AUC Score: {np.mean(auc_scores_tp):.4f}, after {len(auc_scores_tp)} folds.")
    print(f"RS: Best AUC Score across folds: {best_auc_rs:.4f} and mean AUC Score: {np.mean(auc_scores_rs):.4f}, after {len(auc_scores_rs)} folds.")

    torch.save(best_model_tp.state_dict(), "models/best_model_AUC_TP.pt")
    torch.save(best_model_rs.state_dict(), "models/best_model_AUC_RS.pt")

    # Model evaluation
    print(f"\nRe-trained models evaluation")
    evaluate_and_plot_results(best_model_tp, X_test_tp, y_test_tp, best_model_rs, X_test_rs, y_test_rs, device=device, save_path="images/test_results_AUC.png", roc_curve_path="images/auc_opt_roc_curve.png")


# === Run Optuna Optimization ===
def run_optimization(sampler, pruner, csv_path='data/DATA_ABS_&_PP_Binary.csv'):
    n_trials = 20
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(lambda trial: objective(trial, csv_path=csv_path), n_trials=n_trials, timeout=3600)

    # Best trial
    print("\nBest trial:")
    trial = study.best_trial
    print(f"  AUC Score: {trial.value:.4f}")
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
    
    # Run HPO otpimization with TPE sampler and HyperbandPruner
    sampler = optuna.samplers.TPESampler(seed=1) #(n_startup_trials=10, seed=31) # Here tried to add some startup trials
    pruner = optuna.pruners.HyperbandPruner(min_resource=1, max_resource=80, reduction_factor=3)
    best_trial_tpe = run_optimization(sampler, pruner, csv_path)

    # Run HPO otpimization with TPE sampler and HyperbandPruner
    sampler = optuna.samplers.RandomSampler(seed=1)  # Use RandomSampler for simplicity
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5, n_startup_trials=10)
    best_trial_rs = run_optimization(sampler, pruner, csv_path)

    # Retrain the best models
    train_and_save_best_model(params_tpe=best_trial_tpe.params, params_rs=best_trial_rs.params, epochs=200, csv_path=csv_path)
    
    # Print total time taken
    end_time = time.time()
    print(f"\nTotal time taken: {end_time - start_time:.2f} seconds")






###### OLD VERSION OF THE CODE, NOT USED ANYMORE, BUT KEPT FOR REFERENCE
# # === Retrain Final Model ===
# def train_and_save_best_model(params_tpe, params_rs, epochs=100, csv_path='data/DATA_ABS_&_PP_Binary.csv', save_path="models/best_model_AUC.pt"):
#     global best_model_global
#     global batch_size
#     global dropout

#     # batch_size = params["batch_size"]
#     # dropout = params["dropout"]

#     print(f"\nTraining the best model...")

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     X, y = load_dataset(csv_path)
#     print(f"The Data has {X.shape[0]} samples and {X.shape[1]} features.")

#     skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#     # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

#     # train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
#     #                          torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
#     # val_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
#     #                            torch.tensor(y_test, dtype=torch.float32).unsqueeze(1))
#     # train_loader = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=True)
#     # val_loader = DataLoader(val_ds, batch_size=params["batch_size"])

#     batch_size = 32
#     dropout = 0.0
#     model = BinaryClassifier(input_size=X.shape[1], layers_dim=[params_tpe["size_layer{}".format(i)] for i in range(params_tpe["n_layers"])], dropout=dropout).to(device)
#     criterion = BinaryFocalLoss(alpha=params_tpe["alpha"], gamma=params_tpe["gamma"])
#     optimizer = torch.optim.Adam(model.parameters(), lr=params_tpe["lr"])

#     auc_scores = []
#     best_auc = 0
#     best_model = None

#     for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
#         X_train, X_val = X[train_idx], X[val_idx]
#         y_train, y_val = y[train_idx], y[val_idx]

#         train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
#                                  torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
#         val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
#                                torch.tensor(y_val, dtype=torch.float32).unsqueeze(1))

#         train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
#         val_loader = DataLoader(val_ds, batch_size=batch_size)

#         # Train
#         model = train_one_fold(model, train_loader, val_loader, device, criterion, optimizer)

#         # Evaluate
#         auc_score = evaluate_model(model, val_loader, device, 'auc')
#         auc_scores.append(auc_score)

#         if auc_score > best_auc:
#             best_auc = auc_score
#             best_model = model
#             X_test, y_test = X_val, y_val  # Save the test set for final evaluation
    
#     print(f"Best AUC Score across folds: {best_auc:.4f} and mean AUC Score: {np.mean(auc_scores):.4f}, after {len(auc_scores)} folds.")
    
#     # model = train_one_fold(model, train_loader, val_loader, device, criterion, optimizer)

#     # early_stopping = EarlyStopping(patience=25)
#     # for epoch in range(epochs):
#     #     model.train()
#     #     for xb, yb in train_loader:
#     #         xb, yb = xb.to(device), yb.to(device)
#     #         optimizer.zero_grad()
#     #         loss = criterion(model(xb), yb)
#     #         loss.backward()
#     #         optimizer.step()

#     #     model.eval()
#     #     with torch.no_grad():
#     #         outputs = model(torch.tensor(X_test, dtype=torch.float32).to(device))
#     #         prods = (torch.sigmoid(outputs)).float().cpu().numpy()
#     #     auc_score = roc_auc_score(y_test, prods)
#     #     # loss = criterion(outputs, torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device))
#     #     early_stopping(auc_score, model)
#     #     if early_stopping.early_stop:
#     #         print(f"Early stopping at epoch {epoch + 1} with AUC Score: {auc_score:.4f}")
#     #         break
#     # model.load_state_dict(early_stopping.best_model_state)

#     torch.save(best_model.state_dict(), save_path)

#     # Model evaluation
#     print(f"\nRe-trained model evaluation")
#     evaluate_and_plot_results(best_model, X_test, y_test, device=device, save_path="images/test_results_AUC.png", roc_curve_path="images/auc_opt_roc_curve.png")

#     # ---- This makes no sense because this model has probably been trianed on some of the data that now is used for evaluation ----
#     # Load and evaluate best model trained with Optuna
#     # print(f"\nModel trained by otpuna evaluation")
#     # print(f"Optuna model architecture: {best_model_global}")
#     # evaluate_and_plot_results(best_model_global, X_test, y_test, device=device, save_path="images/test_results_AUC_optuna.png", roc_curve_path="images/auc_opt_roc_curve_optuna.png")
