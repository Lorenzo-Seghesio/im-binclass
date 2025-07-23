import pandas as pd
import torch, torch.cuda
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, auc, roc_curve, RocCurveDisplay
import time
import argparse
import matplotlib.pyplot as plt


# === Model ===
class SimpleBinaryClassifier(nn.Module):
    def __init__(self, input_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Linear(input_dim * 2, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, 1),  # logits output
        )

    def forward(self, x):
        return self.net(x)
    

# === Binary Focal Loss Callback ===
class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.2, gamma=2.0, reduction='mean'):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.clamp(min=1e-7, max=1 - 1e-7)  # numerical stability

        loss_pos = -self.alpha * (1 - probs) ** self.gamma * targets * torch.log(probs)
        loss_neg = -(1 - self.alpha) * probs ** self.gamma * (1 - targets) * torch.log(1 - probs)
        loss = loss_pos + loss_neg

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
        

# === EarlyStopping Callback ===
class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, patience=3, delta=0.0, path='checkpoint.pt', verbose=True):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self._save_checkpoint(model)
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self._save_checkpoint(model)
            self.counter = 0

    def _save_checkpoint(self, model):
        """Saves model when validation loss decreases."""
        torch.save(model.state_dict(), self.path)
        if self.verbose:
            print(f" Validation loss improved. Model saved to {self.path}")


# === Data Loading ===
def load_data(csv_path, data_augmentation=False):
    df = pd.read_csv(csv_path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    X_train, X_val, Y_train, Y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Data augmentation - duplication of the faulty parts
    if data_augmentation == True:
        faulty_mask = Y_train == 0
        X_faulty = X_train[faulty_mask]
        Y_faulty = Y_train[faulty_mask]
        X_train_aug = np.concatenate([X_train, X_faulty], axis=0)
        Y_train_aug = np.concatenate([Y_train, Y_faulty], axis=0)
    else:
        X_train_aug = X_train
        Y_train_aug = Y_train


    X_train_tensor = torch.tensor(X_train_aug, dtype=torch.float32)
    y_train_tensor = torch.tensor(Y_train_aug, dtype=torch.float32).unsqueeze(1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(Y_val, dtype=torch.float32).unsqueeze(1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    # Data for check
    len_train_aug = len(Y_train_aug)
    len_train = len(Y_train)
    len_val = len(Y_val)
    healthy_percent_training_aug = (Y_train_aug == 1).sum() / len_train_aug
    healthy_percent_training = (Y_train == 1).sum() / len_train
    healthy_percent_validation = (Y_val == 1).sum() / len_val

    return train_dataset, val_dataset, len_train, len_train_aug, len_val, healthy_percent_training, healthy_percent_training_aug, healthy_percent_validation


# === Training ===
def train(model, train_loader, val_loader, device, epochs=20, learning_rate=0.001, patience=5, pos_weight=None, alpha=None, gamma=None):
    model.to(device)

    if pos_weight is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif alpha is not None and gamma is not None:
        criterion = BinaryFocalLoss(alpha=alpha, gamma=gamma)  # Using Binary Focal Loss
    else:
        criterion = nn.BCEWithLogitsLoss()  # Default loss function
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    early_stopper = EarlyStopping(patience=patience, delta=0.0, path='models/best_model_FixedArchitecture.pt')

    for epoch in range(1, epochs + 1):
        print(f"\nEPOCH {epoch}")
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validation
        val_loss = 0.0
        correct = 0
        total = 0
        j = 1
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                # Validation Loss
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                # Validation accuracy
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                print(f" cycle {j}\n  output: {probs.T}\n  predicted: {preds.T}\n  correct: {labels.T}")
                j += 1
        val_loss /= len(val_loader)
        val_acc = correct/total

        print(f" Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.2%}")
        early_stopper(val_loss, model)

        if early_stopper.early_stop:
            print("\nEarly stopping triggered. Training halted.")
            break


# === Metric Computation ===
def metric_computation(model, val_loader, device):
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).float()
            preds = (probs > 0.5).float()
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # # Convert to NumPy arrays
    # preds_np = np.array(all_preds)
    # labels_np = np.array(all_labels)
    #
    # # Calculate precision, recall, F1
    # tp = np.sum((preds_np == 1) & (labels_np == 1))
    # fp = np.sum((preds_np == 1) & (labels_np == 0))
    # fn = np.sum((preds_np == 0) & (labels_np == 1))
    #
    # precision = tp / (tp + fp + 1e-8)
    # recall = tp / (tp + fn + 1e-8)
    # f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    f1 = f1_score(all_labels, all_preds)
    accuracy = accuracy_score(all_labels, all_preds)
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    return f1, accuracy, fpr, tpr, roc_auc


# === Main ===
def main(args):
    # Time tracking
    start_time = time.time()

    # Check for CUDA availability
    if torch.cuda.is_available():
        print("CUDA is available")
        print("Number of GPUs:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print("Current used GPU:", torch.cuda.current_device())
        device = torch.device("cuda")
    else:
        print("CUDA is not available, using CPU instead.")
        device = torch.device("cpu")

    # --- Immproving techniques for training ---

    # 1. Data augmentation: Not implemented here, but can be done by duplicating faulty parts or adding noise.
    #    Choose True if you want to use data augmentation.
    data_augmentation = False

    # 2. Class weight for imbalanced data: Using pos_weight in BCEWithLogitsLoss or Binary Focal Loss.
    #    If you want to use pos_weight, set it to a tensor with the appropriate value.
    pos_weight = None # torch.tensor([0.4], device=device)  # Example weight, adjust based on your data
    # pos_weight = torch.tensor([(1 - healthy_percent_training_aug) / healthy_percent_training_aug], device=device)

    # 3. Focal loss: Implemented in the BinaryFocalLoss class.
    #    If you want to use focal loss, set pos_weight to None and specify alpha and gamma.
    alpha = 0.4  # Focal loss alpha
    gamma = 1.2  # Focal loss gamma

    # Other hyperparameters that can be adjusted in the train function.
    epochs = 20  # Number of training epochs
    patience = 5  # Early stopping patience
    batch_size = 32  # Batch size for training and validation
    learning_rate = 0.001  # Learning rate for the optimizer

    # Load data
    if args.full_data:
        print("\nUsing the full dataset.")
        csv_path = 'data/IM_Data_Full.csv'
    else:
        print("\nUsing the dataset without measurments.")
        csv_path = 'data/IM_Data.csv'
    train_dataset, val_dataset, len_train, len_train_aug, len_val, healthy_percent_training, healthy_percent_training_aug, healthy_percent_validation = load_data(csv_path, data_augmentation)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize model, loss function, optimizer, and early stopping
    model = SimpleBinaryClassifier(input_dim=train_dataset.tensors[0].shape[1])

    # Train the model
    train(model, train_loader, val_loader, device, epochs=epochs, learning_rate=learning_rate, patience=patience, pos_weight=pos_weight, alpha=alpha, gamma=gamma)
    print('\nFinished training. Best model saved to best_model.pt')

    # Print the used techniques
    if pos_weight is not None:
        print(f"\nUsing BCEWithLogitsLoss with pos_weight={pos_weight.item()}")
    elif alpha is not None and gamma is not None:
        print(f"\nUsing Binary Focal Loss with alpha={alpha}, gamma={gamma}")
    else:
        print("\nUsing BCEWithLogitsLoss without pos_weight")
    if data_augmentation:
        print("\nData augmentation is enabled: duplicating faulty parts in training data.")

    # Metric computation
    f1, accuracy, fpr, tpr, roc_auc  = metric_computation(model, val_loader, device)
    print(f"\nF1 score: {f1:.2%}, Accuracy: {accuracy:.2%}, ROC AUC: {roc_auc:.2%}")
    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, name='MLP')
    display.plot()
    plt.title('ROC Curve')
    plt.savefig('images/roc_curve.png')
    plt.show()

    # Check of the data
    print(f"\nHealthy percentage in original training data: {healthy_percent_training:.2%}\n Total number of training data: {len_train}")
    print(f"Healthy percentage in augmented training data (the used one): {healthy_percent_training_aug:.2%}\n Total number of training augmentd data: {len_train_aug}")
    print(f"Healthy percentage in validation data: {healthy_percent_validation:.2%}\n Total number of validation data: {len_val}")

    edn_time = time.time()
    print(f"\nTotal time taken: {edn_time - start_time:.2f} seconds")


if __name__ == '__main__':

    # Argument parser
    parser = argparse.ArgumentParser(description="Train a binary classification model.")
    parser.add_argument('--full_data', action='store_true', help="Decice whether to use the full dataset or the one without measurments with NaN values.")
    args = parser.parse_args()

    # Run the main function with parsed arguments
    main(args)
