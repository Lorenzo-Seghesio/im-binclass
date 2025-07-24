import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# === Load CSV ===
df = pd.read_csv("data/IM_Data.csv")

# === Split features and labels ===
X = df.iloc[:, :18].values  # first 18 columns
y = df["ground_truth"].values  # binary label column

# === Train/validation split ===
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# === Normalize features ===
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# === Convert to tensors ===
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

# === Create DataLoaders ===
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

# === Define the Neural Network ===
class SimpleBinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(18, 36),
            nn.ReLU(),
            nn.Linear(36, 18),
            nn.ReLU(),
            nn.Linear(18, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

model = SimpleBinaryClassifier()

# === Loss and Optimizer and Epochs ===
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 10

# === Training loop ===
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}")
    model.train()
    train_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # === Validation ===
    model.eval()
    correct = 0
    total = 0
    print(f"Validation:")
    with torch.no_grad():
        j = 1
        for inputs, labels in val_loader:
            outputs = model(inputs)
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            print(f"cycle {j}\noutput: {outputs.T}\npredicted: {predicted.T}\ncorrect: {labels.T}")
            j += 1

    val_accuracy = correct / total
    print(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f} | Val Accuracy: {val_accuracy:.2%}\n")

## proof of accuracy
len_train = len(y_train)
len_val = len(y_val)
healthy_percent_training = ( y_train == 1 ).sum() /len_train
healthy_percent_validation = ( y_val == 1 ).sum() / len_val

print(f"\n Healthy percentage in training data: {healthy_percent_training:.2%}\n Total number of training data: {len_train}")
print(f"Healthy percentage in validation data: {healthy_percent_validation:.2%}\n Total number of validation data: {len_val}")

print("Training complete.")