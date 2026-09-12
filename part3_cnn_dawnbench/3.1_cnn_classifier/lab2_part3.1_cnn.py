import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# ---- Load data (same as Part 2) ----
print("Fetching LFW dataset...")
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

X = lfw_people.images   # shape [n_samples, h, w]
Y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print("X_min:", X.min(), "X_max:", X.max())

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

# ---- Add channel dimension: [N, H, W] -> [N, 1, H, W] ----
X_train = X_train[:, np.newaxis, :, :]
X_test = X_test[:, np.newaxis, :, :]
print("X_train shape:", X_train.shape)

# ---- To tensors, onto device ----
X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_t = torch.tensor(y_test, dtype=torch.long).to(device)

h, w = X_train.shape[2], X_train.shape[3]

# ---- CNN: two 3x3 conv layers (32 filters each) + dense classifier ----
class SimpleCNN(nn.Module):
    def __init__(self, h, w, n_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        # after two 2x2 pools, spatial dims shrink by factor 4
        flat_dim = 32 * (h // 4) * (w // 4)
        self.fc1 = nn.Linear(flat_dim, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.flatten(1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleCNN(h, w, n_classes).to(device)
print(model)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# ---- Train ----
epochs = 30
batch_size = 32
n_train = X_train_t.shape[0]

loss_history = []

for epoch in range(epochs):
    model.train()
    perm = torch.randperm(n_train)
    total_loss = 0.0
    for i in range(0, n_train, batch_size):
        idx = perm[i:i+batch_size]
        xb, yb = X_train_t[idx], y_train_t[idx]

        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)

    avg_loss = total_loss / n_train
    loss_history.append(avg_loss)
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{epochs}  loss: {avg_loss:.4f}")

# ---- Plot training loss ----
plt.figure()
plt.plot(range(1, epochs + 1), loss_history)
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("CNN Training Loss vs Epoch")
plt.grid(True)
plt.savefig("cnn_loss_curve.png")
print("Saved cnn_loss_curve.png")

# ---- Evaluate ----
model.eval()
with torch.no_grad():
    preds = model(X_test_t).argmax(dim=1)
    accuracy = (preds == y_test_t).float().mean().item()

print(f"\nTest Accuracy: {accuracy:.4f}")
print("Done.")