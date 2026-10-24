import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ---------- 1. Load real data ----------
df = pd.read_csv("data/household_power_consumption_days.csv", parse_dates=["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)

series = df["Global_active_power"].values.astype(np.float32)

# ---------- 2. Train/test split FIRST, then normalize ----------
split = int(len(series) * 0.8)
train_raw, test_raw = series[:split], series[split:]

mean, std = train_raw.mean(), train_raw.std()
train_norm = (train_raw - mean) / std
test_norm = (test_raw - mean) / std

# ---------- 3. Windowing ----------
def create_windows(series, window_size):
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i + window_size])
        y.append(series[i + window_size])
    return np.array(X), np.array(y)

WINDOW = 30
X_train, y_train = create_windows(train_norm, WINDOW)
X_test, y_test = create_windows(test_norm, WINDOW)

X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
X_test_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1)

# ---------- 4. Model: GRU ----------
# nn.GRU returns (output, hidden) — same shape as vanilla RNN, no separate
# cell state like LSTM. It gets similar long-range benefits to LSTM using
# fewer gates (2 vs LSTM's 3), so it's cheaper to train while often performing
# comparably. This comparison is the whole point of this script.
class SimpleGRU(nn.Module):
    def __init__(self, input_size=1, hidden_size=32):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, hidden = self.gru(x)
        last_step = out[:, -1, :]
        return self.fc(last_step)

model = SimpleGRU()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# ---------- 5. Training loop ----------
EPOCHS = 150
for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    preds = model(X_train_t)
    loss = criterion(preds, y_train_t)
    loss.backward()
    optimizer.step()

    if epoch % 25 == 0:
        print(f"Epoch {epoch:3d} | Train Loss: {loss.item():.4f}")

# ---------- 6. Evaluate ----------
model.eval()
with torch.no_grad():
    test_preds_norm = model(X_test_t)

test_preds = test_preds_norm.numpy().flatten() * std + mean
test_actual = y_test * std + mean

mae = np.mean(np.abs(test_preds - test_actual))
print(f"\nTest MAE (real units): {mae:.2f}")

plt.plot(test_actual, label="actual")
plt.plot(test_preds, label="predicted")
plt.legend()
plt.title("GRU: 1-day-ahead power forecast")
plt.xlabel("Day (test set)")
plt.ylabel("Global active power")
plt.show()