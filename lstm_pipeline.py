import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ---------- 1. Load real data ----------
df = pd.read_csv("data/household_power_consumption_days.csv", parse_dates=["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)

# Start with ONE variable, same as the sine-wave version. Once this works,
# swap this line for multiple columns to go multivariate.
series = df["Global_active_power"].values.astype(np.float32)

# ---------- 2. Train/test split FIRST, then normalize ----------
# With real data, normalization stats must come from the TRAIN set only.
# If you scale before splitting, test-set info leaks into training
# (the model implicitly "knows" the future's mean/std). This is the
# real-data equivalent of the "don't shuffle time series" rule.
split = int(len(series) * 0.8)
train_raw, test_raw = series[:split], series[split:]

mean, std = train_raw.mean(), train_raw.std()
train_norm = (train_raw - mean) / std
test_norm = (test_raw - mean) / std  # test uses train's mean/std, never its own

# ---------- 3. Windowing (identical logic to before) ----------
def create_windows(series, window_size):
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i + window_size])
        y.append(series[i + window_size])
    return np.array(X), np.array(y)

WINDOW = 30  # 30 days of history to predict day 31
X_train, y_train = create_windows(train_norm, WINDOW)
X_test, y_test = create_windows(test_norm, WINDOW)

X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
X_test_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1)

# ---------- 4. Model: LSTM ----------
# Same structure as SimpleRNN, but nn.LSTM returns (output, (hidden, cell))
# instead of (output, hidden) — the cell state is the extra "memory lane"
# that lets gradients flow over longer sequences without vanishing.
class SimpleLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, (hidden, cell) = self.lstm(x)
        last_step = out[:, -1, :]
        return self.fc(last_step)

model = SimpleLSTM()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# ---------- 5. Training loop (unchanged) ----------
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

# un-normalize back to real watt-scale for a meaningful plot/metric
test_preds = test_preds_norm.numpy().flatten() * std + mean
test_actual = y_test * std + mean

mae = np.mean(np.abs(test_preds - test_actual))
print(f"\nTest MAE (real units): {mae:.2f}")

plt.plot(test_actual, label="actual")
plt.plot(test_preds, label="predicted")
plt.legend()
plt.title("LSTM: 1-day-ahead power forecast")
plt.xlabel("Day (test set)")
plt.ylabel("Global active power")
plt.show()
