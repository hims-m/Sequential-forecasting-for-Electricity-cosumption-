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

# ---------- 3. Windowing — CHANGED for multi-step ----------
# Before: y was a single next-day value. Now: y is a whole HORIZON-length
# vector of future days. This is the key structural change for multi-step —
# everything else (model shape aside) stays the same.
WINDOW = 30
HORIZON = 7  # predict the next 7 days at once, instead of just 1

def create_windows_multistep(series, window_size, horizon):
    X, y = [], []
    for i in range(len(series) - window_size - horizon + 1):
        X.append(series[i:i + window_size])
        y.append(series[i + window_size : i + window_size + horizon])
    return np.array(X), np.array(y)

X_train, y_train = create_windows_multistep(train_norm, WINDOW, HORIZON)
X_test, y_test = create_windows_multistep(test_norm, WINDOW, HORIZON)

X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
y_train_t = torch.tensor(y_train, dtype=torch.float32)  # shape: (batch, HORIZON)
X_test_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
y_test_t = torch.tensor(y_test, dtype=torch.float32)

# ---------- 4. Model — output layer now produces HORIZON values ----------
class MultiStepGRU(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, horizon=HORIZON):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, horizon)  # <- only real architecture change

    def forward(self, x):
        out, hidden = self.gru(x)
        last_step = out[:, -1, :]
        return self.fc(last_step)  # shape: (batch, HORIZON)

model = MultiStepGRU()
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

test_preds = test_preds_norm.numpy() * std + mean
test_actual = y_test * std + mean

# MAE per horizon step — this is the interesting part: error should GROW
# the further out you predict, since day-7 is much harder than day-1.
mae_per_step = np.mean(np.abs(test_preds - test_actual), axis=0)
for day, mae in enumerate(mae_per_step, start=1):
    print(f"Day +{day} MAE: {mae:.2f}")
print(f"\nOverall Test MAE: {mae_per_step.mean():.2f}")

# Plot one example window from the test set to see the shape of a forecast
sample_idx = 0
plt.plot(range(1, HORIZON + 1), test_actual[sample_idx], label="actual", marker="o")
plt.plot(range(1, HORIZON + 1), test_preds[sample_idx], label="predicted", marker="o")
plt.legend()
plt.title("GRU: 7-day-ahead forecast (one example window)")
plt.xlabel("Day ahead")
plt.ylabel("Global active power")
plt.show()