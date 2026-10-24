import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ---------- 1. Load real data ----------
df = pd.read_csv("data/household_power_consumption_days.csv", parse_dates=["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)

series = df["Global_active_power"].values.astype(np.float32)
dates = df["datetime"].values  # keep dates around for the interpretability step

# ---------- 2. Train/test split FIRST, then normalize ----------
split = int(len(series) * 0.8)
train_raw, test_raw = series[:split], series[split:]
test_dates = dates[split:]

mean, std = train_raw.mean(), train_raw.std()
train_norm = (train_raw - mean) / std
test_norm = (test_raw - mean) / std

# ---------- 3. Windowing (single-step, same as before) ----------
WINDOW = 30

def create_windows(series, window_size):
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i + window_size])
        y.append(series[i + window_size])
    return np.array(X), np.array(y)

X_train, y_train = create_windows(train_norm, WINDOW)
X_test, y_test = create_windows(test_norm, WINDOW)
# the date each test prediction corresponds to (the day being predicted, not the window start)
test_target_dates = pd.to_datetime(test_dates[WINDOW:])

X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
X_test_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1)

# ---------- 4. Model — GRU, but forward() optionally returns hidden states per step ----------
class InspectableGRU(nn.Module):
    def __init__(self, input_size=1, hidden_size=32):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, return_all_hidden=False):
        out, hidden = self.gru(x)  # out: (batch, seq_len, hidden_size) — hidden state at EVERY step
        last_step = out[:, -1, :]
        pred = self.fc(last_step)
        if return_all_hidden:
            return pred, out  # expose the full per-timestep hidden states
        return pred

model = InspectableGRU()
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

# ---------- 6. Standard evaluation ----------
model.eval()
with torch.no_grad():
    test_preds_norm = model(X_test_t)

test_preds = test_preds_norm.numpy().flatten() * std + mean
test_actual = y_test * std + mean
abs_errors = np.abs(test_preds - test_actual)
print(f"\nOverall Test MAE: {abs_errors.mean():.2f}")

# ---------- 7. INTERPRETABILITY: error by month ----------
# Does the model do worse in winter (higher heating usage, more volatile
# consumption) than in summer? Group the test errors by calendar month.
error_df = pd.DataFrame({
    "date": test_target_dates,
    "abs_error": abs_errors,
})
error_df["month"] = error_df["date"].dt.month_name()

monthly_mae = error_df.groupby("month")["abs_error"].mean().sort_values(ascending=False)
print("\nMean absolute error by month (worst to best):")
print(monthly_mae)

plt.figure()
monthly_mae.plot(kind="bar")
plt.title("Forecast error by month")
plt.ylabel("Mean Absolute Error")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ---------- 8. INTERPRETABILITY: hidden state behavior within one window ----------
# Take ONE test window and look at how the hidden state's magnitude evolves
# across the 30 days fed into it. A rising norm suggests the model is
# accumulating information; a flat/decaying norm suggests it's "forgetting"
# older days rapidly — visual evidence for how much history actually matters.
with torch.no_grad():
    sample_window = X_test_t[0:1]  # first test window, shape (1, WINDOW, 1)
    _, all_hidden = model(sample_window, return_all_hidden=True)

hidden_norms = all_hidden[0].norm(dim=1).numpy()  # magnitude of hidden state at each of the 30 steps

plt.figure()
plt.plot(range(1, WINDOW + 1), hidden_norms, marker="o")
plt.title("Hidden state magnitude across the 30-day input window")
plt.xlabel("Day within window")
plt.ylabel("Hidden state norm")
plt.show()