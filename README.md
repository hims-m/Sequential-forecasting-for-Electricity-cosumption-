# Sequential Data Forecasting & Understanding Engine

Learning project: build intuition for RNNs, LSTMs, and GRUs via a real forecasting pipeline.

## Structure

```
sequential-forecasting-project/
├── data/
│   └── household_power_consumption_days.csv   # UCI household power data, daily-aggregated
├── rnn_pipeline.py                             # Vanilla RNN baseline
├── lstm_pipeline.py                            # LSTM version (same pipeline, swapped cell)
└── README.md
```

## Dataset

UCI "Individual Household Electric Power Consumption" — real electricity meter readings
from a house in Sceaux, France, Dec 2006–Nov 2010. This is the daily-aggregated version
(~1,442 rows, 8 columns, no missing values).

## Running

From this folder, using your virtual environment's Python:

```powershell
.\.venv\Scripts\python.exe rnn_pipeline.py
.\.venv\Scripts\python.exe lstm_pipeline.py
```

Each script trains, prints a Test MAE (in real watt-units), and shows a plot of
actual vs. predicted values on the held-out test set.

## Results so far

| Model | Test MAE |
|---|---|
| Vanilla RNN | _fill in_ |
| LSTM | 236.45 |

## Next steps

1. Fill in the RNN's Test MAE above and compare against the LSTM.
2. Add `gru_pipeline.py` — same pattern, swap `nn.LSTM` for `nn.GRU`.
3. Multi-step forecasting (predict several days ahead, not just one).
4. Interpretability layer — hidden state visualization, error analysis by season.
5. Try a second, structurally different dataset.
