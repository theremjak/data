"""
bitcoin_price_predictor.py

Compares two models for Bitcoin price prediction:
  1. Power Law  — log-log regression on days since genesis (long-term trend)
  2. Conv-LSTM  — Multivariate Conv1D+LSTM with:
       - Power law fair-value deviation as a feature
       - Rolling expanding-window retrain (every 30 days)
       - Predicts deviation from power law, then adds trend back

Both models are trained on the first 80% of BTC history and evaluated on the
same held-out 20% test period. Results are charted and exported to CSV.
"""

import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import date, timedelta
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

tf.random.set_seed(42)
np.random.seed(42)

GENESIS     = date(2009, 1, 3)
WINDOW      = 30        # lookback window (days)
ROLL_STEP   = 30        # retrain every N days
TEST_FRAC   = 0.30
EPOCHS      = 80
BATCH_SIZE  = 64

S_SLOPE     = 5.84      # Santostasi canonical model
S_INTERCEPT = -17.01
SUP_OFF     = -0.5
RES_OFF     = +0.4

# Historical + estimated future halvings (future dates are approximate)
HALVING_DATES = [
    date(2012, 11, 28),   # Halving 1
    date(2016,  7, 16),   # Halving 2
    date(2020,  5, 11),   # Halving 3
    date(2024,  4, 20),   # Halving 4
    date(2028,  4, 15),   # Halving 5 (estimated)
    date(2032,  4, 15),   # Halving 6 (estimated)
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def days_since_genesis(d):
    if isinstance(d, pd.Timestamp):
        d = d.date()
    return (d - GENESIS).days


def power_law_price(days, slope, intercept):
    return 10 ** (slope * np.log10(np.maximum(days, 1)) + intercept)

def pl_log_deviation(log_price, days, slope, intercept):
    """Deviation of log10(price) from the power law — the 'cyclical' component."""
    return log_price - (slope * np.log10(np.maximum(days, 1)) + intercept)

def mape(actual, pred):
    return np.mean(np.abs((actual - pred) / actual)) * 100

def rmse(actual, pred):
    return np.sqrt(mean_squared_error(actual, pred))

def directional_accuracy(actual, pred):
    return np.mean((np.diff(actual) > 0) == (np.diff(pred) > 0)) * 100

def rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    return 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

def build_model(window, n_feat):
    m = Sequential([
        Conv1D(64, kernel_size=3, activation="relu", padding="same",
               input_shape=(window, n_feat)),
        BatchNormalization(),
        Conv1D(32, kernel_size=3, activation="relu", padding="same"),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="huber")
    return m

def make_windows(X, y, window):
    Xs, ys = [], []
    for i in range(window, len(X)):
        Xs.append(X[i - window:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


# ---------------------------------------------------------------------------
# 1. Fetch & engineer features
# ---------------------------------------------------------------------------
print("Fetching BTC-USD...")
raw = yf.download("BTC-USD", start="2010-07-17", auto_adjust=True, progress=False)
raw.columns = raw.columns.get_level_values(0)
raw = raw[["Close", "Volume"]].dropna()
raw.index = pd.to_datetime(raw.index)

df = pd.DataFrame(index=raw.index)
df["close"]      = raw["Close"].values.flatten()
df["log_close"]  = np.log10(df["close"])
df["log_return"] = df["log_close"].diff()
df["log_volume"] = np.log1p(raw["Volume"].values.flatten())
df["ma7"]        = df["close"].rolling(7).mean()
df["ma30"]       = df["close"].rolling(30).mean()
df["ma7_ratio"]  = df["close"] / df["ma7"]
df["ma30_ratio"] = df["close"] / df["ma30"]
df["rsi14"]      = rsi(df["close"], 14)
df["days"]       = [days_since_genesis(d) for d in df.index]

# --- power law fair-value deviation (Santostasi canonical) ---
df["pl_deviation"] = pl_log_deviation(
    df["log_close"], df["days"], S_SLOPE, S_INTERCEPT
)


df = df.dropna()

print(f"  {len(df):,} trading days  ({df.index[0].date()} → {df.index[-1].date()})")
print(f"  Latest close: ${df['close'].iloc[-1]:,.0f}")

# Features now include pl_deviation
FEATURES = ["log_return", "log_volume", "ma7_ratio", "ma30_ratio", "rsi14", "pl_deviation"]
n_feat   = len(FEATURES)

split    = int(len(df) * (1 - TEST_FRAC))
train_df = df.iloc[:split]
test_df  = df.iloc[split:]
print(f"  Train: {len(train_df):,} days  |  Test: {len(test_df):,} days")
print(f"  Test period: {test_df.index[0].date()} → {test_df.index[-1].date()}")


# ---------------------------------------------------------------------------
# 2. Power Law — use Santostasi canonical model (train R² for reference only)
# ---------------------------------------------------------------------------
print("\n[1/2] Fitting Power Law...")
pl_slope, pl_intercept = S_SLOPE, S_INTERCEPT

# Compute train R² against Santostasi line for reference
pl_r2_train = r2_score(
    np.log10(train_df["close"].values),
    np.log10(power_law_price(train_df["days"].values, pl_slope, pl_intercept))
)

# Align to same date window as LSTM (skip first WINDOW rows of test)
test_dates_aligned = test_df.index[WINDOW:]
test_actual_prices = test_df["close"].values[WINDOW:]
test_days_aligned  = test_df["days"].values[WINDOW:]

pl_pred_test = power_law_price(test_days_aligned, pl_slope, pl_intercept)

pl_mape = mape(test_actual_prices, pl_pred_test)
pl_rmse = rmse(test_actual_prices, pl_pred_test)
pl_r2   = r2_score(np.log10(test_actual_prices), np.log10(pl_pred_test))
pl_dir  = directional_accuracy(test_actual_prices, pl_pred_test)

print(f"  Slope={pl_slope:.4f}  Intercept={pl_intercept:.4f}  Train R²={pl_r2_train:.4f}")
print(f"  Test  MAPE={pl_mape:.1f}%  RMSE=${pl_rmse:,.0f}  Log-R²={pl_r2:.4f}  DirAcc={pl_dir:.1f}%")


# ---------------------------------------------------------------------------
# 3. Conv-LSTM with rolling retrain + pl_deviation target
#
# Strategy:
#   - TARGET: predict tomorrow's pl_deviation (cyclical component)
#   - PREDICTION: power_law_tomorrow + predicted_deviation → price
#   - Rolling expanding window: initial train = first 80%, then retrain
#     every ROLL_STEP days, adding new observations each time.
# ---------------------------------------------------------------------------
print(f"\n[2/2] Training Conv-LSTM (rolling retrain every {ROLL_STEP} days)...")

all_features = df[FEATURES].values
all_target   = df["pl_deviation"].values   # predict tomorrow's deviation
all_days     = df["days"].values
all_prices   = df["close"].values
all_dates    = df.index

# We predict pl_deviation at t+1 using features at t, t-1, ..., t-WINDOW+1
# y[i] = pl_deviation[i+1] so we can reconstruct price[i+1]
# Shift target by 1 to avoid lookahead
target_shifted = np.roll(all_target, -1)   # target_shifted[i] = pl_deviation[i+1]

lstm_pred_prices = np.full(len(test_df), np.nan)

# Rolling loop
n_test    = len(test_df)
n_steps   = (n_test - WINDOW) // ROLL_STEP + 1
retrain_count = 0

for step_i in range(n_steps):
    # Expanding window end index (in global df index)
    train_end  = split + step_i * ROLL_STEP
    pred_start = train_end          # first test index to predict
    pred_end   = min(pred_start + ROLL_STEP, split + n_test - 1)

    if pred_start >= split + n_test - 1:
        break

    # Need at least WINDOW rows for the first window
    if train_end - WINDOW < 0:
        continue

    # Scale using only training data up to train_end
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_tr_raw = scaler_X.fit_transform(all_features[:train_end])
    y_tr_raw = scaler_y.fit_transform(target_shifted[:train_end].reshape(-1, 1))

    X_tr, y_tr = make_windows(X_tr_raw, y_tr_raw, WINDOW)

    if len(X_tr) < BATCH_SIZE:
        continue

    # Build & train fresh model each roll
    model = build_model(WINDOW, n_feat)
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(patience=5, factor=0.5, verbose=0),
    ]
    model.fit(X_tr, y_tr,
              validation_split=0.1,
              epochs=EPOCHS,
              batch_size=BATCH_SIZE,
              callbacks=callbacks,
              verbose=0)
    retrain_count += 1

    # Predict the next ROLL_STEP days
    for pred_idx in range(pred_start, pred_end):
        if pred_idx - WINDOW < 0:
            continue
        window_X = scaler_X.transform(all_features[pred_idx - WINDOW:pred_idx])
        window_X = window_X.reshape(1, WINDOW, n_feat)
        pred_dev_scaled = model.predict(window_X, verbose=0)
        pred_dev = scaler_y.inverse_transform(pred_dev_scaled)[0, 0]

        # Reconstruct price: power_law(t) + predicted deviation
        pl_at_t = S_SLOPE * np.log10(all_days[pred_idx]) + S_INTERCEPT
        pred_log_price = pl_at_t + pred_dev
        lstm_pred_prices[pred_idx - split] = 10 ** pred_log_price

    progress = (step_i + 1) / n_steps * 100
    print(f"  [{progress:5.1f}%] Retrain #{retrain_count:2d}  "
          f"window={train_end:,} days  "
          f"predicting days {pred_start-split}–{pred_end-split} of test set",
          flush=True)

# Drop NaN edges (first WINDOW test rows have no prediction)
valid_mask         = ~np.isnan(lstm_pred_prices)
lstm_pred_prices_v = lstm_pred_prices[valid_mask]
test_actual_v      = test_df["close"].values[valid_mask]
test_dates_v       = test_df.index[valid_mask]
test_days_v        = test_df["days"].values[valid_mask]
pl_pred_v          = power_law_price(test_days_v, pl_slope, pl_intercept)

lstm_mape = mape(test_actual_v, lstm_pred_prices_v)
lstm_rmse = rmse(test_actual_v, lstm_pred_prices_v)
lstm_r2   = r2_score(np.log10(test_actual_v), np.log10(np.maximum(lstm_pred_prices_v, 1)))
lstm_dir  = directional_accuracy(test_actual_v, lstm_pred_prices_v)

print(f"\n  Retrains: {retrain_count}  |  Predictions: {valid_mask.sum():,} days")
print(f"  Test  MAPE={lstm_mape:.1f}%  RMSE=${lstm_rmse:,.0f}  Log-R²={lstm_r2:.4f}  DirAcc={lstm_dir:.1f}%")


# ---------------------------------------------------------------------------
# 4. Summary statistics
# ---------------------------------------------------------------------------
# Align power law to same valid mask for apples-to-apples
pl_mape_v = mape(test_actual_v, pl_pred_v)
pl_rmse_v = rmse(test_actual_v, pl_pred_v)
pl_r2_v   = r2_score(np.log10(test_actual_v), np.log10(pl_pred_v))
pl_dir_v  = directional_accuracy(test_actual_v, pl_pred_v)

print("\n" + "=" * 65)
print(f"{'Model Comparison — Test Period':^65}")
print("=" * 65)
print(f"  {'Metric':<28} {'Power Law':>15}  {'Conv-LSTM':>15}")
print("  " + "-" * 60)
print(f"  {'MAPE (%)':<28} {pl_mape_v:>14.1f}%  {lstm_mape:>14.1f}%")
print(f"  {'RMSE ($)':<28} ${pl_rmse_v:>13,.0f}  ${lstm_rmse:>13,.0f}")
print(f"  {'Log-space R²':<28} {pl_r2_v:>15.4f}  {lstm_r2:>15.4f}")
print(f"  {'Directional Accuracy (%)':<28} {pl_dir_v:>14.1f}%  {lstm_dir:>14.1f}%")

from collections import Counter
metrics = [
    ("MAPE",      pl_mape_v < lstm_mape),
    ("RMSE",      pl_rmse_v < lstm_rmse),
    ("R²",        pl_r2_v   > lstm_r2),
    ("Direction", pl_dir_v  > lstm_dir),
]
pl_wins   = sum(1 for _, v in metrics if v)
lstm_wins = 4 - pl_wins
winner    = "Power Law" if pl_wins > lstm_wins else "Conv-LSTM"
print(f"\n  Overall winner ({max(pl_wins,lstm_wins)}/4 metrics): {winner}")
print("=" * 65)


# ---------------------------------------------------------------------------
# 5. Export CSV
# ---------------------------------------------------------------------------
results_df = pd.DataFrame({
    "date":            test_dates_v,
    "actual_price":    test_actual_v.round(2),
    "power_law_pred":  pl_pred_v.round(2),
    "conv_lstm_pred":  lstm_pred_prices_v.round(2),
    "pl_error_pct":    ((pl_pred_v - test_actual_v) / test_actual_v * 100).round(2),
    "lstm_error_pct":  ((lstm_pred_prices_v - test_actual_v) / test_actual_v * 100).round(2),
})
results_df.to_csv("bitcoin_model_comparison.csv", index=False)

summary_df = pd.DataFrame({
    "metric":    ["MAPE (%)", "RMSE ($)", "Log-R²", "Directional Accuracy (%)"],
    "power_law": [round(pl_mape_v,2), round(pl_rmse_v,2), round(pl_r2_v,4), round(pl_dir_v,2)],
    "conv_lstm": [round(lstm_mape,2), round(lstm_rmse,2), round(lstm_r2,4), round(lstm_dir,2)],
})
summary_df.to_csv("bitcoin_model_summary.csv", index=False)
print(f"\nResults → bitcoin_model_comparison.csv  ({len(results_df):,} rows)")
print(f"Summary → bitcoin_model_summary.csv")


# ---------------------------------------------------------------------------
# 6. Chart
# ---------------------------------------------------------------------------
fig = make_subplots(
    rows=3, cols=1,
    row_heights=[0.55, 0.25, 0.20],
    shared_xaxes=True,
    subplot_titles=(
        "Actual vs Predicted Price (Test Period)",
        "Prediction Error (%)",
        "Power Law Deviation — Actual vs LSTM Predicted",
    ),
    vertical_spacing=0.07,
)

DARK = dict(paper_bgcolor="#0d1117", plot_bgcolor="#0d1117", font=dict(color="#e6edf3"))

# Row 1: prices
fig.add_trace(go.Scatter(x=test_dates_v, y=test_actual_v,
    mode="lines", line=dict(color="white", width=1.5), name="BTC Actual"), row=1, col=1)
fig.add_trace(go.Scatter(x=test_dates_v, y=pl_pred_v,
    mode="lines", line=dict(color="#facc15", width=1.5, dash="dash"),
    name=f"Power Law (MAPE {pl_mape_v:.1f}%)"), row=1, col=1)
fig.add_trace(go.Scatter(x=test_dates_v, y=lstm_pred_prices_v,
    mode="lines", line=dict(color="#60a5fa", width=1.5),
    name=f"Conv-LSTM rolling (MAPE {lstm_mape:.1f}%)"), row=1, col=1)

# Row 2: error %
fig.add_trace(go.Scatter(x=test_dates_v,
    y=((pl_pred_v - test_actual_v) / test_actual_v * 100),
    mode="lines", line=dict(color="#facc15", width=1), name="PL Error %",
    showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=test_dates_v,
    y=((lstm_pred_prices_v - test_actual_v) / test_actual_v * 100),
    mode="lines", line=dict(color="#60a5fa", width=1), name="LSTM Error %",
    showlegend=False), row=2, col=1)
fig.add_hline(y=0, line_color="rgba(255,255,255,0.2)", row=2, col=1)

# Row 3: pl_deviation actual vs predicted
actual_dev   = pl_log_deviation(np.log10(test_actual_v), test_days_v, S_SLOPE, S_INTERCEPT)
pred_dev     = pl_log_deviation(np.log10(np.maximum(lstm_pred_prices_v, 1)), test_days_v, S_SLOPE, S_INTERCEPT)
fig.add_trace(go.Scatter(x=test_dates_v, y=actual_dev,
    mode="lines", line=dict(color="white", width=1), name="Actual deviation",
    showlegend=False), row=3, col=1)
fig.add_trace(go.Scatter(x=test_dates_v, y=pred_dev,
    mode="lines", line=dict(color="#60a5fa", width=1), name="LSTM predicted deviation",
    showlegend=False), row=3, col=1)
fig.add_hline(y=0, line_color="#facc15", line_dash="dot",
    annotation_text="Power Law Fair Value", annotation_font_color="#facc15",
    row=3, col=1)

fig.update_layout(
    title=f"BTC Power Law vs Rolling Conv-LSTM — Test Period "
          f"({test_dates_v[0].date()} → {test_dates_v[-1].date()})",
    **DARK,
    legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1),
    hovermode="x unified",
)
for row in range(1, 4):
    fig.update_xaxes(gridcolor="#21262d", hoverformat="%b %d, %Y", row=row, col=1)
    fig.update_yaxes(gridcolor="#21262d", row=row, col=1)

fig.update_yaxes(title_text="Price (USD)", type="log", hoverformat="$,.0f", row=1, col=1)
fig.update_yaxes(title_text="Error (%)", hoverformat=".2f", row=2, col=1)
fig.update_yaxes(title_text="PL Deviation (log10)", hoverformat=".4f", row=3, col=1)

fig.write_html("bitcoin_power_law.html")
print("Chart     → bitcoin_power_law.html")

# ---------------------------------------------------------------------------
# 7. Standalone Power Law chart (full history + projection to 2035)
# ---------------------------------------------------------------------------
end_proj   = date(2035, 1, 1)
proj_days  = np.linspace(
    days_since_genesis(df.index[0].date()),
    days_since_genesis(end_proj),
    3000,
)
proj_dates = [GENESIS + timedelta(days=int(d)) for d in proj_days]

san_line  = power_law_price(proj_days, S_SLOPE,    S_INTERCEPT)
fit_line  = power_law_price(proj_days, pl_slope,   pl_intercept)
supp_line = power_law_price(proj_days, S_SLOPE,    S_INTERCEPT + SUP_OFF)
res_line  = power_law_price(proj_days, S_SLOPE,    S_INTERCEPT + RES_OFF)

today      = date.today()
today_days = days_since_genesis(today)
cur_px     = float(df["close"].iloc[-1])

# Zone label
san_now  = power_law_price(today_days, S_SLOPE, S_INTERCEPT)
supp_now = power_law_price(today_days, S_SLOPE, S_INTERCEPT + SUP_OFF)
res_now  = power_law_price(today_days, S_SLOPE, S_INTERCEPT + RES_OFF)
if   cur_px < supp_now:           zone = "BELOW SUPPORT"
elif cur_px < san_now  * 0.80:    zone = "UNDERVALUED"
elif cur_px < san_now  * 1.20:    zone = "FAIRLY VALUED"
elif cur_px < res_now:            zone = "OVERVALUED"
else:                              zone = "ABOVE RESISTANCE"

# Halving dates for reference (use only historical ones for chart annotations)
halvings = [(d, f"Halving {i+1}") for i, d in enumerate(HALVING_DATES) if d <= date.today()]

pl_fig = go.Figure()

# Corridor fill
pl_fig.add_trace(go.Scatter(
    x=proj_dates, y=res_line,
    fill=None, mode="lines",
    line=dict(color="rgba(0,0,0,0)"), showlegend=False,
))
pl_fig.add_trace(go.Scatter(
    x=proj_dates, y=supp_line,
    fill="tonexty", mode="lines",
    fillcolor="rgba(100,200,100,0.07)",
    line=dict(color="rgba(0,0,0,0)"),
    name="Power Law Corridor",
))

# Band lines
pl_fig.add_trace(go.Scatter(
    x=proj_dates, y=supp_line, mode="lines",
    line=dict(color="#4ade80", width=1.5, dash="dot"),
    name=f"Support floor  (${supp_now:,.0f})",
))
pl_fig.add_trace(go.Scatter(
    x=proj_dates, y=san_line, mode="lines",
    line=dict(color="#facc15", width=2),
    name=f"Santostasi Fair Value  (${san_now:,.0f})",
))
pl_fig.add_trace(go.Scatter(
    x=proj_dates, y=res_line, mode="lines",
    line=dict(color="#f87171", width=1.5, dash="dot"),
    name=f"Resistance ceiling  (${res_now:,.0f})",
))
pl_fig.add_trace(go.Scatter(
    x=proj_dates, y=fit_line, mode="lines",
    line=dict(color="#60a5fa", width=1.5, dash="dash"),
    name=f"Fitted model  R²={pl_r2_train:.3f}",
))

# Actual price
pl_fig.add_trace(go.Scatter(
    x=df.index, y=df["close"],
    mode="lines", line=dict(color="white", width=1.2),
    name="BTC-USD (actual)",
))

# Halving markers
for hdate, hlabel in halvings:
    if hdate >= df.index[0].date():
        pl_fig.add_vline(
            x=pd.Timestamp(hdate).timestamp() * 1000,
            line_width=1, line_dash="dot", line_color="rgba(168,85,247,0.5)",
        )
        pl_fig.add_annotation(
            x=pd.Timestamp(hdate), y=1,
            text=hlabel, textangle=-90,
            showarrow=False, yref="paper", yanchor="bottom",
            font=dict(color="rgba(168,85,247,0.8)", size=10),
        )

# Today marker
pl_fig.add_vline(
    x=pd.Timestamp(today).timestamp() * 1000,
    line_width=1, line_dash="dash", line_color="rgba(255,255,255,0.4)",
)
pl_fig.add_annotation(
    x=pd.Timestamp(today), y=np.log10(cur_px),
    text=f"  Today  ${cur_px:,.0f}<br>  Zone: {zone}",
    showarrow=False, xanchor="left",
    font=dict(color="white", size=11),
    bgcolor="rgba(0,0,0,0.55)", bordercolor="rgba(255,255,255,0.2)",
)

pl_fig.update_layout(
    title=dict(
        text="Bitcoin Price Power Law  —  Full History + Projection to 2035",
        font=dict(size=18),
    ),
    paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
    font=dict(color="#e6edf3"),
    xaxis=dict(title="Date", gridcolor="#21262d", showgrid=True),
    yaxis=dict(title="Price (USD)", type="log", gridcolor="#21262d", showgrid=True,
               tickformat="$,.0f"),
    legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1,
                x=0.01, y=0.99, xanchor="left", yanchor="top"),
    hovermode="x unified",
    height=700,
)

pl_fig.write_html("bitcoin_power_law_model.html")
print("PL Chart  → bitcoin_power_law_model.html")

import subprocess
subprocess.Popen(["open", "bitcoin_power_law_model.html"])
subprocess.Popen(["open", "bitcoin_power_law.html"])


# ---------------------------------------------------------------------------
# 8. Conv-LSTM Forecast Chart — full history fit + future predictions
# ---------------------------------------------------------------------------
print("\n[3/3] Building Conv-LSTM forecast chart (train on all data)...")

FORECAST_DAYS = 180   # how many days forward to project

# --- 8a. Train final model on recent data only (since Halving 3, May 2020) ---
# Early Bitcoin volatility (2014-2017) is a different regime — training on it
# causes the forecast model to reproduce unrealistic extreme movements.
FORECAST_TRAIN_START = pd.Timestamp("2020-05-11")
forecast_train_mask  = df.index >= FORECAST_TRAIN_START
forecast_features    = all_features[forecast_train_mask]
forecast_target      = target_shifted[forecast_train_mask]

scaler_X_all = MinMaxScaler()
scaler_y_all = MinMaxScaler()

X_all_scaled = scaler_X_all.fit_transform(forecast_features)
y_all_scaled = scaler_y_all.fit_transform(forecast_target.reshape(-1, 1))

X_full, y_full = make_windows(X_all_scaled, y_all_scaled, WINDOW)

final_model = build_model(WINDOW, n_feat)
final_callbacks = [
    EarlyStopping(patience=12, restore_best_weights=True, verbose=0),
    ReduceLROnPlateau(patience=6, factor=0.5, verbose=0),
]
n_forecast_train = forecast_train_mask.sum()
print(f"  Training final model on recent history ({FORECAST_TRAIN_START.date()} → {df.index[-1].date()}, {n_forecast_train:,} days)...")
final_model.fit(
    X_full, y_full,
    validation_split=0.05,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=final_callbacks,
    verbose=0,
)
print("  Final model trained.")

# --- 8b. In-sample fitted values (starting from WINDOW+1) ---
# In-sample fit only over the recent training window
forecast_start_iloc = df.index.searchsorted(FORECAST_TRAIN_START)
in_sample_prices = np.full(len(df), np.nan)
for i in range(WINDOW, len(forecast_features)):
    win = X_all_scaled[i - WINDOW:i].reshape(1, WINDOW, n_feat)
    pred_dev_scaled = final_model.predict(win, verbose=0)
    pred_dev = scaler_y_all.inverse_transform(pred_dev_scaled)[0, 0]
    global_i = forecast_start_iloc + i
    pl_t = S_SLOPE * np.log10(all_days[global_i]) + S_INTERCEPT
    in_sample_prices[global_i] = 10 ** (pl_t + pred_dev)

print(f"  In-sample fit computed for {(~np.isnan(in_sample_prices)).sum():,} days.")

# --- 8c. Recursive future predictions ---
# Seed the buffer with the last WINDOW rows of actual data
price_buf  = list(all_prices[-WINDOW:])          # rolling price history
volume_buf = list(np.expm1(all_features[-WINDOW:, 1]))  # log_volume → raw volume

future_dates  = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=FORECAST_DAYS, freq="D")
future_prices = []
future_days   = [days_since_genesis(d.date()) for d in future_dates]

# Compute test-period residuals for confidence bands
valid_in = ~np.isnan(in_sample_prices)
test_residuals = np.log10(np.maximum(all_prices[valid_in], 1)) - np.log10(np.maximum(in_sample_prices[valid_in], 1))
resid_std = np.std(test_residuals)

# Keep a sliding window of the last 30 raw prices for MA computation
def compute_features_from_buf(buf_prices, buf_volume_log, days_val):
    """Re-derive all features from the rolling price buffer."""
    prices = np.array(buf_prices)
    log_prices = np.log10(np.maximum(prices, 1e-6))
    log_ret = np.diff(log_prices)[-1] if len(prices) > 1 else 0.0

    log_vol = buf_volume_log[-1] if buf_volume_log else 0.0

    ma7  = np.mean(prices[-7:])  if len(prices) >= 7  else np.mean(prices)
    ma30 = np.mean(prices[-30:]) if len(prices) >= 30 else np.mean(prices)
    ma7_r  = prices[-1] / ma7  if ma7 > 0 else 1.0
    ma30_r = prices[-1] / ma30 if ma30 > 0 else 1.0

    # RSI from last 15 prices
    if len(prices) >= 15:
        deltas = np.diff(prices[-15:])
        gains  = deltas.clip(min=0)
        losses = (-deltas.clip(max=0))
        avg_g  = gains.mean()
        avg_l  = losses.mean()
        rsi_val = 100 - (100 / (1 + avg_g / avg_l)) if avg_l > 0 else 50.0
    else:
        rsi_val = 50.0

    pl_dev = log_prices[-1] - (S_SLOPE * np.log10(max(days_val, 1)) + S_INTERCEPT)

    return np.array([log_ret, log_vol, ma7_r, ma30_r, rsi_val, pl_dev])

# Build the rolling feature window for future steps
future_feature_buf = [
    compute_features_from_buf(
        price_buf[:i+1],
        list(all_features[-WINDOW:, 1]),  # log_volume column
        all_days[-WINDOW + i]
    )
    for i in range(WINDOW)
]
future_feature_buf = list(future_feature_buf)  # list of feature vectors

# Historical pl_deviation bounds — hard clamp only, no step-by-step pull
# (step-by-step mean reversion inside the loop causes oscillations via ma feedback)
hist_dev_std  = float(df["pl_deviation"].std())
hist_dev_mean = float(df["pl_deviation"].mean())
DEV_CLAMP     = 2.0 * hist_dev_std   # allow ±2σ, hard-clamp beyond that

for step, (fdate, fdays) in enumerate(zip(future_dates, future_days)):
    win_X = np.array(future_feature_buf[-WINDOW:])
    win_scaled = scaler_X_all.transform(win_X).reshape(1, WINDOW, n_feat)
    pred_dev_scaled = final_model.predict(win_scaled, verbose=0)
    pred_dev = float(np.clip(
        scaler_y_all.inverse_transform(pred_dev_scaled)[0, 0],
        hist_dev_mean - DEV_CLAMP,
        hist_dev_mean + DEV_CLAMP,
    ))

    pl_t = S_SLOPE * np.log10(max(fdays, 1)) + S_INTERCEPT
    pred_price = 10 ** (pl_t + pred_dev)
    future_prices.append(pred_price)

    # Update buffers
    price_buf.append(pred_price)
    if len(price_buf) > 60:
        price_buf.pop(0)

    # Estimate log_volume: carry forward last known
    last_log_vol = future_feature_buf[-1][1]
    future_feature_buf.append(
        compute_features_from_buf(price_buf, [last_log_vol], fdays)
    )

future_prices = np.array(future_prices)

# Post-hoc smoothing: blend forecast toward power law with growing weight,
# applied after recursion so it doesn't feed back into the feature loop.
BLEND_HORIZON = FORECAST_DAYS
pl_forecast   = power_law_price(np.array(future_days), S_SLOPE, S_INTERCEPT)
blend_weights = np.linspace(0, 0.5, BLEND_HORIZON)   # 0% → 50% PL by end
future_prices = (1 - blend_weights) * future_prices + blend_weights * pl_forecast

print(f"  Recursive forecast: {FORECAST_DAYS} days  "
      f"(final price ${future_prices[-1]:,.0f})")

# --- 8d. Confidence bands: ±1σ·√t growing uncertainty ---
t_arr    = np.arange(1, FORECAST_DAYS + 1)
log_mean = np.log10(np.maximum(future_prices, 1))
band_lo  = 10 ** (log_mean - resid_std * np.sqrt(t_arr))
band_hi  = 10 ** (log_mean + resid_std * np.sqrt(t_arr))

# --- 8e. Build chart ---
fc_fig = go.Figure()

# Full actual history
fc_fig.add_trace(go.Scatter(
    x=df.index, y=df["close"],
    mode="lines", line=dict(color="rgba(200,200,200,0.7)", width=1.2),
    name="BTC Actual",
))

# In-sample fitted line
valid_idx = ~np.isnan(in_sample_prices)
fc_fig.add_trace(go.Scatter(
    x=df.index[valid_idx], y=in_sample_prices[valid_idx],
    mode="lines", line=dict(color="#4cc9f0", width=1.2, dash="dot"),
    name="Conv-LSTM Fit (in-sample)",
    opacity=0.8,
))

# Test-period overlay (highlight in brighter color)
test_valid = valid_idx[split:]
test_dates_all = df.index[split:]
fc_fig.add_trace(go.Scatter(
    x=test_dates_all[test_valid], y=in_sample_prices[split:][test_valid],
    mode="lines", line=dict(color="#f72585", width=2),
    name="Conv-LSTM (test period)",
))

# Future confidence band
fc_fig.add_trace(go.Scatter(
    x=list(future_dates) + list(future_dates[::-1]),
    y=list(band_hi) + list(band_lo[::-1]),
    fill="toself", fillcolor="rgba(251,133,0,0.12)",
    line=dict(color="rgba(0,0,0,0)"),
    name="±1σ confidence band",
    showlegend=True,
))

# Future forecast line
fc_fig.add_trace(go.Scatter(
    x=future_dates, y=future_prices,
    mode="lines", line=dict(color="#fb8500", width=2.5),
    name=f"Conv-LSTM Forecast (+{FORECAST_DAYS}d)",
))

# Divider: today / forecast start
fc_fig.add_vline(
    x=pd.Timestamp(df.index[-1]).timestamp() * 1000,
    line_width=1, line_dash="dash", line_color="rgba(255,255,255,0.4)",
)
fc_fig.add_annotation(
    x=df.index[-1], y=1, xref="x", yref="paper",
    text="  Forecast →", showarrow=False, xanchor="left",
    font=dict(color="rgba(255,255,255,0.6)", size=11),
)

# Halving markers
for hdate, hlabel in halvings:
    if hdate >= df.index[0].date():
        fc_fig.add_vline(
            x=pd.Timestamp(hdate).timestamp() * 1000,
            line_width=1, line_dash="dot", line_color="rgba(168,85,247,0.4)",
        )
        fc_fig.add_annotation(
            x=pd.Timestamp(hdate), y=0.97,
            text=hlabel, textangle=-90,
            showarrow=False, yref="paper", yanchor="top",
            font=dict(color="rgba(168,85,247,0.7)", size=9),
        )

fc_fig.update_layout(
    title=dict(
        text=(
            f"Bitcoin Conv-LSTM Model — Full History Fit + {FORECAST_DAYS}-Day Forecast<br>"
            f"<sup>Test MAPE: {lstm_mape:.1f}%  |  Log-R²: {lstm_r2:.4f}  |  "
            f"Final forecast: ${future_prices[-1]:,.0f}</sup>"
        ),
        font=dict(size=17),
    ),
    paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
    font=dict(color="#e6edf3"),
    xaxis=dict(title="Date", gridcolor="#21262d", showgrid=True,
               hoverformat="%b %d, %Y"),
    yaxis=dict(
        title="Price (USD)", type="log", gridcolor="#21262d", showgrid=True,
        tickformat="$,.0f", hoverformat="$,.0f",
    ),
    legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1,
                x=0.01, y=0.99, xanchor="left", yanchor="top"),
    hovermode="x unified",
    height=700,
)

fc_fig.write_html("bitcoin_lstm_forecast.html")
print("Forecast chart → bitcoin_lstm_forecast.html")
subprocess.Popen(["open", "bitcoin_lstm_forecast.html"])

# --- Save forecast cache for daily update script ---
import json
valid_idx_fc = ~np.isnan(in_sample_prices)
forecast_cache = {
    "generated_date":   str(df.index[-1].date()),
    "lstm_mape":        lstm_mape,
    "lstm_r2":          lstm_r2,
    "forecast_dates":   [str(d.date()) for d in future_dates],
    "forecast_prices":  future_prices.tolist(),
    "band_lo":          band_lo.tolist(),
    "band_hi":          band_hi.tolist(),
    "in_sample_dates":  [str(d.date()) for d in df.index[valid_idx_fc]],
    "in_sample_prices": in_sample_prices[valid_idx_fc].tolist(),
}
with open("bitcoin_forecast_cache.json", "w") as f:
    json.dump(forecast_cache, f)
print("Forecast cache  → bitcoin_forecast_cache.json")

# --- Export forecast chart data to CSV ---
hist_df      = df[["close"]].rename(columns={"close": "actual_price"})
insample_s   = pd.Series(in_sample_prices, index=df.index, name="in_sample_fit")
insample_s   = insample_s[~np.isnan(insample_s)]
forecast_s   = pd.DataFrame({
    "forecast_price": future_prices,
    "band_lo":        band_lo,
    "band_hi":        band_hi,
}, index=future_dates)
forecast_s.index.name = "date"

lstm_chart_csv = (
    hist_df
    .join(insample_s, how="outer")
    .join(forecast_s, how="outer")
    .sort_index()
    .round(2)
)
lstm_chart_csv.index.name = "date"
lstm_chart_csv.to_csv("bitcoin_lstm_forecast_data.csv")
print(f"Forecast data   → bitcoin_lstm_forecast_data.csv  ({len(lstm_chart_csv):,} rows)")


# ---------------------------------------------------------------------------
# 9. Cycle-Aware Walk-Forward Cross-Validation
#
# Strategy:
#   Each fold trains ONLY on data before the halving that starts that cycle,
#   then tests across the full cycle. This forces the model to predict
#   genuine out-of-sample bull-run peaks — not a regime it has already seen.
#
#   Fold  |  Train through        |  Test window
#   ------|-----------------------|-----------------------------
#     2   |  2nd halving Jul 2016 |  Jul 2016 → May 2020
#     3   |  3rd halving May 2020 |  May 2020 → Apr 2024
#     4   |  4th halving Apr 2024 |  Apr 2024 → today
# ---------------------------------------------------------------------------
print("\n[4/4] Cycle-aware walk-forward cross-validation...")

CV_ROLL_STEP = 60   # larger step than main eval for speed (~halves retrains)

def run_conv_lstm_fold(all_features, target_shifted, all_days, all_prices, all_dates,
                       train_end_idx, test_end_idx, fold_label):
    """Rolling-retrain Conv-LSTM for one cycle fold. Returns arrays + metrics dict."""
    n_fold  = test_end_idx - train_end_idx
    n_steps = max((n_fold - WINDOW) // CV_ROLL_STEP + 1, 0)
    pred_buf = np.full(n_fold, np.nan)
    count    = 0

    for step_i in range(n_steps):
        te = train_end_idx + step_i * CV_ROLL_STEP
        ps = te
        pe = min(ps + CV_ROLL_STEP, test_end_idx)
        if ps >= test_end_idx or te - WINDOW < 0:
            continue

        sx = MinMaxScaler(); sy = MinMaxScaler()
        Xr = sx.fit_transform(all_features[:te])
        yr = sy.fit_transform(target_shifted[:te].reshape(-1, 1))
        Xt, yt = make_windows(Xr, yr, WINDOW)
        if len(Xt) < BATCH_SIZE:
            continue

        m = build_model(WINDOW, n_feat)
        m.fit(Xt, yt,
              validation_split=0.1, epochs=EPOCHS, batch_size=BATCH_SIZE,
              callbacks=[EarlyStopping(patience=10, restore_best_weights=True, verbose=0),
                         ReduceLROnPlateau(patience=5, factor=0.5, verbose=0)],
              verbose=0)
        count += 1

        for pi in range(ps, pe):
            if pi - WINDOW < 0:
                continue
            wx = sx.transform(all_features[pi - WINDOW:pi]).reshape(1, WINDOW, n_feat)
            pds = m.predict(wx, verbose=0)
            pdv = sy.inverse_transform(pds)[0, 0]
            pl_t = S_SLOPE * np.log10(all_days[pi]) + S_INTERCEPT
            pred_buf[pi - train_end_idx] = 10 ** (pl_t + pdv)

        pct = (step_i + 1) / n_steps * 100
        print(f"    [{pct:5.1f}%] {fold_label}  retrain #{count}  "
              f"days {ps - train_end_idx}–{pe - train_end_idx}", flush=True)

    v = ~np.isnan(pred_buf)
    dates_v  = all_dates[train_end_idx:test_end_idx][v]
    actual_v = all_prices[train_end_idx:test_end_idx][v]
    pred_v   = pred_buf[v]
    days_v   = all_days[train_end_idx:test_end_idx][v]
    pl_v     = power_law_price(days_v, pl_slope, pl_intercept)

    mx = dict(
        lstm_mape = mape(actual_v, pred_v),
        lstm_rmse = rmse(actual_v, pred_v),
        lstm_r2   = r2_score(np.log10(actual_v), np.log10(np.maximum(pred_v, 1))),
        lstm_dir  = directional_accuracy(actual_v, pred_v),
        pl_mape   = mape(actual_v, pl_v),
        pl_rmse   = rmse(actual_v, pl_v),
        pl_r2     = r2_score(np.log10(actual_v), np.log10(pl_v)),
        pl_dir    = directional_accuracy(actual_v, pl_v),
        n_days    = int(v.sum()),
        retrains  = count,
    )
    return dates_v, actual_v, pred_v, pl_v, mx


# --- Define folds from halving dates ---
halving_cv   = [date(2016, 7, 9), date(2020, 5, 11), date(2024, 4, 19)]
halving_idxs = [df.index.searchsorted(pd.Timestamp(d)) for d in halving_cv]

folds = [
    ("Cycle 2 (2016–2020)", halving_idxs[0], halving_idxs[1]),
    ("Cycle 3 (2020–2024)", halving_idxs[1], halving_idxs[2]),
    ("Cycle 4 (2024–now)",  halving_idxs[2], len(df)),
]

FOLD_COLORS = {
    "Cycle 2 (2016–2020)": "#4cc9f0",
    "Cycle 3 (2020–2024)": "#f72585",
    "Cycle 4 (2024–now)":  "#fb8500",
}

fold_results = {}
for label, tr_end, te_end in folds:
    n_train = tr_end
    n_test  = te_end - tr_end
    print(f"\n  → {label}  |  training on {n_train:,} days  |  testing on {n_test:,} days")
    if n_test < WINDOW + CV_ROLL_STEP:
        print(f"    (skipping — test period too short)")
        continue
    dates, actual, pred, pl_pred_fold, mx = run_conv_lstm_fold(
        all_features, target_shifted, all_days, all_prices, df.index.values,
        tr_end, te_end, label,
    )
    fold_results[label] = dict(dates=dates, actual=actual, pred=pred, pl=pl_pred_fold, metrics=mx)
    print(f"    Conv-LSTM  MAPE={mx['lstm_mape']:.1f}%  RMSE=${mx['lstm_rmse']:,.0f}  "
          f"Log-R²={mx['lstm_r2']:.4f}  DirAcc={mx['lstm_dir']:.1f}%")
    print(f"    Power Law  MAPE={mx['pl_mape']:.1f}%  RMSE=${mx['pl_rmse']:,.0f}  "
          f"Log-R²={mx['pl_r2']:.4f}  DirAcc={mx['pl_dir']:.1f}%")


# --- Print cross-fold summary table ---
print("\n" + "=" * 80)
print(f"{'Cycle-Aware Cross-Validation Summary':^80}")
print("=" * 80)
print(f"  {'Fold':<24} {'N days':>7}  {'LSTM MAPE':>10}  {'PL MAPE':>9}  {'LSTM R²':>8}  {'PL R²':>7}")
print("  " + "-" * 74)
for label, res in fold_results.items():
    mx = res["metrics"]
    print(f"  {label:<24} {mx['n_days']:>7,}  {mx['lstm_mape']:>9.1f}%  "
          f"{mx['pl_mape']:>8.1f}%  {mx['lstm_r2']:>8.4f}  {mx['pl_r2']:>7.4f}")

# Averages
if fold_results:
    avg_lstm_mape = np.mean([r["metrics"]["lstm_mape"] for r in fold_results.values()])
    avg_pl_mape   = np.mean([r["metrics"]["pl_mape"]   for r in fold_results.values()])
    avg_lstm_r2   = np.mean([r["metrics"]["lstm_r2"]   for r in fold_results.values()])
    avg_pl_r2     = np.mean([r["metrics"]["pl_r2"]     for r in fold_results.values()])
    print("  " + "-" * 74)
    print(f"  {'Average':<24} {'':>7}  {avg_lstm_mape:>9.1f}%  "
          f"{avg_pl_mape:>8.1f}%  {avg_lstm_r2:>8.4f}  {avg_pl_r2:>7.4f}")
print("=" * 80)


# --- Export CSV ---
cv_rows = []
for label, res in fold_results.items():
    mx = res["metrics"]
    cv_rows.append({
        "fold": label, "n_days": mx["n_days"], "retrains": mx["retrains"],
        "lstm_mape": round(mx["lstm_mape"], 2), "pl_mape": round(mx["pl_mape"], 2),
        "lstm_rmse": round(mx["lstm_rmse"], 2), "pl_rmse": round(mx["pl_rmse"], 2),
        "lstm_log_r2": round(mx["lstm_r2"], 4),  "pl_log_r2": round(mx["pl_r2"], 4),
        "lstm_dir_acc": round(mx["lstm_dir"], 2), "pl_dir_acc": round(mx["pl_dir"], 2),
    })
pd.DataFrame(cv_rows).to_csv("bitcoin_cycle_cv.csv", index=False)
print("CV metrics → bitcoin_cycle_cv.csv")


# --- Chart ---
cv_fig = go.Figure()

# Faint full-history baseline
cv_fig.add_trace(go.Scatter(
    x=df.index, y=df["close"],
    mode="lines", line=dict(color="rgba(200,200,200,0.25)", width=1),
    name="BTC Actual (full history)", showlegend=True,
))

for label, res in fold_results.items():
    color = FOLD_COLORS[label]
    mx    = res["metrics"]

    # Bright actual line for this fold's period
    cv_fig.add_trace(go.Scatter(
        x=res["dates"], y=res["actual"],
        mode="lines", line=dict(color="rgba(230,230,230,0.85)", width=1.5),
        name=f"{label} — actual", showlegend=False,
    ))
    # Conv-LSTM prediction
    cv_fig.add_trace(go.Scatter(
        x=res["dates"], y=res["pred"],
        mode="lines", line=dict(color=color, width=2.2),
        name=f"{label}  LSTM: MAPE {mx['lstm_mape']:.1f}%  R²={mx['lstm_r2']:.3f}",
    ))

# Halving markers
for hdate, hlabel in halvings:
    if hdate >= df.index[0].date():
        cv_fig.add_vline(
            x=pd.Timestamp(hdate).timestamp() * 1000,
            line_width=1, line_dash="dot", line_color="rgba(168,85,247,0.5)",
        )
        cv_fig.add_annotation(
            x=pd.Timestamp(hdate), y=0.97,
            text=hlabel, textangle=-90,
            showarrow=False, yref="paper", yanchor="top",
            font=dict(color="rgba(168,85,247,0.7)", size=9),
        )

# Annotation box: average metrics
avg_text = (
    f"<b>Cross-Cycle Averages</b><br>"
    f"Conv-LSTM MAPE: {avg_lstm_mape:.1f}%<br>"
    f"Power Law MAPE: {avg_pl_mape:.1f}%<br>"
    f"Conv-LSTM Log-R²: {avg_lstm_r2:.3f}<br>"
    f"Power Law Log-R²: {avg_pl_r2:.3f}"
)
cv_fig.add_annotation(
    x=0.99, y=0.04, xref="paper", yref="paper",
    text=avg_text, showarrow=False, xanchor="right", yanchor="bottom",
    font=dict(color="#e6edf3", size=11),
    bgcolor="rgba(22,27,34,0.85)", bordercolor="#30363d", borderwidth=1,
)

cv_fig.update_layout(
    title=dict(
        text=(
            "Bitcoin Conv-LSTM — Cycle-Aware Walk-Forward Cross-Validation<br>"
            "<sup>Each fold trained only on data before that halving cycle; "
            "tested across the full subsequent cycle</sup>"
        ),
        font=dict(size=17),
    ),
    paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
    font=dict(color="#e6edf3"),
    xaxis=dict(title="Date", gridcolor="#21262d", showgrid=True),
    yaxis=dict(title="Price (USD)", type="log", gridcolor="#21262d", showgrid=True,
               tickformat="$,.0f"),
    legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1,
                x=0.01, y=0.99, xanchor="left", yanchor="top"),
    hovermode="x unified",
    height=700,
)

cv_fig.write_html("bitcoin_cycle_cv.html")
print("Cycle CV chart → bitcoin_cycle_cv.html")
subprocess.Popen(["open", "bitcoin_cycle_cv.html"])
