"""
bitcoin_daily_update.py

Lightweight daily update — runs in ~30 seconds:
  1. Fetches latest BTC-USD price
  2. Appends new rows to bitcoin_daily_prices.csv
  3. Rebuilds bitcoin_lstm_forecast.html — plots real data against the saved forecast
  4. Refreshes bitcoin_power_law_model.html with updated Today marker + zone

Run daily. Re-run bitcoin_price_predictor.py monthly to retrain the model.
"""

import warnings; warnings.filterwarnings("ignore")
import argparse
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from datetime import date, datetime, timedelta
import subprocess

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Bitcoin daily update")
parser.add_argument("date", nargs="?", default=None,
                    help="Backfill a specific date, e.g. 4/11 or 2026-04-11")
args = parser.parse_args()

target_date = None
if args.date:
    for fmt in ("%m/%d", "%Y-%m-%d", "%m/%d/%Y"):
        try:
            parsed = datetime.strptime(args.date, fmt)
            if fmt == "%m/%d":
                parsed = parsed.replace(year=date.today().year)
            target_date = parsed.date()
            break
        except ValueError:
            continue
    if target_date is None:
        print(f"Could not parse date: {args.date}")
        exit(1)

# ── Constants (must match main script) ───────────────────────────────────────
GENESIS     = date(2009, 1, 3)
S_SLOPE     = 5.84
S_INTERCEPT = -17.01
SUP_OFF     = -0.5
RES_OFF     = +0.4
DARK        = dict(paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                   font=dict(color="#e6edf3"))
HALVING_DATES = [
    date(2012, 11, 28),
    date(2016,  7, 16),
    date(2020,  5, 11),
    date(2024,  4, 20),
    date(2028,  4, 15),
    date(2032,  4, 15),
]

def days_since_genesis(d):
    if isinstance(d, pd.Timestamp):
        d = d.date()
    return (d - GENESIS).days

def power_law_price(days, slope=S_SLOPE, intercept=S_INTERCEPT):
    return 10 ** (slope * np.log10(np.maximum(days, 1)) + intercept)


# ── 1. Load forecast cache ────────────────────────────────────────────────────
print("Loading forecast cache...")
with open("bitcoin_forecast_cache.json") as f:
    cache = json.load(f)

generated_date   = pd.Timestamp(cache["generated_date"])
forecast_dates   = pd.to_datetime(cache["forecast_dates"])
forecast_prices  = np.array(cache["forecast_prices"])
band_lo          = np.array(cache["band_lo"])
band_hi          = np.array(cache["band_hi"])
in_sample_dates  = pd.to_datetime(cache["in_sample_dates"])
in_sample_prices = np.array(cache["in_sample_prices"])
lstm_mape        = cache["lstm_mape"]
lstm_r2          = cache["lstm_r2"]

print(f"  Forecast generated: {generated_date.date()}  "
      f"({len(forecast_dates)}-day window through {forecast_dates[-1].date()})")


# ── 2. Fetch latest BTC data ──────────────────────────────────────────────────
print("Fetching latest BTC-USD...")
raw = yf.download("BTC-USD", start="2020-01-01", auto_adjust=True, progress=False)
raw.columns = raw.columns.get_level_values(0)
raw = raw[["Close"]].dropna()
raw.index = pd.to_datetime(raw.index)

latest_price = float(raw["Close"].iloc[-1])
latest_date  = raw.index[-1]
print(f"  Latest: ${latest_price:,.0f}  ({latest_date.date()})")


# ── 3. Append new rows to daily CSV ──────────────────────────────────────────
daily_csv = "bitcoin_daily_prices.csv"
try:
    existing  = pd.read_csv(daily_csv, parse_dates=["date"])
    last_saved = existing["date"].max()
except FileNotFoundError:
    existing   = pd.DataFrame(columns=["date", "close"])
    last_saved = pd.Timestamp("2000-01-01")

if target_date:
    # Backfill mode: look for target_date or target_date+1 (UTC offset) in raw
    backfill_price = None
    for lookup in [pd.Timestamp(target_date), pd.Timestamp(target_date + timedelta(days=1))]:
        if lookup in raw.index:
            backfill_price = round(float(raw.loc[lookup, "Close"]), 2)
            break
    if backfill_price is None:
        print(f"  No data found for {target_date} in yfinance feed.")
        exit(1)
    backfill_ts = pd.Timestamp(target_date)
    if (existing["date"] == backfill_ts).any():
        print(f"  {target_date} already in CSV, skipping.")
    else:
        new_row = pd.DataFrame([{"date": backfill_ts, "close": backfill_price}])
        updated = pd.concat([existing, new_row], ignore_index=True).sort_values("date").reset_index(drop=True)
        updated.to_csv(daily_csv, index=False)
        print(f"  Backfilled {target_date} → ${backfill_price:,.0f}")
else:
    new_rows = raw[raw.index > last_saved].reset_index()
    new_rows.columns = ["date", "close"]
    new_rows["close"] = new_rows["close"].round(2)
    if len(new_rows):
        updated = pd.concat([existing, new_rows], ignore_index=True)
        updated.to_csv(daily_csv, index=False)
        print(f"  Appended {len(new_rows)} new row(s) to {daily_csv}")
    else:
        print("  No new rows to append.")


# ── 4. Export forecast chart CSV ─────────────────────────────────────────────
print("Exporting forecast chart data to CSV...")

# Historical actuals (everything shown in the faint background line)
hist_df = raw[["Close"]].copy()
hist_df.index.name = "date"
hist_df.columns = ["actual_price"]

# In-sample fit
insample_df = pd.Series(in_sample_prices, index=in_sample_dates, name="in_sample_fit")

# Forecast + bands
forecast_df = pd.DataFrame({
    "forecast_price": forecast_prices,
    "band_lo":        band_lo,
    "band_hi":        band_hi,
}, index=forecast_dates)
forecast_df.index.name = "date"

# Actual post-forecast prices
actual_post = raw[raw.index >= forecast_dates[0]][["Close"]].copy()
actual_post.columns = ["actual_post_forecast"]

# Combine everything into one wide CSV, aligning on date
chart_csv = (
    hist_df
    .join(insample_df, how="outer")
    .join(forecast_df, how="outer")
    .join(actual_post, how="outer")
    .sort_index()
)
chart_csv.index = pd.to_datetime(chart_csv.index).normalize()
chart_csv.index.name = "date"
chart_csv = chart_csv.round(2)
chart_csv.to_csv("bitcoin_lstm_forecast_data.csv")
print(f"  → bitcoin_lstm_forecast_data.csv  ({len(chart_csv):,} rows)")

# ── 5. Rebuild bitcoin_lstm_forecast.html ─────────────────────────────────────
print("Rebuilding forecast chart...")

halvings = [(d, f"Halving {i+1}") for i, d in enumerate(HALVING_DATES) if d <= date.today()]

days_elapsed   = (latest_date - generated_date).days
days_remaining = max((forecast_dates[-1] - latest_date).days, 0)

fc_fig = go.Figure()

# Full price history (faint background)
fc_fig.add_trace(go.Scatter(
    x=raw.index, y=raw["Close"],
    mode="lines", line=dict(color="rgba(200,200,200,0.35)", width=1),
    name="BTC Actual (history)",
))

# In-sample model fit (dotted)
fc_fig.add_trace(go.Scatter(
    x=in_sample_dates, y=in_sample_prices,
    mode="lines", line=dict(color="#4cc9f0", width=1.2, dash="dot"),
    name="Conv-LSTM Fit (in-sample)", opacity=0.8,
))

# Confidence band
fc_fig.add_trace(go.Scatter(
    x=list(forecast_dates) + list(forecast_dates[::-1]),
    y=list(band_hi) + list(band_lo[::-1]),
    fill="toself", fillcolor="rgba(251,133,0,0.12)",
    line=dict(color="rgba(0,0,0,0)"),
    name="±1σ confidence band",
))

# Forecast line
fc_fig.add_trace(go.Scatter(
    x=forecast_dates, y=forecast_prices,
    mode="lines", line=dict(color="#fb8500", width=2.5),
    name=f"Conv-LSTM Forecast (from {generated_date.date()})",
))

# Actual prices since forecast was generated (bright overlay)
if len(actual_post):
    fc_fig.add_trace(go.Scatter(
        x=actual_post.index, y=actual_post["actual_post_forecast"],
        mode="lines", line=dict(color="#f72585", width=2),
        name="BTC Actual (post-forecast)",
    ))

# Forecast-start divider
fc_fig.add_vline(
    x=pd.Timestamp(generated_date).timestamp() * 1000,
    line_width=1, line_dash="dash", line_color="rgba(255,255,255,0.4)",
)
fc_fig.add_annotation(
    x=generated_date, y=1, xref="x", yref="paper",
    text="  Forecast →", showarrow=False, xanchor="left",
    font=dict(color="rgba(255,255,255,0.6)", size=11),
)

# Halving markers
for hdate, hlabel in halvings:
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
            f"Bitcoin Conv-LSTM Forecast — Generated {generated_date.date()}<br>"
            f"<sup>Test MAPE: {lstm_mape:.1f}%  |  Log-R²: {lstm_r2:.4f}  |  "
            f"{days_elapsed}d elapsed  |  {days_remaining}d remaining  |  "
            f"Latest: ${latest_price:,.0f}</sup>"
        ),
        font=dict(size=17),
    ),
    **DARK,
    xaxis=dict(title="Date", gridcolor="#21262d", showgrid=True,
               hoverformat="%b %d, %Y",
               range=["2025-06-30", str(forecast_dates[-1].date())]),
    yaxis=dict(title="Price (USD)", gridcolor="#21262d", showgrid=True,
               tickformat="$,.0f", hoverformat="$,.0f"),
    legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1,
                x=0.01, y=0.99, xanchor="left", yanchor="top"),
    hovermode="x unified",
    height=700,
    autosize=True,
    margin=dict(l=60, r=30, t=80, b=50),
)

fc_fig.write_html("bitcoin_lstm_forecast.html", config={"responsive": True})
print("  → bitcoin_lstm_forecast.html")


# ── 6. Refresh bitcoin_power_law_model.html ───────────────────────────────────
print("Refreshing power law chart...")

end_proj    = date(2035, 1, 1)
chart_start = raw.index[0].date()
pl_halvings = [(d, f"Halving {i+1}") for i, d in enumerate(HALVING_DATES)
               if chart_start <= d < end_proj]
proj_days  = np.linspace(
    days_since_genesis(raw.index[0].date()),
    days_since_genesis(end_proj),
    3000,
)
proj_dates = [GENESIS + timedelta(days=int(d)) for d in proj_days]

san_line  = power_law_price(proj_days)
supp_line = power_law_price(proj_days, intercept=S_INTERCEPT + SUP_OFF)
res_line  = power_law_price(proj_days, intercept=S_INTERCEPT + RES_OFF)

today      = date.today()
today_days = days_since_genesis(today)
san_now    = power_law_price(today_days)
supp_now   = power_law_price(today_days, intercept=S_INTERCEPT + SUP_OFF)
res_now    = power_law_price(today_days, intercept=S_INTERCEPT + RES_OFF)

if   latest_price < supp_now:             zone = "BELOW SUPPORT"
elif latest_price < san_now  * 0.80:      zone = "UNDERVALUED"
elif latest_price < san_now  * 1.20:      zone = "FAIRLY VALUED"
elif latest_price < res_now:              zone = "OVERVALUED"
else:                                      zone = "ABOVE RESISTANCE"

pl_fig = go.Figure()

# Corridor fill
pl_fig.add_trace(go.Scatter(x=proj_dates, y=res_line, fill=None, mode="lines",
    line=dict(color="rgba(0,0,0,0)"), showlegend=False))
pl_fig.add_trace(go.Scatter(x=proj_dates, y=supp_line, fill="tonexty", mode="lines",
    fillcolor="rgba(100,200,100,0.07)", line=dict(color="rgba(0,0,0,0)"),
    name="Power Law Corridor"))

pl_fig.add_trace(go.Scatter(x=proj_dates, y=supp_line, mode="lines",
    line=dict(color="#4ade80", width=1.5, dash="dot"),
    name=f"Support floor  (${supp_now:,.0f})"))
pl_fig.add_trace(go.Scatter(x=proj_dates, y=san_line, mode="lines",
    line=dict(color="#facc15", width=2),
    name=f"Santostasi Fair Value  (${san_now:,.0f})"))
pl_fig.add_trace(go.Scatter(x=proj_dates, y=res_line, mode="lines",
    line=dict(color="#f87171", width=1.5, dash="dot"),
    name=f"Resistance ceiling  (${res_now:,.0f})"))
pl_fig.add_trace(go.Scatter(x=raw.index, y=raw["Close"], mode="lines",
    line=dict(color="white", width=1.2), name="BTC-USD (actual)"))

for hdate, hlabel in pl_halvings:
    pl_fig.add_vline(x=pd.Timestamp(hdate).timestamp() * 1000,
        line_width=1, line_dash="dot", line_color="rgba(168,85,247,0.5)")
    pl_fig.add_annotation(x=pd.Timestamp(hdate), y=1, text=hlabel, textangle=-90,
        showarrow=False, yref="paper", yanchor="bottom",
        font=dict(color="rgba(168,85,247,0.8)", size=10))
pl_fig.add_annotation(
    x=pd.Timestamp(today), y=np.log10(latest_price),
    ax=60, ay=-80, axref="pixel", ayref="pixel",
    text=f"{today.strftime('%b %d, %Y')}  ${latest_price:,.0f}<br>Zone: {zone}",
    showarrow=True, arrowhead=2, arrowwidth=1.2,
    arrowcolor="rgba(255,255,255,0.5)",
    xanchor="left",
    font=dict(color="white", size=11),
    bgcolor="rgba(0,0,0,0.65)", bordercolor="rgba(255,255,255,0.3)", borderwidth=1,
)

pl_fig.update_layout(
    title=dict(text="Bitcoin Price Power Law  —  Full History + Projection to 2035",
               font=dict(size=18)),
    **DARK,
    xaxis=dict(title="Date", gridcolor="#21262d", showgrid=True,
               hoverformat="%b %d, %Y",
               range=[str(chart_start), str(end_proj)]),
    yaxis=dict(title="Price (USD)", type="log", gridcolor="#21262d", showgrid=True,
               tickformat="$,.0f", hoverformat="$,.0f"),
    legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1,
                x=0.01, y=0.99, xanchor="left", yanchor="top"),
    hovermode="x unified",
    height=700,
)

pl_fig.write_html("bitcoin_power_law_model.html")
print("  → bitcoin_power_law_model.html")

import platform
if platform.system() == "Darwin":
    subprocess.Popen(["open", "bitcoin_lstm_forecast.html"])
    subprocess.Popen(["open", "bitcoin_power_law_model.html"])

print(f"\nDone.  Latest BTC: ${latest_price:,.0f}  |  Zone: {zone}  |  "
      f"{days_remaining}d left in forecast window")
