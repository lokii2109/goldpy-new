import sys
import os
import numpy as np

# --- Patch for pandas_ta / numpy ---
try:
    if not hasattr(np, "NaN"):
        np.NaN = np.nan
except Exception as e:
    print(f"NumPy patch error: {e}")

import yfinance as yf
import pandas as pd
import pandas_ta as ta
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# --- Streamlit UI ---
st.title("📈 Dual Supertrend Trading Strategy Backtester")
st.markdown("Backtests a strategy using Supertrend on dual timeframes for Gold (GC=F / XAUUSD=X).")

# --- Sidebar ---
st.sidebar.header("Strategy Parameters")
ticker = st.sidebar.text_input(
    "Ticker Symbol", 
    value="XAUUSD=X", 
    help="Try XAUUSD=X for spot gold, GC=F for futures, or BTC-USD, AAPL, etc."
)

period = st.sidebar.selectbox(
    "Data Period",
    ["1d", "5d", "30d", "60d", "3mo", "6mo", "1y"],
    index=1,
)
st.sidebar.subheader("Supertrend Settings")
st_length = st.sidebar.number_input("Length", value=7, min_value=1)
st_multiplier = st.sidebar.number_input("Multiplier", value=3.0, min_value=0.1, format="%.1f")
st.sidebar.subheader("Risk Management")
sl_points = st.sidebar.number_input("Stop Loss", value=5.0, min_value=0.1)
tp_points = st.sidebar.number_input("Take Profit", value=15.0, min_value=0.1)
lookahead_candles = st.sidebar.number_input("Lookahead Candles", value=20, min_value=1, max_value=200)

st.subheader("Data Download")
st.info(f"Downloading {period} of 5m / 15m / 1h / 1d data for {ticker}...")

df_5m, df_15m = pd.DataFrame(), pd.DataFrame()
interval_used = None

try:
    # 1️⃣ Try 5-minute
    df_5m = yf.download(ticker, interval="5m", period=period, auto_adjust=False)
    interval_used = "5m"

    # 2️⃣ Fallback to 15m
    if df_5m.empty:
        st.warning("No 5-minute data found. Trying 15-minute...")
        df_5m = yf.download(ticker, interval="15m", period=period, auto_adjust=False)
        interval_used = "15m"

    # 3️⃣ Fallback to 1h
    if df_5m.empty:
        st.warning("No 15-minute data found. Trying 1-hour...")
        df_5m = yf.download(ticker, interval="1h", period=period, auto_adjust=False)
        interval_used = "1h"

    # 4️⃣ Fallback to daily
    if df_5m.empty:
        st.warning("No intraday data found. Using daily candles instead.")
        df_5m = yf.download(ticker, interval="1d", period=period, auto_adjust=False)
        interval_used = "1d"

    if df_5m.empty:
        st.error(f"❌ No data available for {ticker}. Try again later or different symbol.")
        st.stop()

    st.success(f"✅ Successfully downloaded {len(df_5m)} rows ({interval_used} interval).")

    # Prepare main DataFrame
    if isinstance(df_5m.columns, pd.MultiIndex):
        df_5m.columns = df_5m.columns.droplevel(1)
    df_5m.reset_index(inplace=True)
    df_5m.rename(columns={"index": "Datetime", "Date": "Datetime"}, inplace=True, errors="ignore")
    df_5m["Datetime"] = pd.to_datetime(df_5m["Datetime"])
    df_5m.set_index("Datetime", inplace=True)
    df_5m.dropna(inplace=True)

    # Secondary 15m/1h for trend confirmation (skip if daily)
    if interval_used != "1d":
        next_interval = "15m" if interval_used == "5m" else "1h"
        df_15m = yf.download(ticker, interval=next_interval, period=period, auto_adjust=False)
        if isinstance(df_15m.columns, pd.MultiIndex):
            df_15m.columns = df_15m.columns.droplevel(1)
        df_15m.reset_index(inplace=True)
        df_15m.rename(columns={"index": "Datetime", "Date": "Datetime"}, inplace=True, errors="ignore")
        df_15m["Datetime"] = pd.to_datetime(df_15m["Datetime"])
        df_15m.set_index("Datetime", inplace=True)
        df_15m.dropna(inplace=True)
        st.success(f"✅ Secondary timeframe ({next_interval}) loaded successfully.")
    else:
        df_15m = df_5m.copy()

except Exception as e:
    st.error(f"⚠️ Data download failed: {e}")
    st.stop()

# --- Supertrend Calculation ---
st.subheader("Supertrend Calculation")
try:
    st_5m = ta.supertrend(df_5m["High"], df_5m["Low"], df_5m["Close"], length=st_length, multiplier=st_multiplier)
    df_5m["Supertrend"] = st_5m[f"SUPERT_{st_length}_{st_multiplier}"]

    st_15m = ta.supertrend(df_15m["High"], df_15m["Low"], df_15m["Close"], length=st_length, multiplier=st_multiplier)
    df_15m["Supertrend"] = st_15m[f"SUPERT_{st_length}_{st_multiplier}"]

    # Align secondary trend
    df_5m["Supertrend_Secondary"] = df_15m["Supertrend"].resample(interval_used).ffill().reindex(df_5m.index, method="ffill")

    st.success("✅ Supertrend indicators calculated.")
except Exception as e:
    st.error(f"Supertrend calculation failed: {e}")
    st.stop()

# --- Backtesting ---
st.subheader("Backtesting Simulation")
trades, position, entry_price = [], None, 0.0

for i in range(len(df_5m)):
    row = df_5m.iloc[i]
    if pd.isna(row["Close"]) or pd.isna(row["Supertrend"]) or pd.isna(row["Supertrend_Secondary"]):
        continue

    close = row["Close"]
    primary = row["Supertrend"]
    secondary = row["Supertrend_Secondary"]

    buy_signal = (close > primary) and (close > secondary)
    sell_signal = (close < primary) and (close < secondary)

    if position is None:
        if buy_signal:
            position = "Buy"
            entry_price = close
            entry_time = row.name
        elif sell_signal:
            position = "Sell"
            entry_price = close
            entry_time = row.name
        continue

    # Exit check
    exit_price, exit_time, pnl = None, None, 0
    if position == "Buy":
        if close <= entry_price - sl_points or close >= entry_price + tp_points or sell_signal:
            exit_price = close
            exit_time = row.name
            pnl = exit_price - entry_price
    elif position == "Sell":
        if close >= entry_price + sl_points or close <= entry_price - tp_points or buy_signal:
            exit_price = close
            exit_time = row.name
            pnl = entry_price - exit_price

    if exit_price is not None:
        trades.append({
            "Entry Time": entry_time,
            "Exit Time": exit_time,
            "Direction": position,
            "Entry Price": round(entry_price, 2),
            "Exit Price": round(exit_price, 2),
            "PnL": round(pnl, 2),
        })
        position = None

# --- Results ---
if not trades:
    st.warning("No trades executed. Try different period or parameters.")
    st.stop()

trades_df = pd.DataFrame(trades)
total_pnl = trades_df["PnL"].sum()
win_rate = (trades_df["PnL"] > 0).mean() * 100

st.metric("Total Net PnL", f"{total_pnl:.2f}")
st.metric("Win Rate", f"{win_rate:.2f}%")
st.dataframe(trades_df.tail(20))

# --- Charts ---
st.subheader("Charts")
fig = go.Figure()
fig.add_trace(go.Candlestick(x=df_5m.index, open=df_5m["Open"], high=df_5m["High"], low=df_5m["Low"], close=df_5m["Close"], name="Candles"))
fig.add_trace(go.Scatter(x=df_5m.index, y=df_5m["Supertrend"], mode="lines", name="Primary Supertrend", line=dict(color="blue", width=1.5)))
fig.add_trace(go.Scatter(x=df_5m.index, y=df_5m["Supertrend_Secondary"], mode="lines", name="Secondary Supertrend", line=dict(color="purple", dash="dot")))
fig.update_layout(xaxis_rangeslider_visible=False, title=f"Price Chart ({interval_used})")
st.plotly_chart(fig, use_container_width=True)
