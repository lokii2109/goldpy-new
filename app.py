import sys
import os
import numpy as np  # Import numpy early

# --- START OF PRE-IMPORT PATCH FOR PANDAS_TA / NUMPY ---
try:
    if not hasattr(np, 'NaN'):
        np.NaN = np.nan  # Create an alias for np.nan as np.NaN
        print("Successfully patched numpy.NaN alias.")
    else:
        print("numpy.NaN already exists or patch not needed.")
except Exception as e:
    print(f"Error during numpy.NaN pre-import patch: {e}")
# --- END OF PRE-IMPORT PATCH FOR PANDAS_TA / NUMPY ---

import yfinance as yf
import pandas as pd
import pandas_ta as ta
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# --- Streamlit App UI ---
st.title("📈 Dual Supertrend Trading Strategy Backtester")
st.markdown("Backtests a strategy using Supertrend on 5-minute and 15-minute timeframes for Gold Futures (GC=F).")
st.markdown(
    "Buy/Sell conditions: Current candle closes above/below 5m Supertrend AND current price is above/below 15m Supertrend."
)

# --- Sidebar Configuration ---
st.sidebar.header("Strategy Parameters")
ticker = st.sidebar.text_input("Ticker Symbol", value="XAUUSD=X")
period = st.sidebar.selectbox(
    "Data Period", ["1d", "5d", "30d", "60d", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"],
    index=1
)

st.sidebar.subheader("Supertrend Settings (7, 3.0)")
st_length = st.sidebar.number_input("Supertrend Length", value=7, min_value=1)
st_multiplier = st.sidebar.number_input("Supertrend Multiplier", value=3.0, min_value=0.1, format="%.1f")

st.sidebar.subheader("Risk Management")
sl_points = st.sidebar.number_input("Stop Loss (points)", value=5.0, min_value=0.1, format="%.1f")
tp_points = st.sidebar.number_input("Take Profit (points)", value=15.0, min_value=0.1, format="%.1f")
lookahead_candles = st.sidebar.number_input(
    "Lookahead Candles (5m)", value=20, min_value=1, max_value=200,
    help="Number of 5-min candles to look for SL/TP hit after entry."
)

# --- Step 1: Download Historical Data ---
st.subheader("Data Download")
st.info(f"Downloading {period} of 5-minute and 15-minute data for {ticker}...")

df_5m = pd.DataFrame()
df_15m = pd.DataFrame()

try:
    # --- Primary timeframe (5m with fallback) ---
    df_5m = yf.download(ticker, interval="5m", period=period, auto_adjust=False)

    if df_5m.empty:
        st.warning(f"No 5-minute data found for {ticker} ({period}). Trying 15-minute interval instead...")
        df_5m = yf.download(ticker, interval="15m", period=period, auto_adjust=False)

    if df_5m.empty:
        st.warning(f"No 15-minute data available either. Trying 1-hour interval instead...")
        df_5m = yf.download(ticker, interval="1h", period=period, auto_adjust=False)

    if df_5m.empty:
        st.error(f"Error: No data available for {ticker}. Try a shorter period (like '1d').")
        st.stop()

    # --- Clean primary data ---
    if isinstance(df_5m.columns, pd.MultiIndex):
        df_5m.columns = df_5m.columns.droplevel(1)
    df_5m.reset_index(inplace=True)
    df_5m.rename(columns={'index': 'Datetime', 'Date': 'Datetime'}, inplace=True, errors='ignore')
    df_5m['Datetime'] = pd.to_datetime(df_5m['Datetime'])
    df_5m.set_index('Datetime', inplace=True)
    df_5m.dropna(inplace=True)
    st.success(f"Successfully downloaded {len(df_5m)} rows of intraday data.")

    # --- Secondary timeframe (15m) ---
    df_15m = yf.download(ticker, interval="15m", period=period, auto_adjust=False)
    if df_15m.empty:
        st.warning("No 15-minute data found, trying 1-hour secondary timeframe.")
        df_15m = yf.download(ticker, interval="1h", period=period, auto_adjust=False)

    if df_15m.empty:
        st.error("Error: No secondary timeframe data available.")
        st.stop()

    if isinstance(df_15m.columns, pd.MultiIndex):
        df_15m.columns = df_15m.columns.droplevel(1)
    df_15m.reset_index(inplace=True)
    df_15m.rename(columns={'index': 'Datetime', 'Date': 'Datetime'}, inplace=True, errors='ignore')
    df_15m['Datetime'] = pd.to_datetime(df_15m['Datetime'])
    df_15m.set_index('Datetime', inplace=True)
    df_15m.dropna(inplace=True)
    st.success(f"Successfully downloaded {len(df_15m)} rows of secondary timeframe data.")

except Exception as e:
    st.error(f"Failed to download data: {e}")
    st.stop()

# Ensure numeric OHLC
for col in ['Open', 'High', 'Low', 'Close']:
    df_5m[col] = pd.to_numeric(df_5m[col], errors='coerce')
    df_15m[col] = pd.to_numeric(df_15m[col], errors='coerce')
df_5m.dropna(inplace=True)
df_15m.dropna(inplace=True)

# --- Step 2: Compute Supertrend ---
st.subheader("Supertrend Calculation")
with st.spinner(f"Calculating Supertrend ({st_length},{st_multiplier})..."):
    st_5m = ta.supertrend(df_5m['High'], df_5m['Low'], df_5m['Close'], length=st_length, multiplier=st_multiplier)
    st_15m = ta.supertrend(df_15m['High'], df_15m['Low'], df_15m['Close'], length=st_length, multiplier=st_multiplier)

    if st_5m is None or st_15m is None or st_5m.empty or st_15m.empty:
        st.error("Error: Supertrend calculation failed.")
        st.stop()

    df_5m[f'SUPERT_{st_length}_{st_multiplier}'] = st_5m[f'SUPERT_{st_length}_{st_multiplier}']
    df_15m[f'SUPERT_{st_length}_{st_multiplier}'] = st_15m[f'SUPERT_{st_length}_{st_multiplier}']

    df_5m.dropna(inplace=True)
    df_15m.dropna(inplace=True)
st.success("Supertrend calculations complete.")

# --- Step 3: Align Secondary to Primary Timeframe ---
st.subheader("Data Alignment")
st_15m_resampled = df_15m[f'SUPERT_{st_length}_{st_multiplier}'].resample('5min').ffill()
df_5m = df_5m.merge(st_15m_resampled.rename(f'SUPERT_15m_{st_length}_{st_multiplier}'),
                    left_index=True, right_index=True, how='inner')
df_5m.dropna(inplace=True)
st.success("Supertrend data aligned and prepared.")

# --- Step 4: Backtesting ---
st.subheader("Backtesting Strategy")
trades = []
position = None
entry_price = 0.0
entry_time = None
entry_direction = None

with st.spinner("Running backtest..."):
    for i in range(len(df_5m)):
        candle = df_5m.iloc[i]
        close = candle['Close']
        st5 = candle[f'SUPERT_{st_length}_{st_multiplier}']
        st15 = candle[f'SUPERT_15m_{st_length}_{st_multiplier}']

        if pd.isna(close) or pd.isna(st5) or pd.isna(st15):
            continue

        buy_condition = (close > st5) and (close > st15)
        sell_condition = (close < st5) and (close < st15)

        if position is None:
            if buy_condition:
                position = 'Buy'
                entry_price = close
                entry_time = candle.name
                entry_direction = 'Buy'
            elif sell_condition:
                position = 'Sell'
                entry_price = close
                entry_time = candle.name
                entry_direction = 'Sell'

        if position is not None and candle.name == entry_time:
            continue

        if position is not None:
            trade_closed = False
            exit_price = 0.0
            exit_time = None
            pnl = 0

            # Look ahead
            if (i + 1 + lookahead_candles) > len(df_5m):
                lookahead = df_5m.iloc[i + 1:].copy()
            else:
                lookahead = df_5m.iloc[i + 1: i + 1 + lookahead_candles].copy()

            if lookahead.empty:
                continue

            for _, row in lookahead.iterrows():
                low, high = row['Low'], row['High']
                if position == 'Buy':
                    sl_hit = entry_price - sl_points
                    tp_hit = entry_price + tp_points
                    if low <= sl_hit:
                        exit_price, exit_time = sl_hit, row.name
                        pnl = exit_price - entry_price
                        trade_closed = True
                        break
                    elif high >= tp_hit:
                        exit_price, exit_time = tp_hit, row.name
                        pnl = exit_price - entry_price
                        trade_closed = True
                        break
                elif position == 'Sell':
                    sl_hit = entry_price + sl_points
                    tp_hit = entry_price - tp_points
                    if high >= sl_hit:
                        exit_price, exit_time = sl_hit, row.name
                        pnl = entry_price - exit_price
                        trade_closed = True
                        break
                    elif low <= tp_hit:
                        exit_price, exit_time = tp_hit, row.name
                        pnl = entry_price - exit_price
                        trade_closed = True
                        break

            if trade_closed:
                trades.append({
                    'Entry Time': entry_time,
                    'Entry Price': round(entry_price, 2),
                    'Direction': entry_direction,
                    'Exit Time': exit_time,
                    'Exit Price': round(exit_price, 2),
                    'PnL': round(pnl, 2),
                    'Result': 'Profit' if pnl > 0 else 'Loss'
                })
                position = None
            else:
                if (position == 'Buy' and sell_condition) or (position == 'Sell' and buy_condition):
                    exit_price = close
                    pnl = close - entry_price if position == 'Buy' else entry_price - close
                    exit_time = candle.name
                    trades.append({
                        'Entry Time': entry_time,
                        'Entry Price': round(entry_price, 2),
                        'Direction': entry_direction,
                        'Exit Time': exit_time,
                        'Exit Price': round(exit_price, 2),
                        'PnL': round(pnl, 2),
                        'Result': 'Counter-Signal Exit'
                    })
                    position = None

st.success("Backtest complete.")

# --- Results ---
trades_df = pd.DataFrame(trades)
if trades_df.empty:
    st.warning("No trades executed. Try adjusting parameters.")
    st.stop()

st.subheader("Backtest Summary")
total_pnl = trades_df['PnL'].sum()
wins = len(trades_df[trades_df['Result'] == 'Profit'])
losses = len(trades_df[trades_df['Result'] == 'Loss'])
total = len(trades_df)
win_rate = (wins / (wins + losses)) * 100 if (wins + losses) > 0 else 0

st.metric("Total Net PnL", f"{total_pnl:.2f}")
st.metric("Win Rate", f"{win_rate:.2f}%")
st.write(f"Total Trades: {total}")

st.dataframe(trades_df.tail(30))

# --- Charts ---
fig_pie = px.pie(trades_df, names='Result', title="Trade Outcomes",
                 color_discrete_map={'Profit':'green','Loss':'red','Counter-Signal Exit':'orange'})
st.plotly_chart(fig_pie, use_container_width=True)

trades_df['Cumulative PnL'] = trades_df['PnL'].cumsum()
fig_pnl = px.line(trades_df, x='Exit Time', y='Cumulative PnL', title="Cumulative PnL Over Time")
st.plotly_chart(fig_pnl, use_container_width=True)

fig_candle = go.Figure(data=[go.Candlestick(
    x=df_5m.index, open=df_5m['Open'], high=df_5m['High'], low=df_5m['Low'], close=df_5m['Close'], name='Candles'
)])
fig_candle.add_trace(go.Scatter(x=df_5m.index, y=df_5m[f'SUPERT_{st_length}_{st_multiplier}'],
                                name='Primary Supertrend', line=dict(color='blue')))
fig_candle.add_trace(go.Scatter(x=df_5m.index, y=df_5m[f'SUPERT_15m_{st_length}_{st_multiplier}'],
                                name='Secondary Supertrend', line=dict(color='purple', dash='dot')))
fig_candle.update_layout(title="Price Chart with Supertrends", xaxis_rangeslider_visible=False)
st.plotly_chart(fig_candle, use_container_width=True)
