import streamlit as st
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Streamlit Page Config
st.set_page_config(
    page_title="Dual Supertrend Trading Strategy Backtester",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Dual Supertrend Trading Strategy Backtester")
st.markdown("""
Backtests a strategy using Supertrend on two configurable timeframes for **Gold Futures (GC=F)**.

**Buy/Sell conditions:**
- **Buy:** Price closes above both Supertrends  
- **Sell:** Price closes below both Supertrends
""")

# Sidebar Inputs
st.sidebar.header("Strategy Parameters")
symbol = st.sidebar.text_input("Ticker Symbol", "GC=F")
period = st.sidebar.selectbox("Data Period", ["30d", "60d", "90d"], index=1)
primary_tf = st.sidebar.selectbox("Primary Timeframe", ["5m", "15m", "1h"], index=0)
multiplier = st.sidebar.number_input("Supertrend Multiplier", value=3.0, step=0.5)
length = st.sidebar.number_input("Supertrend Length", value=7, step=1)

# Derived timeframe
tf_map = {"5m": "15m", "15m": "1h", "1h": "1d"}
secondary_tf = tf_map.get(primary_tf, "15m")
st.sidebar.write(f"Secondary Timeframe will be: **{secondary_tf}**")

# Data Download
st.subheader("Data Download")
st.info(f"Downloading {period} of {primary_tf} and {secondary_tf} data for {symbol}...")

try:
    df_primary = yf.download(symbol, period=period, interval=primary_tf)
    df_secondary = yf.download(symbol, period=period, interval=secondary_tf)

    if df_primary.empty or df_secondary.empty:
        st.error("❌ DataFrames are empty after download. Adjust parameters or check ticker symbol.")
    else:
        # Supertrend Calculations
        df_primary["ST"] = ta.supertrend(df_primary["High"], df_primary["Low"], df_primary["Close"],
                                         length=length, multiplier=multiplier)["SUPERT_7_3.0"]
        df_secondary["ST"] = ta.supertrend(df_secondary["High"], df_secondary["Low"], df_secondary["Close"],
                                           length=length, multiplier=multiplier)["SUPERT_7_3.0"]

        # Align DataFrames
        df_primary = df_primary.join(df_secondary["ST"], rsuffix="_2", how="inner")

        # Buy/Sell Logic
        df_primary["Signal"] = 0
        df_primary.loc[(df_primary["Close"] > df_primary["ST"]) & (df_primary["Close"] > df_primary["ST_2"]), "Signal"] = 1
        df_primary.loc[(df_primary["Close"] < df_primary["ST"]) & (df_primary["Close"] < df_primary["ST_2"]), "Signal"] = -1

        # Plotly Chart
        st.subheader("📊 Strategy Chart")

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df_primary.index,
            open=df_primary["Open"],
            high=df_primary["High"],
            low=df_primary["Low"],
            close=df_primary["Close"],
            name="Price"
        ))
        fig.add_trace(go.Scatter(
            x=df_primary.index,
            y=df_primary["ST"],
            line=dict(color="orange", width=1.5),
            name=f"Supertrend ({primary_tf})"
        ))
        fig.add_trace(go.Scatter(
            x=df_primary.index,
            y=df_primary["ST_2"],
            line=dict(color="green", width=1.5),
            name=f"Supertrend ({secondary_tf})"
        ))
        fig.add_trace(go.Scatter(
            x=df_primary[df_primary["Signal"] == 1].index,
            y=df_primary[df_primary["Signal"] == 1]["Close"],
            mode="markers",
            marker=dict(symbol="triangle-up", color="lime", size=10),
            name="Buy Signal"
        ))
        fig.add_trace(go.Scatter(
            x=df_primary[df_primary["Signal"] == -1].index,
            y=df_primary[df_primary["Signal"] == -1]["Close"],
            mode="markers",
            marker=dict(symbol="triangle-down", color="red", size=10),
            name="Sell Signal"
        ))
        fig.update_layout(
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            height=700,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig, use_container_width=True)

        # Display Data
        st.subheader("📄 Backtest Data (Last 20 Rows)")
        st.dataframe(df_primary.tail(20))

except Exception as e:
    st.error(f"⚠️ Error: {e}")
