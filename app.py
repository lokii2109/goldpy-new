import streamlit as st
import yfinance as yf
import pandas_ta as ta
import plotly.graph_objects as go

# App title and info
st.set_page_config(page_title="Gold Trading Strategy Backtester", layout="wide")
st.title("🏆 Automated Gold Trading Strategy Backtester")
st.caption("Backtest EMA crossover strategy on live gold futures data")

# Symbol input
symbol = st.text_input("Enter the symbol (default: GC=F for Gold Futures):", "GC=F")

# Interval selection
interval = st.selectbox("Select Time Interval", ["5m", "15m", "1h", "1d"], index=0)

st.write(f"📥 Downloading 30d of {interval} data for **{symbol}**...")

try:
    # Download market data
    df = yf.download(symbol, period="30d", interval=interval)

    # Handle empty or missing data
    if df is None or df.empty:
        st.error("⚠️ No data fetched! Try again later or use a higher interval (like 1h or 1d).")

    else:
        # Calculate EMAs
        df["EMA20"] = ta.ema(df["Close"], length=20)
        df["EMA50"] = ta.ema(df["Close"], length=50)

        # Generate buy/sell signals
        df["Signal"] = 0
        df.loc[df["EMA20"] > df["EMA50"], "Signal"] = 1
        df.loc[df["EMA20"] < df["EMA50"], "Signal"] = -1

        # Plotly chart
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Candlestick"
        ))

        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["EMA20"],
            mode="lines",
            name="EMA 20",
            line=dict(width=1.5)
        ))

        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["EMA50"],
            mode="lines",
            name="EMA 50",
            line=dict(width=1.5)
        ))

        # Highlight buy/sell zones
        buy_signals = df[df["Signal"] == 1]
        sell_signals = df[df["Signal"] == -1]

        fig.add_trace(go.Scatter(
            x=buy_signals.index,
            y=buy_signals["Close"],
            mode="markers",
            marker=dict(color="green", size=8, symbol="triangle-up"),
            name="Buy Signal"
        ))

        fig.add_trace(go.Scatter(
            x=sell_signals.index,
            y=sell_signals["Close"],
            mode="markers",
            marker=dict(color="red", size=8, symbol="triangle-down"),
            name="Sell Signal"
        ))

        fig.update_layout(
            title=f"{symbol} ({interval}) - EMA Strategy Chart",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            height=700,
        )

        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"⚠️ Error: {e}")
