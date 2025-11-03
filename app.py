import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# --- Manual Supertrend Calculation Functions ---
# Function to calculate Average True Range (ATR)
def calculate_atr(df, length=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    atr = true_range.ewm(span=length, adjust=False).mean()
    return atr


# Function to calculate Supertrend
def calculate_supertrend(df, length=7, multiplier=3.0):
    df_copy = df.copy()
    atr = calculate_atr(df_copy, length)

    basic_upper_band = ((df_copy['High'] + df_copy['Low']) / 2) + (multiplier * atr)
    basic_lower_band = ((df_copy['High'] + df_copy['Low']) / 2) - (multiplier * atr)

    final_upper_band = pd.Series(index=df_copy.index, dtype='float64')
    final_lower_band = pd.Series(index=df_copy.index, dtype='float64')
    supertrend = pd.Series(index=df_copy.index, dtype='float64')

    # Initialize first valid values
    first_valid_idx = atr.first_valid_index()
    if first_valid_idx is not None:
        first_valid_pos = df_copy.index.get_loc(first_valid_idx)
        if first_valid_pos < len(df_copy):
            final_upper_band.iloc[first_valid_pos] = basic_upper_band.iloc[first_valid_pos]
            final_lower_band.iloc[first_valid_pos] = basic_lower_band.iloc[first_valid_pos]
            # Initial Supertrend direction based on first valid close vs bands
            if df_copy['Close'].iloc[first_valid_pos] > basic_upper_band.iloc[first_valid_pos]:
                supertrend.iloc[first_valid_pos] = basic_lower_band.iloc[first_valid_pos]
            else:
                supertrend.iloc[first_valid_pos] = basic_upper_band.iloc[first_valid_pos]

    for i in range(len(df_copy)):  # Iterate through all indices
        if i == 0 or pd.isna(supertrend.iloc[i - 1]):  # Handle initial NaN values or first valid entry
            if not pd.isna(basic_upper_band.iloc[i]):
                final_upper_band.iloc[i] = basic_upper_band.iloc[i]
                final_lower_band.iloc[i] = basic_lower_band.iloc[i]
                supertrend.iloc[i] = basic_upper_band.iloc[i]  # Default initial state
            continue  # Skip to next iteration if still NaN or first candle

        # Final Upper Band logic
        if basic_upper_band.iloc[i] < final_upper_band.iloc[i - 1] or df_copy['Close'].iloc[i - 1] > \
                final_upper_band.iloc[i - 1]:
            final_upper_band.iloc[i] = basic_upper_band.iloc[i]
        else:
            final_upper_band.iloc[i] = final_upper_band.iloc[i - 1]

        # Final Lower Band logic
        if basic_lower_band.iloc[i] > final_lower_band.iloc[i - 1] or df_copy['Close'].iloc[i - 1] < \
                final_lower_band.iloc[i - 1]:
            final_lower_band.iloc[i] = basic_lower_band.iloc[i]
        else:
            final_lower_band.iloc[i] = final_lower_band.iloc[i - 1]

        # Supertrend logic
        # Determine the trend direction from the previous candle's Supertrend state
        # If previous Supertrend was the final_upper_band, it means downtrend or flat
        # If previous Supertrend was the final_lower_band, it means uptrend or flat

        # Check for flip from downtrend to uptrend
        if df_copy['Close'].iloc[i] > final_lower_band.iloc[i]:
            supertrend.iloc[i] = final_lower_band.iloc[i]
        # Check for flip from uptrend to downtrend
        elif df_copy['Close'].iloc[i] < final_upper_band.iloc[i]:
            supertrend.iloc[i] = final_upper_band.iloc[i]
        else:  # Maintain previous Supertrend value if no flip
            supertrend.iloc[i] = supertrend.iloc[i - 1]

    return supertrend


# --- Streamlit App UI ---
st.title("📈 Dual Supertrend Trading Strategy Backtester")
st.markdown("Backtests a strategy using Supertrend on two configurable timeframes for Gold Futures (GC=F).")
st.markdown(
    "Buy/Sell conditions: Current candle closes above/below Primary TF Supertrend AND current price is above/below Secondary TF Supertrend.")

# --- Sidebar Configuration ---
st.sidebar.header("Strategy Parameters")
ticker = st.sidebar.text_input("Ticker Symbol", value="GC=F")
period = st.sidebar.selectbox("Data Period", ["30d", "60d", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"],
                              index=1)

# Dynamic Timeframe Selection
st.sidebar.subheader("Timeframe Settings")
primary_tf_str = st.sidebar.selectbox("Primary Timeframe", ["5m", "15m", "30m", "60m"], index=0)
# Map selected string to yfinance interval
primary_interval_map = {
    "5m": "5m", "15m": "15m", "30m": "30m", "60m": "60m"
}
primary_interval = primary_interval_map[primary_tf_str]

# Determine secondary interval based on primary (e.g., 3x primary)
secondary_interval_map = {
    "5m": "15m",
    "15m": "60m",  # yfinance doesn't have 45m, use 60m
    "30m": "90m",
    "60m": "1d"
}
secondary_interval = secondary_interval_map[primary_tf_str]
st.sidebar.write(f"Secondary Timeframe will be: **{secondary_interval}**")

st.sidebar.subheader("Supertrend Settings")
st_length = st.sidebar.number_input("Supertrend Length", value=7, min_value=1)
st_multiplier = st.sidebar.number_input("Supertrend Multiplier", value=3.0, min_value=0.1, format="%.1f")

st.sidebar.subheader("Risk Management")
sl_tp_method = st.sidebar.radio("SL/TP Method", ["Fixed Points", "ATR Multiplier"])
if sl_tp_method == "Fixed Points":
    sl_points = st.sidebar.number_input("Stop Loss (points)", value=5.0, min_value=0.1, format="%.1f")
    tp_points = st.sidebar.number_input("Take Profit (points)", value=15.0, min_value=0.1, format="%.1f")
    atr_length_sl_tp = None  # Not used for fixed points
    sl_multiplier_atr = None
    tp_multiplier_atr = None
else:  # ATR Multiplier
    atr_length_sl_tp = st.sidebar.number_input("ATR Length for SL/TP", value=14, min_value=1)
    sl_multiplier_atr = st.sidebar.number_input("SL ATR Multiplier", value=2.0, min_value=0.1, format="%.1f")
    tp_multiplier_atr = st.sidebar.number_input("TP ATR Multiplier", value=6.0, min_value=0.1, format="%.1f")
    sl_points = None  # Not used for ATR multiplier
    tp_points = None

lookahead_candles = st.sidebar.number_input(f"Lookahead Candles ({primary_tf_str})", value=20, min_value=1,
                                            max_value=200,
                                            help=f"Number of {primary_tf_str} candles to look for SL/TP hit after entry.")

st.sidebar.subheader("Entry Filters")
enable_time_filter = st.sidebar.checkbox("Enable Time of Day Filter", value=False)
trading_start_hour = 0
trading_end_hour = 23
if enable_time_filter:
    trading_start_hour = st.sidebar.slider("Trading Start Hour (24h)", 0, 23, 9)
    trading_end_hour = st.sidebar.slider("Trading End Hour (24h)", 0, 23, 17)
    if trading_start_hour >= trading_end_hour:
        st.sidebar.error("Start hour must be before End hour.")
        st.stop()

# Number of trades to display
st.sidebar.subheader("Display Options")
num_trades_to_display = st.sidebar.number_input("Number of Last Trades to Display", value=30, min_value=1,
                                                max_value=500)

# --- Step 1: Download Historical Data for both timeframes ---
st.subheader("Data Download")
st.info(f"Downloading {period} of {primary_interval} and {secondary_interval} data for {ticker}...")


@st.cache_data  # Cache data download
def get_data(ticker_symbol, p_interval, s_interval, data_period):
    df_p = yf.download(ticker_symbol, interval=p_interval, period=data_period, auto_adjust=False)
    df_s = yf.download(ticker_symbol, interval=s_interval, period=data_period, auto_adjust=False)

    # Process Primary DF
    if isinstance(df_p.columns, pd.MultiIndex):
        df_p.columns = df_p.columns.droplevel(1)
    df_p.reset_index(inplace=True)
    df_p.rename(columns={'index': 'Datetime', 'Date': 'Datetime'}, inplace=True, errors='ignore')
    df_p['Datetime'] = pd.to_datetime(df_p['Datetime'])
    df_p.set_index('Datetime', inplace=True)
    df_p.dropna(inplace=True)

    # Process Secondary DF
    if isinstance(df_s.columns, pd.MultiIndex):
        df_s.columns = df_s.columns.droplevel(1)
    df_s.reset_index(inplace=True)
    df_s.rename(columns={'index': 'Datetime', 'Date': 'Datetime'}, inplace=True, errors='ignore')
    df_s['Datetime'] = pd.to_datetime(df_s['Datetime'])
    df_s.set_index('Datetime', inplace=True)
    df_s.dropna(inplace=True)

    # Ensure OHLC columns are numeric
    for df_temp in [df_p, df_s]:
        for col in ['Open', 'High', 'Low', 'Close']:
            df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce')
        df_temp.dropna(inplace=True)

    return df_p, df_s


df_primary, df_secondary = get_data(ticker, primary_interval, secondary_interval, period)

if df_primary.empty or df_secondary.empty:
    st.error("Error: DataFrames are empty after download and cleanup. Adjust parameters or check data integrity.")
    st.stop()

st.success(
    f"Successfully downloaded {len(df_primary)} rows of {primary_interval} data and {len(df_secondary)} rows of {secondary_interval} data.")

# --- Step 2: Compute Supertrend Indicator for both timeframes ---
st.subheader("Supertrend Calculation")


@st.cache_data  # Cache Supertrend calculation
def get_supertrend_data(df_p_in, df_s_in, st_len, st_mult):
    df_p_copy = df_p_in.copy()
    df_s_copy = df_s_in.copy()

    # Primary Timeframe Supertrend
    df_p_copy[f'SUPERT_{st_len}_{st_mult}'] = calculate_supertrend(df_p_copy, length=st_len, multiplier=st_mult)
    df_p_copy.dropna(subset=[f'SUPERT_{st_len}_{st_mult}'], inplace=True)

    # Secondary Timeframe Supertrend
    df_s_copy[f'SUPERT_{st_len}_{st_mult}'] = calculate_supertrend(df_s_copy, length=st_len, multiplier=st_mult)
    df_s_copy.dropna(subset=[f'SUPERT_{st_len}_{st_mult}'], inplace=True)

    return df_p_copy, df_s_copy


df_primary, df_secondary = get_supertrend_data(df_primary, df_secondary, st_length, st_multiplier)

if df_primary.empty or df_secondary.empty:
    st.error("Error: DataFrames are empty after Supertrend calculation and NaN removal. Exiting.")
    st.stop()
st.success("Supertrend calculations complete.")

# --- Step 3: Align Secondary Supertrend to Primary timeframe ---
st.subheader("Data Alignment")
with st.spinner(f"Aligning {secondary_interval} Supertrend to {primary_interval} timeframe..."):
    primary_resample_freq = primary_interval.replace('m', 'min').replace('h', 'H')

    st_secondary_resampled = df_secondary[f'SUPERT_{st_length}_{st_multiplier}'].resample(primary_resample_freq).ffill()

    df_primary = df_primary.merge(st_secondary_resampled.rename(f'SUPERT_Secondary_{st_length}_{st_multiplier}'),
                                  left_index=True, right_index=True, how='inner')
    df_primary.dropna(inplace=True)

    if df_primary.empty:
        st.error("Error: DataFrame became empty after aligning Supertrend data. Adjust period or check data integrity.")
        st.stop()
st.success("Supertrend data aligned and prepared.")

# --- Step 4: Generate Buy/Sell Signals and Backtest ---
st.subheader("Backtesting Strategy")
trades = []
position = None  # 'Buy', 'Sell', or None
entry_price = 0.0
entry_time = None
entry_direction = None

with st.spinner("Running backtest simulation..."):
    for i in range(len(df_primary)):  # Iterate over the primary timeframe DataFrame
        current_candle = df_primary.iloc[i]

        if pd.isna(current_candle['Close']) or \
                pd.isna(current_candle[f'SUPERT_{st_length}_{st_multiplier}']) or \
                pd.isna(current_candle[f'SUPERT_Secondary_{st_length}_{st_multiplier}']):
            continue

        close_primary = current_candle['Close']
        supertrend_primary = current_candle[f'SUPERT_{st_length}_{st_multiplier}']
        supertrend_secondary = current_candle[f'SUPERT_Secondary_{st_length}_{st_multiplier}']

        # Time of Day Filter
        if enable_time_filter:
            current_hour = current_candle.name.hour
            if not (trading_start_hour <= current_hour < trading_end_hour):
                continue  # Skip trade entry if outside trading hours

        # Buy Condition
        buy_condition = (close_primary > supertrend_primary) and (close_primary > supertrend_secondary)

        # Sell Condition
        sell_condition = (close_primary < supertrend_primary) and (close_primary < supertrend_secondary)

        if position is None:  # No open position, look for entry
            if buy_condition:
                position = 'Buy'
                entry_price = close_primary
                entry_time = current_candle.name  # Datetime index
                entry_direction = 'Buy'
            elif sell_condition:
                position = 'Sell'
                entry_price = close_primary
                entry_time = current_candle.name  # Datetime index
                entry_direction = 'Sell'

        # If a position is open, check for SL/TP or counter-signal
        # Skip exit check on the entry candle itself
        if position is not None and current_candle.name == entry_time:
            continue

        if position is not None:
            trade_closed = False
            exit_price = 0.0
            exit_time = None
            pnl = 0

            # Determine SL/TP prices based on chosen method
            current_sl_points = sl_points
            current_tp_points = tp_points
            if sl_tp_method == "ATR Multiplier":
                # Calculate ATR for the current candle for dynamic SL/TP
                current_atr = \
                calculate_atr(df_primary.iloc[i - atr_length_sl_tp + 1: i + 1], length=atr_length_sl_tp).iloc[-1]
                if pd.isna(current_atr):  # If ATR is NaN, cannot set dynamic SL/TP, close trade
                    exit_price = close_primary
                    pnl = exit_price - entry_price if position == 'Buy' else entry_price - exit_price
                    exit_time = current_candle.name
                    trades.append({
                        'Entry Time': entry_time,
                        'Entry Price': round(entry_price, 2),
                        'Direction': entry_direction,
                        'Exit Time': exit_time,
                        'Exit Price': round(exit_price, 2),
                        'PnL': round(pnl, 2),
                        'Result': 'ATR Calc Error/Timeout'  # Indicate specific exit reason
                    })
                    position = None
                    continue

                current_sl_points = current_atr * sl_multiplier_atr
                current_tp_points = current_atr * tp_multiplier_atr

            sl_hit_price = entry_price - current_sl_points if position == 'Buy' else entry_price + current_sl_points
            tp_hit_price = entry_price + current_tp_points if position == 'Buy' else entry_price - current_tp_points

            # Look ahead for SL/TP hit in the next LOOKAHEAD_CANDLES
            # Ensure there are enough candles for the lookahead slice
            if (i + 1 + lookahead_candles) > len(df_primary):
                lookahead_for_exit_df = df_primary.iloc[i + 1:].copy()  # Slice till end if not enough candles
            else:
                lookahead_for_exit_df = df_primary.iloc[i + 1: i + 1 + lookahead_candles].copy()

            # If no lookahead candles available, handle as a timeout at the very end of data
            if lookahead_for_exit_df.empty:
                # This specific case will be handled by the final open position check outside the loop
                continue

            for exit_idx, exit_row in lookahead_for_exit_df.iterrows():
                current_lookahead_low = float(exit_row['Low'])
                current_lookahead_high = float(exit_row['High'])

                if position == 'Buy':
                    if current_lookahead_low <= sl_hit_price:
                        exit_price = sl_hit_price
                        pnl = exit_price - entry_price
                        exit_time = exit_row.name
                        trade_closed = True
                        break
                    elif current_lookahead_high >= tp_hit_price:
                        exit_price = tp_hit_price
                        pnl = exit_price - entry_price
                        exit_time = exit_row.name
                        trade_closed = True
                        break

                elif position == 'Sell':
                    if current_lookahead_high >= sl_hit_price:
                        exit_price = sl_hit_price
                        pnl = entry_price - exit_price
                        exit_time = exit_row.name
                        trade_closed = True
                        break
                    elif current_lookahead_low <= tp_hit_price:
                        exit_price = tp_hit_price
                        pnl = entry_price - exit_price
                        exit_time = exit_row.name
                        trade_closed = True
                        break

            # If trade was closed within lookahead window by SL/TP
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
                position = None  # Reset position after closing trade
            else:
                # If not closed by SL/TP within lookahead, check for counter-signal on current candle
                # This is important to prevent holding positions indefinitely
                if (position == 'Buy' and sell_condition) or \
                        (position == 'Sell' and buy_condition):
                    exit_price = close_primary  # Close at current candle's close due to counter-signal
                    pnl = exit_price - entry_price if position == 'Buy' else entry_price - exit_price
                    exit_time = current_candle.name
                    trades.append({
                        'Entry Time': entry_time,
                        'Entry Price': round(entry_price, 2),
                        'Direction': entry_direction,
                        'Exit Time': exit_time,
                        'Exit Price': round(exit_price, 2),
                        'PnL': round(pnl, 2),
                        'Result': 'Counter-Signal Exit'
                    })
                    position = None  # Reset position after closing trade

                # If still open after counter-signal check and end of loop, it's a timeout
                # This handles the very last open position if no exit condition is met
                elif i == len(df_primary) - 1:  # If this is the last candle in the dataframe
                    exit_price = close_primary
                    pnl = exit_price - entry_price if position == 'Buy' else entry_price - close_primary  # Use close_primary for final PnL calc
                    exit_time = current_candle.name
                    trades.append({
                        'Entry Time': entry_time,
                        'Entry Price': round(entry_price, 2),
                        'Direction': entry_direction,
                        'Exit Time': exit_time,
                        'Exit Price': round(exit_price, 2),
                        'PnL': round(pnl, 2),
                        'Result': 'Timeout'
                    })
                    position = None  # Reset position

st.success("Backtest simulation complete.")

# --- Convert trades to DataFrame and display results ---
trades_df = pd.DataFrame(trades)

if trades_df.empty:
    st.warning("No trades were executed based on the defined strategy and parameters. Adjust parameters or data range.")
else:
    st.subheader("Backtest Summary")

    total_pnl = trades_df['PnL'].sum()
    gross_profit = trades_df[trades_df['PnL'] > 0]['PnL'].sum()
    gross_loss = trades_df[trades_df['PnL'] < 0]['PnL'].sum()  # This will be negative

    wins = len(trades_df[trades_df['Result'] == 'Profit'])
    losses = len(trades_df[trades_df['Result'] == 'Loss'])
    timeouts = len(trades_df[trades_df['Result'] == 'Timeout'])
    counter_exits = len(trades_df[trades_df['Result'] == 'Counter-Signal Exit'])
    atr_calc_errors = len(trades_df[trades_df['Result'] == 'ATR Calc Error/Timeout'])

    total_trades = len(trades_df)

    # Calculate win rate for Profit vs Loss trades only
    total_win_loss_trades = wins + losses
    win_rate = (wins / total_win_loss_trades) * 100 if total_win_loss_trades > 0 else 0.0

    # Profit Factor
    profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else float('inf')

    # Average Win/Loss
    avg_win = gross_profit / wins if wins > 0 else 0.0
    avg_loss = gross_loss / losses if losses > 0 else 0.0

    # Max Drawdown
    trades_df['Cumulative PnL'] = trades_df['PnL'].cumsum()
    peak = trades_df['Cumulative PnL'].expanding(min_periods=1).max()
    drawdown = trades_df['Cumulative PnL'] - peak
    max_drawdown = drawdown.min() if not drawdown.empty else 0.0

    st.metric("Total Net PnL (points)", f"{total_pnl:.2f}")
    st.metric("Win Rate (Profit vs Loss)", f"{win_rate:.2f}%")
    st.metric("Profit Factor", f"{profit_factor:.2f}")
    st.metric("Max Drawdown (points)", f"{max_drawdown:.2f}")

    st.write(f"Total Trades: {total_trades}")
    st.write(f"Gross Profit: {gross_profit:.2f}")
    st.write(f"Gross Loss: {gross_loss:.2f}")
    st.write(f"Average Win: {avg_win:.2f}")
    st.write(f"Average Loss: {avg_loss:.2f}")
    st.write(f"Profitable Trades: {wins}")
    st.write(f"Losing Trades: {losses}")
    st.write(f"Counter-Signal Exits: {counter_exits}")
    st.write(f"Timeout Trades: {timeouts}")
    st.write(f"ATR Calculation Error Exits: {atr_calc_errors}")

    # --- Trade Log (Dynamic Number of Trades) ---
    st.subheader(f"Trade Log (Last {num_trades_to_display} Trades)")
    st.dataframe(trades_df.tail(num_trades_to_display))

    # --- CSV Export ---
    csv_export = trades_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Trades as CSV",
        data=csv_export,
        file_name="dual_supertrend_trades.csv",
        mime="text/csv",
    )

    # --- Trade Outcome Distribution Chart ---
    st.subheader("Trade Outcome Distribution")
    outcome_counts = trades_df['Result'].value_counts().reset_index()
    outcome_counts.columns = ['Outcome', 'Count']
    fig_outcome_pie = px.pie(outcome_counts,
                             values='Count',
                             names='Outcome',
                             title="Distribution of Trade Outcomes",
                             color_discrete_map={'Profit': 'green', 'Loss': 'red', 'Counter-Signal Exit': 'orange',
                                                 'Timeout': 'blue', 'ATR Calc Error/Timeout': 'gray'})
    st.plotly_chart(fig_outcome_pie, use_container_width=True)

    # --- Cumulative PnL Chart ---
    fig_cumulative_pnl = px.line(trades_df, x='Exit Time', y='Cumulative PnL',
                                 title="Cumulative PnL Over Time",
                                 labels={'Cumulative PnL': 'Cumulative PnL (points)'})
    st.plotly_chart(fig_cumulative_pnl, use_container_width=True)

    # --- Candlestick Chart with Supertrends and Trade Markers ---
    st.subheader(f"Gold Price ({primary_interval} Candles with Supertrends & Trades)")
    fig_candlestick = go.Figure(data=[go.Candlestick(
        x=df_primary.index,  # Use the Datetime index for x-axis
        open=df_primary['Open'],
        high=df_primary['High'],
        low=df_primary['Low'],
        close=df_primary['Close'],
        name=f'{primary_interval} Candles'
    )])

    # Add Primary Supertrend
    fig_candlestick.add_trace(go.Scatter(x=df_primary.index, y=df_primary[f'SUPERT_{st_length}_{st_multiplier}'],
                                         mode='lines',
                                         name=f'{primary_interval} Supertrend ({st_length},{st_multiplier})',
                                         line=dict(color='blue', width=1.5)))

    # Add Secondary Supertrend (aligned to primary timeframe)
    fig_candlestick.add_trace(
        go.Scatter(x=df_primary.index, y=df_primary[f'SUPERT_Secondary_{st_length}_{st_multiplier}'],
                   mode='lines', name=f'{secondary_interval} Supertrend ({st_length},{st_multiplier})',
                   line=dict(color='purple', width=1.5, dash='dot')))

    # Add Trade Entry Markers
    entry_trades = trades_df[trades_df['Direction'] == 'Buy']
    fig_candlestick.add_trace(go.Scatter(
        x=entry_trades['Entry Time'],
        y=entry_trades['Entry Price'],
        mode='markers',
        marker=dict(symbol='triangle-up', size=10, color='green'),
        name='Buy Entries'
    ))

    entry_trades = trades_df[trades_df['Direction'] == 'Sell']
    fig_candlestick.add_trace(go.Scatter(
        x=entry_trades['Entry Time'],
        y=entry_trades['Entry Price'],
        mode='markers',
        marker=dict(symbol='triangle-down', size=10, color='red'),
        name='Sell Entries'
    ))

    # Add Trade Exit Markers (for all exit types)
    fig_candlestick.add_trace(go.Scatter(
        x=trades_df['Exit Time'],
        y=trades_df['Exit Price'],
        mode='markers',
        marker=dict(symbol='circle', size=8, color='orange', line=dict(width=1, color='DarkSlateGrey')),
        name='Exits'
    ))

    fig_candlestick.update_layout(xaxis_rangeslider_visible=False,
                                  title=f"Gold Price ({primary_interval} Candles with Supertrends & Trades)")
    st.plotly_chart(fig_candlestick, use_container_width=True)

    # --- PnL Distribution ---
    st.subheader("PnL Distribution")
    fig_pnl_hist = px.histogram(trades_df, x='PnL', nbins=20, title="Distribution of Trade PnL (points)")
    st.plotly_chart(fig_pnl_hist, use_container_width=True)

    # --- Daily Performance Analysis ---
    st.subheader("Daily Performance Analysis")
    if not trades_df.empty:
        # Ensure 'Exit Time' is datetime for day_name()
        trades_df['Exit Time'] = pd.to_datetime(trades_df['Exit Time'])
        trades_df['DayOfWeek'] = trades_df['Exit Time'].dt.day_name()

        # Calculate metrics per day of week
        daily_summary = trades_df.groupby('DayOfWeek').agg(
            Total_PnL=('PnL', 'sum'),
            Profitable_Trades=('Result', lambda x: (x == 'Profit').sum()),
            Losing_Trades=('Result', lambda x: (x == 'Loss').sum()),
            Timeout_Trades=('Result', lambda x: (x == 'Timeout').sum()),
            Counter_Signal_Exits=('Result', lambda x: (x == 'Counter-Signal Exit').sum()),
            ATR_Calc_Errors=('Result', lambda x: (x == 'ATR Calc Error/Timeout').sum())  # New metric
        ).reset_index()

        # Calculate Win Rate for each day
        daily_summary['Total_Win_Loss_Trades'] = daily_summary['Profitable_Trades'] + daily_summary['Losing_Trades']
        daily_summary['Win_Rate (%)'] = (daily_summary['Profitable_Trades'] / daily_summary[
            'Total_Win_Loss_Trades']) * 100
        daily_summary['Win_Rate (%)'] = daily_summary['Win_Rate (%)'].fillna(0).round(2)  # Handle division by zero

        # Order days of the week for consistent display
        ordered_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_summary['DayOfWeek'] = pd.Categorical(daily_summary['DayOfWeek'], categories=ordered_days, ordered=True)
        daily_summary = daily_summary.sort_values('DayOfWeek')

        # Display table
        st.dataframe(daily_summary.set_index('DayOfWeek'))

        # Find day with highest win rate
        if not daily_summary.empty:
            # Filter out days with 0 total_win_loss_trades to avoid idxmax on all zeros
            highest_win_rate_day_df = daily_summary[daily_summary['Total_Win_Loss_Trades'] > 0]
            if not highest_win_rate_day_df.empty:
                highest_win_rate_day = highest_win_rate_day_df.loc[highest_win_rate_day_df['Win_Rate (%)'].idxmax()]
                st.info(
                    f"**Day with Highest Win Rate:** {highest_win_rate_day['DayOfWeek']} with {highest_win_rate_day['Win_Rate (%)']:.2f}%")
            else:
                st.info("No days with winning or losing trades to determine highest win rate.")
        else:
            st.info("No daily performance data to analyze.")