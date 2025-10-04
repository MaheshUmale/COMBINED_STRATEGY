import pandas as pd
import numpy as np
import glob
import os
from typing import List, Dict

# --- Indicator Helper Functions ---

def calculate_ema(series, period):
    """Calculates the Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()

def calculate_atr(df, period=14):
    """Calculates the Average True Range."""
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def calculate_sma(series, period):
    """Calculates the Simple Moving Average."""
    return series.rolling(window=period).mean()

def calculate_bollinger_bands(df, period=20, std_dev=2):
    """Calculates Bollinger Bands."""
    sma = calculate_sma(df['Close'], period)
    std = df['Close'].rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, lower_band

def calculate_keltner_channels(df, period=20, atr_multiplier=1.5):
    """Calculates Keltner Channels."""
    sma = calculate_sma(df['Close'], period)
    atr = calculate_atr(df, period)
    upper_channel = sma + (atr * atr_multiplier)
    lower_channel = sma - (atr * atr_multiplier)
    return upper_channel, lower_channel

class Trade:
    """Represents an active or closed trade position."""
    def __init__(self, symbol, entry_time, entry_price, direction, sl, tp, size, initial_risk, trade_type='FULL'):
        self.symbol = symbol
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.direction = direction
        self.sl = sl
        self.tp = tp
        self.size = size
        self.initial_risk = initial_risk
        self.status = 'OPEN'
        self.close_time = None
        self.close_price = None
        self.pnl = 0.0
        self.sl_history = [(entry_time, sl)]
        self.moved_to_be = False
        self.special_exit_mode = False
        self.special_exit_target = None
        self.trade_type = trade_type

    def close(self, price, time):
        self.close_price = price
        self.close_time = time
        if self.direction == 'LONG':
            self.pnl = (price - self.entry_price) * self.size
        else:
            self.pnl = (self.entry_price - price) * self.size
        self.status = 'CLOSED'


class CombinedStrategy:
    """
    A combined trading strategy that identifies inside bar breakouts occurring
    within a higher timeframe squeeze.

    Entry Logic:
    1. A "strong" TTM Squeeze must be active on the higher timeframe (HTF).
    2. An inside bar pattern must form on the lower timeframe (LTF).
    3. A breakout from the inside bar must occur on the LTF.
    4. The breakout must be accompanied by high relative volume (RVOL).

    Exit Logic:
    1. Default Trailing Stop:
        - Move to Break-Even (BE) when the trade reaches 1R (1x risk) in profit.
        - Trail with a 2-ATR stop-loss after reaching BE.
    2. Special Squeeze-Fire Exit:
        - If the HTF squeeze "fires" (breaks out), take half profit immediately.
        - The remaining half is exited based on the 7-bar high/low following the fire.
    """
    def __init__(self, data_path, higher_tf='30min', lower_tf='15min', rvol_threshold=1.5, risk_reward=3.0, initial_capital=100000, risk_per_trade_percent=0.01, atr_trailing_multiplier=2.0, special_exit_lookback=7):
        self.data_path = data_path
        self.higher_tf = higher_tf
        self.lower_tf = lower_tf
        self.rvol_threshold = rvol_threshold
        self.risk_reward = risk_reward
        self.initial_capital = initial_capital
        self.risk_per_trade_percent = risk_per_trade_percent
        self.atr_trailing_multiplier = atr_trailing_multiplier
        self.special_exit_lookback = special_exit_lookback

        self.ohlcv_data: Dict[str, Dict[str, pd.DataFrame]] = {}
        self.trades = []
        self.open_trades: Dict[str, Trade] = {}

    def load_and_prepare_data(self):
        """Loads 1-minute data and resamples it to the required timeframes."""
        print("Loading and preparing data...")
        csv_files = glob.glob(os.path.join(self.data_path, '*_minute.csv'))
        for file_path in csv_files:
            symbol = os.path.basename(file_path).split('_minute.csv')[0]
            df_1m = pd.read_csv(file_path, parse_dates=['date'])
            df_1m.set_index('date', inplace=True)
            df_1m.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            df_1m = df_1m.sort_index().replace([np.inf, -np.inf], np.nan).dropna()
            if df_1m.empty: continue
            self.ohlcv_data[symbol] = {}
            for tf in [self.lower_tf, self.higher_tf]:
                resampled_df = df_1m.resample(tf).agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}).dropna()
                if not resampled_df.empty:
                    resampled_df['ATR'] = calculate_atr(resampled_df, period=14)
                    resampled_df['BB_Upper'], resampled_df['BB_Lower'] = calculate_bollinger_bands(resampled_df)
                    resampled_df['KC_Upper'], resampled_df['KC_Lower'] = calculate_keltner_channels(resampled_df)
                    resampled_df['Avg_Vol_10'] = resampled_df['Volume'].rolling(window=10).mean()
                    self.ohlcv_data[symbol][tf] = resampled_df.dropna()
        print("Data loaded successfully.")

    def check_squeeze(self, df_htf):
        """Identifies squeeze conditions and squeeze fire events on the HTF."""
        df_htf['in_squeeze'] = (df_htf['BB_Lower'] > df_htf['KC_Lower']) & (df_htf['BB_Upper'] < df_htf['KC_Upper'])
        df_htf['squeeze_fired'] = (df_htf['in_squeeze'].shift(1) == True) & (df_htf['in_squeeze'] == False)
        return df_htf

    def find_inside_bar(self, df_ltf):
        """Identifies inside bar patterns on the LTF."""
        df_ltf['MB_High'] = df_ltf['High'].shift(1)
        df_ltf['MB_Low'] = df_ltf['Low'].shift(1)
        df_ltf['MB_Volume'] = df_ltf['Volume'].shift(1)
        df_ltf['is_inside_bar'] = (df_ltf['High'] < df_ltf['MB_High']) & (df_ltf['Low'] > df_ltf['MB_Low'])
        df_ltf['volume_confirms'] = df_ltf['Volume'] < df_ltf['MB_Volume']
        df_ltf['valid_ib_setup'] = df_ltf['is_inside_bar'] & df_ltf['volume_confirms']
        return df_ltf

    def run_backtest(self):
        """Executes the backtest logic for all symbols."""
        for symbol, data in self.ohlcv_data.items():
            if self.higher_tf not in data or self.lower_tf not in data: continue
            df_htf = self.check_squeeze(data[self.higher_tf].copy())
            df_ltf = self.find_inside_bar(data[self.lower_tf].copy())
            aligned_df = pd.merge(df_ltf, df_htf[['in_squeeze', 'squeeze_fired']], left_index=True, right_index=True, how='left')
            aligned_df[['in_squeeze', 'squeeze_fired']] = aligned_df[['in_squeeze', 'squeeze_fired']].ffill()
            print(f"--- Running backtest for {symbol} ---")
            for i in range(1, len(aligned_df)):
                current_bar = aligned_df.iloc[i]
                prev_bar = aligned_df.iloc[i-1]
                if symbol in self.open_trades:
                    self._manage_open_trade(symbol, current_bar, df_htf)
                    if symbol not in self.open_trades: continue
                if prev_bar['valid_ib_setup'] and current_bar['in_squeeze']:
                    rvol = current_bar['Volume'] / current_bar['Avg_Vol_10']
                    if rvol > self.rvol_threshold:
                        if current_bar['Close'] > prev_bar['High']:
                            self._execute_trade(symbol, current_bar, prev_bar, 'LONG')
                        elif current_bar['Close'] < prev_bar['Low']:
                            self._execute_trade(symbol, current_bar, prev_bar, 'SHORT')

            # --- Post-backtest debug info ---
            if not any(t.symbol == symbol for t in self.trades) and symbol not in self.open_trades:
                print(f"DEBUG: No trades were executed for {symbol}.")
                print(f"DEBUG: Total 'valid_ib_setup' signals: {aligned_df['valid_ib_setup'].sum()}")
                print(f"DEBUG: Total 'in_squeeze' bars: {aligned_df['in_squeeze'].sum()}")
                # Check for overlaps. Note: prev_bar is used for IB, so we shift valid_ib_setup to align with current_bar's squeeze status
                crossovers = aligned_df[(aligned_df['valid_ib_setup'].shift(1) == True) & (aligned_df['in_squeeze'] == True)]
                print(f"DEBUG: Total potential entry moments (IB in Squeeze): {len(crossovers)}")
                if len(crossovers) > 0:
                    print("DEBUG: Potential entries were found, but RVOL or breakout condition was not met.")
                else:
                    print("DEBUG: The conditions for 'Inside Bar' and 'Squeeze' never occurred at the same time.")

    def _execute_trade(self, symbol, current_bar, ib_bar, direction):
        """Calculates trade parameters and executes a new trade."""
        entry_price = current_bar['Close']
        if direction == 'LONG':
            sl = ib_bar['MB_Low']
            risk = entry_price - sl
            tp = entry_price + (risk * self.risk_reward)
        else:
            sl = ib_bar['MB_High']
            risk = sl - entry_price
            tp = entry_price - (risk * self.risk_reward)
        if risk <= 0: return
        trade_size = (self.initial_capital * self.risk_per_trade_percent) / risk
        trade = Trade(symbol, current_bar.name, entry_price, direction, sl, tp, trade_size, risk)
        self.open_trades[symbol] = trade
        print(f"{current_bar.name} | {symbol} | NEW TRADE [{direction}] | Entry: {entry_price:.2f} | SL: {sl:.2f}")

    def _manage_open_trade(self, symbol, current_bar, df_htf):
        """Manages all exit logic for an open trade."""
        trade = self.open_trades[symbol]

        # --- Special Exit on Squeeze Fire ---
        if trade.special_exit_mode:
            if (trade.direction == 'LONG' and current_bar['Close'] < trade.special_exit_target) or \
               (trade.direction == 'SHORT' and current_bar['Close'] > trade.special_exit_target):
                print(f"{current_bar.name} | {symbol} | CLOSED 2nd HALF [{trade.direction}] on Special Exit. Target: {trade.special_exit_target:.2f}")
                trade.close(current_bar['Close'], current_bar.name)
                self.trades.append(trade)
                del self.open_trades[symbol]
            return

        if current_bar['squeeze_fired'] and not trade.special_exit_mode:
            self._handle_squeeze_fire(symbol, current_bar, df_htf)
            return

        # --- Standard Exit Logic (SL, TP, Trailing) ---
        self._check_standard_exits(symbol, current_bar)

    def _handle_squeeze_fire(self, symbol, current_bar, df_htf):
        """Handles the logic for taking partial profit when a squeeze fires."""
        trade = self.open_trades[symbol]
        profit_price = current_bar['Close']
        partial_size = trade.size / 2
        partial_trade = Trade(symbol, trade.entry_time, trade.entry_price, trade.direction, trade.sl, profit_price, partial_size, trade.initial_risk, trade_type='PARTIAL_PROFIT')
        partial_trade.close(profit_price, current_bar.name)
        self.trades.append(partial_trade)
        print(f"{current_bar.name} | {symbol} | SQUEEZE FIRED: Took PARTIAL PROFIT [{trade.direction}] at {profit_price:.2f}. PnL: {partial_trade.pnl:.2f}")

        trade.size /= 2
        trade.special_exit_mode = True

        htf_fire_index = df_htf.index.get_loc(current_bar.name, method='ffill')
        if htf_fire_index + self.special_exit_lookback + 1 < len(df_htf):
            following_bars = df_htf.iloc[htf_fire_index + 1 : htf_fire_index + self.special_exit_lookback + 1]
            trade.special_exit_target = following_bars['Low'].min() if trade.direction == 'LONG' else following_bars['High'].max()
            print(f"{current_bar.name} | {symbol} | New exit target for 2nd half: {trade.special_exit_target:.2f}")
        else:
            print(f"{current_bar.name} | {symbol} | Squeeze Fired, but not enough HTF bars to set new target. Closing remaining position.")
            trade.close(current_bar['Close'], current_bar.name)
            self.trades.append(trade)
            del self.open_trades[symbol]

    def _check_standard_exits(self, symbol, current_bar):
        """Checks for standard SL, TP, and trailing stop exits."""
        trade = self.open_trades[symbol]

        if (trade.direction == 'LONG' and current_bar['Low'] <= trade.sl) or \
           (trade.direction == 'SHORT' and current_bar['High'] >= trade.sl):
            trade.close(trade.sl, current_bar.name); self.trades.append(trade); del self.open_trades[symbol]
            print(f"{current_bar.name} | {symbol} | CLOSED TRADE [{trade.direction}] at SL. PnL: {trade.pnl:.2f}")
            return

        if (trade.direction == 'LONG' and current_bar['High'] >= trade.tp) or \
           (trade.direction == 'SHORT' and current_bar['Low'] <= trade.tp):
            trade.close(trade.tp, current_bar.name); self.trades.append(trade); del self.open_trades[symbol]
            print(f"{current_bar.name} | {symbol} | CLOSED TRADE [{trade.direction}] at TP. PnL: {trade.pnl:.2f}")
            return

        if not trade.moved_to_be:
            if (trade.direction == 'LONG' and current_bar['Close'] >= trade.entry_price + trade.initial_risk) or \
               (trade.direction == 'SHORT' and current_bar['Close'] <= trade.entry_price - trade.initial_risk):
                trade.sl = trade.entry_price; trade.moved_to_be = True; trade.sl_history.append((current_bar.name, trade.sl))
                print(f"{current_bar.name} | {symbol} | Moved SL to Break-Even: {trade.sl:.2f}")

        if trade.moved_to_be:
            atr_val = current_bar['ATR']
            if pd.notna(atr_val) and atr_val > 0:
                new_sl = (current_bar['Close'] - (self.atr_trailing_multiplier * atr_val)) if trade.direction == 'LONG' else (current_bar['Close'] + (self.atr_trailing_multiplier * atr_val))
                if (trade.direction == 'LONG' and new_sl > trade.sl) or (trade.direction == 'SHORT' and new_sl < trade.sl):
                    trade.sl = new_sl; trade.sl_history.append((current_bar.name, trade.sl))

if __name__ == '__main__':
    if not glob.glob("*_minute.csv"):
        print("No CSV data found. Creating dummy data for testing...")
        dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=20000, freq='min'))
        price = 100 + np.random.randn(20000).cumsum() * 0.1
        volume = np.random.randint(100, 1000, size=20000)
        dummy_df = pd.DataFrame({'Open':price, 'High':price+np.random.uniform(0, 1, 20000), 'Low':price-np.random.uniform(0, 1, 20000), 'Close':price+np.random.uniform(-0.5, 0.5, 20000), 'Volume':volume}, index=dates)
        dummy_df.to_csv('DUMMY_minute.csv', header=['Open', 'High', 'Low', 'Close', 'Volume'], index_label='date')

    strategy = CombinedStrategy(
        data_path=".",
        higher_tf='60min',
        lower_tf='15min',
        rvol_threshold=1.5,
        risk_reward=3.0,
        initial_capital=100000,
        risk_per_trade_percent=0.01,
        atr_trailing_multiplier=2.0,
        special_exit_lookback=7
    )
    strategy.load_and_prepare_data()
    strategy.run_backtest()

    print("\n--- Backtest Complete ---")
    print(f"Total Trades Logged (incl. partials): {len(strategy.trades)}")
    total_pnl = sum(t.pnl for t in strategy.trades)
    print(f"Total PnL: {total_pnl:.2f}")

    wins = sum(1 for t in strategy.trades if t.pnl > 0)
    losses = sum(1 for t in strategy.trades if t.pnl <= 0)
    win_rate = (wins / len(strategy.trades) * 100) if strategy.trades else 0
    print(f"Win Rate: {win_rate:.2f}%")