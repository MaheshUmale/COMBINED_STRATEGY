# Combined Squeeze and Inside Bar Breakout Strategy

## 1. Overview

This project implements a powerful, multi-timeframe trading strategy that combines two well-known patterns for high-probability setups: the **TTM Squeeze** and the **Inside Bar breakout**.

The core idea is to identify a major period of market consolidation on a **higher timeframe (HTF)** using the Squeeze. Then, we zoom into a **lower timeframe (LTF)** to find a secondary, tighter compression in the form of an Inside Bar. A breakout from this Inside Bar, confirmed by a spike in volume, serves as our entry trigger.

This "fractal compression" approach—a small pattern occurring within a larger pattern—is designed to pinpoint moments of significant potential energy release in the market.

## 2. Core Concepts

### The Squeeze (Higher Timeframe)

-   **Definition:** A TTM Squeeze occurs when the **Bollinger Bands** move completely inside the **Keltner Channels**. This indicates a period of low volatility and market consolidation, often preceding a powerful directional move.
-   **Role in this Strategy:** We use the Squeeze on a higher timeframe (e.g., 30-minute, 60-minute) to establish the overall market context. We only look for trading opportunities when the market is "coiling" for a big move.

### The Inside Bar (Lower Timeframe)

-   **Definition:** An Inside Bar is a candle whose entire range (high and low) is contained within the range of the preceding candle (the "Mother Bar").
-   **Role in this Strategy:** We use the Inside Bar on a lower timeframe (e.g., 15-minute) to identify a precise, short-term point of consolidation *within* the broader HTF Squeeze. A breakout from this Inside Bar acts as our specific entry trigger.

## 3. Strategy Logic

The strategy is executed through the `combined_strategy.py` script.

### Entry Condition

A trade is entered ONLY when all of the following conditions are met:

1.  **HTF Squeeze:** The higher timeframe (e.g., `60min`) is in a confirmed, strong Squeeze.
2.  **LTF Inside Bar:** A valid Inside Bar pattern forms on the lower timeframe (e.g., `15min`). This requires the inside bar's volume to be lower than the Mother Bar's volume, indicating a true contraction in activity.
3.  **LTF Breakout:** The price breaks above the high of the Inside Bar (for a long trade) or below its low (for a short trade).
4.  **Volume Confirmation:** The breakout is confirmed by a high **Relative Volume (RVOL)**, which must be greater than the `rvol_threshold` (e.g., > 1.5) to ensure strong momentum.

### Multi-Stage Exit Logic

The exit strategy is dynamic and adapts to the market's behavior, particularly how the HTF Squeeze resolves.

#### Phase 1: Default Trailing Stop

-   **Initial Stop-Loss:** The initial stop is placed just beyond the Mother Bar's range (below the low for longs, above the high for shorts).
-   **Move to Break-Even:** Once the trade moves 1R (one unit of risk) in profit, the stop-loss is moved to the entry price to eliminate risk.
-   **ATR Trailing Stop:** After reaching break-even, a trailing stop based on the **Average True Range (ATR)** is activated. The stop is placed at a distance of `atr_trailing_multiplier` (e.g., 2) times the ATR from the current price, locking in profits as the trade moves favorably.

#### Phase 2: The Squeeze Fire Event

-   The strategy continuously monitors the HTF for the Squeeze to "fire" (i.e., for the Bollinger Bands to expand back outside the Keltner Channels).
-   When a `squeeze_fired` event is detected:
    1.  **Take Partial Profit:** Half of the position is immediately sold at the current market price to lock in gains.
    2.  **Deactivate Standard Exits:** The standard trailing stop and initial take-profit target are deactivated for the remaining half of the position.

#### Phase 3: Special Post-Fire Exit Rule

-   A new, special exit target is calculated for the remaining half of the position.
-   The strategy looks at the `special_exit_lookback` period (e.g., 7 bars) on the **higher timeframe** immediately *following* the squeeze fire.
-   **For a Long Trade:** The exit signal is the first bar that closes **below the low** of those 7 bars.
-   **For a Short Trade:** The exit signal is the first bar that closes **above the high** of those 7 bars.

## 4. How to Use

1.  **Data:** Place your historical 1-minute data in the root directory. The files must be in CSV format and named according to the convention: `SYMBOL_minute.csv` (e.g., `DUMMY_minute.csv`).
2.  **Configuration:** Open the `combined_strategy.py` file. At the bottom, in the `if __name__ == '__main__':` block, you can adjust the strategy's parameters.
3.  **Execution:** Run the backtest from your terminal:
    ```bash
    python combined_strategy.py
    ```
4.  **Results:** The script will print trade-by-trade logs to the console and a final summary of the performance, including total PnL and win rate.

## 5. Tunable Parameters

You can fine-tune the strategy's behavior by modifying the parameters in the `__init__` method of the `CombinedStrategy` class:

-   `higher_tf`: The timeframe for detecting the Squeeze (e.g., `'60min'`).
-   `lower_tf`: The timeframe for identifying the Inside Bar breakout (e.g., `'15min'`).
-   `rvol_threshold`: The minimum Relative Volume required to confirm a breakout (e.g., `1.5`).
-   `risk_reward`: The initial risk-to-reward ratio for the take-profit target (e.g., `3.0`).
-   `initial_capital`: The starting capital for the backtest.
-   `risk_per_trade_percent`: The percentage of capital to risk on each trade.
-   `atr_trailing_multiplier`: The ATR multiplier for the trailing stop-loss (e.g., `2.0`).
-   `special_exit_lookback`: The number of HTF bars to look at after a squeeze fire to set the special exit target (e.g., `7`).