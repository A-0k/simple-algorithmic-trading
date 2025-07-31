"""
moving_average_crossover.py
---------------------------------
This script implements a simple moving‑average crossover strategy for educational
purposes.  It downloads historical price data for a specified ticker, computes
short‑ and long‑term moving averages, generates buy/sell signals when the
averages cross, and then simulates the portfolio value over time.  Finally, it
prints a performance summary and plots the price series alongside the moving
averages with markers for entry and exit points.

Usage (from the command line):
    python moving_average_crossover.py --symbol AAPL --start 2015-01-01 \
        --short_window 50 --long_window 200

Dependencies:
    - pandas
    - numpy
    - yfinance (for downloading data)
    - matplotlib (for plotting the results)

NOTE: This script is intended for learning.  It does not include transaction
costs, slippage or other market frictions.  Use it as a starting point for
experimentation; do not trade real money based solely on this example.
"""

import argparse
import datetime
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


@dataclass
class TradeResult:
    symbol: str
    start_date: str
    short_window: int
    long_window: int
    total_return: float
    annualized_return: float
    max_drawdown: float
    trades: int
    history: pd.DataFrame = field(repr=False)


def download_data(symbol: str, start: str) -> pd.DataFrame:
    """Download historical adjusted closing price data using yfinance."""
    try:
        data = yf.download(symbol, start=start, progress=False)
    except Exception as exc:
        raise RuntimeError(f"Failed to download data for {symbol}: {exc}") from exc
    if data.empty:
        raise ValueError(f"No data returned for symbol {symbol}. Check the ticker symbol.")
    return data


def compute_signals(data: pd.DataFrame, short_window: int, long_window: int) -> pd.DataFrame:
    """Compute moving averages and generate buy/sell signals."""
    # Calculate moving averages
    data = data.copy()
    data['short_ma'] = data['Adj Close'].rolling(window=short_window, min_periods=1).mean()
    data['long_ma'] = data['Adj Close'].rolling(window=long_window, min_periods=1).mean()

    # Generate signals: 1 when short_ma > long_ma, else 0
    data['signal'] = 0
    data.loc[data['short_ma'] > data['long_ma'], 'signal'] = 1
    
    # Create positions by taking the difference of signals (1 = buy, -1 = sell)
    data['position'] = data['signal'].diff().fillna(0)
    return data


def backtest(data: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate the portfolio value based on generated signals.  Assumes buying one
    unit at each buy signal and selling at each sell signal.  Cash balance is
    tracked implicitly by computing the cumulative returns of positions.
    """
    data = data.copy()
    # Compute daily returns of the underlying asset
    data['return'] = data['Adj Close'].pct_change().fillna(0)
    
    # Strategy return: when in position (signal == 1), take the asset return
    data['strategy_return'] = data['return'] * data['signal'].shift(1).fillna(0)
    
    # Cumulative returns
    data['cumulative_asset'] = (1 + data['return']).cumprod()
    data['cumulative_strategy'] = (1 + data['strategy_return']).cumprod()
    return data


def evaluate_performance(data: pd.DataFrame, symbol: str, start: str,
                         short_window: int, long_window: int) -> TradeResult:
    """Evaluate portfolio performance and compute metrics."""
    # Total return and annualized return
    total_return = data['cumulative_strategy'].iloc[-1] - 1
    # Annualization: approximate number of trading days per year (252)
    trading_days = (data.index[-1] - data.index[0]).days
    annualized_return = (1 + total_return) ** (252 / trading_days) - 1

    # Maximum drawdown (peak-to-trough decline)
    cumulative = data['cumulative_strategy']
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    # Count number of trades (buy signals)
    trades = int((data['position'] == 1).sum())

    return TradeResult(
        symbol=symbol,
        start_date=start,
        short_window=short_window,
        long_window=long_window,
        total_return=total_return,
        annualized_return=annualized_return,
        max_drawdown=max_drawdown,
        trades=trades,
        history=data
    )


def plot_results(data: pd.DataFrame, symbol: str, result: TradeResult) -> None:
    """Plot the price and moving averages with trade signals."""
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Adj Close'], label='Adj Close', alpha=0.6)
    plt.plot(data.index, data['short_ma'], label=f'Short MA ({result.short_window})', alpha=0.8)
    plt.plot(data.index, data['long_ma'], label=f'Long MA ({result.long_window})', alpha=0.8)

    # Mark buy and sell signals
    buys = data[data['position'] == 1]
    sells = data[data['position'] == -1]
    plt.scatter(buys.index, buys['Adj Close'], marker='^', color='green', label='Buy Signal')
    plt.scatter(sells.index, sells['Adj Close'], marker='v', color='red', label='Sell Signal')

    plt.title(f'Moving‑Average Crossover Strategy for {symbol}')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description='Backtest a moving‑average crossover strategy.')
    parser.add_argument('--symbol', type=str, default='AAPL',
                        help='Ticker symbol to download (default: AAPL)')
    parser.add_argument('--start', type=str, default='2015-01-01',
                        help='Start date for historical data (YYYY-MM-DD)')
    parser.add_argument('--short_window', type=int, default=50,
                        help='Window length for the short moving average (default: 50 days)')
    parser.add_argument('--long_window', type=int, default=200,
                        help='Window length for the long moving average (default: 200 days)')
    parser.add_argument('--no_plot', action='store_true',
                        help='Suppress plotting of the results.')
    args = parser.parse_args()

    # Validate window sizes
    if args.short_window >= args.long_window:
        parser.error('The short_window must be less than the long_window.')

    # Suppress potential warnings from yfinance about missing columns
    warnings.simplefilter('ignore', category=UserWarning)

    data = download_data(args.symbol, args.start)
    data = compute_signals(data, args.short_window, args.long_window)
    data = backtest(data)
    result = evaluate_performance(data, args.symbol, args.start, args.short_window, args.long_window)

    # Display performance summary
    print(f"\nPerformance Summary for {args.symbol}")
    print(f"Start Date      : {args.start}")
    print(f"Short Window    : {args.short_window}")
    print(f"Long Window     : {args.long_window}")
    print(f"Total Return    : {result.total_return:.2%}")
    print(f"Annualized Return: {result.annualized_return:.2%}")
    print(f"Max Drawdown    : {result.max_drawdown:.2%}")
    print(f"Number of Trades: {result.trades}\n")

    if not args.no_plot:
        plot_results(data, args.symbol, result)


if __name__ == '__main__':
    main()