# Simple Algorithmic Trading (A‑T) Project

This repository is a starting point for learning **algorithmic trading** with Python.  It contains a basic example of a moving‑average crossover strategy, which is one of the simplest systematic trading techniques.  The purpose of this project is educational: it illustrates how to fetch historical price data, compute indicators, generate trading signals and backtest the performance using pure Python.

## Contents

* `moving_average_crossover.py` – a Python script implementing a simple moving‑average crossover backtest.
* `README.md` – this file; contains instructions and explanations.

## How the moving‑average crossover strategy works

The moving‑average crossover strategy uses two moving averages with different window lengths:

1. **Short‑term moving average**: captures recent price trends (e.g., 50‑day average).
2. **Long‑term moving average**: captures longer‑term trends (e.g., 200‑day average).

When the short‑term average crosses **above** the long‑term average, the strategy generates a **buy** signal (expecting an upward trend).  When the short‑term average crosses **below** the long‑term average, it generates a **sell** signal (expecting a downward trend).  The script then simulates buying and selling the asset based on these signals and computes the resulting portfolio value over time.

## Running the example

1. Install the required Python packages if you don’t already have them:

   ```bash
   pip install yfinance pandas matplotlib
   ```

2. Execute the script from the command line, specifying a ticker (e.g. AAPL) and optional parameters:

   ```bash
   python moving_average_crossover.py --symbol AAPL --start 2015-01-01 --short_window 50 --long_window 200
   ```

   The script fetches historical price data from Yahoo Finance, calculates the moving averages, generates buy/sell signals, and prints a summary of the strategy’s performance.  It will also produce a plot of the closing price along with the short and long moving averages and mark the entry/exit points.

3. Experiment with different parameters (e.g. change the window lengths or the ticker symbol) to see how the strategy performs on other assets.

## Notes

* This example does **not** include transaction fees, bid–ask spreads or slippage; real‑world results will differ.
* The script is intended for educational purposes and should not be considered financial advice.  Always research and test thoroughly before trading real money.
* See the comments in the source code for explanations of each step.
