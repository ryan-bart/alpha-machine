# Quant Trading Research Platform

This is a Python-based trading research platform built to test algorithmic strategies. It is designed for flexibility, modularity, and speed, leveraging **Polars** for data handling and **yfinance** for historical data.  

The system currently supports:

- Loading historical price data
- Moving Average Crossover strategy
- Portfolio simulation with trade tracking
- Calculation of annualized return (CAGR)

Future upgrades will include performance metrics, equity curve plots, parameter sweeps, and multi-asset portfolios.

---

## Project Structure

```
alpha-machine/
├── data/
│   └── loader.py          # Loads historical data from yfinance into Polars
├── strategies/
│   ├── base_strategy.py   # Base strategy interface
│   └── ma_crossover.py    # Simple moving average crossover strategy
├── backtester/
│   ├── portfolio.py       # Portfolio simulation with trade tracking
│   └── engine.py          # Backtest engine that runs strategy on data
├── run_backtest.py        # Main script to run backtest
├── requirements.txt       # Dependencies
└── README.md
```
---

