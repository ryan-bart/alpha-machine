# Quant Trading System – Next Steps & Todo

## ✅ Current Status

You currently have a **working Polars-based backtester** with:

- `data/loader.py` → fetches and cleans SPY data
- `strategies/base_strategy.py` → strategy interface
- `strategies/ma_crossover.py` → moving average crossover strategy
- `backtester/portfolio.py` → executes trades and logs them (with trade tracking)
- `backtester/engine.py` → runs backtest
- `run_backtest.py` → prints final portfolio value, trades, CAGR

Current metrics include:

- Portfolio growth over time
- Trade profits & holding periods
- CAGR (annualized return)

---

## 📝 Next Improvements (Todo)

### 1️⃣ Vectorized Backtesting (Big Speed / “Looping” Improvement)
- Replace row-by-row iteration in `BacktestEngine` with Polars vectorized calculations
- Compute positions, trades, and equity curve in one pass
- Benefits: 100–1000x faster, scalable to multi-asset backtests
- Update in:
  - `backtester/engine.py`
  - `backtester/portfolio.py`

---

### 2️⃣ Performance Metrics
Add metrics every quant needs:

- Sharpe Ratio
- Max Drawdown
- Total Return / CAGR (already done)
- Win Rate
- Average Trade Return

Files to create/update:

- `analytics/metrics.py`
- Update `run_backtest.py` to call metrics

---

### 3️⃣ Equity Curve + Drawdown Visualization
- Plot portfolio growth over time and drawdowns
- Tools: `matplotlib` or `plotly`
- Files:
  - `analytics/plots.py`
  - `run_backtest.py` → call plot functions after backtest

---

### 4️⃣ Benchmark Comparison
- Compare strategy performance vs **buy-and-hold SPY**
- Include benchmark equity curve and CAGR for sanity check

---

### 5️⃣ Strategy Parameter Sweeps / Hyperparameter Testing
- Automatically test multiple strategy parameter combinations:
  - fast MA: 10 → 50
  - slow MA: 50 → 200
- Produce results table with:
  - CAGR
  - Sharpe
  - Max Drawdown
- Files:
  - `experiments/parameter_sweep.py` (new)
- Optional: vectorize sweep with Polars or `itertools.product`

---

### 6️⃣ Multi-Asset Support (Optional Later)
- Load multiple tickers
- Portfolio handles multiple positions
- Future-proof system for realistic quant research

---

### 7️⃣ Other Small Improvements
- Add `__init__.py` in `analytics/` and `experiments/` folders
- Add logging via `utils/logger.py`
- Add config file (`config/settings.yaml`) for default parameters (start date, initial cash, symbol list)

---

## 📁 Suggested Next File Updates

| File | Next Steps |
|------|------------|
| `backtester/engine.py` | Implement vectorized backtester |
| `backtester/portfolio.py` | Track positions in vectorized format; keep per-day portfolio value |
| `analytics/metrics.py` | Compute Sharpe, Max Drawdown, Win Rate, Avg Trade Return |
| `analytics/plots.py` | Equity curve + drawdown charts |
| `run_backtest.py` | Integrate metrics + plots + benchmark comparison |
| `experiments/parameter_sweep.py` | Run multiple MA parameter combinations |

---

> **Next Recommended Step:** Rewrite the Polars backtester to be fully vectorized while keeping trade tracking, portfolio values, and metrics. This will make your system significantly faster and more scalable.