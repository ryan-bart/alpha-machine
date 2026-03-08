# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Generate today's stock recommendations (prints + saves cache/recommendation_report.md)
python run_score.py

# Run historical walk-forward backtest (prints metrics + saves plots to cache/)
python run_backtest.py

# Experimental weight optimizer (fast numpy proxy, not fully validated)
python run_optimize.py
```

First run is slow (~1,000 tickers × 10 years of price data). Subsequent runs use parquet cache in `cache/` and only fetch new dates.

No test suite or linter is configured.

## Architecture

Multi-factor stock scoring system: screens Russell 1000 using 10 price/volume factors, selects a 20-stock equal-weight portfolio, rebalanced quarterly.

**Data flow:** `data/` → `factors/` → `scoring/` → `portfolio/` → `backtester/` or `output/`

1. **data/**: Universe from Wikipedia (Russell 1000 or S&P 500), prices via yfinance, risk-free rate from FRED. All cached as parquet with staleness checks.
2. **factors/**: 10 factors subclassing `BaseFactor` (momentum, mean-reversion, volume, volatility, quality). Each implements `name()`, `compute()`, and `invert` property. Computed per-ticker, then cross-sectionally percentile-ranked.
3. **scoring/**: `combine.py` orchestrates factor computation and produces weighted composite scores. `normalize.py` does cross-sectional percentile ranking via Polars. `missing.py` filters tickers with insufficient history and imputes NaN ranks to 0.5.
4. **portfolio/**: `construct.py` selects top N with turnover dampening (retain holdings above `SELL_THRESHOLD_RANK`) and optional sector caps. `risk.py` computes HHI and sector exposure.
5. **backtester/**: Walk-forward simulator (`engine.py`) with quarterly/monthly rebalance, in-sample/out-of-sample split. `metrics.py` computes CAGR, Sharpe, max drawdown, Calmar, hit rate.
6. **output/**: Markdown report generation and matplotlib plots (equity curve, monthly heatmap).

## Key Conventions

- **Polars-first**: All internal data processing uses `pl.DataFrame`. Pandas only for yfinance compatibility and `pd.read_html()`.
- **All config in one place**: `config/settings.py` holds every tunable parameter (weights, thresholds, universe source, rebalance frequency). Factor weights are a dict summing to 1.0.
- **No lookahead bias**: Backtester strictly uses data available up to each rebalance date.
- **Rank-based scoring**: Factors are percentile-ranked (0–1) for comparability before weighted combination. Inverted factors (lower=better like volatility) have ranks flipped.
- **Caching**: Parquet files in `cache/` (gitignored). Universe refreshes every 7 days; prices update incrementally.

## Adding a New Factor

1. Create a class in `factors/` subclassing `BaseFactor` from `factors/base.py`
2. Implement `name` (string ID matching the key in `FACTOR_WEIGHTS`), `compute(df)` (returns df with new column), and `invert` (True if lower=better)
3. Add instance to `ALL_FACTORS` list in `scoring/combine.py`
4. Add weight entry to `FACTOR_WEIGHTS` in `config/settings.py` (all weights must sum to 1.0)
