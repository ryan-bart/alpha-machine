# Backlog

## Completed
- ~~Transaction cost modeling (spread + slippage)~~ — 10 bps per trade
- ~~Capital gains tax modeling (ST/LT rates, tax loss carry-forward)~~
- ~~Tax-aware trading rules (rebalance bands, LT tax protection)~~
- ~~Strategy profiles (tax_advantaged vs taxable)~~
- ~~Score-proportional position sizing~~ — tested via sweep, equal weight optimal
- ~~RSP (equal-weight S&P) benchmark~~
- ~~Sell threshold optimization~~ — rank 150 optimal
- ~~Russell 1000 universe expansion~~

## Near-Term (Improve Current Strategy)

### Historical Universe Reconstruction
- **Point-in-time Russell 1000 membership**: Use the actual Russell 1000 constituents for each year (reconstitutes every June) instead of today's list for the entire backtest. Fixes survivorship bias and look-ahead bias in universe selection. Cannot be done accurately with yfinance alone (no historical shares outstanding, no delisted tickers). Data options: paid FTSE Russell data, Bloomberg/FactSet, or scrape iShares IWB ETF historical holdings as a free approximation.

### Factor & Signal Improvements
- **Factor/market timing (regime filters)**: Reduce exposure during bear markets. Ideas: moving average regime filter (e.g., reduce position size when SPY < 200-day MA), VIX-based scaling, or trend-following overlay. Could significantly reduce max drawdown (-37%) without hurting CAGR much.
- **Fundamental factors**: Add earnings momentum, revenue growth, or profitability factors. Requires paid data (or scraping SEC filings). Would diversify away from pure price/volume signals.
- **Factor decay analysis**: Measure how quickly each factor's predictive power decays after portfolio formation. Could inform whether monthly rebalancing is worth the extra tax drag for specific factors.

### Portfolio & Execution
- **Correlation-aware position sizing**: Use rolling pairwise correlations to reduce weight on clustered positions. Currently equal-weight ignores that holding 5 tech stocks is less diversified than 5 stocks across sectors.
- **Dynamic position count**: Currently fixed at 20. Could vary based on signal strength — hold fewer stocks when conviction is high, more when scores are bunched.
- **Rebalance timing**: Test rebalancing on different days within the quarter (e.g., avoiding quarter-end rebalancing when institutional flows distort prices).

### Risk Management
- **Drawdown-triggered deleveraging**: Automatically reduce exposure (raise cash) when the portfolio draws down more than X%. Simple rule: if trailing drawdown > 15%, go to 50% cash.
- **Beta hedging**: Hedge market risk with SPY puts or short positions during high-vol regimes. Requires options/futures data.

## Medium-Term (New Capabilities)

### Options
- Options screener: high IV rank + bullish factor score = sell put spreads
- Covered call overlay on long positions
- IV term structure analysis

### Futures
- Managed futures momentum signals
- Roll yield factor

### ML Scoring
- Replace fixed weights with learned weights (ridge regression on forward returns)
- Ensemble: gradient boosted trees on factor ranks
- Walk-forward cross-validation framework
- **Risk**: easy to overfit with ML on financial data. Would need rigorous OOS validation.

## Long-Term (Infrastructure & Scale)

### Paid Data Sources
- Quarterly earnings data (for earnings momentum factor)
- Short interest data
- Insider trading data
- Analyst estimate revisions

### Infrastructure
- Database backend (DuckDB) instead of parquet files
- Scheduled daily runs (cron / Airflow)
- Email/Slack alerts on rebalance days
- Web dashboard (Streamlit)
