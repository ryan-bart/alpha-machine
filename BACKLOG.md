# Backlog

## Options
- Options screener: high IV rank + bullish factor score = sell put spreads
- Covered call overlay on long positions
- IV term structure analysis

## Futures
- Managed futures momentum signals
- Roll yield factor

## Paid Data Sources
- Quarterly earnings data (for earnings momentum factor)
- Short interest data
- Insider trading data
- Analyst estimate revisions

## ML Scoring
- Replace fixed weights with learned weights (ridge regression)
- Ensemble: gradient boosted trees on factor ranks
- Walk-forward cross-validation framework

## Infrastructure
- Database backend (DuckDB) instead of parquet files
- Scheduled daily runs (cron / Airflow)
- Email/Slack alerts on rebalance days
- Web dashboard (Streamlit)

## Risk
- Correlation-based position sizing
- Drawdown-triggered deleveraging
- Beta hedging with SPY shorts
