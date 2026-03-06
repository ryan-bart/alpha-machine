# Alpha-Machine

A multi-factor stock scoring system that screens the S&P 500 and outputs actionable quarterly stock recommendations. Built with Python, Polars, and free data sources (yfinance, Wikipedia, FRED).

---

## What Does This Do?

Alpha-Machine looks at every stock in the S&P 500 and ranks them using 10 different measurable signals (called "factors") derived from price and volume data. It combines those rankings into a single composite score per stock, then picks the top 20 to form a portfolio. Think of it like a systematic stock screener that replaces gut feelings with math.

**It answers one question:** "If I had to pick 20 stocks to hold for the next quarter, which ones have the best combination of momentum, stability, and volume characteristics?"

### What It Recommends

When you run `python run_score.py`, you get a ranked list of 20 stocks to buy. The intended holding period is **one quarter** — on the first trading day of each quarter, you would:

1. Sell any stocks that dropped out of the top 20
2. Buy any new stocks that entered the top 20
3. Rebalance so each position is an equal ~5% of your portfolio

You are not meant to day-trade these picks or hold them for years. The factors are tuned for a **quarterly horizon** where momentum and low-volatility effects are strongest.

### Why Would This Work?

The strategy is based on well-documented effects in academic finance:

- **Momentum**: Stocks that have gone up over the past 3-12 months tend to continue going up for a while. This is the most robust and widely studied factor in finance (Jegadeesh & Titman, 1993).
- **Low volatility**: Less volatile stocks tend to deliver better risk-adjusted returns than highly volatile ones. The "low-volatility anomaly" contradicts the textbook idea that more risk = more reward.
- **Volume confirmation**: Rising prices on rising volume are more likely to continue than rising prices on falling volume.

None of these are guaranteed to work in any given quarter. They are statistical tendencies that show up over many quarters and many stocks.

---

## How the Scoring Works

### The 10 Factors

Each stock gets scored on 10 factors. Every factor produces a single number per stock per day.

| # | Factor | What It Measures | Weight |
|---|--------|-----------------|--------|
| 1 | **Momentum 12-1mo** | Price change over the past year, excluding the most recent month (to avoid short-term reversal noise) | 25% |
| 2 | **Momentum 6-1mo** | Same idea but over 6 months — captures medium-term trends | 15% |
| 3 | **Relative Strength 3mo** | Price change over the past 3 months — shorter-term momentum | 10% |
| 4 | **Short-Term Reversal** | Price change over the past week, *inverted* — disabled (0% weight) because it decays too fast for quarterly rebalancing | 0% |
| 5 | **Distance from 50-day MA** | How far the stock is above its 50-day moving average, *inverted* — prefers stocks near their average rather than overextended | 5% |
| 6 | **Volume Trend** | Is trading volume increasing or decreasing? Compares recent 20-day average volume to 60-day average | 5% |
| 7 | **OBV Slope** | On-Balance Volume trend — tracks whether volume flows into or out of a stock over 40 days | 5% |
| 8 | **Realized Volatility** | How much the stock's price fluctuates day-to-day (60-day window), *inverted* — calmer stocks score higher | 15% |
| 9 | **Volatility Trend** | Is volatility increasing or decreasing? *Inverted* — stocks with declining volatility score higher | 5% |
| 10 | **Price Consistency** | What fraction of the last 60 trading days had positive returns — measures how "steadily" a stock has been climbing | 15% |

"Inverted" means lower raw values get higher scores. For example, a stock with low volatility scores *better* on the realized volatility factor.

The weights are tuned for quarterly rebalancing: persistent signals (momentum, low-vol, price consistency) are weighted heavily, while fast-decaying signals (short-term reversal, volume spikes) are reduced or disabled.

### How Factors Become Scores

1. **Compute raw factor values** for every stock on a given date
2. **Percentile rank** each factor across all ~500 stocks. The stock with the highest momentum gets a rank of 1.0, the lowest gets 0.0, and everything else falls in between. This makes factors comparable — you can't directly compare a momentum percentage to a volatility number, but you can compare "top 10% in momentum" to "top 10% in volatility"
3. **Weighted average** of all 10 percentile ranks using the weights above, producing a single composite score between 0 and 1
4. **Select the top 20** stocks by composite score

### No Lookahead Bias

A critical rule: when scoring stocks on date X, only data from date X and earlier is used. The system never "peeks" at future prices. This seems obvious but is the #1 source of bugs in backtesting systems.

---

## Portfolio Construction

Picking the top 20 stocks isn't quite as simple as sorting by score. There are constraints:

- **Minimum score**: A stock must be above the 50th percentile composite score to be included. If fewer than 20 stocks pass this bar, the portfolio holds fewer positions (and implicitly holds cash).
- **Turnover dampening**: A stock already in the portfolio doesn't get removed unless it drops below rank 15. This prevents churning — without this rule, a stock oscillating between rank 19 and 21 would be bought and sold every quarter, generating unnecessary trading costs.
- **Equal weight**: Each position gets an equal ~5% allocation. This is simpler and more diversified than market-cap weighting where a few mega-caps dominate.

---

## What Is Backtesting?

Backtesting answers: "If I had followed this exact strategy in the past, how would I have done?"

### How It Works

The backtester simulates running the strategy quarter by quarter starting from the beginning of the historical data:

1. On the first trading day of Quarter 1, score all stocks using only data available at that point. Pick the top 20. "Buy" them.
2. On the first trading day of Quarter 2, score all stocks again (still only using data up to that day). Sell any that dropped out, buy new ones. Track the portfolio value.
3. Repeat for every quarter in the dataset.

At the end, you have a full history of what the portfolio would have been worth on every day — this is the **equity curve**.

### Why It's Valid

The key insight: **the backtest never uses future information**. On each simulated rebalance date, the system only sees data that was actually available at that time. So the recommendations you get today use the exact same logic that was used in the backtest — just applied to today's data instead of historical data.

The backtest doesn't prove the strategy *will* work going forward. Markets change. But it does show whether the strategy's logic *would have* captured the effects it's designed to capture. If a momentum-based strategy can't even beat buy-and-hold in a backtest, it's probably not worth running live.

### Overfitting Safeguards

The biggest risk in backtesting is **overfitting** — tuning your strategy to perfectly fit historical data in a way that won't generalize to the future. Like memorizing test answers instead of learning the material.

Alpha-Machine guards against this in three ways:

1. **Out-of-sample holdout**: The most recent 4 quarters of data are reserved as a "holdout" set. The strategy is developed and evaluated on older data first, then checked against the holdout. If performance collapses on the holdout, it's a red flag for overfitting.
2. **No parameter optimization**: The factor weights and thresholds are set based on academic research and intuition, not by searching for the combination that maximizes backtest returns.
3. **Simple, transparent rules**: 10 factors, fixed weights, quarterly rebalance. There are very few "knobs to turn," which limits the opportunity to overfit.

---

## Performance Metrics Explained

When the backtest runs, it produces several metrics. Here's what each one means:

### CAGR (Compound Annual Growth Rate)

The average annual return, accounting for compounding. If you start with $100,000 and end with $184,510 after 3 years, your CAGR is 22.67%. This is the single most intuitive performance number — "how much did it grow per year, on average?"

### Sharpe Ratio

Measures **return per unit of risk**. Calculated as: (strategy return - risk-free rate) / volatility of returns.

- A Sharpe of 1.0 means you earned 1% of excess return for every 1% of volatility
- Above 1.0 is generally considered good
- Above 2.0 is exceptional (and rare for a long-only stock strategy)

The "risk-free rate" is what you'd earn with zero risk (3-month Treasury bills, fetched from FRED). The Sharpe ratio asks: "Was the extra return worth the extra risk compared to just parking money in T-bills?"

### Max Drawdown

The worst peak-to-trough decline during the backtest period. If the portfolio went from $175,000 to $140,000 at its worst point, that's a -20% max drawdown. This is arguably the most important risk metric — it tells you the worst pain you would have experienced. A strategy with great returns but a -50% drawdown means you'd have watched half your money evaporate at some point.

### Calmar Ratio

CAGR divided by the absolute value of max drawdown. A Calmar of 1.12 means you earned 1.12% annually for every 1% of worst-case decline. Higher is better — it rewards high returns with shallow drawdowns.

### Annual Volatility

The annualized standard deviation of daily returns. Measures how much the portfolio value bounces around day to day. Lower volatility means smoother returns. A 16% annual vol means roughly a 1% daily standard deviation (16% / sqrt(252 trading days)).

### Monthly Hit Rate

The percentage of months with positive returns. A 67.6% hit rate means roughly 2 out of 3 months were profitable. Even good strategies lose money some months — a 60%+ hit rate is solid.

### Total Return

Simple cumulative return over the entire period. Starting with $100,000 and ending at $184,510 is an 84.51% total return. Unlike CAGR, this doesn't account for the time period — an 84% return over 3 years is very different from 84% over 10 years.

### HHI (Herfindahl-Hirschman Index)

A concentration measure used in the sector exposure section. It's the sum of squared portfolio weights. For 20 equal-weight positions (5% each), HHI = 20 * 0.05^2 = 0.05. Lower HHI means more diversified. An HHI of 1.0 would mean the entire portfolio is in one stock.

---

## What Is an Equity Curve?

An equity curve is a line chart showing the portfolio's total value over time. It's the single most important visualization in backtesting.

The X-axis is time (dates), the Y-axis is portfolio value in dollars. If you started with $100,000, the line shows what your account balance would have been on every trading day.

In the Alpha-Machine equity curve plot:
- The **blue line** is the strategy's portfolio value
- The **orange line** is SPY (S&P 500 ETF) buy-and-hold — the benchmark you're trying to beat
- The **red dashed line** marks where the out-of-sample holdout period begins

When the blue line is above the orange line, the strategy is outperforming the market. The slope of the line shows the growth rate. Steep drops are drawdowns. A smooth, steadily rising line is ideal; a jagged, volatile line means the ride was bumpy.

---

## Project Structure

```
alpha-machine/
├── config/
│   └── settings.py              # All tunable parameters (weights, thresholds, paths)
├── data/
│   ├── universe.py              # Fetches S&P 500 ticker list from Wikipedia
│   ├── prices.py                # Downloads price data via yfinance, caches as parquet
│   └── macro.py                 # Fetches risk-free rate from FRED
├── factors/
│   ├── base.py                  # Base interface for all factors
│   ├── momentum.py              # Momentum factors (12-1mo, 6-1mo, 3mo)
│   ├── mean_reversion.py        # Reversal factors (5-day reversal, distance from MA)
│   ├── volume.py                # Volume factors (volume trend, OBV slope)
│   ├── volatility.py            # Volatility factors (realized vol, vol trend)
│   └── quality.py               # Quality factors (price consistency)
├── scoring/
│   ├── normalize.py             # Converts raw factors to percentile ranks
│   ├── combine.py               # Combines ranked factors into composite score
│   └── missing.py               # Handles missing data (filters + imputation)
├── portfolio/
│   ├── construct.py             # Selects top N stocks with constraints
│   └── risk.py                  # Sector exposure and concentration checks
├── backtester/
│   ├── engine.py                # Walk-forward quarterly backtest simulation
│   └── metrics.py               # Performance metric calculations
├── output/
│   ├── report.py                # Generates markdown recommendation report
│   └── plots.py                 # Equity curve and monthly returns heatmap
├── cache/                       # Local data cache (gitignored)
├── run_score.py                 # Score today's S&P 500 and produce recommendations
├── run_backtest.py              # Run historical backtest and generate performance report
├── run_optimize.py              # Weight optimizer (experimental)
├── BACKLOG.md                   # Future work: options, ML scoring, paid data
└── requirements.txt
```

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate today's stock recommendations
python run_score.py

# Run the historical backtest
python run_backtest.py
```

The first run downloads ~3 years of daily price data for all S&P 500 stocks (~500 tickers). This takes a few minutes. Subsequent runs use cached data and only download new days.

### Output

- `run_score.py` prints a ranked table of 20 stock recommendations and saves it to `cache/recommendation_report.md`
- `run_backtest.py` prints performance metrics and saves charts to `cache/equity_curve.png` and `cache/monthly_returns.png`

---

## Data Sources

All data is free and requires no API keys:

| Source | What | How |
|--------|------|-----|
| **yfinance** | Daily OHLCV price data for all S&P 500 stocks | `yf.download()` — pulls from Yahoo Finance |
| **Wikipedia** | Current list of S&P 500 constituents and their sectors | Scraped from the S&P 500 companies table |
| **FRED** | 3-month Treasury bill rate (risk-free rate for Sharpe ratio) | CSV download from Federal Reserve Economic Data |

---

## Limitations

- **Survivorship bias**: The system uses today's S&P 500 list for the entire backtest. Stocks that were removed from the index (because they crashed, got acquired, etc.) are not included, which makes historical results look slightly better than they really were.
- **No transaction costs**: The backtest doesn't account for trading commissions or bid-ask spreads. With 20 stocks rebalanced quarterly, real-world costs would be modest but nonzero.
- **Price-only factors**: The system only uses price and volume data. Fundamental data (earnings, revenue, debt) could improve accuracy but requires paid data sources.
- **3-year history**: With only ~3 years of data from yfinance, the backtest covers limited market conditions. A full market cycle (bull market, bear market, recovery) typically takes 7-10 years.

---

## Disclaimer

This is a research and educational tool. It is not financial advice. The recommendations are generated by a mechanical system with no awareness of earnings reports, Fed decisions, geopolitical events, or any other real-world context. Past backtest performance does not predict future results. Use at your own risk.
