# Alpha-Machine

A multi-factor stock scoring system that screens the Russell 1000 and outputs actionable quarterly stock recommendations. Built with Python, Polars, and free data sources (yfinance, Wikipedia, FRED).

---

## What Does This Do?

Alpha-Machine looks at every stock in the Russell 1000 (~1,005 stocks) and ranks them using 10 different measurable signals (called "factors") derived from price and volume data. It combines those rankings into a single composite score per stock, then picks the top 20 to form a portfolio. Think of it like a systematic stock screener that replaces gut feelings with math.

**It answers one question:** "If I had to pick 20 stocks to hold for the next quarter, which ones have the best combination of momentum, stability, and volume characteristics?"

### What It Recommends

When you run `python run_score.py`, you get a ranked list of 20 stocks to buy. The intended holding period is **one quarter** — on the first trading day of each quarter, you would:

1. Sell any stocks that dropped out of the top 20 (unless they're still ranked within the top 150 — see turnover dampening below)
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
2. **Percentile rank** each factor across all ~1,005 stocks. The stock with the highest momentum gets a rank of 1.0, the lowest gets 0.0, and everything else falls in between. This makes factors comparable — you can't directly compare a momentum percentage to a volatility number, but you can compare "top 10% in momentum" to "top 10% in volatility"
3. **Weighted average** of all 10 percentile ranks using the weights above, producing a single composite score between 0 and 1
4. **Select the top 20** stocks by composite score

### No Lookahead Bias

A critical rule: when scoring stocks on date X, only data from date X and earlier is used. The system never "peeks" at future prices. This seems obvious but is the #1 source of bugs in backtesting systems.

---

## Portfolio Construction

Picking the top 20 stocks isn't quite as simple as sorting by score. There are constraints:

- **Minimum score**: A stock must be above the 50th percentile composite score to be included. If fewer than 20 stocks pass this bar, the portfolio holds fewer positions (and implicitly holds cash).
- **Turnover dampening**: A stock already in the portfolio doesn't get removed unless it drops below rank 150. This prevents churning — without this rule, a stock oscillating between rank 19 and 21 would be bought and sold every quarter, generating unnecessary trading costs. Rank 150 was selected via parameter sweep as the best tradeoff between turnover reduction and performance.
- **Equal weight**: Each position gets an equal ~5% allocation. This is simpler and more diversified than market-cap weighting where a few mega-caps dominate. Score-proportional sizing was tested via sweep but showed no benefit — with 20 stocks selected from the top of the distribution, composite scores are too bunched together for tilting to add signal.
- **Delta-based rebalancing**: When rebalancing, only the difference between current and target allocations is traded, not full liquidation and re-purchase.

---

## Strategy Profiles

Alpha-Machine supports two strategy profiles optimized for different account types:

### Tax-Advantaged (`python run_backtest.py tax_advantaged`)

For IRA, 401(k), and other tax-deferred accounts. No tax-aware rules — reports pre-tax results only. Tight rebalance bands (1%) for precise allocation tracking.

### Taxable (`python run_backtest.py taxable`)

For taxable brokerage accounts. Includes:

- **Transaction costs**: 10 basis points (0.10%) per trade, modeled as price slippage
- **Capital gains taxes**: Short-term gains (<1 year held) taxed at 35%, long-term gains (>=1 year) at 15%
- **Tax loss carry-forward**: Realized losses offset future gains before taxes are applied
- **Wider rebalance bands** (5%): Only rebalance a position if it drifts more than 5% from its target weight, reducing unnecessary taxable events
- **Long-term tax protection** (90 days): Defers selling a profitable position if it's within 90 days of qualifying for long-term capital gains treatment

---

## Backtest Results (Russell 1000, 10-Year, 2016-2026)

40 quarterly rebalance periods. 32 in-sample, 8 out-of-sample (2-year holdout).

### Tax-Advantaged (Pre-Tax)

| Period | CAGR | Sharpe | Max Drawdown |
|--------|------|--------|-------------|
| **Full (2016-2026)** | 17.8% | 0.71 | -37.4% |
| **In-Sample (2016-2024)** | 16.8% | 0.67 | -37.4% |
| **Out-of-Sample (2024-2026)** | 22.8% | 1.05 | -17.1% |

### Benchmarks (Same Periods)

| Benchmark | Period | CAGR | Sharpe | Max Drawdown |
|-----------|--------|------|--------|-------------|
| **SPY** (S&P 500) | Full | 14.9% | 0.67 | -33.7% |
| **SPY** | OOS | 15.3% | 0.73 | -18.8% |
| **RSP** (Equal-Weight S&P) | Full | 11.7% | 0.50 | -39.7% |
| **RSP** | OOS | 9.1% | 0.37 | -16.5% |

### Taxable (After Costs & Taxes)

| Period | CAGR | Sharpe | Max Drawdown | Tax Drag |
|--------|------|--------|-------------|----------|
| **Full** | 12.0% | 0.47 | -36.1% | -3.50%/yr |

Note: After-tax returns trail SPY buy-and-hold (14.9%) because SPY defers all capital gains taxes indefinitely. This strategy is best deployed in tax-advantaged accounts where the full pre-tax edge is realized.

### Key Takeaways

- Beats SPY by +2.9% CAGR pre-tax over the full 10-year period
- OOS Sharpe (1.05) improved vs. in-sample (0.67) — no evidence of overfitting
- Russell 1000's wider mid-cap pool gives the scoring system more dispersion to exploit vs. S&P 500 alone
- Tax-aware rules (wider bands + LT protection) reduced drag from -4.37%/yr to -3.50%/yr

---

## What Is Backtesting?

Backtesting answers: "If I had followed this exact strategy in the past, how would I have done?"

### How It Works

The backtester simulates running the strategy quarter by quarter starting from 2016:

1. On the first trading day of Quarter 1, score all stocks using only data available at that point. Pick the top 20. "Buy" them.
2. On the first trading day of Quarter 2, score all stocks again (still only using data up to that day). Sell any that dropped out, buy new ones. Track the portfolio value.
3. Repeat for every quarter in the dataset (40 quarters total).

At the end, you have a full history of what the portfolio would have been worth on every day — this is the **equity curve**.

### Overfitting Safeguards

The biggest risk in backtesting is **overfitting** — tuning your strategy to perfectly fit historical data in a way that won't generalize to the future.

Alpha-Machine guards against this in three ways:

1. **Out-of-sample holdout**: The most recent 8 quarters (2 years) of data are reserved as a "holdout" set. The strategy is developed and evaluated on older data first, then checked against the holdout. If performance collapses on the holdout, it's a red flag for overfitting.
2. **No parameter optimization**: The factor weights and thresholds are set based on academic research and intuition, not by searching for the combination that maximizes backtest returns.
3. **Simple, transparent rules**: 10 factors, fixed weights, quarterly rebalance. There are very few "knobs to turn," which limits the opportunity to overfit.

---

## Performance Metrics Explained

### CAGR (Compound Annual Growth Rate)
The average annual return, accounting for compounding.

### Sharpe Ratio
Return per unit of risk: (strategy return - risk-free rate) / volatility. Above 1.0 is generally good.

### Max Drawdown
The worst peak-to-trough decline. Tells you the worst pain you would have experienced.

### Calmar Ratio
CAGR / |max drawdown|. Rewards high returns with shallow drawdowns.

### Annual Volatility
Annualized standard deviation of daily returns. Lower = smoother ride.

### Monthly Hit Rate
Percentage of months with positive returns. 60%+ is solid.

---

## Project Structure

```
alpha-machine/
├── config/
│   └── settings.py              # All tunable parameters (weights, thresholds, strategies)
├── data/
│   ├── universe.py              # Fetches Russell 1000 ticker list from Wikipedia
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
│   ├── engine.py                # Walk-forward backtest with position tracking and tax modeling
│   └── metrics.py               # Performance metric calculations
├── output/
│   ├── report.py                # Generates markdown recommendation report
│   └── plots.py                 # Equity curve and monthly returns heatmap
├── cache/                       # Local data cache (gitignored)
├── run_score.py                 # Score today's Russell 1000 and produce recommendations
├── run_backtest.py              # Run historical backtest (supports strategy profiles)
├── run_sweep.py                 # Sweep parameters (sell threshold, score tilt)
├── run_optimize.py              # Weight optimizer (experimental)
└── requirements.txt
```

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate today's stock recommendations
python run_score.py

# Run the historical backtest (default: taxable strategy)
python run_backtest.py

# Run for IRA/401k (pre-tax only)
python run_backtest.py tax_advantaged

# Sweep parameters to test tradeoffs
python run_sweep.py threshold   # sell threshold rank values
python run_sweep.py tilt        # score tilt (position sizing) values
```

The first run downloads ~10 years of daily price data for ~1,005 Russell 1000 stocks. This takes several minutes. Subsequent runs use cached data and only download new days.

### Output

- `run_score.py` prints a ranked table of 20 stock recommendations and saves it to `cache/recommendation_report.md`
- `run_backtest.py` prints performance metrics and saves charts to `cache/equity_curve.png` and `cache/monthly_returns.png`

---

## Data Sources

All data is free and requires no API keys:

| Source | What | How |
|--------|------|-----|
| **yfinance** | Daily OHLCV price data for ~1,005 stocks | `yf.download()` — pulls from Yahoo Finance |
| **Wikipedia** | Current Russell 1000 constituents and their sectors | Scraped from the Russell 1000 Index table |
| **FRED** | 3-month Treasury bill rate (risk-free rate for Sharpe ratio) | CSV download from Federal Reserve Economic Data |

---

## Limitations

- **Survivorship bias**: The system uses today's Russell 1000 list for the entire backtest. Stocks that were removed from the index (because they crashed, got acquired, etc.) are not included, which makes historical results look slightly better than they really were.
- **Price-only factors**: The system only uses price and volume data. Fundamental data (earnings, revenue, debt) could improve accuracy but requires paid data sources.
- **Static universe**: The Russell 1000 membership changes annually; using today's list for a 10-year backtest introduces some look-ahead bias in universe selection.

---

## Disclaimer

This is a research and educational tool. It is not financial advice. The recommendations are generated by a mechanical system with no awareness of earnings reports, Fed decisions, geopolitical events, or any other real-world context. Past backtest performance does not predict future results. Use at your own risk.
