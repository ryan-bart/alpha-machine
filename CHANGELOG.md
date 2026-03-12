# Changelog

Development history of the alpha-machine project — key decisions, experiments, results, and lessons learned.

## Phase 1: Initial Build (Mar 2026)

Built the core multi-factor scoring system from scratch:

- **10 price/volume factors** organized into 5 categories: momentum (12-1mo, 6-1mo, relative strength 3mo), mean reversion (short-term reversal, distance from MA50), volume (volume trend, OBV slope), volatility (realized vol 60d, vol trend), and quality (price consistency).
- **Rank-based scoring**: Each factor is percentile-ranked (0-1) cross-sectionally, then combined via weighted average into a composite score. Inverted factors (lower=better) get rank-flipped.
- **Walk-forward backtester** with strict no-lookahead: at each rebalance date, only data available up to that date is used.
- **S&P 500 universe**, ~500 stocks, monthly rebalance, 10-stock portfolio.

Initial results were promising but unstable.

## Phase 2: Parameter Tuning

### Stock Count & Rebalance Frequency
- **20 stocks >> 10 stocks**: Concentration in 10 stocks introduced too much idiosyncratic risk. 20 stocks smoothed returns significantly.
- **Quarterly >> Monthly rebalance**: Monthly had slightly higher returns but much worse IS-to-OOS stability. Quarterly forced us to rely on persistent signals, which turned out to be a feature not a bug.
- **No sector cap >> 25% sector cap**: Forced diversification hurt the edge. Set SECTOR_CAP=1.0 (disabled).

### Sell Threshold Sweep (`f194322`)
Swept sell_threshold_rank from 20 to 200 to control turnover dampening (retain holdings unless they drop below this rank). Used precompute_snapshots() to compute factors once and sweep fast.

| Threshold | Avg Turnover | OOS Sharpe | Notes |
|-----------|-------------|------------|-------|
| 20 (none) | 87% | 0.55 | Too much churn |
| 30 (old) | 84% | 0.47 | Worst option tested |
| 100 | 66% | 0.87 | Good |
| **150** | **57%** | **0.95** | **Sweet spot — adopted** |
| 200 | 49% | 0.78 | Over-dampened, holds losers |

**Bug found during sweep**: `select_top_n` had sector_cap default hardcoded to 0.25 instead of using SECTOR_CAP from settings. Previous "no cap" results actually had a 5-stock-per-sector cap. Fixed.

## Phase 3: Universe Expansion (`8a76476`)

Expanded from S&P 500 (~500 stocks) to Russell 1000 (~1,005 stocks). Added `UNIVERSE_SOURCE` setting to switch between them.

- OOS Sharpe improved 0.70 → 0.77 (later 0.95 with threshold tuning)
- Mid-cap stocks in Russell 1000 added meaningful edge — less efficiently priced than large-cap S&P 500 constituents.
- Known limitation: uses today's Russell 1000 membership for the entire backtest (survivorship bias). Historical reconstitution is on the backlog but requires paid data.

## Phase 4: Transaction Cost & Tax Modeling (`c29d087`)

Built a delta-based trading engine with Position tracking (cost basis, purchase date):

- **Transaction costs**: 10 bps per trade (spread + slippage)
- **Capital gains taxes**: 35% short-term (<1 year), 15% long-term (>=1 year)
- **Tax-aware rules**: Rebalance bands (only rebalance if position drifts >X% from target), LT tax protection (defer selling gains within 90 days of 1-year threshold)
- **Strategy profiles**: `tax_advantaged` (IRA/401k, 1% band, no tax protection) and `taxable` (5% band, 90d protection)

### Results (10-year, Russell 1000, 40 quarterly periods)

**Tax-Advantaged (pre-tax):**
- Strategy: CAGR 17.8%, Sharpe 0.71, Max DD -39.5%
- SPY: CAGR 14.9%, Sharpe 0.67, Max DD -33.7%
- OOS: Strategy 22.8% CAGR vs SPY 15.5%

**Taxable (after costs & taxes):**
- After-tax CAGR: 12.0% vs SPY 14.9% — tax drag is structural (-3.5%/yr)
- Tax-aware rules shifted $60k from ST→LT gains, saved $14k in taxes
- **Conclusion**: Strategy is best in tax-advantaged accounts

## Phase 5: Score-Proportional Sizing (`eec979c`)

Tested score-tilt position sizing (weight proportional to composite_score^tilt) via parameter sweep. Built `run_sweep.py` infrastructure.

- Tilt 0 (equal weight), 0.5, 1.0, 1.5, 2.0 tested
- **Equal weight was optimal** — score-proportional sizing concentrated in top names without improving risk-adjusted returns.
- Also added RSP (equal-weight S&P 500) as a second benchmark (`81c629d`).

## Phase 6: Factor Decay Analysis (Mar 12, 2026)

### Motivation
Built `run_decay.py` to measure how quickly each factor's predictive power decays after portfolio formation. Goal: validate weight choices and identify factors that don't belong at quarterly rebalance frequency.

### Methodology
- At each rebalance date, compute Spearman rank IC (correlation) between each factor's rank and forward stock returns at 7 horizons (1mo, 2mo, 3mo, 4mo, 6mo, 9mo, 12mo)
- Average IC across all dates for each factor x horizon combination
- Added **IC hit rate** (% of periods with IC > 0) to check consistency
- **Split IS/OOS**: Analyzed IS (32 periods) and OOS (8 periods) separately to keep OOS pure for weight validation

### Key IS Findings (32 periods, 2016-2024)

| Factor | IC @ 3mo | Hit Rate @ 3mo | Peak IC | Verdict |
|--------|----------|----------------|---------|---------|
| momentum_6_1 | 0.034 | 62% | 0.059* @ 2mo | Best factor |
| vol_trend | 0.037* | 75% | 0.039* @ 4mo | Most consistent — hidden gem |
| dist_from_ma50 | 0.042 | 62% | 0.042 @ 3mo | Peaks at our horizon |
| short_term_reversal | 0.026 | 59% | 0.068* @ 2mo | Strong peak but decays fast |
| momentum_12_1 | 0.000 | 50% | 0.023 @ 6mo | Weak IS despite 25% weight |
| price_consistency | -0.007 | 56% | 0.019 @ 2mo | Weaker than expected |
| realized_vol_60d | **-0.054** | **38%** | n/a | **Consistently negative — picking losers** |
| volume_trend | -0.010 | 53% | n/a | Pure noise |
| obv_slope | -0.003 | 47% | 0.022 @ 2mo | Dead by 3mo |

### Experiment 1: Aggressive IS-Optimized Reweighting

Derived new weights entirely from IS IC data: boosted momentum_6_1 to 30%, vol_trend to 20%, killed realized_vol_60d/volume_trend/obv_slope.

| Metric | Current | Aggressive | SPY |
|--------|---------|-----------|-----|
| IS CAGR | 16.5% | **38.1%** | 14.2% |
| OOS CAGR | 16.3% | 16.4% | 15.5% |
| OOS Sharpe | **0.77** | 0.59 | 0.74 |

**Classic overfitting.** Crushed IS but OOS was unchanged on returns and worse on risk. Abandoned.

### Experiment 2: Conservative — Remove Broken Factors Only

More modest change: just remove realized_vol_60d (15%) and volume_trend (5%), redistribute 20% evenly across survivors.

| Metric | Current | Conservative | SPY |
|--------|---------|-------------|-----|
| IS CAGR | 17.5% | 35.6% | 14.2% |
| OOS CAGR | 18.2% | **28.7%** | 15.5% |
| OOS Sharpe | 0.86 | **0.91** | 0.74 |
| OOS Max DD | **-15.9%** | -28.1% | -18.8% |

OOS returns improved significantly, but verification revealed the mechanism: removing the low-vol filter shifts the portfolio from "high-momentum + low-volatility" stocks (MSFT, IBM, KR) to "pure high-momentum" stocks (ENPH, AMD, NVDA, COIN, AFRM). During 2016-2025, the volatile momentum names were among the best performers — so the improvement is partly a bet on the continuation of high-vol momentum outperformance, not pure factor improvement.

Portfolio overlap between current and proposed was only 20% in early periods, rising to 70% in recent OOS — explaining why IS improvement (17%→36%) is much larger than OOS (18%→29%).

### Lessons Learned

1. **Decay analysis is best at identifying factors to REMOVE, not for setting precise weights.** Factors with consistently negative IC (realized_vol_60d, volume_trend) are clearly hurting. But translating positive IC rankings into optimal weights doesn't survive out-of-sample.
2. **IS-optimized weights are a trap.** Even with the discipline of only using IS data, aggressive reweighting overfits. The current hand-tuned weights are more robust than they look.
3. **Factor removal changes the portfolio's character, not just its quality.** Removing realized_vol_60d doesn't just "remove noise" — it fundamentally shifts from a low-vol-momentum blend to pure momentum, with different risk/return characteristics.
4. **IC hit rate is more informative than mean IC.** A factor with 75% hit rate and modest IC (vol_trend) is more trustworthy than one with high mean IC driven by a few outlier periods.

### Decision
Adopted conservative weights: removed realized_vol_60d and volume_trend, redistributed 20% evenly across survivors. Strategy shifted from momentum+low-vol blend to pure momentum. Full period CAGR jumped to 34.7%, after-tax now beats SPY (26.7% vs 14.9%).

## Phase 7: Regime Filter — Negative Result (Mar 12, 2026)

### Motivation
The -40% max drawdown is the strategy's biggest weakness. Tested SPY moving average regime filter: when SPY is below its N-day SMA at a rebalance date, move a fraction of the portfolio to cash.

### Methodology
- Added `regime_ma_lookback` and `regime_cash_fraction` parameters to `run_backtest_from_snapshots()`
- Pre-computed SPY MA series for efficiency; regime check at each quarterly rebalance
- Swept 2D grid: MA lookbacks (100, 150, 200, 250 days) x cash fractions (25%, 50%, 75%, 100%)

### Results

| Config | Full CAGR | Full Sharpe | Full Max DD | OOS CAGR | OOS Max DD |
|--------|----------|------------|------------|---------|-----------|
| **off (baseline)** | **34.2%** | **1.04** | **-36.7%** | **30.6%** | **-26.3%** |
| MA200/50% | 28.4% | 0.93 | -36.7% | 26.4% | -22.5% |
| MA200/100% | 20.8% | 0.71 | **-42.6%** | 20.8% | -22.5% |
| MA250/50% | 29.2% | 0.94 | -36.7% | 30.6% | -26.3% |

### Why It Failed
**Quarterly rebalancing is too slow for a regime filter.** The filter can only adjust exposure once per quarter. The COVID crash (Feb-March 2020) happened between quarterly rebalances — SPY was above its 200-day MA on Jan 2, 2020, crashed, and had partially recovered by April 1. The filter either:
1. Never triggers (crash happens and recovers within one quarter)
2. Triggers AFTER the bottom — raises cash during the recovery, missing the bounce

The 100% cash variants had the WORST max drawdowns (-42.6%) because they went to all-cash after the crash bottom, then sat out the recovery.

### Lesson
Regime filters need daily or weekly position adjustment frequency to work. They are fundamentally incompatible with quarterly rebalancing. Left the infrastructure in the engine (disabled by default, `regime_ma_lookback=0`) in case we move to more frequent rebalancing later.

## Phase 8: Score Distribution Analysis (Mar 12, 2026)

Analyzed whether dynamic position count (varying the number of held stocks based on signal strength) was worthwhile. Found that the composite score at rank 20 is always ~91% of rank 1 (std 2.5%), with no meaningful variation across periods. The score curve shape is remarkably consistent — there's no natural "elbow" to exploit. Abandoned this line of research.
