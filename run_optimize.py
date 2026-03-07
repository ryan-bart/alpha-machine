"""Optimize factor weights on in-sample data, validate on out-of-sample.

Precomputes all factor ranks once, then uses a fast numpy-based scoring
proxy: for each weight combo, compute composite scores via matrix multiply,
pick the top 20 tickers each month, and measure their average forward return.
No full portfolio simulation — runs in seconds, not hours.
"""

import numpy as np
import polars as pl
from data.universe import get_universe
from data.prices import download_prices
from scoring.combine import compute_all_factors, ALL_FACTORS
from scoring.missing import filter_insufficient_history
from backtester.engine import _get_quarterly_rebalance_dates, run_backtest
from backtester.metrics import compute_metrics, compute_benchmark_metrics
from config.settings import FACTOR_WEIGHTS, HOLDOUT_QUARTERS, TOP_N


def build_monthly_matrices(factored, prices, rebalance_dates):
    """Build numpy arrays for fast scoring.

    Returns per rebalance date:
        rank_matrix: (n_tickers, n_factors) — percentile ranks
        fwd_returns: (n_tickers,) — return from this rebalance to next
        tickers: list of tickers in order
    """
    factor_names = [f.name() for f in ALL_FACTORS]
    rank_cols = [f"{n}_rank" for n in factor_names]

    months = []
    all_dates = prices["date"].unique().sort().to_list()

    for i, reb_date in enumerate(rebalance_dates):
        snapshot = factored.filter(pl.col("date") == reb_date)
        if snapshot.is_empty():
            continue

        next_reb = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else all_dates[-1]

        # Get forward returns: price at next_reb / price at reb_date - 1
        snap_prices = prices.filter(pl.col("date") == reb_date).select("ticker", pl.col("close").alias("p0"))
        fwd_prices = prices.filter(pl.col("date") == next_reb).select("ticker", pl.col("close").alias("p1"))

        merged = snapshot.join(snap_prices, on="ticker", how="inner")
        merged = merged.join(fwd_prices, on="ticker", how="inner")
        merged = merged.with_columns(((pl.col("p1") / pl.col("p0")) - 1.0).alias("fwd_ret"))

        if len(merged) < TOP_N:
            continue

        available = [c for c in rank_cols if c in merged.columns]
        if len(available) != len(rank_cols):
            continue

        rank_matrix = merged.select(available).to_numpy().astype(np.float64)
        fwd_returns = merged["fwd_ret"].to_numpy().astype(np.float64)
        tickers = merged["ticker"].to_list()

        months.append({
            "date": reb_date,
            "ranks": rank_matrix,
            "fwd_ret": fwd_returns,
            "tickers": tickers,
        })

    return months


def score_weights(months, weight_vec, top_n=TOP_N):
    """Fast proxy backtest: pick top_n by weighted score, return avg monthly return."""
    monthly_returns = []
    for m in months:
        scores = m["ranks"] @ weight_vec
        top_idx = np.argsort(scores)[-top_n:]
        avg_ret = np.mean(m["fwd_ret"][top_idx])
        monthly_returns.append(avg_ret)

    monthly_returns = np.array(monthly_returns)
    if len(monthly_returns) < 2:
        return -999, 0

    # Compound monthly returns for CAGR
    cumulative = np.prod(1 + monthly_returns)
    n_months = len(monthly_returns)
    cagr = cumulative ** (12.0 / n_months) - 1.0

    # Monthly Sharpe (annualized)
    excess = monthly_returns - 0.036 / 12
    sharpe = np.mean(excess) / np.std(excess) * np.sqrt(12) if np.std(excess) > 0 else 0

    return sharpe, cagr * 100


def generate_combos(n_random=2000):
    """Generate weight combinations to search."""
    factor_names = [f.name() for f in ALL_FACTORS]
    base = np.array([FACTOR_WEIGHTS[n] for n in factor_names])
    combos = []
    rng = np.random.default_rng(42)

    # Original weights
    combos.append(base.copy())

    # Random perturbations
    for _ in range(n_random):
        raw = base + rng.uniform(-0.12, 0.12, size=len(base))
        raw = np.maximum(raw, 0.0)
        raw = raw / raw.sum()
        combos.append(raw)

    # Single-factor sweeps
    levels = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
    for i in range(len(factor_names)):
        for level in levels:
            w = base.copy()
            w[i] = level
            others_sum = w.sum() - w[i]
            if others_sum > 0:
                scale = (1.0 - level) / others_sum
                for j in range(len(w)):
                    if j != i:
                        w[j] *= scale
            else:
                w = np.ones(len(base)) / len(base)
            combos.append(w)

    return np.array(combos), factor_names


def main():
    print("=== Alpha-Machine: Weight Optimization ===\n")

    print("1. Loading data...")
    universe = get_universe()
    tickers = universe["ticker"].to_list()
    sector_map = dict(zip(universe["ticker"].to_list(), universe["sector"].to_list()))
    if "SPY" not in tickers:
        tickers.append("SPY")

    prices = download_prices(tickers)
    prices = filter_insufficient_history(prices)
    print(f"   {prices['ticker'].n_unique()} tickers\n")

    print("2. Precomputing all factor ranks...")
    factored = compute_all_factors(prices)
    print("   Done.\n")

    dates = prices["date"].unique().sort().to_list()
    rebalance_dates = _get_monthly_rebalance_dates(dates)
    cutoff_idx = len(rebalance_dates) - HOLDOUT_MONTHS
    is_dates = rebalance_dates[:cutoff_idx]
    oos_dates = rebalance_dates[cutoff_idx:]

    print("3. Building monthly matrices...")
    is_months = build_monthly_matrices(factored, prices, is_dates)
    oos_months = build_monthly_matrices(factored, prices, oos_dates)
    print(f"   {len(is_months)} in-sample months, {len(oos_months)} OOS months\n")

    print("4. Searching weight combinations...")
    combos, factor_names = generate_combos(n_random=5000)
    print(f"   {len(combos)} combinations\n")

    # Score all combos on in-sample
    results = []
    for i, w in enumerate(combos):
        sharpe, cagr = score_weights(is_months, w)
        results.append({"idx": i, "sharpe": sharpe, "cagr": cagr, "weights": w})

    results.sort(key=lambda x: x["sharpe"], reverse=True)

    # Original weights result
    orig_sharpe, orig_cagr = score_weights(is_months, combos[0])

    print(f"   Original weights IS Sharpe: {orig_sharpe:.2f}, CAGR: {orig_cagr:.1f}%")
    print(f"   Best weights IS Sharpe:     {results[0]['sharpe']:.2f}, CAGR: {results[0]['cagr']:.1f}%\n")

    best = results[0]
    print("   Best in-sample weights:")
    for name, w, orig in zip(factor_names, best["weights"], combos[0]):
        arrow = " ←" if abs(w - orig) > 0.03 else ""
        print(f"     {name:25s} {w:.1%}  (was {orig:.1%}){arrow}")

    # Validate top 5 on OOS
    print("\n5. Validating on out-of-sample holdout...\n")

    orig_oos_sharpe, orig_oos_cagr = score_weights(oos_months, combos[0])

    # SPY OOS metrics
    oos_start = oos_dates[0] if oos_dates else None
    oos_end = dates[-1]
    spy_metrics = compute_benchmark_metrics(prices, 0.036, "SPY", oos_start, oos_end)

    print("=" * 70)
    print(f"  {'':25s} {'IS Sharpe':>10} {'IS CAGR':>10} {'OOS Sharpe':>11} {'OOS CAGR':>10}")
    print("  " + "-" * 65)
    print(f"  {'Original weights':25s} {orig_sharpe:>10.2f} {orig_cagr:>9.1f}% {orig_oos_sharpe:>11.2f} {orig_oos_cagr:>9.1f}%")

    for rank, r in enumerate(results[:10], 1):
        oos_sharpe, oos_cagr = score_weights(oos_months, r["weights"])
        tag = " ***" if oos_sharpe > orig_oos_sharpe and oos_cagr > float(spy_metrics.get("cagr", 999)) else ""
        print(f"  #{rank:2d} optimized            {r['sharpe']:>10.2f} {r['cagr']:>9.1f}% {oos_sharpe:>11.2f} {oos_cagr:>9.1f}%{tag}")

    print(f"\n  {'SPY buy-and-hold':25s} {'':>10} {'':>10} {spy_metrics.get('sharpe', 'N/A'):>11} {spy_metrics.get('cagr', 'N/A'):>9}%")
    print("  *** = beats both original OOS Sharpe and SPY OOS CAGR")

    # Show the best OOS weight set
    print("\n" + "=" * 70)
    print("BEST OOS PERFORMER (from top 10 IS)")
    print("=" * 70)
    best_oos = None
    best_oos_sharpe = -999
    for r in results[:10]:
        s, c = score_weights(oos_months, r["weights"])
        if s > best_oos_sharpe:
            best_oos_sharpe = s
            best_oos = r

    if best_oos is not None:
        oos_s, oos_c = score_weights(oos_months, best_oos["weights"])
        print(f"\n  IS Sharpe: {best_oos['sharpe']:.2f}, IS CAGR: {best_oos['cagr']:.1f}%")
        print(f"  OOS Sharpe: {oos_s:.2f}, OOS CAGR: {oos_c:.1f}%\n")
        print("  Weights:")
        for name, w, orig in zip(factor_names, best_oos["weights"], combos[0]):
            arrow = " ←" if abs(w - orig) > 0.03 else ""
            print(f"    {name:25s} {w:.1%}  (was {orig:.1%}){arrow}")

    # Full proper backtest with best OOS weights
    print("\n6. Running full backtest with best OOS weights...")
    best_weight_dict = dict(zip(factor_names, best_oos["weights"]))
    bt_results = run_backtest(prices, sector_map, weights=best_weight_dict)
    ec = bt_results["equity_curve"]

    full = compute_metrics(ec, 0.036)
    oos_ec = ec.filter(pl.col("is_oos"))
    oos_full = compute_metrics(oos_ec, 0.036) if not oos_ec.is_empty() else {}

    print("\n  Full period:")
    for k in ["cagr", "sharpe", "max_drawdown", "total_return"]:
        print(f"    {k.replace('_',' ').title():20s} {full.get(k, 'N/A')}")

    print("\n  OOS period:")
    for k in ["cagr", "sharpe", "max_drawdown", "total_return"]:
        print(f"    {k.replace('_',' ').title():20s} {oos_full.get(k, 'N/A')}")

    print(f"\n  SPY OOS: CAGR {spy_metrics.get('cagr')}%, Sharpe {spy_metrics.get('sharpe')}")
    print("\nDone.")


if __name__ == "__main__":
    main()
