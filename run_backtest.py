"""Backtest the multi-factor scoring system historically.

Usage:
    python run_backtest.py                # uses default strategy
    python run_backtest.py tax_advantaged  # IRA/401k (pre-tax only)
    python run_backtest.py taxable         # taxable account (after-tax metrics)
"""

import sys
from data.universe import get_universe
from data.prices import download_prices
from data.macro import get_risk_free_rate
from scoring.missing import filter_insufficient_history
from backtester.engine import precompute_snapshots, run_backtest_from_snapshots
from backtester.metrics import compute_metrics, compute_benchmark_metrics
from output.plots import plot_equity_curve, plot_monthly_heatmap
from config.settings import (
    FACTOR_WEIGHTS, TOP_N, SECTOR_CAP, SELL_THRESHOLD_RANK, REBALANCE_FREQ,
    WEIGHTING, UNIVERSE_SOURCE, TRANSACTION_COST_BPS, SHORT_TERM_TAX_RATE,
    LONG_TERM_TAX_RATE, INITIAL_CASH, STRATEGIES, DEFAULT_STRATEGY,
)
import polars as pl


def main():
    # Strategy selection
    strategy_name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_STRATEGY
    if strategy_name not in STRATEGIES:
        print(f"Unknown strategy '{strategy_name}'. Available: {', '.join(STRATEGIES.keys())}")
        sys.exit(1)
    strategy = STRATEGIES[strategy_name]

    print("=== Alpha-Machine: Historical Backtest ===\n")

    print(f"Strategy: {strategy_name} — {strategy['description']}")
    print(f"  Universe: {UNIVERSE_SOURCE} | Positions: {TOP_N} | Weighting: {WEIGHTING} | Sector cap: {'off' if SECTOR_CAP >= 1.0 else f'{SECTOR_CAP:.0%}'} | Sell threshold: rank {SELL_THRESHOLD_RANK} | Rebalance: {'quarterly' if REBALANCE_FREQ == 'QS' else 'monthly'}")
    if strategy["show_after_tax"]:
        print(f"  Transaction cost: {TRANSACTION_COST_BPS} bps | Tax rates: {SHORT_TERM_TAX_RATE:.0%} short-term, {LONG_TERM_TAX_RATE:.0%} long-term | Rebalance band: {strategy['rebalance_band']:.0%} | Tax protection: {strategy['tax_protection_days']}d")
    else:
        print(f"  Rebalance band: {strategy['rebalance_band']:.0%}")
    print(f"  Weights:")
    for name, w in FACTOR_WEIGHTS.items():
        if w > 0:
            print(f"    {name:25s} {w:.0%}")
    print()

    print("1. Fetching universe and prices...")
    universe = get_universe()
    tickers = universe["ticker"].to_list()
    sector_map = dict(zip(universe["ticker"].to_list(), universe["sector"].to_list()))

    for bench_ticker in ["SPY", "RSP"]:
        if bench_ticker not in tickers:
            tickers.append(bench_ticker)

    prices = download_prices(tickers)
    prices = filter_insufficient_history(prices)
    print(f"   {prices['ticker'].n_unique()} tickers ready\n")

    risk_free = get_risk_free_rate()
    print(f"   Risk-free rate: {risk_free:.2%}\n")

    print("2. Pre-computing factor scores...")
    precomputed = precompute_snapshots(prices)
    n_periods = len(precomputed["is_dates"]) + len(precomputed["oos_dates"])
    print(f"   {n_periods} rebalance periods\n")

    # Pre-tax backtest (always run)
    print("3. Running backtest (pre-tax)...")
    results_pretax = run_backtest_from_snapshots(
        precomputed, prices, sector_map,
        rebalance_band=strategy["rebalance_band"],
        tax_protection_days=strategy["tax_protection_days"],
    )

    # After-tax backtest (only for taxable strategy)
    results_aftertax = None
    if strategy["show_after_tax"]:
        print("   Running backtest (after-tax)...")
        results_aftertax = run_backtest_from_snapshots(
            precomputed, prices, sector_map,
            apply_costs=True,
            rebalance_band=strategy["rebalance_band"],
            tax_protection_days=strategy["tax_protection_days"],
        )

    print("\n4. Computing metrics...")
    ec_pretax = results_pretax["equity_curve"]

    def _split_curves(ec):
        is_c = ec.filter(~pl.col("is_oos"))
        oos_c = ec.filter(pl.col("is_oos"))
        return is_c, oos_c

    def _metrics(ec):
        return compute_metrics(ec, risk_free) if not ec.is_empty() else {}

    is_pretax, oos_pretax = _split_curves(ec_pretax)
    metrics = {
        "pretax": {"full": _metrics(ec_pretax), "is": _metrics(is_pretax), "oos": _metrics(oos_pretax)},
    }

    if results_aftertax:
        ec_aftertax = results_aftertax["equity_curve"]
        is_aftertax, oos_aftertax = _split_curves(ec_aftertax)
        metrics["aftertax"] = {"full": _metrics(ec_aftertax), "is": _metrics(is_aftertax), "oos": _metrics(oos_aftertax)}

    # Date ranges for benchmarks
    is_start = is_pretax["date"].min() if not is_pretax.is_empty() else None
    is_end = is_pretax["date"].max() if not is_pretax.is_empty() else None
    oos_start = oos_pretax["date"].min() if not oos_pretax.is_empty() else None
    oos_end = oos_pretax["date"].max() if not oos_pretax.is_empty() else None

    benchmarks = {}
    for bench_ticker in ["SPY", "RSP"]:
        benchmarks[bench_ticker] = {
            "full": compute_benchmark_metrics(prices, risk_free, bench_ticker),
            "is": compute_benchmark_metrics(prices, risk_free, bench_ticker, is_start, is_end) if is_start else {},
            "oos": compute_benchmark_metrics(prices, risk_free, bench_ticker, oos_start, oos_end) if oos_start else {},
        }

    # Print results
    for period, label in [("full", "FULL PERIOD"), ("is", "IN-SAMPLE"), ("oos", "OUT-OF-SAMPLE (holdout)")]:
        if not metrics["pretax"][period]:
            continue

        if period == "full":
            print("\n" + "=" * 50)
            print(label)
            print("=" * 50)
        else:
            print(f"\n{label}")
            print("-" * 50)

        label_suffix = " (pre-tax)" if strategy["show_after_tax"] else ""
        print(f"  --- Strategy{label_suffix} ---")
        _print_metrics(metrics["pretax"][period])

        if "aftertax" in metrics and metrics["aftertax"][period]:
            print("\n  --- Strategy (after costs & taxes) ---")
            _print_metrics(metrics["aftertax"][period])

        for name, bm in benchmarks.items():
            if bm[period]:
                print(f"\n  --- {name} buy-and-hold (same period) ---")
                _print_metrics(bm[period])

    # Cost/tax summary (taxable strategy only)
    if results_aftertax:
        cost_summary = results_aftertax.get("cost_summary", {})
        if cost_summary:
            final_pretax = ec_pretax.sort("date")["value"][-1]
            final_aftertax = ec_aftertax.sort("date")["value"][-1]
            total_drag = final_pretax - final_aftertax

            dates = ec_pretax.sort("date")["date"].to_list()
            years = (dates[-1] - dates[0]).days / 365.25

            print("\n" + "=" * 50)
            print("COST & TAX SUMMARY")
            print("=" * 50)
            print(f"  Transaction costs:  ${cost_summary['total_transaction_costs']:>10,.0f}")
            print(f"  Taxes paid:         ${cost_summary['total_taxes']:>10,.0f}")
            print(f"    Short-term gains: ${cost_summary['short_term_gains_taxed']:>10,.0f}")
            print(f"    Long-term gains:  ${cost_summary['long_term_gains_taxed']:>10,.0f}")
            print(f"  Tax loss carry:     ${cost_summary['tax_loss_carry']:>10,.0f}")
            print(f"  Total drag:         ${total_drag:>10,.0f}  ({total_drag / final_pretax * 100:.1f}% of pre-tax value)")
            print(f"  Annualized drag:    {((final_aftertax / final_pretax) ** (1 / years) - 1) * 100:.2f}% per year")

    print(f"\n5. Generating plots...")
    bench_prices = {
        name: prices.filter(pl.col("ticker") == name)
        for name in ["SPY", "RSP"]
    }
    plot_equity_curve(ec_pretax, bench_prices)
    plot_monthly_heatmap(ec_pretax)

    print("\nBacktest complete.")


def _print_metrics(metrics):
    for key, val in metrics.items():
        label = key.replace("_", " ").title()
        if isinstance(val, float):
            print(f"  {label}: {val}")
        else:
            print(f"  {label}: {val}")


if __name__ == "__main__":
    main()
