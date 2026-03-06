"""Backtest the multi-factor scoring system historically."""

from data.universe import get_sp500_tickers
from data.prices import download_prices
from data.macro import get_risk_free_rate
from scoring.missing import filter_insufficient_history
from backtester.engine import run_backtest
from backtester.metrics import compute_metrics, compute_benchmark_metrics
from output.plots import plot_equity_curve, plot_monthly_heatmap
from config.settings import FACTOR_WEIGHTS, TOP_N, SECTOR_CAP, SELL_THRESHOLD_RANK, REBALANCE_FREQ, WEIGHTING
import polars as pl


def main():
    print("=== Alpha-Machine: Historical Backtest ===\n")

    print("Settings:")
    print(f"  Positions: {TOP_N} | Weighting: {WEIGHTING} | Sector cap: {'off' if SECTOR_CAP >= 1.0 else f'{SECTOR_CAP:.0%}'} | Sell threshold: rank {SELL_THRESHOLD_RANK} | Rebalance: {'quarterly' if REBALANCE_FREQ == 'QS' else 'monthly'}")
    print(f"  Weights:")
    for name, w in FACTOR_WEIGHTS.items():
        if w > 0:
            print(f"    {name:25s} {w:.0%}")
    print()

    print("1. Fetching universe and prices...")
    universe = get_sp500_tickers()
    tickers = universe["ticker"].to_list()
    sector_map = dict(zip(universe["ticker"].to_list(), universe["sector"].to_list()))

    if "SPY" not in tickers:
        tickers.append("SPY")

    prices = download_prices(tickers)
    prices = filter_insufficient_history(prices)
    print(f"   {prices['ticker'].n_unique()} tickers ready\n")

    risk_free = get_risk_free_rate()
    print(f"   Risk-free rate: {risk_free:.2%}\n")

    print("2. Running walk-forward backtest...")
    results = run_backtest(prices, sector_map)
    equity_curve = results["equity_curve"]
    print(f"   {len(results['rebalance_log'])} rebalance periods\n")

    print("3. Computing metrics...")
    is_curve = equity_curve.filter(~pl.col("is_oos"))
    oos_curve = equity_curve.filter(pl.col("is_oos"))

    full_metrics = compute_metrics(equity_curve, risk_free)
    is_metrics = compute_metrics(is_curve, risk_free) if not is_curve.is_empty() else {}
    oos_metrics = compute_metrics(oos_curve, risk_free) if not oos_curve.is_empty() else {}

    bench_full = compute_benchmark_metrics(prices, risk_free, "SPY")

    is_start = is_curve["date"].min() if not is_curve.is_empty() else None
    is_end = is_curve["date"].max() if not is_curve.is_empty() else None
    oos_start = oos_curve["date"].min() if not oos_curve.is_empty() else None
    oos_end = oos_curve["date"].max() if not oos_curve.is_empty() else None

    bench_is = compute_benchmark_metrics(prices, risk_free, "SPY", is_start, is_end) if is_start else {}
    bench_oos = compute_benchmark_metrics(prices, risk_free, "SPY", oos_start, oos_end) if oos_start else {}

    print("\n" + "=" * 50)
    print("FULL PERIOD")
    print("=" * 50)
    _print_metrics(full_metrics)
    if bench_full:
        print("\n  --- SPY buy-and-hold (same period) ---")
        _print_metrics(bench_full)

    if is_metrics:
        print("\nIN-SAMPLE")
        print("-" * 50)
        _print_metrics(is_metrics)
        if bench_is:
            print("\n  --- SPY buy-and-hold (same period) ---")
            _print_metrics(bench_is)

    if oos_metrics:
        print("\nOUT-OF-SAMPLE (holdout)")
        print("-" * 50)
        _print_metrics(oos_metrics)
        if bench_oos:
            print("\n  --- SPY buy-and-hold (same period) ---")
            _print_metrics(bench_oos)

    print("\n4. Generating plots...")
    spy_prices = prices.filter(pl.col("ticker") == "SPY")
    plot_equity_curve(equity_curve, spy_prices)
    plot_monthly_heatmap(equity_curve)

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
