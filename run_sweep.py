"""Sweep parameters to measure performance tradeoffs.

Usage:
    python run_sweep.py threshold    # sweep sell_threshold_rank (default)
    python run_sweep.py tilt         # sweep score_tilt values
"""

import sys
from data.universe import get_universe
from data.prices import download_prices
from data.macro import get_risk_free_rate
from scoring.missing import filter_insufficient_history
from backtester.engine import precompute_snapshots, run_backtest_from_snapshots
from backtester.metrics import compute_metrics, compute_benchmark_metrics
from config.settings import TOP_N, SELL_THRESHOLD_RANK, SCORE_TILT
import polars as pl


THRESHOLD_VALUES = [20, 30, 50, 75, 100, 150, 200]
TILT_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]


def _load_data():
    """Load universe, prices, risk-free rate. Returns (prices, sector_map, risk_free)."""
    print("1. Loading data...")
    universe = get_universe()
    tickers = universe["ticker"].to_list()
    sector_map = dict(zip(universe["ticker"].to_list(), universe["sector"].to_list()))

    for bench in ["SPY", "RSP"]:
        if bench not in tickers:
            tickers.append(bench)

    prices = download_prices(tickers)
    prices = filter_insufficient_history(prices)
    print(f"   {prices['ticker'].n_unique()} tickers ready\n")

    risk_free = get_risk_free_rate()
    return prices, sector_map, risk_free


def _run_and_collect(precomputed, prices, sector_map, risk_free, **kwargs):
    """Run backtest and return metrics dict."""
    bt = run_backtest_from_snapshots(precomputed, prices, sector_map, **kwargs)
    ec = bt["equity_curve"]
    is_c = ec.filter(~pl.col("is_oos"))
    oos_c = ec.filter(pl.col("is_oos"))

    full_m = compute_metrics(ec, risk_free)
    is_m = compute_metrics(is_c, risk_free) if not is_c.is_empty() else {}
    oos_m = compute_metrics(oos_c, risk_free) if not oos_c.is_empty() else {}

    turnovers = [r["turnover"] for r in bt["rebalance_log"] if r["turnover"] < 1.0]
    avg_turnover = sum(turnovers) / len(turnovers) if turnovers else 0.0

    return {"full": full_m, "is": is_m, "oos": oos_m, "avg_turnover": avg_turnover}


def _get_spy_benchmarks(precomputed, prices, risk_free):
    """Compute SPY benchmarks for full/IS/OOS periods."""
    is_dates = precomputed["is_dates"]
    oos_dates = precomputed["oos_dates"]
    all_dates = precomputed["all_dates"]

    bench_full = compute_benchmark_metrics(prices, risk_free, "SPY")
    bench_is = compute_benchmark_metrics(prices, risk_free, "SPY", is_dates[0],
                                          oos_dates[0] if oos_dates else all_dates[-1]) if is_dates else {}
    bench_oos = compute_benchmark_metrics(prices, risk_free, "SPY", oos_dates[0],
                                          all_dates[-1]) if oos_dates else {}
    return bench_full, bench_is, bench_oos


def _print_table(rows, param_label, current_label, spy):
    """Print a results table."""
    print(
        f"{param_label:>10} | {'Turnover':>8} |"
        f" {'CAGR':>7} {'Sharpe':>7} {'MaxDD':>7} |"
        f" {'CAGR':>7} {'Sharpe':>7} {'MaxDD':>7} |"
        f" {'CAGR':>7} {'Sharpe':>7} {'MaxDD':>7}"
    )
    print(
        f"{'':>10} | {'avg/reb':>8} |"
        f" {'--- FULL PERIOD ---':^23}|"
        f" {'--- IN-SAMPLE ----':^23}|"
        f" {'-- OUT-OF-SAMPLE -':^23}"
    )
    print("-" * 120)

    for label, r in rows:
        f, i, o = r["full"], r["is"], r["oos"]
        print(
            f"{label:>10} | {r['avg_turnover']:>7.1%} |"
            f" {f.get('cagr', 0):>6.1f}% {f.get('sharpe', 0):>7.2f} {f.get('max_drawdown', 0):>6.1f}% |"
            f" {i.get('cagr', 0):>6.1f}% {i.get('sharpe', 0):>7.2f} {i.get('max_drawdown', 0):>6.1f}% |"
            f" {o.get('cagr', 0):>6.1f}% {o.get('sharpe', 0):>7.2f} {o.get('max_drawdown', 0):>6.1f}%"
        )

    spy_f, spy_i, spy_o = spy
    print("-" * 120)
    print(
        f"{'SPY':>10} | {'n/a':>8} |"
        f" {spy_f.get('cagr', 0):>6.1f}% {spy_f.get('sharpe', 0):>7.2f} {spy_f.get('max_drawdown', 0):>6.1f}% |"
        f" {spy_i.get('cagr', 0):>6.1f}% {spy_i.get('sharpe', 0):>7.2f} {spy_i.get('max_drawdown', 0):>6.1f}% |"
        f" {spy_o.get('cagr', 0):>6.1f}% {spy_o.get('sharpe', 0):>7.2f} {spy_o.get('max_drawdown', 0):>6.1f}%"
    )


def sweep_threshold():
    """Sweep sell_threshold_rank values."""
    print("=== Alpha-Machine: Sell Threshold Sweep ===\n")
    print(f"Testing thresholds: {THRESHOLD_VALUES}")
    print(f"Current setting: SELL_THRESHOLD_RANK = {SELL_THRESHOLD_RANK}\n")

    prices, sector_map, risk_free = _load_data()

    print("2. Pre-computing factor scores (one-time, slow)...")
    precomputed = precompute_snapshots(prices)
    print("   Done.\n")

    print("3. Running sweep...\n")
    rows = []
    for t in THRESHOLD_VALUES:
        label = f"{t} (none)" if t <= TOP_N else (f"{t} *" if t == SELL_THRESHOLD_RANK else str(t))
        print(f"   Threshold: rank {t}...")
        r = _run_and_collect(precomputed, prices, sector_map, risk_free, sell_threshold_rank=t)
        rows.append((label, r))

    spy = _get_spy_benchmarks(precomputed, prices, risk_free)

    print("\n" + "=" * 120)
    print("SELL THRESHOLD SWEEP RESULTS")
    print("=" * 120)
    _print_table(rows, "Threshold", f"SELL_THRESHOLD_RANK = {SELL_THRESHOLD_RANK}", spy)
    print(f"\n* = current setting")
    print("Done.")


def sweep_tilt():
    """Sweep score_tilt values."""
    print("=== Alpha-Machine: Score Tilt Sweep ===\n")
    print(f"Testing tilts: {TILT_VALUES}")
    print(f"Current setting: SCORE_TILT = {SCORE_TILT}")
    print(f"0 = equal weight, 1 = proportional, >1 = concentrated\n")

    prices, sector_map, risk_free = _load_data()

    print("2. Pre-computing factor scores (one-time, slow)...")
    precomputed = precompute_snapshots(prices)
    print("   Done.\n")

    print("3. Running sweep...\n")
    rows = []
    for tilt in TILT_VALUES:
        label = f"{tilt:.2f} *" if tilt == SCORE_TILT else f"{tilt:.2f}"
        if tilt == 0:
            label = "0 (equal)"
        print(f"   Tilt: {tilt}...")
        r = _run_and_collect(precomputed, prices, sector_map, risk_free, score_tilt=tilt)
        rows.append((label, r))

    spy = _get_spy_benchmarks(precomputed, prices, risk_free)

    print("\n" + "=" * 120)
    print("SCORE TILT SWEEP RESULTS")
    print("=" * 120)
    _print_table(rows, "Tilt", f"SCORE_TILT = {SCORE_TILT}", spy)
    print(f"\n* = current setting")
    print("Done.")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "threshold"
    if mode == "tilt":
        sweep_tilt()
    elif mode == "threshold":
        sweep_threshold()
    else:
        print(f"Unknown sweep mode: {mode}")
        print("Usage: python run_sweep.py [threshold|tilt]")
        sys.exit(1)
