"""Sweep sell_threshold_rank to measure turnover vs performance tradeoff."""

from data.universe import get_universe
from data.prices import download_prices
from data.macro import get_risk_free_rate
from scoring.missing import filter_insufficient_history
from backtester.engine import precompute_snapshots, run_backtest_from_snapshots
from backtester.metrics import compute_metrics, compute_benchmark_metrics
from config.settings import TOP_N, SELL_THRESHOLD_RANK
import polars as pl


THRESHOLDS = [20, 30, 50, 75, 100, 150, 200]


def main():
    print("=== Alpha-Machine: Sell Threshold Sweep ===\n")
    print(f"Testing thresholds: {THRESHOLDS}")
    print(f"Current setting: SELL_THRESHOLD_RANK = {SELL_THRESHOLD_RANK}")
    print(f"Threshold = {TOP_N} means no dampening (only top-{TOP_N} retained)\n")

    print("1. Loading data...")
    universe = get_universe()
    tickers = universe["ticker"].to_list()
    sector_map = dict(zip(universe["ticker"].to_list(), universe["sector"].to_list()))

    if "SPY" not in tickers:
        tickers.append("SPY")

    prices = download_prices(tickers)
    prices = filter_insufficient_history(prices)
    print(f"   {prices['ticker'].n_unique()} tickers ready\n")

    risk_free = get_risk_free_rate()

    print("2. Pre-computing factor scores (one-time, slow)...")
    precomputed = precompute_snapshots(prices)
    print("   Done.\n")

    print("3. Running sweep...\n")

    results = []
    for threshold in THRESHOLDS:
        label = f"rank {threshold}" if threshold > TOP_N else f"rank {threshold} (no dampening)"
        print(f"   Threshold: {label}...")

        bt = run_backtest_from_snapshots(
            precomputed, prices, sector_map,
            sell_threshold_rank=threshold,
        )

        equity_curve = bt["equity_curve"]
        is_curve = equity_curve.filter(~pl.col("is_oos"))
        oos_curve = equity_curve.filter(pl.col("is_oos"))

        full_m = compute_metrics(equity_curve, risk_free)
        is_m = compute_metrics(is_curve, risk_free) if not is_curve.is_empty() else {}
        oos_m = compute_metrics(oos_curve, risk_free) if not oos_curve.is_empty() else {}

        # Average turnover (exclude first rebalance which is always 100%)
        turnovers = [r["turnover"] for r in bt["rebalance_log"] if r["turnover"] < 1.0]
        avg_turnover = sum(turnovers) / len(turnovers) if turnovers else 0.0

        # Unique stocks used across all periods
        all_tickers_used = set()
        for r in bt["rebalance_log"]:
            all_tickers_used.update(r["tickers"])

        results.append({
            "threshold": threshold,
            "avg_turnover": avg_turnover,
            "unique_stocks": len(all_tickers_used),
            "full": full_m,
            "is": is_m,
            "oos": oos_m,
        })

    # SPY benchmark for all periods
    bench_full = compute_benchmark_metrics(prices, risk_free, "SPY")

    is_dates = precomputed["is_dates"]
    oos_dates = precomputed["oos_dates"]
    all_dates = precomputed["all_dates"]

    if is_dates:
        is_end = oos_dates[0] if oos_dates else all_dates[-1]
        bench_is = compute_benchmark_metrics(prices, risk_free, "SPY", is_dates[0], is_end)
    else:
        bench_is = {}

    if oos_dates:
        bench_oos = compute_benchmark_metrics(prices, risk_free, "SPY", oos_dates[0], all_dates[-1])
    else:
        bench_oos = {}

    # Print results table
    print("\n" + "=" * 130)
    print("SELL THRESHOLD SWEEP RESULTS")
    print("=" * 130)

    print(
        f"{'Threshold':>10} | {'Turnover':>8} | {'Unique':>6} |"
        f" {'CAGR':>7} {'Sharpe':>7} {'MaxDD':>7} |"
        f" {'CAGR':>7} {'Sharpe':>7} {'MaxDD':>7} |"
        f" {'CAGR':>7} {'Sharpe':>7} {'MaxDD':>7}"
    )
    print(
        f"{'':>10} | {'avg/reb':>8} | {'stocks':>6} |"
        f" {'--- FULL PERIOD ---':^23}|"
        f" {'--- IN-SAMPLE ----':^23}|"
        f" {'-- OUT-OF-SAMPLE -':^23}"
    )
    print("-" * 130)

    for r in results:
        t = r["threshold"]
        if t <= TOP_N:
            label = f"{t} (none)"
        elif t == SELL_THRESHOLD_RANK:
            label = f"{t} *"
        else:
            label = str(t)

        f = r["full"]
        i = r["is"]
        o = r["oos"]

        print(
            f"{label:>10} | {r['avg_turnover']:>7.1%} | {r['unique_stocks']:>6} |"
            f" {f.get('cagr', 0):>6.1f}% {f.get('sharpe', 0):>7.2f} {f.get('max_drawdown', 0):>6.1f}% |"
            f" {i.get('cagr', 0):>6.1f}% {i.get('sharpe', 0):>7.2f} {i.get('max_drawdown', 0):>6.1f}% |"
            f" {o.get('cagr', 0):>6.1f}% {o.get('sharpe', 0):>7.2f} {o.get('max_drawdown', 0):>6.1f}%"
        )

    print("-" * 130)
    print(
        f"{'SPY':>10} | {'n/a':>8} | {'1':>6} |"
        f" {bench_full.get('cagr', 0):>6.1f}% {bench_full.get('sharpe', 0):>7.2f} {bench_full.get('max_drawdown', 0):>6.1f}% |"
        f" {bench_is.get('cagr', 0):>6.1f}% {bench_is.get('sharpe', 0):>7.2f} {bench_is.get('max_drawdown', 0):>6.1f}% |"
        f" {bench_oos.get('cagr', 0):>6.1f}% {bench_oos.get('sharpe', 0):>7.2f} {bench_oos.get('max_drawdown', 0):>6.1f}%"
    )

    print(f"\n* = current setting (SELL_THRESHOLD_RANK = {SELL_THRESHOLD_RANK})")
    print(f"Threshold = {TOP_N} means no dampening (top-{TOP_N} only)")
    print(f"Higher threshold = more dampening = lower turnover = hold losers longer")
    print("\nDone.")


if __name__ == "__main__":
    main()
