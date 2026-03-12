"""Compare current vs decay-informed factor weights via backtest.

Computes factor snapshots once, then runs backtests with both weight sets
and prints side-by-side results.
"""

import polars as pl
from data.universe import get_universe
from data.prices import download_prices
from data.macro import get_risk_free_rate
from scoring.missing import filter_insufficient_history
from scoring.combine import ALL_FACTORS, composite_score
from backtester.engine import precompute_snapshots, run_backtest_from_snapshots
from backtester.metrics import compute_metrics, compute_benchmark_metrics
from config.settings import FACTOR_WEIGHTS

CURRENT_WEIGHTS = dict(FACTOR_WEIGHTS)

PROPOSED_WEIGHTS = {
    "momentum_12_1": 0.30,       # was 0.25 — absorb some freed weight
    "momentum_6_1": 0.20,        # was 0.15 — absorb some freed weight
    "rel_strength_3mo": 0.10,    # keep
    "short_term_reversal": 0.00, # keep off
    "dist_from_ma50": 0.05,      # keep
    "volume_trend": 0.00,        # kill — negative IC in both IS and OOS
    "obv_slope": 0.05,           # keep
    "realized_vol_60d": 0.00,    # kill — consistently negative IC in both IS and OOS
    "vol_trend": 0.10,           # was 0.05 — absorb some freed weight, strong IS consistency
    "price_consistency": 0.20,   # was 0.15 — absorb some freed weight
}

assert abs(sum(PROPOSED_WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"


def rescore_snapshots(precomputed, weights):
    """Recompute composite_score on existing snapshots with new weights."""
    new_snapshots = {}
    for date, snapshot in precomputed["snapshots"].items():
        if snapshot.is_empty():
            new_snapshots[date] = snapshot
            continue
        rescored = composite_score(snapshot, weights=weights)
        new_snapshots[date] = rescored.sort("composite_score", descending=True)
    return {**precomputed, "snapshots": new_snapshots}


def main():
    print("=== Factor Weight Comparison ===\n")

    print("Current weights:")
    for name, w in CURRENT_WEIGHTS.items():
        if w > 0:
            print(f"  {name:25s} {w:.0%}")

    print("\nProposed weights (decay-informed):")
    for name, w in PROPOSED_WEIGHTS.items():
        if w > 0:
            print(f"  {name:25s} {w:.0%}")

    changes = []
    for name in CURRENT_WEIGHTS:
        old, new = CURRENT_WEIGHTS[name], PROPOSED_WEIGHTS[name]
        if old != new:
            changes.append(f"  {name:25s} {old:.0%} → {new:.0%}")
    print(f"\nChanges:")
    for c in changes:
        print(c)

    print("\n1. Loading data...")
    universe = get_universe()
    tickers = universe["ticker"].to_list()
    sector_map = dict(zip(universe["ticker"].to_list(), universe["sector"].to_list()))
    for bench in ["SPY", "RSP"]:
        if bench not in tickers:
            tickers.append(bench)

    prices = download_prices(tickers)
    prices = filter_insufficient_history(prices)
    risk_free = get_risk_free_rate()
    print(f"   {prices['ticker'].n_unique()} tickers, risk-free: {risk_free:.2%}\n")

    print("2. Computing factor snapshots...")
    precomputed = precompute_snapshots(prices)
    n_periods = len(precomputed["is_dates"]) + len(precomputed["oos_dates"])
    print(f"   {n_periods} periods ({len(precomputed['is_dates'])} IS + {len(precomputed['oos_dates'])} OOS)\n")

    # Rescore with proposed weights
    precomputed_proposed = rescore_snapshots(precomputed, PROPOSED_WEIGHTS)

    print("3. Running backtests...\n")
    results = {}
    for label, pre in [("current", precomputed), ("proposed", precomputed_proposed)]:
        results[label] = run_backtest_from_snapshots(
            pre, prices, sector_map,
            rebalance_band=0.01,
            tax_protection_days=0,
        )

    # Benchmarks
    ec = results["current"]["equity_curve"]
    is_c = ec.filter(~pl.col("is_oos"))
    oos_c = ec.filter(pl.col("is_oos"))

    benchmarks = {}
    for bench in ["SPY"]:
        benchmarks[bench] = {
            "full": compute_benchmark_metrics(prices, risk_free, bench),
            "is": compute_benchmark_metrics(prices, risk_free, bench,
                                            is_c["date"].min(), is_c["date"].max()) if not is_c.is_empty() else {},
            "oos": compute_benchmark_metrics(prices, risk_free, bench,
                                             oos_c["date"].min(), oos_c["date"].max()) if not oos_c.is_empty() else {},
        }

    # Print comparison
    for period, label in [("full", "FULL PERIOD"), ("is", "IN-SAMPLE"), ("oos", "OUT-OF-SAMPLE")]:
        print(f"\n{'=' * 60}")
        print(f"{label}")
        print(f"{'=' * 60}")
        print(f"  {'Metric':<20s}  {'Current':>10s}  {'Proposed':>10s}  {'SPY':>10s}")
        print(f"  {'-' * 55}")

        metrics = {}
        for name, res in results.items():
            ec = res["equity_curve"]
            if period == "full":
                sub = ec
            elif period == "is":
                sub = ec.filter(~pl.col("is_oos"))
            else:
                sub = ec.filter(pl.col("is_oos"))
            metrics[name] = compute_metrics(sub, risk_free) if not sub.is_empty() else {}

        spy = benchmarks["SPY"][period]

        for key in ["cagr", "sharpe", "max_drawdown", "calmar", "total_return"]:
            current_val = metrics["current"].get(key, "")
            proposed_val = metrics["proposed"].get(key, "")
            spy_val = spy.get(key, "")

            row = f"  {key:<20s}"
            for val in [current_val, proposed_val, spy_val]:
                if isinstance(val, (int, float)):
                    row += f"  {val:>10s}" if isinstance(val, str) else f"  {val:>10.2f}"
                else:
                    row += f"  {str(val):>10s}"
            print(row)

        # Turnover for full and oos
        if period in ("full", "oos"):
            for name in ["current", "proposed"]:
                log = results[name]["rebalance_log"]
                if period == "oos":
                    log = [r for r in log if r["is_oos"]]
                turnovers = [r.get("turnover", 0) for r in log if r.get("turnover") is not None]
                if turnovers:
                    avg_t = sum(turnovers) / len(turnovers)
                    print(f"  {name + ' avg turnover':<20s}  {avg_t:>10.1%}")


if __name__ == "__main__":
    main()
