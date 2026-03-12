"""Factor decay analysis: measure how quickly each factor's predictive power decays.

For each factor and forward-return horizon, computes the average Spearman rank IC
(information coefficient) between the factor rank at portfolio formation and
subsequent stock returns. This reveals which factors are persistent (good for
quarterly rebalancing) vs fast-decaying (need more frequent rebalancing).

Usage:
    python run_decay.py
"""

import numpy as np
import polars as pl
from data.universe import get_universe
from data.prices import download_prices
from scoring.missing import filter_insufficient_history
from backtester.engine import precompute_snapshots, BENCHMARK_TICKERS
from scoring.combine import ALL_FACTORS
from config.settings import FACTOR_WEIGHTS, CACHE_DIR

# Forward return horizons in approximate trading days
HORIZONS = {
    "1mo": 21,
    "2mo": 42,
    "3mo": 63,
    "4mo": 84,
    "6mo": 126,
    "9mo": 189,
    "12mo": 252,
}


def spearman_corr(x, y):
    """Spearman rank correlation via numpy (no scipy dependency)."""
    def _rank(arr):
        order = arr.argsort()
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(arr), dtype=float)
        return ranks

    rx, ry = _rank(x), _rank(y)
    return np.corrcoef(rx, ry)[0, 1]


def compute_forward_returns(prices: pl.DataFrame, snapshot_date, horizons: dict) -> pl.DataFrame:
    """Compute forward returns from snapshot_date for each horizon."""
    all_dates = prices["date"].unique().sort().to_list()

    try:
        base_idx = next(i for i, d in enumerate(all_dates) if d >= snapshot_date)
    except StopIteration:
        return pl.DataFrame()

    base_date = all_dates[base_idx]
    base_prices = prices.filter(pl.col("date") == base_date).select(
        "ticker", pl.col("close").alias("base_close")
    )

    result = base_prices.clone()

    for label, days in horizons.items():
        target_idx = base_idx + days
        if target_idx >= len(all_dates):
            result = result.with_columns(pl.lit(None).cast(pl.Float64).alias(f"fwd_{label}"))
            continue

        target_date = all_dates[target_idx]
        target_prices = prices.filter(pl.col("date") == target_date).select(
            "ticker", pl.col("close").alias(f"close_{label}")
        )

        result = result.join(target_prices, on="ticker", how="left")
        result = result.with_columns(
            ((pl.col(f"close_{label}") / pl.col("base_close")) - 1.0).alias(f"fwd_{label}")
        ).drop(f"close_{label}")

    return result.drop("base_close")


def compute_factor_ics(snapshot, forward_returns, factor_col, horizons):
    """Compute Spearman rank IC between a factor column and forward returns."""
    merged = snapshot.select("ticker", factor_col).join(forward_returns, on="ticker", how="inner")

    ics = {}
    for label in horizons:
        fwd_col = f"fwd_{label}"
        valid = merged.filter(pl.col(factor_col).is_not_null() & pl.col(fwd_col).is_not_null())

        if len(valid) < 30:
            ics[label] = np.nan
            continue

        x = valid[factor_col].to_numpy().astype(float)
        y = valid[fwd_col].to_numpy().astype(float)
        ics[label] = spearman_corr(x, y)

    return ics


def main():
    print("=== Factor Decay Analysis ===\n")

    print("1. Loading data...")
    universe = get_universe()
    tickers = universe["ticker"].to_list()
    for bench in ["SPY", "RSP"]:
        if bench not in tickers:
            tickers.append(bench)

    prices = download_prices(tickers)
    prices = filter_insufficient_history(prices)
    scoring_prices = prices.filter(~pl.col("ticker").is_in(BENCHMARK_TICKERS))
    print(f"   {scoring_prices['ticker'].n_unique()} tickers ready\n")

    print("2. Computing factor snapshots (this takes a while)...")
    precomputed = precompute_snapshots(prices)
    snapshots = precomputed["snapshots"]
    is_dates = set(precomputed["is_dates"])
    oos_dates = set(precomputed["oos_dates"])
    rebalance_dates = precomputed["is_dates"] + precomputed["oos_dates"]
    print(f"   {len(rebalance_dates)} rebalance dates ({len(is_dates)} IS + {len(oos_dates)} OOS)\n")

    print("3. Computing rank ICs per factor × horizon...")
    factor_names = [f.name() for f in ALL_FACTORS]
    all_columns = factor_names + ["composite"]
    rank_cols = {f: f"{f}_rank" for f in factor_names}
    rank_cols["composite"] = "composite_score"

    # Separate IS and OOS IC collections
    is_ics = {name: {h: [] for h in HORIZONS} for name in all_columns}
    oos_ics = {name: {h: [] for h in HORIZONS} for name in all_columns}

    for idx, reb_date in enumerate(rebalance_dates):
        print(f"   Processing {idx + 1}/{len(rebalance_dates)}...", end="\r")

        snapshot = snapshots.get(reb_date)
        if snapshot is None or snapshot.is_empty():
            continue

        fwd = compute_forward_returns(scoring_prices, reb_date, HORIZONS)
        if fwd.is_empty():
            continue

        target = is_ics if reb_date in is_dates else oos_ics

        for name in all_columns:
            ics = compute_factor_ics(snapshot, fwd, rank_cols[name], HORIZONS)
            for h, ic in ics.items():
                target[name][h].append(ic)

    print()

    # --- Print tables for a given IC dataset ---
    horizon_labels = list(HORIZONS.keys())
    header = f"{'Factor':<25s}"
    for h in horizon_labels:
        header += f"  {h:>6s}"
    header += f"  {'wt':>4s}"

    def _print_tables(ic_data, label, show_peak=False):
        print(f"--- {label}: Mean Rank IC ---")
        print(header)
        print("-" * len(header))

        decay = {}
        hit_rates = {}

        for name in all_columns:
            row = f"{name:<25s}"
            means = []
            hrs = []
            for h in horizon_labels:
                values = [v for v in ic_data[name][h] if not np.isnan(v)]
                if len(values) >= 5:
                    mean_ic = np.mean(values)
                    means.append(mean_ic)
                    std_ic = np.std(values, ddof=1)
                    t_stat = mean_ic / (std_ic / np.sqrt(len(values))) if std_ic > 0 else 0
                    marker = "*" if abs(t_stat) >= 2.0 else " "
                    row += f"  {mean_ic:>5.3f}{marker}"
                    hrs.append(sum(1 for v in values if v > 0) / len(values))
                else:
                    means.append(np.nan)
                    hrs.append(np.nan)
                    row += f"  {'n/a':>6s}"

            w = FACTOR_WEIGHTS.get(name, None)
            if w is not None:
                row += f"  {w:>3.0%} " if w > 0 else f"  {'off':>4s}"
            else:
                row += f"  {'':>4s}"

            print(row)
            decay[name] = means
            hit_rates[name] = hrs

        print("  * = statistically significant (|t| >= 2.0)")

        print(f"\n--- {label}: IC Hit Rate (% of periods with IC > 0) ---")
        print(header)
        print("-" * len(header))

        for name in all_columns:
            row = f"{name:<25s}"
            for i, h in enumerate(horizon_labels):
                hr = hit_rates[name][i]
                if not np.isnan(hr):
                    row += f"  {hr:>5.0%} "
                else:
                    row += f"  {'n/a':>6s}"
            w = FACTOR_WEIGHTS.get(name, None)
            if w is not None:
                row += f"  {w:>3.0%} " if w > 0 else f"  {'off':>4s}"
            else:
                row += f"  {'':>4s}"
            print(row)
        print("  >60% = consistent signal, <40% = consistently wrong direction")

        if show_peak:
            print(f"\n--- {label}: Peak & Half-life ---")
            print(f"{'Factor':<25s}  {'Peak IC':>7s}  {'Peak':>4s}  {'Half-life':>10s}")
            print("-" * 55)
            for name in all_columns:
                means = decay[name]
                valid = [(i, m) for i, m in enumerate(means) if not np.isnan(m)]
                if not valid:
                    continue
                peak_idx, peak_ic = max(valid, key=lambda x: x[1])
                peak_label = horizon_labels[peak_idx]
                half_life = "n/a"
                if peak_ic > 0:
                    for i, m in valid:
                        if i > peak_idx and m < peak_ic / 2:
                            half_life = horizon_labels[i]
                            break
                    else:
                        half_life = ">12mo"
                w = FACTOR_WEIGHTS.get(name, None)
                mk = "" if w is None or w > 0 else " (off)"
                print(f"{name:<25s}  {peak_ic:>7.3f}  {peak_label:>4s}  {half_life:>10s}{mk}")

        return decay

    print("\n4. Results\n")
    print(f"{'=' * 60}")
    print(f"IN-SAMPLE ({len(is_dates)} periods) — use this to set weights")
    print(f"{'=' * 60}\n")
    is_decay = _print_tables(is_ics, "IS", show_peak=True)

    print(f"\n{'=' * 60}")
    print(f"OUT-OF-SAMPLE ({len(oos_dates)} periods) — validation only")
    print(f"{'=' * 60}\n")
    oos_decay = _print_tables(oos_ics, "OOS", show_peak=True)

    # --- Plot (IS-only for weight decisions) ---
    print("\n5. Generating decay plot (IS-only)...")
    _plot_decay(is_decay, horizon_labels, factor_names)
    print("   Saved to cache/factor_decay.png")


def _plot_decay(decay_data, horizon_labels, factor_names):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    x = range(len(horizon_labels))

    # Left panel: active factors + composite
    active = [f for f in factor_names if FACTOR_WEIGHTS.get(f, 0) > 0]
    for name in active:
        ax1.plot(x, decay_data[name], marker="o",
                 label=f"{name} ({FACTOR_WEIGHTS[name]:.0%})", linewidth=2)
    ax1.plot(x, decay_data["composite"], marker="s",
             label="composite", linewidth=2.5, color="black", linestyle="--")

    ax1.set_xticks(list(x))
    ax1.set_xticklabels(horizon_labels)
    ax1.set_xlabel("Forward Return Horizon")
    ax1.set_ylabel("Mean Rank IC (Spearman)")
    ax1.set_title("Active Factors — IC Decay")
    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax1.legend(fontsize=8, loc="best")
    ax1.grid(True, alpha=0.3)

    # Right panel: all factors including disabled
    inactive = [f for f in factor_names if FACTOR_WEIGHTS.get(f, 0) == 0]
    for name in inactive:
        ax2.plot(x, decay_data[name], marker="o",
                 label=f"{name} (off)", linewidth=2, linestyle=":")
    for name in active:
        ax2.plot(x, decay_data[name], marker="o",
                 label=name, linewidth=1.5, alpha=0.4)

    ax2.set_xticks(list(x))
    ax2.set_xticklabels(horizon_labels)
    ax2.set_xlabel("Forward Return Horizon")
    ax2.set_ylabel("Mean Rank IC (Spearman)")
    ax2.set_title("All Factors — IC Decay")
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax2.legend(fontsize=8, loc="best")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Factor Decay Analysis — Rank IC vs Forward Return Horizon",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(CACHE_DIR / "factor_decay.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
