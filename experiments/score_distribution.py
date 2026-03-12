"""Analyze composite score distributions across rebalance dates.

Answers: does the score distribution vary enough across periods to make
dynamic position sizing worthwhile? If the top-20 cutoff always falls
at a similar score level, dynamic count won't add much.
"""

import numpy as np
import polars as pl
from data.universe import get_universe
from data.prices import download_prices
from scoring.missing import filter_insufficient_history
from backtester.engine import precompute_snapshots
from config.settings import CACHE_DIR, TOP_N


def main():
    print("=== Score Distribution Analysis ===\n")

    print("1. Loading data...")
    universe = get_universe()
    tickers = universe["ticker"].to_list()
    for bench in ["SPY", "RSP"]:
        if bench not in tickers:
            tickers.append(bench)
    prices = download_prices(tickers)
    prices = filter_insufficient_history(prices)
    print(f"   {prices['ticker'].n_unique()} tickers\n")

    print("2. Computing snapshots...")
    precomputed = precompute_snapshots(prices, holdout_periods=0)
    snapshots = precomputed["snapshots"]
    rebalance_dates = precomputed["rebalance_dates"]
    print(f"   {len(rebalance_dates)} periods\n")

    print("3. Analyzing score distributions...\n")

    records = []
    for reb_date in rebalance_dates:
        snapshot = snapshots.get(reb_date)
        if snapshot is None or snapshot.is_empty():
            continue

        scores = snapshot.sort("composite_score", descending=True)["composite_score"].to_numpy()
        n = len(scores)

        # Score at various rank positions
        top1 = scores[0]
        top10 = scores[min(9, n - 1)]
        top20 = scores[min(19, n - 1)]
        top30 = scores[min(29, n - 1)]
        top50 = scores[min(49, n - 1)]
        median = np.median(scores)

        # Score gap: difference between rank 20 and rank 21
        gap_at_20 = scores[19] - scores[20] if n > 20 else 0

        # Spread metrics
        top20_std = np.std(scores[:20])
        top50_std = np.std(scores[:50])
        full_std = np.std(scores)

        # Ratio: how much of the top score does rank 20 retain?
        pct_of_top = top20 / top1 if top1 > 0 else 0

        # "Natural break" — find where score drops most between consecutive ranks (top 50)
        diffs = np.diff(scores[:min(50, n)])  # negative values (scores descending)
        biggest_drop_idx = np.argmin(diffs)  # most negative diff = biggest drop
        biggest_drop_rank = biggest_drop_idx + 2  # rank is 1-indexed

        # How many stocks within X% of top score?
        within_90pct = np.sum(scores >= top1 * 0.90)
        within_95pct = np.sum(scores >= top1 * 0.95)
        within_80pct = np.sum(scores >= top1 * 0.80)

        records.append({
            "date": reb_date,
            "top1": top1,
            "top10": top10,
            "top20": top20,
            "top30": top30,
            "median": median,
            "gap_at_20": gap_at_20,
            "top20_std": top20_std,
            "full_std": full_std,
            "pct_of_top": pct_of_top,
            "biggest_drop_rank": biggest_drop_rank,
            "within_90pct": within_90pct,
            "within_95pct": within_95pct,
            "within_80pct": within_80pct,
        })

    # --- Print summary ---
    dates = [r["date"] for r in records]
    print(f"{'Date':<12s}  {'Top1':>5s}  {'Top10':>5s}  {'Top20':>5s}  {'Top30':>5s}  {'Gap@20':>6s}  {'%ofTop':>6s}  {'Break':>5s}  {'90%':>4s}  {'80%':>4s}")
    print("-" * 85)
    for r in records:
        print(f"{str(r['date'])[:10]:<12s}  {r['top1']:>5.3f}  {r['top10']:>5.3f}  {r['top20']:>5.3f}  {r['top30']:>5.3f}  {r['gap_at_20']:>6.4f}  {r['pct_of_top']:>5.1%}  {r['biggest_drop_rank']:>5d}  {r['within_90pct']:>4d}  {r['within_80pct']:>4d}")

    # --- Summary stats ---
    print(f"\n--- Summary ---")
    pcts = [r["pct_of_top"] for r in records]
    gaps = [r["gap_at_20"] for r in records]
    breaks = [r["biggest_drop_rank"] for r in records]
    w90 = [r["within_90pct"] for r in records]
    w80 = [r["within_80pct"] for r in records]

    print(f"  Score at rank 20 as % of rank 1:  mean {np.mean(pcts):.1%}, min {np.min(pcts):.1%}, max {np.max(pcts):.1%}, std {np.std(pcts):.1%}")
    print(f"  Gap at rank 20-21:                mean {np.mean(gaps):.4f}, min {np.min(gaps):.4f}, max {np.max(gaps):.4f}, std {np.std(gaps):.4f}")
    print(f"  Biggest score drop (top 50):      mean rank {np.mean(breaks):.0f}, min {np.min(breaks)}, max {np.max(breaks)}, std {np.std(breaks):.1f}")
    print(f"  Stocks within 90% of top score:   mean {np.mean(w90):.0f}, min {np.min(w90)}, max {np.max(w90)}")
    print(f"  Stocks within 80% of top score:   mean {np.mean(w80):.0f}, min {np.min(w80)}, max {np.max(w80)}")

    # --- Plot ---
    print("\n4. Generating plots...")
    _plot_distributions(records, snapshots, rebalance_dates)
    print(f"   Saved to cache/score_distributions.png")


def _plot_distributions(records, snapshots, rebalance_dates):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Top-left: Score curves for 6 sample periods (spread across time)
    ax = axes[0][0]
    sample_indices = np.linspace(0, len(rebalance_dates) - 1, 6, dtype=int)
    for idx in sample_indices:
        date = rebalance_dates[idx]
        snapshot = snapshots.get(date)
        if snapshot is None or snapshot.is_empty():
            continue
        scores = snapshot.sort("composite_score", descending=True)["composite_score"].to_numpy()
        ax.plot(range(1, min(51, len(scores) + 1)), scores[:50],
                label=str(date)[:10], linewidth=1.5)
    ax.axvline(x=20, color="red", linestyle="--", alpha=0.5, label="Current TOP_N=20")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Composite Score")
    ax.set_title("Score Curves (Top 50) — Sample Periods")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Top-right: Rank 20 score as % of rank 1 over time
    ax = axes[0][1]
    dates = [r["date"] for r in records]
    pcts = [r["pct_of_top"] for r in records]
    ax.plot(dates, pcts, marker="o", linewidth=1.5)
    ax.axhline(y=np.mean(pcts), color="red", linestyle="--", alpha=0.5,
               label=f"Mean: {np.mean(pcts):.1%}")
    ax.set_ylabel("Rank 20 Score / Rank 1 Score")
    ax.set_title("Score Concentration Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)

    # Bottom-left: Stocks within thresholds over time
    ax = axes[1][0]
    w80 = [r["within_80pct"] for r in records]
    w90 = [r["within_90pct"] for r in records]
    w95 = [r["within_95pct"] for r in records]
    ax.plot(dates, w80, marker="o", label="Within 80% of top", linewidth=1.5)
    ax.plot(dates, w90, marker="s", label="Within 90% of top", linewidth=1.5)
    ax.plot(dates, w95, marker="^", label="Within 95% of top", linewidth=1.5)
    ax.axhline(y=20, color="red", linestyle="--", alpha=0.5, label="Current TOP_N=20")
    ax.set_ylabel("Number of Stocks")
    ax.set_title("Stocks Within X% of Top Score")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)

    # Bottom-right: Natural break rank over time
    ax = axes[1][1]
    breaks = [r["biggest_drop_rank"] for r in records]
    ax.bar(range(len(dates)), breaks, color="steelblue", alpha=0.7)
    ax.axhline(y=20, color="red", linestyle="--", alpha=0.5, label="Current TOP_N=20")
    ax.axhline(y=np.mean(breaks), color="orange", linestyle="--", alpha=0.5,
               label=f"Mean: {np.mean(breaks):.0f}")
    ax.set_xticks(range(0, len(dates), 4))
    ax.set_xticklabels([str(d)[:7] for d in dates[::4]], rotation=45)
    ax.set_ylabel("Rank of Biggest Score Drop")
    ax.set_title("Natural Break Point (Biggest Gap in Top 50)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Score Distribution Analysis — Is Dynamic Position Count Worthwhile?",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(CACHE_DIR / "score_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
