"""Verify rescore_snapshots correctness: compare top-N stocks under two weight sets.

Loads precomputed snapshots, rescores with proposed weights, and for several
rebalance dates shows side-by-side top-30 rankings, score breakdowns, and
portfolio overlap.
"""

import polars as pl
from data.universe import get_universe
from data.prices import download_prices
from data.macro import get_risk_free_rate
from scoring.missing import filter_insufficient_history
from scoring.combine import ALL_FACTORS, composite_score
from backtester.engine import precompute_snapshots

CURRENT = {
    "momentum_12_1": 0.25, "momentum_6_1": 0.15, "rel_strength_3mo": 0.10,
    "short_term_reversal": 0.00, "dist_from_ma50": 0.05, "volume_trend": 0.05,
    "obv_slope": 0.05, "realized_vol_60d": 0.15, "vol_trend": 0.05, "price_consistency": 0.15,
}
PROPOSED = {
    "momentum_12_1": 0.30, "momentum_6_1": 0.20, "rel_strength_3mo": 0.10,
    "short_term_reversal": 0.00, "dist_from_ma50": 0.05, "volume_trend": 0.00,
    "obv_slope": 0.05, "realized_vol_60d": 0.00, "vol_trend": 0.10, "price_consistency": 0.20,
}

RANK_COLS = [f"{f.name()}_rank" for f in ALL_FACTORS]
# Columns of interest for display (the ones with changing weights)
KEY_FACTORS = [
    "momentum_12_1", "momentum_6_1", "realized_vol_60d",
    "volume_trend", "vol_trend", "price_consistency",
]
KEY_RANK_COLS = [f"{f}_rank" for f in KEY_FACTORS]


def rescore(snapshot: pl.DataFrame, weights: dict) -> pl.DataFrame:
    """Recompute composite_score with given weights."""
    if snapshot.is_empty():
        return snapshot
    rescored = composite_score(snapshot, weights=weights)
    return rescored.sort("composite_score", descending=True)


def manual_score(row: dict, weights: dict) -> float:
    """Manually compute composite score from rank columns to double-check."""
    total = 0.0
    for factor in ALL_FACTORS:
        rank_col = f"{factor.name()}_rank"
        total += row.get(rank_col, 0.0) * weights[factor.name()]
    return total


def analyze_date(snapshot: pl.DataFrame, date_label: str, top_n: int = 30):
    """For a single rebalance date, compare top stocks under both weight sets."""
    print(f"\n{'=' * 90}")
    print(f"  REBALANCE DATE: {date_label}")
    print(f"  Tickers in snapshot: {len(snapshot)}")
    print(f"{'=' * 90}")

    current_scored = rescore(snapshot, CURRENT)
    proposed_scored = rescore(snapshot, PROPOSED)

    # Sanity check: manually verify composite_score for top ticker under each
    for label, scored, weights in [("CURRENT", current_scored, CURRENT), ("PROPOSED", proposed_scored, PROPOSED)]:
        top_row = scored.head(1).to_dicts()[0]
        computed = top_row["composite_score"]
        manual = manual_score(top_row, weights)
        match = abs(computed - manual) < 1e-9
        print(f"\n  [{label}] Top ticker: {top_row['ticker']}")
        print(f"    composite_score (from Polars): {computed:.6f}")
        print(f"    composite_score (manual calc):  {manual:.6f}")
        print(f"    Match: {'YES' if match else '*** MISMATCH ***'}")

    # Show score distributions
    for label, scored in [("CURRENT", current_scored), ("PROPOSED", proposed_scored)]:
        scores = scored["composite_score"]
        print(f"\n  [{label}] Score distribution (all {len(scored)} tickers):")
        print(f"    Mean: {scores.mean():.4f}  Std: {scores.std():.4f}")
        print(f"    Min: {scores.min():.4f}  Max: {scores.max():.4f}")
        print(f"    Top-20 range: {scored['composite_score'][0]:.4f} - {scored['composite_score'][min(19, len(scored)-1)]:.4f}")

    # Extract top-N tickers under each
    current_top = current_scored.head(top_n)["ticker"].to_list()
    proposed_top = proposed_scored.head(top_n)["ticker"].to_list()

    current_top20 = set(current_top[:20])
    proposed_top20 = set(proposed_top[:20])
    overlap_20 = current_top20 & proposed_top20
    only_current = current_top20 - proposed_top20
    only_proposed = proposed_top20 - current_top20

    print(f"\n  TOP-20 OVERLAP:")
    print(f"    Shared:         {len(overlap_20)} / 20")
    print(f"    Only current:   {len(only_current)}  {sorted(only_current)}")
    print(f"    Only proposed:  {len(only_proposed)}  {sorted(only_proposed)}")

    # Build a combined view
    # Get all tickers that appear in either top-30
    all_tickers = set(current_top) | set(proposed_top)

    # Build lookup dicts
    current_dict = {row["ticker"]: row for row in current_scored.to_dicts()}
    proposed_dict = {row["ticker"]: row for row in proposed_scored.to_dicts()}

    # Current rank (0-indexed position in sorted order)
    current_rank_map = {t: i for i, t in enumerate(current_scored["ticker"].to_list())}
    proposed_rank_map = {t: i for i, t in enumerate(proposed_scored["ticker"].to_list())}

    # Print side-by-side top-30 for CURRENT
    print(f"\n  TOP-30 UNDER CURRENT WEIGHTS:")
    print(f"  {'Rank':>4s}  {'Ticker':<8s}  {'CurrScore':>9s}  {'PropScore':>9s}  {'PropRank':>8s}  ", end="")
    for col in KEY_RANK_COLS:
        short = col.replace("_rank", "").replace("momentum_", "mom").replace("realized_vol_60d", "rvol60").replace("volume_trend", "voltrd").replace("vol_trend", "vtrnd").replace("price_consistency", "pcons")
        print(f"  {short:>8s}", end="")
    print()

    for i, ticker in enumerate(current_top[:30]):
        crow = current_dict[ticker]
        prow = proposed_dict.get(ticker, {})
        cscore = crow["composite_score"]
        pscore = prow.get("composite_score", float("nan"))
        prank = proposed_rank_map.get(ticker, -1)

        marker = ""
        if i < 20 and ticker not in proposed_top20:
            marker = " <-- dropped"
        elif i >= 20 and ticker in proposed_top20:
            marker = " <-- promoted"

        print(f"  {i+1:>4d}  {ticker:<8s}  {cscore:>9.4f}  {pscore:>9.4f}  {prank+1:>8d}  ", end="")
        for col in KEY_RANK_COLS:
            val = crow.get(col, float("nan"))
            print(f"  {val:>8.3f}", end="")
        print(marker)

    # Print top-30 for PROPOSED
    print(f"\n  TOP-30 UNDER PROPOSED WEIGHTS:")
    print(f"  {'Rank':>4s}  {'Ticker':<8s}  {'PropScore':>9s}  {'CurrScore':>9s}  {'CurrRank':>8s}  ", end="")
    for col in KEY_RANK_COLS:
        short = col.replace("_rank", "").replace("momentum_", "mom").replace("realized_vol_60d", "rvol60").replace("volume_trend", "voltrd").replace("vol_trend", "vtrnd").replace("price_consistency", "pcons")
        print(f"  {short:>8s}", end="")
    print()

    for i, ticker in enumerate(proposed_top[:30]):
        prow = proposed_dict[ticker]
        crow = current_dict.get(ticker, {})
        pscore = prow["composite_score"]
        cscore = crow.get("composite_score", float("nan"))
        crank = current_rank_map.get(ticker, -1)

        marker = ""
        if i < 20 and ticker not in current_top20:
            marker = " <-- new"
        elif i >= 20 and ticker in current_top20:
            marker = " <-- demoted"

        print(f"  {i+1:>4d}  {ticker:<8s}  {pscore:>9.4f}  {cscore:>9.4f}  {crank+1:>8d}  ", end="")
        for col in KEY_RANK_COLS:
            val = prow.get(col, float("nan"))
            print(f"  {val:>8.3f}", end="")
        print(marker)

    # Show the stocks that changed and WHY
    print(f"\n  STOCKS DROPPED FROM TOP-20 (current -> proposed):")
    for ticker in sorted(only_current):
        crow = current_dict[ticker]
        crank = current_rank_map[ticker]
        prank = proposed_rank_map.get(ticker, -1)
        rvol = crow.get("realized_vol_60d_rank", float("nan"))
        vtrd = crow.get("volume_trend_rank", float("nan"))
        print(f"    {ticker:<8s}  current_rank={crank+1:3d}  proposed_rank={prank+1:3d}  "
              f"rvol60_rank={rvol:.3f}  voltrd_rank={vtrd:.3f}")

    print(f"\n  STOCKS ADDED TO TOP-20 (proposed only):")
    for ticker in sorted(only_proposed):
        prow = proposed_dict[ticker]
        crank = current_rank_map.get(ticker, -1)
        prank = proposed_rank_map[ticker]
        rvol = prow.get("realized_vol_60d_rank", float("nan"))
        vtrd = prow.get("volume_trend_rank", float("nan"))
        print(f"    {ticker:<8s}  current_rank={crank+1:3d}  proposed_rank={prank+1:3d}  "
              f"rvol60_rank={rvol:.3f}  voltrd_rank={vtrd:.3f}")

    # Show the score delta breakdown for tickers that changed
    print(f"\n  SCORE DELTA BREAKDOWN for selected tickers:")
    check_tickers = sorted(only_current)[:3] + sorted(only_proposed)[:3]
    if not check_tickers:
        check_tickers = current_top[:3]

    for ticker in check_tickers:
        row = current_dict.get(ticker, proposed_dict.get(ticker, {}))
        if not row:
            continue
        print(f"\n    {ticker}:")
        curr_total = 0.0
        prop_total = 0.0
        for factor in ALL_FACTORS:
            rank_col = f"{factor.name()}_rank"
            rank_val = row.get(rank_col, 0.0)
            cw = CURRENT[factor.name()]
            pw = PROPOSED[factor.name()]
            cc = rank_val * cw
            pc = rank_val * pw
            delta = pc - cc
            curr_total += cc
            prop_total += pc
            if abs(delta) > 0.001 or cw != pw:
                print(f"      {factor.name():25s}  rank={rank_val:.3f}  "
                      f"curr_w={cw:.2f}*{rank_val:.3f}={cc:.4f}  "
                      f"prop_w={pw:.2f}*{rank_val:.3f}={pc:.4f}  "
                      f"delta={delta:+.4f}")
        print(f"      {'TOTAL':25s}  curr={curr_total:.4f}  prop={prop_total:.4f}  delta={prop_total-curr_total:+.4f}")


def main():
    print("=== Rescore Verification ===\n")

    print("Weight changes (current -> proposed):")
    for name in CURRENT:
        c, p = CURRENT[name], PROPOSED[name]
        if c != p:
            print(f"  {name:25s}  {c:.2f} -> {p:.2f}  ({p-c:+.2f})")

    print("\n1. Loading data...")
    universe = get_universe()
    tickers = universe["ticker"].to_list()
    for bench in ["SPY", "RSP"]:
        if bench not in tickers:
            tickers.append(bench)

    prices = download_prices(tickers)
    prices = filter_insufficient_history(prices)
    print(f"   {prices['ticker'].n_unique()} tickers loaded\n")

    print("2. Computing factor snapshots (this takes a while)...")
    precomputed = precompute_snapshots(prices)
    is_dates = precomputed["is_dates"]
    oos_dates = precomputed["oos_dates"]
    all_dates = is_dates + oos_dates
    print(f"   {len(all_dates)} rebalance dates ({len(is_dates)} IS + {len(oos_dates)} OOS)")

    # Pick 4 dates: early IS, mid IS, late IS, mid OOS
    check_indices = [
        ("Early IS", 2),
        ("Mid IS", len(is_dates) // 2),
        ("Late IS (last before OOS)", len(is_dates) - 1),
        ("Mid OOS", len(is_dates) + len(oos_dates) // 2),
    ]

    dates_to_check = []
    for label, idx in check_indices:
        if idx < len(all_dates):
            dates_to_check.append((f"{label} [{all_dates[idx]}]", all_dates[idx]))

    print(f"\n3. Checking {len(dates_to_check)} rebalance dates:\n")
    for label, _ in dates_to_check:
        print(f"   - {label}")

    # Summary stats across all dates
    all_overlaps = []

    for label, date in dates_to_check:
        snapshot = precomputed["snapshots"].get(date)
        if snapshot is None or snapshot.is_empty():
            print(f"\n  *** No snapshot for {date} ***")
            continue
        analyze_date(snapshot, label)

        # Collect overlap stat
        current_top20 = set(rescore(snapshot, CURRENT).head(20)["ticker"].to_list())
        proposed_top20 = set(rescore(snapshot, PROPOSED).head(20)["ticker"].to_list())
        overlap = len(current_top20 & proposed_top20)
        all_overlaps.append((label, overlap))

    print(f"\n\n{'=' * 90}")
    print(f"  SUMMARY ACROSS ALL CHECKED DATES")
    print(f"{'=' * 90}")
    for label, overlap in all_overlaps:
        print(f"  {label:50s}  Top-20 overlap: {overlap}/20 ({overlap/20:.0%})")
    avg_overlap = sum(o for _, o in all_overlaps) / len(all_overlaps) if all_overlaps else 0
    print(f"\n  Average overlap: {avg_overlap:.1f}/20 ({avg_overlap/20:.0%})")
    print(f"  Average turnover from weight change: {(20-avg_overlap)/20:.0%}")

    # Final check: verify rescore_snapshots matches manual rescoring
    print(f"\n\n{'=' * 90}")
    print(f"  CROSS-CHECK: rescore_snapshots() vs manual composite_score()")
    print(f"{'=' * 90}")
    from run_decay_test import rescore_snapshots
    proposed_precomputed = rescore_snapshots(precomputed, PROPOSED)
    for label, date in dates_to_check[:2]:
        snap_from_fn = proposed_precomputed["snapshots"][date]
        snap_manual = rescore(precomputed["snapshots"][date], PROPOSED)

        # Compare top 5 tickers and scores
        fn_top5 = snap_from_fn.head(5).select("ticker", "composite_score").to_dicts()
        man_top5 = snap_manual.head(5).select("ticker", "composite_score").to_dicts()

        match = all(
            fn_top5[i]["ticker"] == man_top5[i]["ticker"] and
            abs(fn_top5[i]["composite_score"] - man_top5[i]["composite_score"]) < 1e-9
            for i in range(min(len(fn_top5), len(man_top5)))
        )
        print(f"\n  {label}:")
        print(f"    rescore_snapshots top-5: {[r['ticker'] for r in fn_top5]}")
        print(f"    manual rescore top-5:    {[r['ticker'] for r in man_top5]}")
        print(f"    Match: {'YES' if match else '*** MISMATCH ***'}")


if __name__ == "__main__":
    main()
