import polars as pl
from config.settings import TOP_N, MIN_COMPOSITE_PERCENTILE, SELL_THRESHOLD_RANK, WEIGHTING, SECTOR_CAP, SCORE_TILT


def select_top_n(
    scored: pl.DataFrame,
    sector_map: dict[str, str],
    previous_holdings: list[str] | None = None,
    sector_cap: float = SECTOR_CAP,
    sell_threshold_rank: int = SELL_THRESHOLD_RANK,
    score_tilt: float = SCORE_TILT,
) -> pl.DataFrame:
    """Select top N stocks with sector constraints and turnover dampening.

    Args:
        scored: DataFrame with [ticker, composite_score] sorted descending
        sector_map: {ticker: sector}
        previous_holdings: tickers held in previous period
        sector_cap: max weight in any one sector
        sell_threshold_rank: retain holdings ranked above this (lower = less dampening)
        score_tilt: 0 = equal weight, 1 = proportional to score, >1 = concentrated
    """
    if previous_holdings is None:
        previous_holdings = []

    scored = scored.with_columns(
        pl.col("ticker").replace_strict(sector_map, default="Unknown").alias("sector")
    )

    min_score = scored["composite_score"].quantile(MIN_COMPOSITE_PERCENTILE)

    eligible = scored.filter(pl.col("composite_score") >= min_score)

    prev_set = set(previous_holdings) if previous_holdings else set()
    eligible_rows = eligible.to_dicts()
    max_per_sector = int(TOP_N * sector_cap)

    # Build rank lookup (0-indexed)
    ticker_rank = {row["ticker"]: i for i, row in enumerate(eligible_rows)}

    # Step 1: Retain previous holdings that haven't dropped below threshold
    retained = []
    sector_counts: dict[str, int] = {}
    if prev_set:
        for row in eligible_rows:
            ticker = row["ticker"]
            if ticker in prev_set and ticker_rank[ticker] < sell_threshold_rank:
                sector = row["sector"]
                if sector_counts.get(sector, 0) < max_per_sector:
                    retained.append(row)
                    sector_counts[sector] = sector_counts.get(sector, 0) + 1

    retained_tickers = {r["ticker"] for r in retained}

    # Step 2: Fill remaining slots from top-ranked stocks not already retained
    selected = list(retained)
    for row in eligible_rows:
        if len(selected) >= TOP_N:
            break
        ticker = row["ticker"]
        if ticker in retained_tickers:
            continue
        sector = row["sector"]
        if sector_counts.get(sector, 0) >= max_per_sector:
            continue
        selected.append(row)
        sector_counts[sector] = sector_counts.get(sector, 0) + 1

    result = pl.DataFrame(selected[:TOP_N])

    if len(result) == 0:
        return result.with_columns(pl.lit(0.0).alias("weight"))

    if score_tilt > 0:
        # weight ∝ score ^ tilt (0 = equal, 1 = proportional, >1 = concentrated)
        result = result.with_columns(
            pl.col("composite_score").pow(score_tilt).alias("_raw_weight")
        )
        total = result["_raw_weight"].sum()
        result = result.with_columns(
            (pl.col("_raw_weight") / total).alias("weight")
        ).drop("_raw_weight")
    else:
        weight = 1.0 / len(result)
        result = result.with_columns(pl.lit(weight).alias("weight"))

    return result
