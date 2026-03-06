import polars as pl
from config.settings import TOP_N, MIN_COMPOSITE_PERCENTILE, SELL_THRESHOLD_RANK, WEIGHTING


def select_top_n(
    scored: pl.DataFrame,
    sector_map: dict[str, str],
    previous_holdings: list[str] | None = None,
    sector_cap: float = 0.25,
) -> pl.DataFrame:
    """Select top N stocks with sector constraints and turnover dampening.

    Args:
        scored: DataFrame with [ticker, composite_score] sorted descending
        sector_map: {ticker: sector}
        previous_holdings: tickers held in previous period
        sector_cap: max weight in any one sector
    """
    if previous_holdings is None:
        previous_holdings = []

    scored = scored.with_columns(
        pl.col("ticker").replace_strict(sector_map, default="Unknown").alias("sector")
    )

    min_score = scored["composite_score"].quantile(MIN_COMPOSITE_PERCENTILE)

    eligible = scored.filter(pl.col("composite_score") >= min_score)

    selected = []
    sector_counts: dict[str, int] = {}
    max_per_sector = int(TOP_N * sector_cap)

    prev_set = set(previous_holdings)
    eligible_rows = eligible.to_dicts()

    for row in eligible_rows:
        if len(selected) >= TOP_N:
            break

        ticker = row["ticker"]
        sector = row["sector"]
        current_count = sector_counts.get(sector, 0)

        if current_count >= max_per_sector:
            continue

        selected.append(row)
        sector_counts[sector] = current_count + 1

    selected_tickers = {r["ticker"] for r in selected}

    if prev_set:
        retained = []
        for row in eligible_rows:
            ticker = row["ticker"]
            if ticker in prev_set and ticker not in selected_tickers:
                rank_in_list = next(
                    (i for i, r in enumerate(eligible_rows) if r["ticker"] == ticker),
                    TOP_N + 1,
                )
                if rank_in_list < SELL_THRESHOLD_RANK:
                    sector = row["sector"]
                    if sector_counts.get(sector, 0) < max_per_sector:
                        retained.append(row)
                        sector_counts[sector] = sector_counts.get(sector, 0) + 1

        if retained:
            selected = selected[: TOP_N - len(retained)] + retained

    result = pl.DataFrame(selected[:TOP_N])

    if len(result) == 0:
        return result.with_columns(pl.lit(0.0).alias("weight"))

    if WEIGHTING == "score":
        scores = result["composite_score"]
        total = scores.sum()
        result = result.with_columns(
            (pl.col("composite_score") / total).alias("weight")
        )
    else:
        weight = 1.0 / len(result)
        result = result.with_columns(pl.lit(weight).alias("weight"))

    return result
