import polars as pl


def percentile_rank(df: pl.DataFrame, factor_name: str, invert: bool = False) -> pl.DataFrame:
    """Cross-sectional percentile rank for a single factor column.

    Expects df with columns [ticker, date, <factor_name>].
    Returns df with additional column <factor_name>_rank in [0, 1].
    """
    rank_col = f"{factor_name}_rank"

    ranked = df.with_columns(
        pl.col(factor_name)
        .rank("ordinal")
        .over("date")
        .alias("_raw_rank")
    ).with_columns(
        pl.col(factor_name)
        .count()
        .over("date")
        .alias("_count")
    ).with_columns(
        ((pl.col("_raw_rank") - 1) / (pl.col("_count") - 1))
        .alias(rank_col)
    ).drop("_raw_rank", "_count")

    if invert:
        ranked = ranked.with_columns(
            (1.0 - pl.col(rank_col)).alias(rank_col)
        )

    return ranked
