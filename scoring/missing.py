import polars as pl
from config.settings import MIN_TRADING_DAYS


def filter_insufficient_history(df: pl.DataFrame) -> pl.DataFrame:
    """Remove tickers with fewer than MIN_TRADING_DAYS of data."""
    counts = df.group_by("ticker").agg(pl.col("date").count().alias("n_days"))
    valid_tickers = counts.filter(pl.col("n_days") >= MIN_TRADING_DAYS)["ticker"]
    return df.filter(pl.col("ticker").is_in(valid_tickers))


def impute_missing_ranks(df: pl.DataFrame, rank_columns: list[str]) -> pl.DataFrame:
    """Fill missing percentile ranks with 0.5 (median rank)."""
    return df.with_columns(
        [pl.col(c).fill_null(0.5) for c in rank_columns]
    )
