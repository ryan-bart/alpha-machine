import polars as pl
from config.settings import FACTOR_WEIGHTS
from factors.momentum import Momentum12M1M, Momentum6M1M, RelativeStrength3M
from factors.mean_reversion import ShortTermReversal, DistFromMA50
from factors.volume import VolumeTrend, OBVSlope
from factors.volatility import RealizedVol60D, VolTrend
from factors.quality import PriceConsistency
from scoring.normalize import percentile_rank
from scoring.missing import impute_missing_ranks


ALL_FACTORS = [
    Momentum12M1M(),
    Momentum6M1M(),
    RelativeStrength3M(),
    ShortTermReversal(),
    DistFromMA50(),
    VolumeTrend(),
    OBVSlope(),
    RealizedVol60D(),
    VolTrend(),
    PriceConsistency(),
]


def compute_all_factors(prices: pl.DataFrame) -> pl.DataFrame:
    """Compute all 10 factors per ticker, then cross-sectional percentile rank."""
    tickers = prices["ticker"].unique().to_list()
    all_frames = []

    for ticker in tickers:
        ticker_df = prices.filter(pl.col("ticker") == ticker).sort("date")
        for factor in ALL_FACTORS:
            ticker_df = factor.compute(ticker_df)
        all_frames.append(ticker_df)

    combined = pl.concat(all_frames)

    rank_columns = []
    for factor in ALL_FACTORS:
        combined = percentile_rank(combined, factor.name(), invert=factor.invert)
        rank_columns.append(f"{factor.name()}_rank")

    combined = impute_missing_ranks(combined, rank_columns)
    return combined


def composite_score(df: pl.DataFrame, weights: dict | None = None) -> pl.DataFrame:
    """Compute weighted composite score from percentile ranks."""
    if weights is None:
        weights = FACTOR_WEIGHTS

    score_expr = pl.lit(0.0)
    for factor in ALL_FACTORS:
        rank_col = f"{factor.name()}_rank"
        weight = weights[factor.name()]
        score_expr = score_expr + pl.col(rank_col) * weight

    df = df.with_columns(score_expr.alias("composite_score"))
    return df


def score_universe(prices: pl.DataFrame, as_of_date=None) -> pl.DataFrame:
    """Full pipeline: factors → ranks → composite → latest snapshot."""
    factored = compute_all_factors(prices)
    scored = composite_score(factored)

    if as_of_date is not None:
        snapshot = scored.filter(pl.col("date") == as_of_date)
    else:
        snapshot = scored.group_by("ticker").agg(
            pl.all().sort_by("date").last()
        )

    return snapshot.sort("composite_score", descending=True)
