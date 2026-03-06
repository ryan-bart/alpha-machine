import polars as pl
from factors.base import BaseFactor


class PriceConsistency(BaseFactor):
    def name(self):
        return "price_consistency"

    def compute(self, df):
        daily_ret = pl.col("close") / pl.col("close").shift(1) - 1.0
        positive = daily_ret.gt(0).cast(pl.Float64)
        return df.with_columns(
            positive.rolling_mean(60).alias(self.name())
        )
