import polars as pl
from factors.base import BaseFactor


class ShortTermReversal(BaseFactor):
    def name(self):
        return "short_term_reversal"

    @property
    def invert(self):
        return True

    def compute(self, df):
        return df.with_columns(
            (pl.col("close") / pl.col("close").shift(5) - 1.0)
            .alias(self.name())
        )


class DistFromMA50(BaseFactor):
    def name(self):
        return "dist_from_ma50"

    @property
    def invert(self):
        return True

    def compute(self, df):
        return df.with_columns(
            ((pl.col("close") - pl.col("close").rolling_mean(50))
             / pl.col("close").rolling_mean(50))
            .alias(self.name())
        )
