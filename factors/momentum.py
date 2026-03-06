import polars as pl
from factors.base import BaseFactor


class Momentum12M1M(BaseFactor):
    def name(self):
        return "momentum_12_1"

    def compute(self, df):
        return df.with_columns(
            (pl.col("close").shift(21) / pl.col("close").shift(252) - 1.0)
            .alias(self.name())
        )


class Momentum6M1M(BaseFactor):
    def name(self):
        return "momentum_6_1"

    def compute(self, df):
        return df.with_columns(
            (pl.col("close").shift(21) / pl.col("close").shift(126) - 1.0)
            .alias(self.name())
        )


class RelativeStrength3M(BaseFactor):
    def name(self):
        return "rel_strength_3mo"

    def compute(self, df):
        return df.with_columns(
            (pl.col("close") / pl.col("close").shift(63) - 1.0)
            .alias(self.name())
        )
