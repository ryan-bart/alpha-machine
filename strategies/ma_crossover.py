import polars as pl
from strategies.base_strategy import BaseStrategy


class MovingAverageCrossStrategy(BaseStrategy):

    def __init__(self, fast=20, slow=50):
        self.fast = fast
        self.slow = slow

    def generate_signals(self, data):

        df = data.with_columns([
            pl.col("Close").rolling_mean(self.fast).alias("fast_ma"),
            pl.col("Close").rolling_mean(self.slow).alias("slow_ma")
        ])

        df = df.with_columns(
            pl.when(pl.col("fast_ma") > pl.col("slow_ma"))
            .then(1)
            .when(pl.col("fast_ma") < pl.col("slow_ma"))
            .then(-1)
            .otherwise(0)
            .alias("signal")
        )

        return df