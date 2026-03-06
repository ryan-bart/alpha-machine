import polars as pl
import numpy as np
from factors.base import BaseFactor


class RealizedVol60D(BaseFactor):
    def name(self):
        return "realized_vol_60d"

    @property
    def invert(self):
        return True

    def compute(self, df):
        log_ret = (pl.col("close") / pl.col("close").shift(1)).log()
        return df.with_columns(
            (log_ret.rolling_std(60) * np.sqrt(252)).alias(self.name())
        )


class VolTrend(BaseFactor):
    def name(self):
        return "vol_trend"

    @property
    def invert(self):
        return True

    def compute(self, df):
        log_ret = (pl.col("close") / pl.col("close").shift(1)).log()
        return df.with_columns(
            (log_ret.rolling_std(20) / log_ret.rolling_std(60) - 1.0)
            .alias(self.name())
        )
