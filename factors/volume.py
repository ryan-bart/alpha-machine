import polars as pl
import numpy as np
from factors.base import BaseFactor


class VolumeTrend(BaseFactor):
    def name(self):
        return "volume_trend"

    def compute(self, df):
        return df.with_columns(
            (pl.col("volume").rolling_mean(20) / pl.col("volume").rolling_mean(60) - 1.0)
            .alias(self.name())
        )


class OBVSlope(BaseFactor):
    def name(self):
        return "obv_slope"

    def compute(self, df):
        price_change = pl.col("close") - pl.col("close").shift(1)
        signed_vol = (
            pl.when(price_change > 0).then(pl.col("volume"))
            .when(price_change < 0).then(-pl.col("volume"))
            .otherwise(pl.lit(0))
        )
        df = df.with_columns(signed_vol.cum_sum().alias("_obv"))

        obv_vals = df["_obv"].to_numpy().astype(float)
        slopes = np.full(len(obv_vals), np.nan)
        window = 40
        x = np.arange(window, dtype=float)
        x_mean = x.mean()
        x_var = ((x - x_mean) ** 2).sum()

        for i in range(window - 1, len(obv_vals)):
            y = obv_vals[i - window + 1: i + 1]
            if np.any(np.isnan(y)):
                continue
            y_mean = y.mean()
            slopes[i] = ((x - x_mean) * (y - y_mean)).sum() / x_var

        df = df.with_columns(
            pl.Series(name=self.name(), values=slopes)
        ).drop("_obv")
        return df
