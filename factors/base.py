import polars as pl
from abc import ABC, abstractmethod


class BaseFactor(ABC):
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute factor for a single ticker.

        Input: DataFrame with columns [date, open, high, low, close, volume, ticker]
        Output: same DataFrame with an additional column named self.name()
        """
        pass

    @property
    def invert(self) -> bool:
        """If True, lower raw values are better (will be rank-inverted)."""
        return False
