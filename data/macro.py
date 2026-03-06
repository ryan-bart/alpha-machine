import polars as pl
import pandas as pd
from io import StringIO
from config.settings import FRED_TB3MS_URL, CACHE_DIR


MACRO_CACHE = CACHE_DIR / "macro_tb3ms.parquet"


def get_risk_free_rate():
    if MACRO_CACHE.exists():
        df = pl.read_parquet(MACRO_CACHE)
        latest = df.filter(pl.col("value").is_not_null()).sort("date").tail(1)
        if not latest.is_empty():
            return latest["value"][0] / 100.0

    try:
        import urllib.request
        with urllib.request.urlopen(FRED_TB3MS_URL) as resp:
            text = resp.read().decode()
        pdf = pd.read_csv(StringIO(text))
        pdf.columns = ["date", "value"]
        pdf["value"] = pd.to_numeric(pdf["value"], errors="coerce")
        pdf["date"] = pd.to_datetime(pdf["date"])
        df = pl.from_pandas(pdf).drop_nulls("value")
        df.write_parquet(MACRO_CACHE)
        latest = df.sort("date").tail(1)
        return latest["value"][0] / 100.0
    except Exception as e:
        print(f"Warning: could not fetch FRED data ({e}), using default risk-free rate 0.05")
        return 0.05
