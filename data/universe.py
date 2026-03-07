import polars as pl
import pandas as pd
import urllib.request
from io import StringIO
from datetime import datetime, timedelta
from config.settings import (
    UNIVERSE_CACHE, UNIVERSE_REFRESH_DAYS, SP500_WIKI_URL,
    RUSSELL1000_WIKI_URL, UNIVERSE_SOURCE,
)


def _fetch_sp500():
    req = urllib.request.Request(SP500_WIKI_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        html = resp.read().decode()
    tables = pd.read_html(StringIO(html))
    df = tables[0]
    return pl.DataFrame({
        "ticker": df["Symbol"].str.replace(".", "-", regex=False).tolist(),
        "company": df["Security"].tolist(),
        "sector": df["GICS Sector"].tolist(),
        "sub_industry": df["GICS Sub-Industry"].tolist(),
        "fetched_on": [datetime.now()] * len(df),
    })


def _fetch_russell1000():
    req = urllib.request.Request(RUSSELL1000_WIKI_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        html = resp.read().decode()
    tables = pd.read_html(StringIO(html))
    df = tables[3]
    df = df.fillna({"GICS Sector": "Unknown", "GICS Sub-Industry": "Unknown"})
    return pl.DataFrame({
        "ticker": df["Symbol"].str.replace(".", "-", regex=False).tolist(),
        "company": df["Company"].tolist(),
        "sector": df["GICS Sector"].tolist(),
        "sub_industry": df["GICS Sub-Industry"].tolist(),
        "fetched_on": [datetime.now()] * len(df),
    })


_FETCHERS = {
    "sp500": _fetch_sp500,
    "russell1000": _fetch_russell1000,
}


def get_universe(force_refresh=False):
    if not force_refresh and UNIVERSE_CACHE.exists():
        cached = pl.read_parquet(UNIVERSE_CACHE)
        fetched_on = cached["fetched_on"][0]
        if datetime.now() - fetched_on < timedelta(days=UNIVERSE_REFRESH_DAYS):
            return cached

    fetcher = _FETCHERS[UNIVERSE_SOURCE]
    result = fetcher()
    result.write_parquet(UNIVERSE_CACHE)
    return result


# Backward compat alias
get_sp500_tickers = get_universe
