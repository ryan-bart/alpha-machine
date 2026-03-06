import yfinance as yf
import polars as pl
import pandas as pd
from datetime import datetime, timedelta
from config.settings import PRICES_CACHE, PRICE_HISTORY_YEARS


def download_prices(tickers, force_refresh=False):
    start_date = (datetime.now() - timedelta(days=PRICE_HISTORY_YEARS * 365)).strftime("%Y-%m-%d")

    if not force_refresh and PRICES_CACHE.exists():
        cached = pl.read_parquet(PRICES_CACHE)
        cached_tickers = set(cached["ticker"].unique().to_list())
        missing = [t for t in tickers if t not in cached_tickers]

        last_date = cached["date"].max()
        today = datetime.now().date()
        needs_update = (today - last_date).days > 1

        if not missing and not needs_update:
            return cached.filter(pl.col("ticker").is_in(tickers))

        if needs_update:
            update_start = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
            new_data = _fetch_batch(tickers, update_start)
            if new_data is not None:
                cached = pl.concat([cached, new_data]).unique(
                    subset=["ticker", "date"], keep="last"
                )

        if missing:
            missing_data = _fetch_batch(missing, start_date)
            if missing_data is not None:
                cached = pl.concat([cached, missing_data]).unique(
                    subset=["ticker", "date"], keep="last"
                )

        cached = cached.sort(["ticker", "date"])
        cached.write_parquet(PRICES_CACHE)
        return cached.filter(pl.col("ticker").is_in(tickers))

    data = _fetch_batch(tickers, start_date)
    if data is not None:
        data = data.sort(["ticker", "date"])
        data.write_parquet(PRICES_CACHE)
    return data


def _fetch_batch(tickers, start_date):
    print(f"Downloading {len(tickers)} tickers from {start_date}...")
    raw = yf.download(tickers, start=start_date, threads=True)

    if raw.empty:
        return None

    # yfinance returns multi-index: (Field, Ticker) for multi-ticker,
    # or (Field, Ticker) even for single-ticker in recent versions
    raw = raw.reset_index()

    if not isinstance(raw.columns, pd.MultiIndex):
        # Simple columns (shouldn't happen in yfinance >=1.2 but handle it)
        ticker = tickers[0]
        raw.columns = [c if c != "Date" else "Date" for c in raw.columns]
        cols = {c: c for c in ["Date", "Open", "High", "Low", "Close", "Volume"] if c in raw.columns}
        df = raw[list(cols.keys())].dropna()
        df["ticker"] = ticker
        combined = df
    else:
        # Multi-index: level 0 = field, level 1 = ticker
        frames = []
        # Get list of tickers present in the data
        available_tickers = raw.columns.get_level_values(1).unique()
        available_tickers = [t for t in available_tickers if t and t != ""]

        for ticker in available_tickers:
            try:
                sub = pd.DataFrame()
                sub["Date"] = raw[("Date", "")].values if ("Date", "") in raw.columns else raw.iloc[:, 0]
                for field in ["Open", "High", "Low", "Close", "Volume"]:
                    sub[field] = raw[(field, ticker)].values
                sub = sub.dropna()
                sub["ticker"] = ticker
                frames.append(sub)
            except (KeyError, TypeError):
                continue

        if not frames:
            return None
        combined = pd.concat(frames, ignore_index=True)

    combined.columns = [c.lower() for c in combined.columns]
    result = pl.from_pandas(combined)
    result = result.with_columns(pl.col("date").cast(pl.Date))
    return result
