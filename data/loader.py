import yfinance as yf
import polars as pl


def load_data(symbol, start="2015-01-01"):

    df = yf.download(symbol, start=start)

    # flatten multi-index columns from yfinance
    if hasattr(df.columns, "levels"):
        df.columns = df.columns.get_level_values(0)

    # reset index so Date becomes a regular column
    df = df.reset_index()

    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]

    df = df.dropna()

    # convert to polars (Date is now a column, not the index)
    pl_df = pl.from_pandas(df)

    return pl_df