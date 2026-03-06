import polars as pl
import numpy as np


def compute_metrics(equity_curve: pl.DataFrame, risk_free_rate: float = 0.05) -> dict:
    """Compute performance metrics from an equity curve DataFrame [date, value]."""
    if equity_curve.is_empty() or len(equity_curve) < 2:
        return {}

    values = equity_curve.sort("date")["value"].to_numpy().astype(float)
    dates = equity_curve.sort("date")["date"].to_list()

    total_days = (dates[-1] - dates[0]).days
    years = total_days / 365.25

    cagr = (values[-1] / values[0]) ** (1.0 / years) - 1.0 if years > 0 else 0.0

    daily_returns = np.diff(values) / values[:-1]
    daily_returns = daily_returns[np.isfinite(daily_returns)]

    if len(daily_returns) == 0:
        return {"cagr": cagr}

    ann_vol = np.std(daily_returns) * np.sqrt(252)
    daily_rf = risk_free_rate / 252
    excess = daily_returns - daily_rf
    sharpe = np.mean(excess) / np.std(excess) * np.sqrt(252) if np.std(excess) > 0 else 0.0

    running_max = np.maximum.accumulate(values)
    drawdowns = (values - running_max) / running_max
    max_drawdown = np.min(drawdowns)

    calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else 0.0

    monthly_returns = _compute_monthly_returns(equity_curve)
    hit_rate = np.mean(monthly_returns > 0) if len(monthly_returns) > 0 else 0.0

    return {
        "cagr": round(cagr * 100, 2),
        "annual_vol": round(ann_vol * 100, 2),
        "sharpe": round(sharpe, 2),
        "max_drawdown": round(max_drawdown * 100, 2),
        "calmar": round(calmar, 2),
        "monthly_hit_rate": round(hit_rate * 100, 1),
        "total_return": round((values[-1] / values[0] - 1) * 100, 2),
        "start_date": str(dates[0]),
        "end_date": str(dates[-1]),
    }


def compute_benchmark_metrics(
    prices: pl.DataFrame, risk_free_rate: float = 0.05, ticker: str = "SPY",
    start_date=None, end_date=None,
) -> dict:
    """Compute buy-and-hold metrics for a benchmark ticker, optionally within a date range."""
    bench = prices.filter(pl.col("ticker") == ticker).sort("date")
    if bench.is_empty():
        return {}

    if start_date is not None:
        bench = bench.filter(pl.col("date") >= start_date)
    if end_date is not None:
        bench = bench.filter(pl.col("date") <= end_date)

    if bench.is_empty():
        return {}

    bench_curve = bench.select(
        pl.col("date"),
        (pl.col("close") / pl.col("close").first() * 100_000).alias("value"),
    )
    return compute_metrics(bench_curve, risk_free_rate)


def _compute_monthly_returns(equity_curve: pl.DataFrame) -> np.ndarray:
    """Compute monthly returns from daily equity curve."""
    ec = equity_curve.sort("date").with_columns(
        pl.col("date").dt.month().alias("month"),
        pl.col("date").dt.year().alias("year"),
    )

    monthly = ec.group_by(["year", "month"]).agg([
        pl.col("value").first().alias("start_val"),
        pl.col("value").last().alias("end_val"),
    ])

    returns = (monthly["end_val"] / monthly["start_val"] - 1.0).to_numpy()
    return returns[np.isfinite(returns)]
