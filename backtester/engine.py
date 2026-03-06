import polars as pl
import numpy as np
from datetime import datetime
from config.settings import INITIAL_CASH, HOLDOUT_QUARTERS, TOP_N, REBALANCE_FREQ
from scoring.combine import compute_all_factors, composite_score
from portfolio.construct import select_top_n


def run_backtest(
    prices: pl.DataFrame,
    sector_map: dict[str, str],
    holdout_periods: int = HOLDOUT_QUARTERS,
    weights: dict | None = None,
    freq: str = REBALANCE_FREQ,
) -> dict:
    """Walk-forward backtest with configurable rebalance frequency.

    Returns dict with equity_curve, rebalance_log, is_dates, oos_dates.
    """
    dates = prices["date"].unique().sort().to_list()

    if freq == "QS":
        rebalance_dates = _get_quarterly_rebalance_dates(dates)
    else:
        rebalance_dates = _get_monthly_rebalance_dates(dates)

    if holdout_periods > 0:
        cutoff_idx = len(rebalance_dates) - holdout_periods
        is_dates = rebalance_dates[:cutoff_idx]
        oos_dates = rebalance_dates[cutoff_idx:]
    else:
        is_dates = rebalance_dates
        oos_dates = []

    cash = float(INITIAL_CASH)
    holdings: dict[str, float] = {}
    previous_tickers: list[str] = []

    equity_curve = []
    rebalance_log = []

    all_rebalance_dates = is_dates + oos_dates

    for i, reb_date in enumerate(all_rebalance_dates):
        available_prices = prices.filter(pl.col("date") <= reb_date)
        factored = compute_all_factors(available_prices)
        scored = composite_score(factored, weights=weights)

        snapshot = scored.filter(pl.col("date") == reb_date)
        if snapshot.is_empty():
            closest = scored.filter(pl.col("date") <= reb_date).sort("date", descending=True)
            if not closest.is_empty():
                latest_date = closest["date"][0]
                snapshot = scored.filter(pl.col("date") == latest_date)

        if snapshot.is_empty():
            continue

        snapshot = snapshot.sort("composite_score", descending=True)

        portfolio = select_top_n(snapshot, sector_map, previous_tickers)

        portfolio_value = _get_portfolio_value(holdings, prices, reb_date) + cash

        selected_tickers = portfolio["ticker"].to_list()
        port_weights = portfolio["weight"].to_list()

        holdings = {}
        cash = 0.0
        for ticker, pw in zip(selected_tickers, port_weights):
            allocation = portfolio_value * pw
            price_row = prices.filter(
                (pl.col("ticker") == ticker) & (pl.col("date") == reb_date)
            )
            if not price_row.is_empty():
                price = price_row["close"][0]
                holdings[ticker] = allocation / price
            else:
                cash += allocation

        previous_tickers = selected_tickers
        is_oos = reb_date in oos_dates

        rebalance_log.append({
            "date": reb_date,
            "tickers": selected_tickers,
            "portfolio_value": portfolio_value,
            "is_oos": is_oos,
        })

        next_reb = all_rebalance_dates[i + 1] if i + 1 < len(all_rebalance_dates) else dates[-1]
        period_dates = [d for d in dates if reb_date <= d <= next_reb]

        for d in period_dates:
            val = _get_portfolio_value(holdings, prices, d) + cash
            equity_curve.append({"date": d, "value": val, "is_oos": is_oos})

    return {
        "equity_curve": pl.DataFrame(equity_curve),
        "rebalance_log": rebalance_log,
        "is_dates": is_dates,
        "oos_dates": oos_dates,
    }


def _get_monthly_rebalance_dates(dates):
    """First trading day of each month."""
    rebalance = []
    current_month = None
    for d in sorted(dates):
        month_key = (d.year, d.month)
        if month_key != current_month:
            rebalance.append(d)
            current_month = month_key
    return rebalance


def _get_quarterly_rebalance_dates(dates):
    """First trading day of each quarter (Jan, Apr, Jul, Oct)."""
    quarter_months = {1, 4, 7, 10}
    rebalance = []
    current_quarter = None
    for d in sorted(dates):
        q_key = (d.year, (d.month - 1) // 3)
        if q_key != current_quarter and d.month in quarter_months:
            rebalance.append(d)
            current_quarter = q_key
    return rebalance


def _get_portfolio_value(holdings, prices, date):
    """Sum of holdings * price at given date."""
    total = 0.0
    for ticker, shares in holdings.items():
        price_row = prices.filter(
            (pl.col("ticker") == ticker) & (pl.col("date") == date)
        )
        if not price_row.is_empty():
            total += shares * price_row["close"][0]
    return total
