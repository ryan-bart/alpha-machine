import polars as pl
import numpy as np
from datetime import datetime
from config.settings import INITIAL_CASH, HOLDOUT_MONTHS, TOP_N
from scoring.combine import compute_all_factors, composite_score
from portfolio.construct import select_top_n


def run_backtest(
    prices: pl.DataFrame,
    sector_map: dict[str, str],
    holdout_months: int = HOLDOUT_MONTHS,
) -> dict:
    """Walk-forward monthly backtest.

    Returns dict with equity_curve, holdings_history, rebalance_dates, and metrics.
    """
    dates = prices["date"].unique().sort().to_list()

    rebalance_dates = _get_monthly_rebalance_dates(dates)

    if holdout_months > 0:
        cutoff_idx = len(rebalance_dates) - holdout_months
        is_dates = rebalance_dates[:cutoff_idx]
        oos_dates = rebalance_dates[cutoff_idx:]
    else:
        is_dates = rebalance_dates
        oos_dates = []

    cash = float(INITIAL_CASH)
    holdings: dict[str, float] = {}  # ticker -> shares
    previous_tickers: list[str] = []

    equity_curve = []
    holdings_history = []
    rebalance_log = []

    all_rebalance_dates = is_dates + oos_dates

    for i, reb_date in enumerate(all_rebalance_dates):
        available_prices = prices.filter(pl.col("date") <= reb_date)
        factored = compute_all_factors(available_prices)
        scored = composite_score(factored)

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
        weights = portfolio["weight"].to_list()

        holdings = {}
        cash = 0.0
        for ticker, weight in zip(selected_tickers, weights):
            allocation = portfolio_value * weight
            price_row = prices.filter(
                (pl.col("ticker") == ticker) & (pl.col("date") == reb_date)
            )
            if not price_row.is_empty():
                price = price_row["close"][0]
                shares = allocation / price
                holdings[ticker] = shares
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
