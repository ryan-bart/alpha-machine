import polars as pl
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from config.settings import (
    INITIAL_CASH, HOLDOUT_QUARTERS, TOP_N, REBALANCE_FREQ, SELL_THRESHOLD_RANK,
    TRANSACTION_COST_BPS, SHORT_TERM_TAX_RATE, LONG_TERM_TAX_RATE,
    REBALANCE_BAND, TAX_PROTECTION_DAYS,
)
from scoring.combine import compute_all_factors, composite_score
from portfolio.construct import select_top_n


@dataclass
class Position:
    shares: float
    cost_basis: float
    purchase_date: object  # datetime.date


BENCHMARK_TICKERS = {"SPY", "RSP"}


def run_backtest(
    prices: pl.DataFrame,
    sector_map: dict[str, str],
    holdout_periods: int = HOLDOUT_QUARTERS,
    weights: dict | None = None,
    freq: str = REBALANCE_FREQ,
    sell_threshold_rank: int = SELL_THRESHOLD_RANK,
) -> dict:
    """Walk-forward backtest with configurable rebalance frequency.

    Returns dict with equity_curve, rebalance_log, is_dates, oos_dates.
    """
    scoring_prices = prices.filter(~pl.col("ticker").is_in(BENCHMARK_TICKERS))
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
        available_prices = scoring_prices.filter(pl.col("date") <= reb_date)
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

        portfolio = select_top_n(snapshot, sector_map, previous_tickers,
                                sell_threshold_rank=sell_threshold_rank)

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


def _get_single_price(prices, ticker, date):
    """Get closing price for a single ticker on a date."""
    row = prices.filter(
        (pl.col("ticker") == ticker) & (pl.col("date") == date)
    )
    if not row.is_empty():
        return row["close"][0]
    return None


def _get_positions_value(positions, prices, date):
    """Sum of all positions' market value at given date."""
    total = 0.0
    for ticker, pos in positions.items():
        price = _get_single_price(prices, ticker, date)
        if price is not None:
            total += pos.shares * price
    return total


def precompute_snapshots(
    prices: pl.DataFrame,
    weights: dict | None = None,
    freq: str = REBALANCE_FREQ,
    holdout_periods: int = HOLDOUT_QUARTERS,
) -> dict:
    """Pre-compute scored snapshots for all rebalance dates (expensive step).

    Returns dict with rebalance_dates, snapshots, all_dates, is_dates, oos_dates.
    """
    scoring_prices = prices.filter(~pl.col("ticker").is_in(BENCHMARK_TICKERS))
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

    all_rebalance_dates = is_dates + oos_dates

    snapshots = {}
    for idx, reb_date in enumerate(all_rebalance_dates):
        print(f"   Computing factors for period {idx + 1}/{len(all_rebalance_dates)}...", end="\r")
        available_prices = scoring_prices.filter(pl.col("date") <= reb_date)
        factored = compute_all_factors(available_prices)
        scored = composite_score(factored, weights=weights)

        snapshot = scored.filter(pl.col("date") == reb_date)
        if snapshot.is_empty():
            closest = scored.filter(pl.col("date") <= reb_date).sort("date", descending=True)
            if not closest.is_empty():
                latest_date = closest["date"][0]
                snapshot = scored.filter(pl.col("date") == latest_date)

        if not snapshot.is_empty():
            snapshot = snapshot.sort("composite_score", descending=True)

        snapshots[reb_date] = snapshot

    print()
    return {
        "rebalance_dates": rebalance_dates,
        "snapshots": snapshots,
        "all_dates": dates,
        "is_dates": is_dates,
        "oos_dates": oos_dates,
    }


def run_backtest_from_snapshots(
    precomputed: dict,
    prices: pl.DataFrame,
    sector_map: dict[str, str],
    sell_threshold_rank: int = SELL_THRESHOLD_RANK,
    apply_costs: bool = False,
    rebalance_band: float | None = None,
    tax_protection_days: int | None = None,
    score_tilt: float = 0.0,
) -> dict:
    """Run backtest using pre-computed snapshots (fast, for parameter sweeps).

    When apply_costs=True, uses delta-based trading with transaction costs
    (modeled as slippage) and capital gains taxes.
    """
    is_dates = precomputed["is_dates"]
    oos_dates = precomputed["oos_dates"]
    snapshots = precomputed["snapshots"]
    all_dates = precomputed["all_dates"]

    tc_rate = TRANSACTION_COST_BPS / 10000 if apply_costs else 0.0
    if rebalance_band is None:
        rebalance_band = REBALANCE_BAND if apply_costs else 0.01
    if tax_protection_days is None:
        tax_protection_days = TAX_PROTECTION_DAYS if apply_costs else 0

    cash = float(INITIAL_CASH)
    positions: dict[str, Position] = {}
    previous_tickers: list[str] = []

    equity_curve = []
    rebalance_log = []

    # Cost/tax tracking
    tax_loss_carry = 0.0
    cumulative_costs = 0.0
    cumulative_taxes = 0.0
    cumulative_st_gains = 0.0
    cumulative_lt_gains = 0.0

    all_rebalance_dates = is_dates + oos_dates

    for i, reb_date in enumerate(all_rebalance_dates):
        snapshot = snapshots.get(reb_date)
        if snapshot is None or snapshot.is_empty():
            continue

        portfolio = select_top_n(
            snapshot, sector_map, previous_tickers,
            sell_threshold_rank=sell_threshold_rank,
            score_tilt=score_tilt,
        )

        selected_tickers = portfolio["ticker"].to_list()
        port_weights = portfolio["weight"].to_list()
        target_set = set(selected_tickers)

        # Track turnover
        prev_set = set(previous_tickers)
        if prev_set:
            turnover = len(target_set - prev_set) / max(len(target_set), 1)
        else:
            turnover = 1.0

        period_costs = 0.0
        period_taxes = 0.0

        # --- Phase 1: Sell removed positions ---
        for ticker in list(positions.keys()):
            if ticker not in target_set:
                price = _get_single_price(prices, ticker, reb_date)
                if price is None:
                    continue
                pos = positions[ticker]

                # Tax protection: defer selling gains close to 1-year LT threshold
                if apply_costs and tax_protection_days > 0:
                    holding_days = (reb_date - pos.purchase_date).days
                    days_to_lt = 365 - holding_days
                    gain_check = pos.shares * price - pos.cost_basis
                    if 0 < days_to_lt <= tax_protection_days and gain_check > 0:
                        continue  # hold for LT treatment, sell next rebalance

                positions.pop(ticker)
                sell_price = price * (1 - tc_rate)
                proceeds = pos.shares * sell_price
                gain = proceeds - pos.cost_basis

                period_costs += pos.shares * price * tc_rate

                if apply_costs and gain > 0:
                    taxable = max(0, gain - tax_loss_carry)
                    tax_loss_carry = max(0, tax_loss_carry - gain)
                    holding_days = (reb_date - pos.purchase_date).days
                    if holding_days >= 365:
                        tax = taxable * LONG_TERM_TAX_RATE
                        cumulative_lt_gains += gain
                    else:
                        tax = taxable * SHORT_TERM_TAX_RATE
                        cumulative_st_gains += gain
                    proceeds -= tax
                    period_taxes += tax
                elif apply_costs and gain < 0:
                    tax_loss_carry += abs(gain)

                cash += proceeds

        # Recalculate portfolio value after sells for accurate targeting
        # Subtract value of tax-protected holds (not in target, but still held)
        portfolio_value = _get_positions_value(positions, prices, reb_date) + cash
        protected_value = sum(
            pos.shares * (_get_single_price(prices, t, reb_date) or 0)
            for t, pos in positions.items() if t not in target_set
        )
        available_capital = portfolio_value - protected_value
        targets = {t: available_capital * w for t, w in zip(selected_tickers, port_weights)}

        # --- Phase 2: Trim overweight retained positions ---
        band = rebalance_band
        for ticker in list(positions.keys()):
            if ticker not in targets:
                continue
            pos = positions[ticker]
            price = _get_single_price(prices, ticker, reb_date)
            if price is None:
                continue

            current_val = pos.shares * price
            target_val = targets[ticker]

            if current_val > target_val * (1 + band):
                # Tax protection: skip trim if close to LT threshold and has gain
                if apply_costs and tax_protection_days > 0:
                    holding_days = (reb_date - pos.purchase_date).days
                    days_to_lt = 365 - holding_days
                    if 0 < days_to_lt <= tax_protection_days:
                        cost_per_share = pos.cost_basis / pos.shares
                        trim_shares = (current_val - target_val) / price
                        if trim_shares * price > trim_shares * cost_per_share:
                            continue  # skip trim, gain would be short-term

                sell_shares = (current_val - target_val) / price
                sell_price = price * (1 - tc_rate)
                proceeds = sell_shares * sell_price

                cost_per_share = pos.cost_basis / pos.shares
                partial_basis = sell_shares * cost_per_share
                gain = proceeds - partial_basis

                period_costs += sell_shares * price * tc_rate

                if apply_costs and gain > 0:
                    taxable = max(0, gain - tax_loss_carry)
                    tax_loss_carry = max(0, tax_loss_carry - gain)
                    holding_days = (reb_date - pos.purchase_date).days
                    if holding_days >= 365:
                        tax = taxable * LONG_TERM_TAX_RATE
                        cumulative_lt_gains += gain
                    else:
                        tax = taxable * SHORT_TERM_TAX_RATE
                        cumulative_st_gains += gain
                    proceeds -= tax
                    period_taxes += tax
                elif apply_costs and gain < 0:
                    tax_loss_carry += abs(gain)

                cash += proceeds
                pos.shares -= sell_shares
                pos.cost_basis -= partial_basis

        # --- Phase 3: Buy new positions and add to underweight ---
        for ticker in selected_tickers:
            price = _get_single_price(prices, ticker, reb_date)
            if price is None:
                continue

            target_val = targets[ticker]

            if ticker in positions:
                current_val = positions[ticker].shares * price
                if current_val < target_val * (1 - band):
                    buy_val = min(target_val - current_val, max(cash, 0))
                    if buy_val < 1:
                        continue
                    buy_price = price * (1 + tc_rate)
                    new_shares = buy_val / buy_price
                    period_costs += buy_val * tc_rate / (1 + tc_rate)
                    positions[ticker].shares += new_shares
                    positions[ticker].cost_basis += new_shares * buy_price
                    cash -= buy_val
            else:
                buy_val = min(target_val, max(cash, 0))
                if buy_val < 1:
                    continue
                buy_price = price * (1 + tc_rate)
                new_shares = buy_val / buy_price
                period_costs += buy_val * tc_rate / (1 + tc_rate)
                positions[ticker] = Position(
                    shares=new_shares,
                    cost_basis=new_shares * buy_price,
                    purchase_date=reb_date,
                )
                cash -= buy_val

        cumulative_costs += period_costs
        cumulative_taxes += period_taxes

        # Include all held positions (target + tax-protected) for dampening
        previous_tickers = list(positions.keys())
        is_oos = reb_date in oos_dates

        rebalance_log.append({
            "date": reb_date,
            "tickers": selected_tickers,
            "portfolio_value": portfolio_value,
            "is_oos": is_oos,
            "turnover": turnover,
        })

        next_reb = all_rebalance_dates[i + 1] if i + 1 < len(all_rebalance_dates) else all_dates[-1]
        period_dates = [d for d in all_dates if reb_date <= d <= next_reb]

        for d in period_dates:
            val = _get_positions_value(positions, prices, d) + cash
            equity_curve.append({"date": d, "value": val, "is_oos": is_oos})

    result = {
        "equity_curve": pl.DataFrame(equity_curve),
        "rebalance_log": rebalance_log,
        "is_dates": is_dates,
        "oos_dates": oos_dates,
    }

    if apply_costs:
        result["cost_summary"] = {
            "total_transaction_costs": cumulative_costs,
            "total_taxes": cumulative_taxes,
            "short_term_gains_taxed": cumulative_st_gains,
            "long_term_gains_taxed": cumulative_lt_gains,
            "tax_loss_carry": tax_loss_carry,
        }

    return result
