"""Track paper trading performance vs SPY and backtest expectations.

Pulls portfolio history from Alpaca and compares against SPY.
Saves equity curve chart to cache/paper_performance.png.

Usage:
    python run_track.py              # show current performance
    python run_track.py --since 30   # last 30 days (default: all history)
"""

import sys
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from config.settings import CACHE_DIR


def get_clients():
    """Create Alpaca trading + data clients."""
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")

    if not api_key or not secret_key:
        print("ERROR: Missing Alpaca credentials in .env")
        sys.exit(1)

    trading = TradingClient(api_key, secret_key, paper=True)
    data = StockHistoricalDataClient(api_key, secret_key)
    return trading, data


def get_portfolio_history(trading_client):
    """Get daily portfolio value history from Alpaca."""
    # Use the REST API directly since alpaca-py's portfolio history
    # access varies by version
    history = trading_client.get(
        "/account/portfolio/history",
        {"period": "all", "timeframe": "1D", "intraday_reporting": "market_hours"},
    )

    timestamps = history.get("timestamp", [])
    equity = history.get("equity", [])
    profit_loss = history.get("profit_loss", [])
    profit_loss_pct = history.get("profit_loss_pct", [])

    if not timestamps:
        return None

    records = []
    for i, ts in enumerate(timestamps):
        dt = datetime.fromtimestamp(ts)
        records.append({
            "date": dt.date(),
            "equity": equity[i] if i < len(equity) else None,
            "profit_loss": profit_loss[i] if i < len(profit_loss) else None,
            "profit_loss_pct": profit_loss_pct[i] if i < len(profit_loss_pct) else None,
        })

    return records


def get_spy_returns(data_client, start_date, end_date):
    """Get SPY daily close prices for comparison."""
    request = StockBarsRequest(
        symbol_or_symbols="SPY",
        timeframe=TimeFrame.Day,
        start=datetime.combine(start_date, datetime.min.time()),
        end=datetime.combine(end_date, datetime.max.time()),
    )
    bars = data_client.get_stock_bars(request)
    spy_bars = bars["SPY"]

    records = []
    for bar in spy_bars:
        records.append({
            "date": bar.timestamp.date(),
            "close": bar.close,
        })
    return records


def print_performance(history, spy_data):
    """Print performance summary."""
    if not history or len(history) < 2:
        print("Not enough history yet. Check back after a few trading days.")
        return

    # Filter out None equity values
    valid = [r for r in history if r["equity"] is not None and r["equity"] > 0]
    if len(valid) < 2:
        print("Not enough valid data points yet.")
        return

    start_equity = valid[0]["equity"]
    end_equity = valid[-1]["equity"]
    start_date = valid[0]["date"]
    end_date = valid[-1]["date"]
    days = (end_date - start_date).days

    total_return = (end_equity / start_equity - 1) * 100
    if days > 0:
        annualized = ((end_equity / start_equity) ** (365.25 / days) - 1) * 100
    else:
        annualized = 0

    print(f"\n{'=' * 50}")
    print(f"PAPER TRADING PERFORMANCE")
    print(f"{'=' * 50}")
    print(f"  Period:         {start_date} to {end_date} ({days} days)")
    print(f"  Start value:    ${start_equity:>12,.2f}")
    print(f"  Current value:  ${end_equity:>12,.2f}")
    print(f"  Total return:   {total_return:>+10.2f}%")
    if days >= 30:
        print(f"  Annualized:     {annualized:>+10.2f}%")

    # SPY comparison
    if spy_data and len(spy_data) >= 2:
        spy_start = spy_data[0]["close"]
        spy_end = spy_data[-1]["close"]
        spy_return = (spy_end / spy_start - 1) * 100
        spy_annual = ((spy_end / spy_start) ** (365.25 / days) - 1) * 100 if days > 0 else 0

        print(f"\n  --- SPY (same period) ---")
        print(f"  Total return:   {spy_return:>+10.2f}%")
        if days >= 30:
            print(f"  Annualized:     {spy_annual:>+10.2f}%")
        print(f"  Alpha:          {total_return - spy_return:>+10.2f}% (total)")

    # Current positions
    positions = trading_client.get_all_positions()
    if positions:
        print(f"\n  --- Current Holdings ({len(positions)} positions) ---")
        total_pnl = 0
        for pos in sorted(positions, key=lambda p: p.symbol):
            pnl = float(pos.unrealized_pl)
            pnl_pct = float(pos.unrealized_plpc) * 100
            total_pnl += pnl
            marker = "+" if pnl >= 0 else ""
            print(f"    {pos.symbol:6s}  {float(pos.qty):>7.1f} sh  "
                  f"${float(pos.market_value):>10,.0f}  "
                  f"{marker}{pnl_pct:.1f}%  ({marker}${pnl:,.0f})")
        print(f"    {'TOTAL':6s}  {'':>7s}     "
              f"${sum(float(p.market_value) for p in positions):>10,.0f}  "
              f"{'+'if total_pnl >= 0 else ''}${total_pnl:,.0f}")


def plot_performance(history, spy_data):
    """Plot equity curve vs SPY."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = [r for r in history if r["equity"] is not None and r["equity"] > 0]
    if len(valid) < 2:
        return

    dates = [r["date"] for r in valid]
    equity = [r["equity"] for r in valid]
    start_equity = equity[0]

    # Normalize to percentage return
    returns = [(e / start_equity - 1) * 100 for e in equity]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates, returns, label="Paper Portfolio", linewidth=2)

    if spy_data and len(spy_data) >= 2:
        spy_start = spy_data[0]["close"]
        spy_dates = [r["date"] for r in spy_data]
        spy_returns = [(r["close"] / spy_start - 1) * 100 for r in spy_data]
        ax.plot(spy_dates, spy_returns, label="SPY", linewidth=1.5, alpha=0.7)

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Return (%)")
    ax.set_title("Paper Trading Performance vs SPY")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = CACHE_DIR / "paper_performance.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Chart saved to {out_path}")


def main():
    global trading_client

    trading_client, data_client = get_clients()

    account = trading_client.get_account()
    print(f"=== Alpha-Machine: Paper Trading Tracker ===\n")
    print(f"  Account: {account.account_number} (paper)")
    print(f"  Status:  {account.status}")
    print(f"  Equity:  ${float(account.portfolio_value):,.2f}")
    print(f"  Cash:    ${float(account.cash):,.2f}")

    # Get portfolio history
    history = get_portfolio_history(trading_client)

    if not history:
        print("\n  No portfolio history yet. Execute a rebalance first with:")
        print("    python run_trade.py --execute")
        return

    # Get SPY data for same period
    start_date = history[0]["date"]
    end_date = history[-1]["date"]
    spy_data = get_spy_returns(data_client, start_date, end_date)

    print_performance(history, spy_data)
    plot_performance(history, spy_data)


if __name__ == "__main__":
    main()
