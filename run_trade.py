"""Execute quarterly rebalance on Alpaca paper trading account.

Scores the Russell 1000, compares to current Alpaca holdings, and
submits orders to rebalance to the target 20-stock equal-weight portfolio.

Usage:
    python run_trade.py              # dry run (show orders, don't execute)
    python run_trade.py --execute    # actually submit orders

Requires .env file with:
    ALPACA_API_KEY=your_key
    ALPACA_SECRET_KEY=your_secret
"""

import sys
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from data.universe import get_universe
from data.prices import download_prices
from scoring.missing import filter_insufficient_history
from scoring.combine import score_universe
from config.settings import TOP_N, SELL_THRESHOLD_RANK, CACHE_DIR


def get_alpaca_client():
    """Create Alpaca trading client from .env credentials."""
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")

    if not api_key or not secret_key:
        print("ERROR: Missing Alpaca credentials.")
        print("Create a .env file in the project root with:")
        print("  ALPACA_API_KEY=your_key")
        print("  ALPACA_SECRET_KEY=your_secret")
        sys.exit(1)

    # paper=True for paper trading
    return TradingClient(api_key, secret_key, paper=True)


def get_current_holdings(client):
    """Get current positions from Alpaca. Returns {ticker: {shares, market_value}}."""
    positions = client.get_all_positions()
    holdings = {}
    for pos in positions:
        holdings[pos.symbol] = {
            "shares": float(pos.qty),
            "market_value": float(pos.market_value),
            "current_price": float(pos.current_price),
        }
    return holdings


def get_account_value(client):
    """Get total portfolio value (cash + positions)."""
    account = client.get_account()
    return float(account.portfolio_value), float(account.cash)


def compute_target_portfolio():
    """Run the scoring pipeline and return target tickers + weights."""
    print("1. Scoring universe...")
    universe = get_universe()
    tickers = universe["ticker"].to_list()
    prices = download_prices(tickers)
    prices = filter_insufficient_history(prices)
    print(f"   {prices['ticker'].n_unique()} tickers scored\n")

    scored = score_universe(prices)

    # Apply turnover dampening: we don't have previous holdings from the model,
    # so just take top N. In practice, the sell_threshold_rank dampening happens
    # through the Alpaca position diff — we only sell what needs selling.
    top = scored.head(TOP_N)
    target_tickers = top["ticker"].to_list()

    weight = 1.0 / len(target_tickers)
    return {t: weight for t in target_tickers}


def compute_orders(target_weights, current_holdings, portfolio_value):
    """Compute buy/sell orders to rebalance from current to target.

    Returns list of {ticker, side, shares, dollar_amount, reason}.
    """
    orders = []
    target_tickers = set(target_weights.keys())
    current_tickers = set(current_holdings.keys())

    # Sells: positions not in target
    for ticker in current_tickers - target_tickers:
        pos = current_holdings[ticker]
        orders.append({
            "ticker": ticker,
            "side": "sell",
            "shares": pos["shares"],
            "dollar_amount": pos["market_value"],
            "reason": "not in target portfolio",
        })

    # Buys and rebalances for target positions
    for ticker, weight in target_weights.items():
        target_value = portfolio_value * weight
        current_value = current_holdings.get(ticker, {}).get("market_value", 0)
        current_price = current_holdings.get(ticker, {}).get("current_price", None)

        diff = target_value - current_value
        band = target_value * 0.05  # 5% rebalance band

        if abs(diff) < band:
            continue  # close enough, skip

        if diff > 0:
            orders.append({
                "ticker": ticker,
                "side": "buy",
                "notional": round(diff, 2),
                "reason": "new position" if ticker not in current_tickers else "underweight",
            })
        else:
            # Trim overweight position
            if current_price and current_price > 0:
                trim_shares = round(abs(diff) / current_price, 4)
                if trim_shares > 0.0001:
                    orders.append({
                        "ticker": ticker,
                        "side": "sell",
                        "shares": trim_shares,
                        "dollar_amount": abs(diff),
                        "reason": "overweight",
                    })

    return orders


def execute_orders(client, orders, dry_run=True):
    """Submit orders to Alpaca. If dry_run=True, just print them."""
    if not orders:
        print("   No orders needed — portfolio is on target.\n")
        return

    sells = [o for o in orders if o["side"] == "sell"]
    buys = [o for o in orders if o["side"] == "buy"]

    # Execute sells first to free up cash
    for group_label, group in [("SELLS", sells), ("BUYS", buys)]:
        if not group:
            continue
        print(f"\n   --- {group_label} ---")
        for o in group:
            if o["side"] == "sell":
                label = f"  SELL {o['shares']:.2f} shares of {o['ticker']} (~${o['dollar_amount']:,.0f}) — {o['reason']}"
            else:
                label = f"  BUY ${o['notional']:,.0f} of {o['ticker']} — {o['reason']}"

            if dry_run:
                print(f"  [DRY RUN] {label}")
            else:
                print(f"  {label}")
                try:
                    if o["side"] == "sell":
                        req = MarketOrderRequest(
                            symbol=o["ticker"],
                            qty=o["shares"],
                            side=OrderSide.SELL,
                            time_in_force=TimeInForce.DAY,
                        )
                    else:
                        req = MarketOrderRequest(
                            symbol=o["ticker"],
                            notional=o["notional"],
                            side=OrderSide.BUY,
                            time_in_force=TimeInForce.DAY,
                        )
                    order = client.submit_order(req)
                    print(f"    → Order submitted: {order.id} ({order.status})")
                except Exception as e:
                    print(f"    → ERROR: {e}")


def log_rebalance(target_weights, orders, portfolio_value, dry_run):
    """Save rebalance log to cache/."""
    log_dir = CACHE_DIR / "trade_logs"
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    mode = "dry_run" if dry_run else "executed"
    log_file = log_dir / f"rebalance_{timestamp}_{mode}.md"

    lines = [
        f"# Rebalance Log — {timestamp}",
        f"Mode: {'DRY RUN' if dry_run else 'EXECUTED'}",
        f"Portfolio value: ${portfolio_value:,.0f}",
        f"Target positions: {len(target_weights)}",
        "",
        "## Target Portfolio",
    ]
    for ticker, weight in sorted(target_weights.items()):
        lines.append(f"- {ticker}: {weight:.1%} (${portfolio_value * weight:,.0f})")

    lines.append("")
    lines.append("## Orders")
    for o in orders:
        if o["side"] == "sell":
            lines.append(f"- SELL {o['shares']:.2f} {o['ticker']} (~${o['dollar_amount']:,.0f}) — {o['reason']}")
        else:
            lines.append(f"- BUY ${o['notional']:,.0f} {o['ticker']} — {o['reason']}")

    with open(log_file, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n   Log saved to {log_file}")


def main():
    dry_run = "--execute" not in sys.argv
    if dry_run:
        print("=== Alpha-Machine: Paper Trade Rebalance (DRY RUN) ===")
        print("    Add --execute to actually submit orders.\n")
    else:
        print("=== Alpha-Machine: Paper Trade Rebalance (LIVE) ===\n")

    # Connect to Alpaca
    client = get_alpaca_client()
    portfolio_value, cash = get_account_value(client)
    print(f"   Account value: ${portfolio_value:,.0f} (${cash:,.0f} cash)\n")

    # Get current holdings
    current = get_current_holdings(client)
    if current:
        print(f"   Current positions ({len(current)}):")
        for ticker, pos in sorted(current.items()):
            print(f"     {ticker:6s}  {pos['shares']:>8.2f} shares  ${pos['market_value']:>10,.0f}")
    else:
        print("   No current positions (fresh account).")
    print()

    # Compute target
    target_weights = compute_target_portfolio()
    target_tickers = sorted(target_weights.keys())
    print(f"2. Target portfolio ({len(target_tickers)} stocks, equal weight):")
    for t in target_tickers:
        print(f"     {t:6s}  {target_weights[t]:.1%}  ${portfolio_value * target_weights[t]:>10,.0f}")
    print()

    # Compute and show diff
    current_set = set(current.keys())
    target_set = set(target_tickers)
    new = target_set - current_set
    removed = current_set - target_set
    kept = target_set & current_set

    print(f"3. Portfolio diff:")
    print(f"     Keep:   {len(kept)} positions")
    if new:
        print(f"     Add:    {', '.join(sorted(new))}")
    if removed:
        print(f"     Remove: {', '.join(sorted(removed))}")
    print()

    # Compute orders
    print(f"4. Orders:")
    orders = compute_orders(target_weights, current, portfolio_value)
    execute_orders(client, orders, dry_run=dry_run)

    # Log
    log_rebalance(target_weights, orders, portfolio_value, dry_run)

    if dry_run:
        print("\n   This was a dry run. Run with --execute to submit orders.")


if __name__ == "__main__":
    main()
