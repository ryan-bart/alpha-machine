"""Score the S&P 500 universe and produce a recommendation report."""

from data.universe import get_sp500_tickers
from data.prices import download_prices
from data.macro import get_risk_free_rate
from scoring.combine import score_universe
from scoring.missing import filter_insufficient_history
from portfolio.construct import select_top_n
from portfolio.risk import check_sector_exposure
from output.report import generate_report


def main():
    print("=== Alpha-Machine: Multi-Factor Stock Scoring ===\n")

    print("1. Fetching S&P 500 universe...")
    universe = get_sp500_tickers()
    tickers = universe["ticker"].to_list()
    sector_map = dict(zip(universe["ticker"].to_list(), universe["sector"].to_list()))
    company_map = dict(zip(universe["ticker"].to_list(), universe["company"].to_list()))
    print(f"   {len(tickers)} tickers loaded\n")

    print("2. Downloading price data...")
    prices = download_prices(tickers)
    prices = filter_insufficient_history(prices)
    print(f"   {prices['ticker'].n_unique()} tickers with sufficient history\n")

    print("3. Computing factors and scoring...")
    scored = score_universe(prices)
    print(f"   Scored {len(scored)} stocks\n")

    print("4. Constructing portfolio...")
    portfolio = select_top_n(scored, sector_map)
    print(f"   Selected {len(portfolio)} positions\n")

    print("5. Generating report...")
    report = generate_report(
        portfolio=portfolio,
        scored=scored,
        sector_map=sector_map,
        company_map=company_map,
    )

    report_path = "cache/recommendation_report.md"
    with open(report_path, "w") as f:
        f.write(report)

    print(report)
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
