import polars as pl
from datetime import datetime
from portfolio.risk import check_sector_exposure, concentration_risk
from scoring.combine import ALL_FACTORS


def generate_report(
    portfolio: pl.DataFrame,
    scored: pl.DataFrame,
    sector_map: dict[str, str],
    company_map: dict[str, str] | None = None,
    backtest_metrics: dict | None = None,
    previous_holdings: list[str] | None = None,
) -> str:
    """Generate a markdown recommendation report."""
    if company_map is None:
        company_map = {}

    lines = []
    lines.append(f"# Alpha-Machine Stock Recommendations")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    lines.append("## Top Picks\n")
    lines.append("| Rank | Ticker | Company | Sector | Score | Weight |")
    lines.append("|------|--------|---------|--------|-------|--------|")

    for i, row in enumerate(portfolio.to_dicts(), 1):
        ticker = row["ticker"]
        company = company_map.get(ticker, "")
        sector = row.get("sector", sector_map.get(ticker, "N/A"))
        score = row["composite_score"]
        weight = row["weight"]
        lines.append(f"| {i} | {ticker} | {company} | {sector} | {score:.3f} | {weight:.1%} |")

    lines.append("")

    if previous_holdings:
        current = set(portfolio["ticker"].to_list())
        prev = set(previous_holdings)
        added = current - prev
        removed = prev - current
        if added or removed:
            lines.append("## Changes from Last Rebalance\n")
            if added:
                lines.append(f"**Added:** {', '.join(sorted(added))}\n")
            if removed:
                lines.append(f"**Removed:** {', '.join(sorted(removed))}\n")

    lines.append("## Sector Exposure\n")
    exposure = check_sector_exposure(portfolio)
    lines.append("| Sector | Weight |")
    lines.append("|--------|--------|")
    for sector, weight in sorted(exposure["exposures"].items(), key=lambda x: -x[1]):
        flag = " ⚠️" if sector in exposure["breaches"] else ""
        lines.append(f"| {sector} | {weight:.1%}{flag} |")

    if exposure["breaches"]:
        lines.append(f"\n⚠️ **Sector cap breached:** {', '.join(exposure['breaches'])}\n")

    conc = concentration_risk(portfolio)
    lines.append(f"\n**Positions:** {conc['n_positions']} | "
                 f"**Max weight:** {conc['max_weight']:.1%} | "
                 f"**HHI:** {conc['hhi']:.4f}\n")

    lines.append("## Factor Breakdown (Top 5)\n")
    top5 = portfolio.head(5)
    factor_names = [f.name() for f in ALL_FACTORS]
    rank_cols = [f"{n}_rank" for n in factor_names]
    available_cols = [c for c in rank_cols if c in top5.columns]

    if available_cols:
        header = "| Ticker | " + " | ".join(c.replace("_rank", "") for c in available_cols) + " |"
        sep = "|--------" + "|------" * len(available_cols) + "|"
        lines.append(header)
        lines.append(sep)
        for row in top5.to_dicts():
            vals = " | ".join(f"{row.get(c, 0):.2f}" for c in available_cols)
            lines.append(f"| {row['ticker']} | {vals} |")

    if backtest_metrics:
        lines.append("\n## Historical Backtest\n")
        for key, val in backtest_metrics.items():
            label = key.replace("_", " ").title()
            lines.append(f"- **{label}:** {val}")

    lines.append("\n---")
    lines.append("*This is not financial advice. Past performance does not guarantee future results.*")

    return "\n".join(lines)
