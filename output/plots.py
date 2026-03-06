import polars as pl
import numpy as np
from config.settings import CACHE_DIR


def plot_equity_curve(equity_curve: pl.DataFrame, benchmark: pl.DataFrame | None = None):
    """Plot equity curve and optionally a benchmark. Saves to cache/equity_curve.png."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plots")
        return

    ec = equity_curve.sort("date")
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(ec["date"].to_list(), ec["value"].to_list(), label="Strategy", linewidth=1.5)

    if benchmark is not None and not benchmark.is_empty():
        bench = benchmark.sort("date")
        initial = ec["value"][0]
        bench_vals = bench["close"].to_numpy()
        bench_normalized = bench_vals / bench_vals[0] * initial
        ax.plot(bench["date"].to_list(), bench_normalized, label="SPY", linewidth=1.0, alpha=0.7)

    if "is_oos" in ec.columns:
        oos = ec.filter(pl.col("is_oos"))
        if not oos.is_empty():
            oos_start = oos["date"].min()
            ax.axvline(x=oos_start, color="red", linestyle="--", alpha=0.5, label="OOS Start")

    ax.set_title("Equity Curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = CACHE_DIR / "equity_curve.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Equity curve saved to {path}")


def plot_monthly_heatmap(equity_curve: pl.DataFrame):
    """Plot monthly returns heatmap. Saves to cache/monthly_returns.png."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plots")
        return

    ec = equity_curve.sort("date").with_columns([
        pl.col("date").dt.year().alias("year"),
        pl.col("date").dt.month().alias("month"),
    ])

    monthly = ec.group_by(["year", "month"]).agg([
        pl.col("value").first().alias("start"),
        pl.col("value").last().alias("end"),
    ]).with_columns(
        ((pl.col("end") / pl.col("start") - 1) * 100).alias("return_pct")
    ).sort(["year", "month"])

    years = sorted(monthly["year"].unique().to_list())
    months = list(range(1, 13))
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    data = np.full((len(years), 12), np.nan)
    for row in monthly.to_dicts():
        yi = years.index(row["year"])
        mi = row["month"] - 1
        data[yi, mi] = row["return_pct"]

    fig, ax = plt.subplots(figsize=(14, max(4, len(years) * 0.6)))
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=-10, vmax=10)

    ax.set_xticks(range(12))
    ax.set_xticklabels(month_names)
    ax.set_yticks(range(len(years)))
    ax.set_yticklabels(years)

    for i in range(len(years)):
        for j in range(12):
            val = data[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=7)

    plt.colorbar(im, label="Monthly Return (%)")
    ax.set_title("Monthly Returns Heatmap")
    plt.tight_layout()

    path = CACHE_DIR / "monthly_returns.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Monthly returns heatmap saved to {path}")
