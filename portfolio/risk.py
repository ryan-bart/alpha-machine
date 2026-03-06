import polars as pl


def check_sector_exposure(portfolio: pl.DataFrame, cap: float = 0.25) -> dict:
    """Return sector exposure and any breaches."""
    if portfolio.is_empty():
        return {"exposures": {}, "breaches": []}

    exposure = (
        portfolio.group_by("sector")
        .agg(pl.col("weight").sum().alias("total_weight"))
        .sort("total_weight", descending=True)
    )

    exposures = dict(zip(
        exposure["sector"].to_list(),
        exposure["total_weight"].to_list(),
    ))

    breaches = [s for s, w in exposures.items() if w > cap]
    return {"exposures": exposures, "breaches": breaches}


def concentration_risk(portfolio: pl.DataFrame) -> dict:
    """Basic concentration metrics."""
    if portfolio.is_empty():
        return {"n_positions": 0, "max_weight": 0, "hhi": 0}

    weights = portfolio["weight"].to_list()
    return {
        "n_positions": len(weights),
        "max_weight": max(weights),
        "hhi": sum(w ** 2 for w in weights),
    }
