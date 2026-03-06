from pathlib import Path

# -- Paths --
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "cache"
CACHE_DIR.mkdir(exist_ok=True)

UNIVERSE_CACHE = CACHE_DIR / "universe.parquet"
PRICES_CACHE = CACHE_DIR / "prices.parquet"
SECTORS_CACHE = CACHE_DIR / "sectors.parquet"

# -- Universe --
UNIVERSE_REFRESH_DAYS = 7
SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# -- Prices --
PRICE_HISTORY_YEARS = 3
MIN_TRADING_DAYS = 200

# -- Factor weights (sum to 1.0) --
FACTOR_WEIGHTS = {
    "momentum_12_1": 0.20,
    "momentum_6_1": 0.15,
    "rel_strength_3mo": 0.10,
    "short_term_reversal": 0.05,
    "dist_from_ma50": 0.05,
    "volume_trend": 0.10,
    "obv_slope": 0.05,
    "realized_vol_60d": 0.10,
    "vol_trend": 0.05,
    "price_consistency": 0.15,
}

# -- Portfolio --
TOP_N = 20
EQUAL_WEIGHT = 1.0 / TOP_N
SECTOR_CAP = 0.25
MIN_COMPOSITE_PERCENTILE = 0.50
SELL_THRESHOLD_RANK = 30
REBALANCE_FREQ = "MS"  # month start

# -- Backtest --
HOLDOUT_MONTHS = 12
INITIAL_CASH = 100_000

# -- FRED --
FRED_TB3MS_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=TB3MS"
