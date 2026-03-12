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
RUSSELL1000_WIKI_URL = "https://en.wikipedia.org/wiki/Russell_1000_Index"
UNIVERSE_SOURCE = "russell1000"  # "sp500" or "russell1000"

# -- Prices --
PRICE_HISTORY_YEARS = 10
MIN_TRADING_DAYS = 200

# -- Factor weights (sum to 1.0) --
# Tuned for quarterly rebalancing: favor persistent signals (momentum, low-vol,
# price consistency) over fast-decaying ones (reversal, volume spikes).
FACTOR_WEIGHTS = {
    "momentum_12_1": 0.25,       # strongest evidence at 3-12mo horizon
    "momentum_6_1": 0.15,        # intermediate momentum
    "rel_strength_3mo": 0.10,    # most recent momentum signal at quarterly freq
    "short_term_reversal": 0.00, # decays in days, useless at quarterly
    "dist_from_ma50": 0.05,      # mild overextension filter
    "volume_trend": 0.05,        # decays over 3 months, reduce
    "obv_slope": 0.05,           # cumulative volume flow, somewhat persistent
    "realized_vol_60d": 0.15,    # low-vol anomaly, very persistent
    "vol_trend": 0.05,           # declining vol signal
    "price_consistency": 0.15,   # steady climbers compound better over quarters
}

# -- Portfolio --
TOP_N = 20
WEIGHTING = "equal"  # "equal" or "score"
SECTOR_CAP = 1.0  # disabled
MIN_COMPOSITE_PERCENTILE = 0.50
SELL_THRESHOLD_RANK = 150
REBALANCE_FREQ = "QS"  # quarter start

# -- Backtest --
HOLDOUT_QUARTERS = 8  # 8 quarters = 2 year holdout
INITIAL_CASH = 100_000

# -- FRED --
FRED_TB3MS_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=TB3MS"
