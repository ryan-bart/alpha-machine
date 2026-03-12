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
SCORE_TILT = 0.0     # 0 = equal weight, 1 = proportional to score, >1 = more concentrated
SECTOR_CAP = 1.0  # disabled
MIN_COMPOSITE_PERCENTILE = 0.50
SELL_THRESHOLD_RANK = 150
REBALANCE_FREQ = "QS"  # quarter start

# -- Transaction costs & taxes --
TRANSACTION_COST_BPS = 10      # basis points per trade (10 = 0.10%)
SHORT_TERM_TAX_RATE = 0.35     # gains on positions held < 1 year
LONG_TERM_TAX_RATE = 0.15      # gains on positions held >= 1 year
REBALANCE_BAND = 0.05          # only rebalance if position drifts >5% from target
TAX_PROTECTION_DAYS = 90       # defer selling gains within this many days of 1-year LT threshold

# -- Strategy profiles --
# Shared scoring/portfolio settings above, only trading rules differ.
# Select via: python run_backtest.py [strategy_name]
STRATEGIES = {
    "tax_advantaged": {
        "description": "For IRA/401k — no tax-aware rules, pre-tax results only",
        "rebalance_band": 0.01,
        "tax_protection_days": 0,
        "show_after_tax": False,
    },
    "taxable": {
        "description": "For taxable accounts — wider bands, LT tax protection",
        "rebalance_band": 0.05,
        "tax_protection_days": 90,
        "show_after_tax": True,
    },
}
DEFAULT_STRATEGY = "taxable"

# -- Backtest --
HOLDOUT_QUARTERS = 8  # 8 quarters = 2 year holdout
INITIAL_CASH = 100_000

# -- FRED --
FRED_TB3MS_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=TB3MS"
