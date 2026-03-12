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
# Informed by factor decay analysis (run_decay.py): removed factors with
# consistently negative IC (realized_vol_60d, volume_trend), redistributed
# weight to survivors. See CHANGELOG.md Phase 6 for full analysis.
FACTOR_WEIGHTS = {
    "momentum_12_1": 0.30,       # primary momentum signal, persistent at 6-12mo
    "momentum_6_1": 0.20,        # best single factor by IC (0.059* at 2mo IS)
    "rel_strength_3mo": 0.10,    # most recent momentum, slow to start but persistent
    "short_term_reversal": 0.00, # strong peak IC but decays by 3mo, too fast for quarterly
    "dist_from_ma50": 0.05,      # mild overextension filter, peaks at 3mo
    "volume_trend": 0.00,        # removed — negative IC, ~50% hit rate (noise)
    "obv_slope": 0.05,           # cumulative volume flow, marginal signal
    "realized_vol_60d": 0.00,    # removed — consistently negative IC (34-38% hit rate)
    "vol_trend": 0.10,           # most consistent factor at 3mo (75% hit rate, 0.037* IC)
    "price_consistency": 0.20,   # steady climbers, 69% hit rate at 6mo
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
