"""
config.py
Central settings for the data pipeline and the value forecast model of the Football Recruitment Decision-Support Tool.
The starting data is the synthetic dataset submitted in Week 5 (data/raw/). Every rule and every
assumption used to clean, calibrate, extend, model and validate it is kept here so it matches the reports.
"""
from pathlib import Path

# ------------------------------------------------------------------ paths
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
INTERMEDIATE = ROOT / "data" / "intermediate"
FINAL = ROOT / "data" / "final"
WITHHELD = ROOT / "data" / "withheld"
MODEL = ROOT / "data" / "model"
OUTPUTS = ROOT / "outputs"

SEED = 20251003            # fixed seed for the few values that are generated in Weeks 7-8

# ------------------------------------------------------------------ the Week 5 files (must stay unchanged)
RAW_FILES = {
    "case_event.csv": "deb5a3045c142498f1d405cd4e03983c86a47d9aba71547354d5de356e546c53",
    "club.csv": "556cd4455a0b0699d6bb9b79feb039f8a67748f486766ca15f9e038d1f1fb2e7",
    "contract.csv": "cd33e9f940706ee7e14034edbd1ec739a72993d5d81950c893790428c352467a",
    "decision_context.csv": "7952b2f1b9443e44d7a834365f9afd0fe3aa5ce8f1b4f66e46c30a61dac9fc41",
    "player.csv": "3624599bc077ec5c5650dfeb8203e6ac5439f432f1e88c9690f0c696275241d8",
    "player_season_stats.csv": "a1b4e37cf4dba229dc6ba924720241da44c118f5ed38cdfe136cbc0f151c5f02",
    "prediction_run.csv": "26c3fd98d8c0abc9ab503583fd383354e2fdaf991a9517524d0b30d28c9ce5cd",
    "scouting_case.csv": "6a9a7c953ea6e8ae4535475a52a2de2e9ad05444e38614eeb17d11c086a04722",
    "squad_need.csv": "aa43c189b8b1b1103540305113c5f95f7beeaf2e3f73d2d8b417190e3591689c",
    "transfer_decision.csv": "bd8bcae0c1d164b27735594e0d13a97989ba7492812723518181d875568f1a3b",
    "transfer_outcome.csv": "c9b3a2b118539d83f5028d0f50c42609888c163d0a82bd7a8d248e3b5372e53e",
    "valuation_history.csv": "5f87b31ea730e13c3ddd60ca04a89dc10aef0a5623d45e12eb3b1dd598b665dd",
}

# ------------------------------------------------------------------ cleaning rules
MIN_PLAYER_AGE_FOR_DATA = 16       # stats / valuations for players younger than 16 are not plausible
SEASON_START_MONTH_DAY = "08-01"   # age for a season is measured on 1 August of season_year
MAX_VALID_PASSES_P90 = 150         # above this a value is impossible -> set to missing
MAX_VALID_CARRIES_P90 = 60

# ------------------------------------------------------------------ calibration to Week 5 targets
TARGET_MEDIAN_MARKET_VALUE_EUR = 275_000   # all money fields are multiplied by one factor
TARGET_OUTFIELD_PASSES_MEAN = 45.0         # rank-preserving mapping to a log-normal shape
OUTFIELD_PASSES_CV = 0.35
TARGET_GK_PASSES_MEAN = 27.0
GK_PASSES_CV = 0.25

# ------------------------------------------------------------------ fields added in Weeks 7-8 (Week 6 needs them)
NEED_SETTINGS = {           # need_id: (need_code, min_age, max_age)
    1: ("N-01", 18, 32),
    2: ("N-02", 18, 32),
}
TARGET_PROFILE_PERCENTILE = 0.75   # target style = 75th percentile of MID players (club wants above average)
ASK_PREMIUM_LOG_MEAN = 0.08        # asking price = latest value x exp(normal(0.08, 0.20))
ASK_PREMIUM_LOG_SD = 0.20
ASK_SHORT_CONTRACT_FACTOR = 0.85   # contract ending within 12 months -> asking price 15% lower
VALIDATION_WINDOW = "Summer 2026"  # the 30 cases of this window are the validation set (Week 5 plan)

# ------------------------------------------------------------------ Week 6 decision rules
TSS_SHORTLIST = 70
TSS_MONITOR_MIN = 55
RAPF_MIN = 60
PREMIUM_BID_MAX = 0.10
PREMIUM_COUNTER_MAX = 0.25
COUNTER_BID_CAP = 1.10
MAX_WINDOWS_IN_MONITOR = 2
PRIORITY_WEIGHTS = (0.5, 0.3, 0.2)
MIN_SEASONS_FOR_RAPF = 2

FMV_AGE_ADJUSTMENT = [(23, 0.08), (26, 0.03), (29, -0.02), (99, -0.10)]   # (max age, adjustment)
FMV_CONTRACT_FACTOR = [(6, 0.80), (12, 0.88), (24, 0.96), (999, 1.00)]    # (months below, factor)
FMV_RANGE = 0.125

# ------------------------------------------------------------------ withheld validation outcomes (simulated)
OUTCOME_HORIZON_DAYS = 365
OUTCOME_VALUE_NOISE_SD = 0.15
AGENT_FEE_SHARE = 0.08
SIGNING_BONUS_SHARE_OF_WAGE = 0.5

# ------------------------------------------------------------------ value forecast model (steps 07-09)
# One forecast is made per player at each "forecast origin" (the day of the summer prediction run). It predicts the
# market value at the NEXT yearly valuation (1 January), i.e. 12 months after the latest valuation known at the origin.
FORECAST_ORIGIN_MONTH_DAY = "06-21"        # same day as the Week 5 prediction runs (21 June 2025 and 21 June 2026)
DEVELOPMENT_ORIGINS = [2023, 2024]         # used to choose and fit the model
BACKTEST_ORIGIN = 2025                     # held back: forecasts made in June 2025, checked against January 2026
LIVE_ORIGIN = 2026                         # the open window: the outcome (January 2027) is not known yet
CV_FOLDS = 5                               # cross-validation inside the development origins, split by player
CV_REPEATS = 3
KNOT_STEP_OPTIONS = [1, 2, 3]              # years between the points of the age curve; chosen by cross-validation
INTERVAL_LEVELS = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]   # prediction intervals checked in the backtest
MAIN_INTERVAL = 0.80                       # interval shown first in the tool
WIDE_INTERVAL = 0.95
COVERAGE_TOLERANCE = 0.07                  # the 80% and 95% intervals must cover within 7 points of their level
CONFIDENCE_PRICE_LEVELS = [0.50, 0.75, 0.90]   # "how sure do you want to be that the fee holds its value"
