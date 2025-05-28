from decimal import Decimal

# --- WebSocket and API Endpoints ---
WEBSOCKET_URI = "wss://ws.gomarket-cpp.goquant.io/ws/l2-orderbook/okx/BTC-USDT-SWAP"
OKX_INSTRUMENTS_API = "https://www.okx.com/api/v5/public/instruments?instType=SPOT"

# --- Default UI Values ---
DEFAULT_USD_EQUIVALENT_QUANTITY = Decimal("10000")
DEFAULT_ANNUAL_VOLATILITY_PCT = Decimal("60")
DEFAULT_ASSUMED_DAILY_VOLUME_ASSET = Decimal("2000") 

# --- Fee Model---
FEE_TIERS_OKX_SPOT = {
    # Regular Users
    "Regular User Lv1 (Taker: 0.100%)": {"rate": Decimal("0.00100"), "id": "REG_LV1"},
    "Regular User Lv2 (Taker: 0.090%)": {"rate": Decimal("0.00090"), "id": "REG_LV2"}, 
    "Regular User Lv3 (Taker: 0.085%)": {"rate": Decimal("0.00085"), "id": "REG_LV3"},
    "Regular User Lv4 (Taker: 0.080%)": {"rate": Decimal("0.00080"), "id": "REG_LV4"},
    "Regular User Lv5 (Taker: 0.070%)": {"rate": Decimal("0.00070"), "id": "REG_LV5"},
    # VIP Tiers (typically based on 30d trading volume in USD and/or Asset Balance)
    # The volume/asset thresholds are not included in this dict for simplicity, only the rates.
    "VIP 1 (Taker: 0.080%)": {"rate": Decimal("0.00080"), "id": "VIP_1"},
    "VIP 2 (Taker: 0.070%)": {"rate": Decimal("0.00070"), "id": "VIP_2"},
    "VIP 3 (Taker: 0.060%)": {"rate": Decimal("0.00060"), "id": "VIP_3"},
    "VIP 4 (Taker: 0.050%)": {"rate": Decimal("0.00050"), "id": "VIP_4"},
    "VIP 5 (Taker: 0.040%)": {"rate": Decimal("0.00040"), "id": "VIP_5"},
    "VIP 6 (Taker: 0.030%)": {"rate": Decimal("0.00030"), "id": "VIP_6"},
    "VIP 7 (Taker: 0.025%)": {"rate": Decimal("0.00025"), "id": "VIP_7"},
    "VIP 8 (Taker: 0.020%)": {"rate": Decimal("0.00020"), "id": "VIP_8"}
}
DEFAULT_FEE_TIER_DISPLAY_NAME = "Regular User Lv1 (Taker: 0.100%)"

# --- Model Placeholder Coefficients ---
SLIPPAGE_REG_INTERCEPT = Decimal("0.0001")      # Base slippage (0.01%) - OK
SLIPPAGE_REG_COEF_DEPTH_INV = Decimal("0.0000001") # If inv_depth is 1M, this adds 0.1 to slippage.
SLIPPAGE_REG_COEF_SIZE_USD = Decimal("0.000000005")# If qty is 1M USD, this adds 0.005 to slippage.

SLIPPAGE_REG_COEF_VOL_DAILY = Decimal("0.1")    # Coef for daily volatility
DEPTH_PROXY_PERCENTAGE = Decimal("0.005")       # 0.5% from mid-price for depth calculation

# Market Impact (Original Simple Almgren-Chriss Style for Instantaneous Impact - Simulated)
AC_ETA_SIMPLE = Decimal("0.7")
AC_DELTA_SIMPLE = Decimal("0.5")

# --- Almgren-Chriss Dynamic Programming (DP) Model Parameters ---
DEFAULT_AC_DP_RISK_AVERSION = Decimal("0.01") 
DEFAULT_AC_DP_TIME_STEPS = 10                  
DEFAULT_AC_DP_TOTAL_EXECUTION_HORIZON_DAYS = Decimal("1.0") 
AC_DP_ALPHA = Decimal("0.5")  
AC_DP_BETA = Decimal("0.5")   
AC_DP_ETA = Decimal("0.0005")
AC_DP_GAMMA = Decimal("0.0005")

# --- Almgren-Chriss Closed-Form (CF) Model Parameters ---
DEFAULT_AC_CF_BID_ASK_SPREAD_USD = Decimal("0.01")
DEFAULT_AC_CF_DAILY_TRADE_VOL_ASSET = Decimal("5000")
DEFAULT_AC_CF_RISK_AVERSION = Decimal("1E-6") # Lambda
DEFAULT_AC_CF_TRADING_DAYS_PER_YEAR = 250


# Maker/Taker Logistic Regression (Simulated)
LR_INTERCEPT = Decimal("-2.0")  # Base tendency (more towards taker if very negative)
LR_COEF_SPREAD_BPS = Decimal("-0.1") # Wider spread -> less likely maker (more likely taker)
LR_COEF_SIZE_USD = Decimal("-0.000002") # Larger size -> less likely maker (more likely taker)

# New features for Maker/Taker model
LR_OBI_NUM_LEVELS = 5 # Number of levels to consider for Order Book Imbalance
LR_COEF_OBI_OPPOSITE_VS_PASSIVE = Decimal("-0.3") # Renamed from LR_COEF_OBI_ASKS_BIDS_RATIO
LR_COEF_OBI_ASKS_BIDS_RATIO = Decimal("0.5") # Positive: if more asks relative to bids, more likely to be a maker when BUYING
                                             # (and vice-versa for selling). This implies a specific definition of OBI.
                                             # Let OBI = (sum_ask_qty / sum_bid_qty).
                                             # If BUYING: P(Maker) increases if OBI is high (lots of asks to potentially sit in front of). Coef should be positive.
                                             # If SELLING: P(Maker) increases if OBI is low (lots of bids to potentially sit in front of). Coef should be negative if OBI is ask/bid.
                                             # Let's define OBI relative to the order side.
                                             # OBI_BUY_SIDE = (sum_ask_N / sum_bid_N). If BUYING, high OBI_BUY_SIDE is good for being maker.
                                             # OBI_SELL_SIDE = (sum_bid_N / sum_ask_N). If SELLING, high OBI_SELL_SIDE is good for being maker.
                                             # Simpler: Define one OBI = sum_far_side_qty / sum_near_side_qty. Coef positive.

LR_COEF_SOB_SIZE_OVER_BEST_QTY = Decimal("-1.0")
#other
DECIMAL_PRECISION = 28