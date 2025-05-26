# utils/config.py
import os
import random
# Directory settings
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE_DIR, "..")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
HISTORICAL_DATA_DIR = os.path.join(DATA_DIR, "historical")
REALTIME_DATA_DIR = os.path.join(DATA_DIR, "realtime")
DB_PATH = os.path.join(DATA_DIR, "stock_data.db")
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "models")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

for path in [DATA_DIR, RAW_DATA_DIR, HISTORICAL_DATA_DIR, REALTIME_DATA_DIR, MODEL_SAVE_PATH, LOG_DIR]:
    if not os.path.exists(path):
        os.makedirs(path)

# ============================
# FinMind API Settings
# ============================

API_ENDPOINT = "https://api.finmindtrade.com/api/v4/data"
TOKEN = "your_FinMind_API_token"
DATASET = "TaiwanStockPrice"
FINANCIAL_BALANCE_SHEET_DATASET = "TaiwanStockBalanceSheet"
FINANCIAL_INCOME_STATEMENT_DATASET = "TaiwanStockFinancialStatements"
FINANCIAL_MONTH_REVENNUE = "TaiwanStockMonthRevenue"
MARGINPURCHASESHORTSALE = "TaiwanStockMarginPurchaseShortSale"
FINANCIAL_INVESTORSBUYSELL="TaiwanStockInstitutionalInvestorsBuySell"
FINANCIAL_PER="TaiwanStockPER"


# Trading Parameters
INITIAL_CAPITAL = 10_000_000
COMMISSION_RATE = 0.001425      # 手續費率 0.1425%
STAMP_TAX_RATE = 0.003          # 證交稅率 0.3%
FEE_DISCOUNT = 0.21             # 手續費折扣（2.1折）
TRADE_COST_MULTIPLIER = 1     # 交易成本乘數，用於加強成本懲罰
FREQUENCY_PENALTY = 0.0000001      # 每股交易懲罰基礎值（後續使用平方懲罰）
# 新增：若每檔股票調整金額低於前一日持倉價值的比例，則不調倉
REBALANCE_THRESHOLD_RATIO = 0.0001

# Candidate stock settings
# 訓練時使用完整候選集 (例如150支)
TRAINING_CANDIDATE_SET_SIZE = 59
# 部署時使用子集 (例如30支)
DEPLOYMENT_CANDIDATE_SET_SIZE = 59



# 模型架構參數
# 訓練時 state 與 action 維度依候選股票數決定
STATE_DIM = TRAINING_CANDIDATE_SET_SIZE
ACTION_DIM = TRAINING_CANDIDATE_SET_SIZE


# ============================
# Data Date Range
# ============================
START_DATE = "2020-01-01"
END_DATE = "2025-03-25"  # 可根據需求動態更新
MIN_START = "2020-01-01"  # 若需要盤中數據

# PPO 超參數
PPO_LEARNING_RATE = 1e-4        # 降低學習率，穩定訓練
PPO_EPOCHS = 10
PPO_BATCH_SIZE = 128
PPO_CLIP_EPSILON = 0.2
PPO_DISCOUNT_FACTOR = 0.99
PPO_VALUE_LOSS_COEF = 0.5
PPO_ENTROPY_COEF = 0.01
PPO_MAX_GRAD_NORM = 0.5


# 訓練參數
TOTAL_TRAINING_STEPS = 500_000
UPDATE_INTERVAL = 4096

# Transformer parameters
D_MODEL = 64
NHEAD = 4
NUM_LAYERS = 2
DROPOUT = 0.1


# Reward Scaling 調整
REWARD_SCALE = 1e-7             # 進一步降低 reward scale，壓縮回報數值

# 風險管理參數：停損閥值 (跌幅超過此比例則觸發停損)
STOP_LOSS_THRESHOLD = 0.1

# 新增 GAE 與優勢標準化參數
USE_GAE = True                  # 使用 Generalized Advantage Estimation
GAE_LAMBDA = 0.95               # GAE 的 lambda 參數
NORMALIZE_ADVANTAGE = True      # 對優勢進行標準化




# 候選股票列表 (這裡示範使用數字字串，實際應用請根據需求調整)
STOCK_IDS = [
    "2330", "2317", "2382", "2449", "3711", "3037", "2308", "2395", "5347", "2486",
    "4919", "6187", "2454", "2609", "2603", "3481", "3019", "2379", "3231", "8064",
    "2356", "3665", "3227", "6223", "3029", "2618", "2383", "2368", "2472", "3596",
    "2303", "3406", "8299", "3706", "1605", "6148", "2344", "2328", "2408", "2027",
    "2409", "2606", "2002", "2637", "2371", "2610", "3260", "9945", "3162", "1101",
    "3264", "5309", "1303", "1326", "2547", "8088", "2324", "5439", "3189"
    
]



BACKTEST_STOCK_LIST = [
    "2330", "2317", "2382", "2449", "3711", "3037", "2308", "2395", "5347", "2486",
    "4919", "6187", "2454", "2609", "2603", "3481", "3019", "2379", "3231", "8064",
    "2356", "3665", "3227", "6223", "3029", "2618", "2383", "2368", "2472", "3596",
    "2303", "3406", "8299", "3706", "1605", "6148", "2344", "2328", "2408", "2027",
    "2409", "2606", "2002", "2637", "2371", "2610", "3260", "9945", "3162", "1101",
    "3264", "5309", "1303", "1326", "2547", "8088", "2324", "5439", "3189"
    
]
# 部署時的股票子集 (取前 DEPLOYMENT_CANDIDATE_SET_SIZE 支)
DEPLOYMENT_STOCK_IDS = STOCK_IDS



# 回測時的股票列表（可根據需求調整）

# [
#     '5274', '3661', '3529', '3008', '6669', '2454', '8299', '4763', '2345',
#     '2317', '8234', '5439', '2408', '1802', '2328', '2027', '6558', '1519', '3131',
#     '3665', '4749', '6446', '3017', '2379', '3034', '2327', '1476', '3563', '6640',
#     '3491', '6442', '3406', '2404', '6691', '3680', '6415', '6488', '3592', '2395',
#     '6121', '6683', '8454', '3693', '6533', '1477', '6890', '2308', '5536', '6491',
#     '4770', '7728', '3081', '3583', '6231', '1264', '4766', '6187', '8114', '2360',
#     '8069', '6531', '6679',
#     '8996', '2392', '6416', '6271', '1584', '4919', '1590', '2231', '1598', '4958',
#     '2357', '3037', '2356', '3588', '2382', '2377', '2353', '4952', '2303', '2352',
#     '2337', '2331', '2351', '2344', '2342', '2340', '2338', '2332',
#     '3711', '2609', '2603', '6526', '3596', '8028', '4979', "2393", "2449", "6148",
#     "6768", "1605", "3706", "2368", "2383", "2618", "3029", "6223", "3227", "3231", "3019", "2486"
# ]
