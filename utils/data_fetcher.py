# utils/data_fetcher.py
import os
import sqlite3
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import pytz
from datetime import datetime, timedelta
import holidays

from utils.config import (
    API_ENDPOINT, TOKEN, DATASET, START_DATE, DB_PATH,
    FINANCIAL_BALANCE_SHEET_DATASET, FINANCIAL_INCOME_STATEMENT_DATASET, FINANCIAL_MONTH_REVENNUE,
    FINANCIAL_INVESTORSBUYSELL, FINANCIAL_PER,
    MARGINPURCHASESHORTSALE
)


def get_last_trading_day(date):
    # 使用 holidays 套件定義台灣假日
    tw_holidays = holidays.Taiwan()
    # 檢查是否為週末（週六或週日）或假日，若是則往前推一天
    while date.weekday() >= 5 or date in tw_holidays:
        date -= timedelta(days=1)
    return date

def get_expected_latest_date():
    """
    根據台灣時區，若當前時間 >= 22:00，則最新資料日期為今日，
    否則為昨日；若該日期為休盤日（週末或假日），則返回最近一個交易日。
    """
    tz = pytz.timezone("Asia/Taipei")
    now = datetime.now(tz)
    # 根據時間決定初步日期
    if now.hour >= 22:
        latest_date = now.date()
    else:
        latest_date = (now - timedelta(days=1)).date()
    # 如果初步日期為非交易日，則往前推到最近一個交易日
    return get_last_trading_day(latest_date)


def create_db_and_table():
    """建立 SQLite 資料庫與各資料表"""
    db_dir = os.path.dirname(DB_PATH)
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # candlesticks_daily table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS candlesticks_daily (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            market TEXT NOT NULL,
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            UNIQUE(market, symbol, date)
        )
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_daily_market_symbol_date 
        ON candlesticks_daily (market, symbol, date)
    """)
    
    # candlesticks_min table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS candlesticks_min (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            market TEXT NOT NULL,
            symbol TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            interval TEXT,
            UNIQUE(market, symbol, timestamp)
        )
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_min_market_symbol_timestamp 
        ON candlesticks_min (market, symbol, timestamp)
    """)
    
    # financials table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS financials (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            cost_of_goods_sold REAL,
            eps REAL,
            pe_ratio REAL,
            equity_attributable_to_owners REAL,
            gross_profit REAL,
            income_after_taxes REAL,
            income_from_continuing_operations REAL,
            noncontrolling_interests REAL,
            operating_expenses REAL,
            operating_income REAL,
            other_comprehensive_income REAL,
            pre_tax_income REAL,
            realized_gain REAL,
            revenue REAL,
            tax REAL,
            total_profit REAL,
            nonoperating_income_expense REAL,
            UNIQUE(symbol, date)
        )
    """)
    
    # technical_indicators table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS technical_indicators (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            market TEXT NOT NULL,
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            sma_5 REAL,
            sma_10 REAL,
            sma_20 REAL,
            sma_50 REAL,
            sma_200 REAL,
            ema_5 REAL,
            ema_10 REAL,
            ema_20 REAL,
            ema_50 REAL,
            ema_200 REAL,
            macd REAL,
            macd_signal REAL,
            macd_hist REAL,
            adx REAL,
            rsi_14 REAL,
            rsi_7 REAL,
            cci_14 REAL,
            cci_20 REAL,
            stoch_k REAL,
            stoch_d REAL,
            roc_10 REAL,
            roc_20 REAL,
            mom_10 REAL,
            mom_20 REAL,
            williams_r REAL,
            atr_14 REAL,
            bollinger_upper REAL,
            bollinger_middle REAL,
            bollinger_lower REAL,
            obv REAL,
            obv_slope REAL,
            mfi_14 REAL,
            volume_sma_5 REAL,
            volume_sma_10 REAL,
            volume_sma_20 REAL,
            UNIQUE(market, symbol, date)
        )
    """)
    
    # monthly_revenue table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS monthly_revenue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            monthly_revenue REAL,
            UNIQUE(symbol, date)
        )
    """)
    
    # margin_purchase_shortsale table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS margin_purchase_shortsale (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            symbol TEXT NOT NULL,
            MarginPurchaseBuy REAL,
            MarginPurchaseCashRepayment REAL,
            MarginPurchaseLimit REAL,
            MarginPurchaseSell REAL,
            MarginPurchaseTodayBalance REAL,
            MarginPurchaseYesterdayBalance REAL,
            OffsetLoanAndShort REAL,
            ShortSaleBuy REAL,
            ShortSaleCashRepayment REAL,
            ShortSaleLimit REAL,
            ShortSaleSell REAL,
            ShortSaleTodayBalance REAL,
            ShortSaleYesterdayBalance REAL,
            UNIQUE(symbol, date)
        )
    """)
    
    # institutional_investors_buy_sell table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS institutional_investors_buy_sell (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            Dealer_Hedging_buy REAL,
            Dealer_self_buy REAL,
            Foreign_Dealer_Self_buy REAL,
            Foreign_Investor_buy REAL,
            Investment_Trust_buy REAL,
            Dealer_Hedging_sell REAL,
            Dealer_self_sell REAL,
            Foreign_Dealer_Self_sell REAL,
            Foreign_Investor_sell REAL,
            Investment_Trust_sell REAL,
            UNIQUE(symbol, date)
        )
    """)
    
    # financial_per table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS financial_per (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            dividend_yield REAL,
            PER REAL,
            PBR REAL,
            UNIQUE(symbol, date)
        )
    """)
    
    conn.commit()
    conn.close()

# --------------------
# FinMind 日線資料抓取與存入
def fetch_stock_data(stock_id, start_date=None, end_date=None):
    if start_date is None:
        start_date = START_DATE
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")
    
    print(f"[data_fetcher] 正在從 FinMind 下載 {stock_id} 的日線資料...")
    
    params = {
        "dataset": DATASET,
        "data_id": stock_id,
        "start_date": start_date,
        "end_date": end_date,
        "token": TOKEN
    }
    response = requests.get(API_ENDPOINT, params=params)
    if response.status_code == 200:
        json_data = response.json()
        if json_data.get("status") == 200 and "data" in json_data:
            df = pd.DataFrame(json_data["data"])
            if not df.empty and "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df.sort_values("date", inplace=True)
                df.reset_index(drop=True, inplace=True)
                return df
            else:
                raise ValueError(f"[data_fetcher] FinMind 回傳資料為空：{stock_id}")
        else:
            raise ValueError(f"[data_fetcher] FinMind 回傳錯誤: {json_data}")
    else:
        raise Exception(f"[data_fetcher] HTTP Error: {response.status_code} {response.text}")

def store_stock_data_to_db(df, stock_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    market = "TW"
    for _, row in df.iterrows():
        date_str = row["date"].strftime("%Y-%m-%d")
        open_price = row.get("open", None)
        high_price = row.get("max", None)
        low_price = row.get("min", None)
        close_price = row.get("close", None)
        volume = row.get("Trading_Volume", None)
        cursor.execute("""
            INSERT OR REPLACE INTO candlesticks_daily 
            (market, symbol, date, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (market, stock_id, date_str, open_price, high_price, low_price, close_price, volume))
    
    conn.commit()
    conn.close()
    print(f"[data_fetcher] {stock_id} 的日線資料已存入資料庫。")

def load_stock_data_from_db(stock_id, start_date=None, end_date=None):
    if start_date is None:
        start_date = START_DATE
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT * FROM candlesticks_daily
    WHERE symbol = ?
      AND date BETWEEN ? AND ?
    ORDER BY date ASC
    """
    df = pd.read_sql_query(query, conn, params=(stock_id, start_date, end_date))
    conn.close()
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df

# --------------------
# 財報資料抓取與存入
def fetch_financial_data(stock_id, start_date=None, end_date=None):
    if start_date is None:
        start_date = START_DATE
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")
    
    print(f"[data_fetcher] 正在從 FinMind 下載 {stock_id} 的財報資料...")
    params_is = {
        "dataset": FINANCIAL_INCOME_STATEMENT_DATASET,
        "data_id": stock_id,
        "start_date": start_date,
        "end_date": end_date,
        "token": TOKEN
    }
    response_is = requests.get(API_ENDPOINT, params=params_is)
    if response_is.status_code != 200:
        raise Exception(f"[data_fetcher] HTTP Error (IncomeStatement): {response_is.status_code} {response_is.text}")
    json_is = response_is.json()
    if json_is.get("status") != 200 or "data" not in json_is:
        raise ValueError(f"[data_fetcher] FinMind 回傳錯誤 (IncomeStatement): {json_is}")
    df_is = pd.DataFrame(json_is["data"])
    df_is.rename(columns={"stock_id": "symbol"}, inplace=True)
    df_is = df_is.pivot(index=["symbol", "date"], columns="type", values="value").reset_index()
    
    income_statement_mapping = {
        "symbol": "symbol",
        "date": "date",
        "CostOfGoodsSold": "cost_of_goods_sold",
        "EPS": "eps",
        "EquityAttributableToOwnersOfParent": "equity_attributable_to_owners",
        "GrossProfit": "gross_profit",
        "IncomeAfterTaxes": "income_after_taxes",
        "IncomeFromContinuingOperations": "income_from_continuing_operations",
        "NoncontrollingInterests": "noncontrolling_interests",
        "OperatingExpenses": "operating_expenses",
        "OperatingIncome": "operating_income",
        "OtherComprehensiveIncome": "other_comprehensive_income",
        "PreTaxIncome": "pre_tax_income",
        "RealizedGain": "realized_gain",
        "Revenue": "revenue",
        "TAX": "tax",
        "TotalConsolidatedProfitForThePeriod": "total_profit",
        "TotalNonoperatingIncomeAndExpense": "nonoperating_income_expense"
    }
    df_is.rename(columns=lambda col: income_statement_mapping.get(col, col), inplace=True)
    df_is["date"] = pd.to_datetime(df_is["date"])
    
    expected_cols = ["symbol", "date", "cost_of_goods_sold", "eps",
                     "equity_attributable_to_owners", "gross_profit", "income_after_taxes",
                     "income_from_continuing_operations", "noncontrolling_interests",
                     "operating_expenses", "operating_income", "other_comprehensive_income",
                     "pre_tax_income", "realized_gain", "revenue", "tax",
                     "total_profit", "nonoperating_income_expense"]
    df_fin = df_is.reindex(columns=expected_cols)
    
    def get_closing_price(report_date):
        report_date_str = report_date.strftime("%Y-%m-%d")
        daily_df = load_stock_data_from_db(stock_id, start_date=report_date_str, end_date=report_date_str)
        if not daily_df.empty:
            return daily_df.iloc[0]["close"]
        start_range = (report_date - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
        end_range = (report_date + pd.Timedelta(days=10)).strftime("%Y-%m-%d")
        daily_df_range = load_stock_data_from_db(stock_id, start_date=start_range, end_date=end_range)
        if daily_df_range.empty:
            return None
        daily_df_range["diff"] = (daily_df_range["date"] - report_date).abs()
        closest_row = daily_df_range.loc[daily_df_range["diff"].idxmin()]
        return closest_row["close"]
    
    pe_list = []
    for idx, row in df_fin.iterrows():
        report_date = row["date"]
        closing_price = get_closing_price(report_date)
        eps_val = row["eps"]
        if closing_price is not None and eps_val not in [None, 0]:
            pe_list.append(closing_price / eps_val)
        else:
            pe_list.append(None)
    df_fin["pe_ratio"] = pe_list
    df_fin["symbol"] = df_fin["symbol"].fillna(stock_id)
    df_fin.sort_values("date", inplace=True)
    df_fin.reset_index(drop=True, inplace=True)
    
    return df_fin

def store_financial_data_to_db(df, stock_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    for _, row in df.iterrows():
        date_str = row["date"].strftime("%Y-%m-%d") if hasattr(row["date"], "strftime") else row["date"]
        cursor.execute("""
            INSERT OR REPLACE INTO financials 
            (symbol, date, cost_of_goods_sold, eps, pe_ratio, equity_attributable_to_owners,
             gross_profit, income_after_taxes, income_from_continuing_operations, noncontrolling_interests,
             operating_expenses, operating_income, other_comprehensive_income, pre_tax_income,
             realized_gain, revenue, tax, total_profit, nonoperating_income_expense)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            row.get("symbol"),
            date_str,
            row.get("cost_of_goods_sold"),
            row.get("eps"),
            row.get("pe_ratio"),
            row.get("equity_attributable_to_owners"),
            row.get("gross_profit"),
            row.get("income_after_taxes"),
            row.get("income_from_continuing_operations"),
            row.get("noncontrolling_interests"),
            row.get("operating_expenses"),
            row.get("operating_income"),
            row.get("other_comprehensive_income"),
            row.get("pre_tax_income"),
            row.get("realized_gain"),
            row.get("revenue"),
            row.get("tax"),
            row.get("total_profit"),
            row.get("nonoperating_income_expense")
        ))
    conn.commit()
    conn.close()
    print(f"[data_fetcher] {stock_id} 的財報資料已存入資料庫。")

def load_financial_data_from_db(stock_id, start_date=None, end_date=None):
    if start_date is None:
        start_date = START_DATE
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT * FROM financials
    WHERE symbol = ?
      AND date BETWEEN ? AND ?
    ORDER BY date ASC
    """
    df = pd.read_sql_query(query, conn, params=(stock_id, start_date, end_date))
    conn.close()
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df

# --------------------
# 月營收資料抓取與存入
def fetch_monthly_revenue(stock_id, start_date=None, end_date=None):
    if start_date is None:
        start_date = START_DATE
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")
    
    print(f"[data_fetcher] 正在從 FinMind 下載 {stock_id} 的月營收資料...")
    
    params = {
        "dataset": FINANCIAL_MONTH_REVENNUE,
        "data_id": stock_id,
        "start_date": start_date,
        "end_date": end_date,
        "token": TOKEN
    }
    response = requests.get(API_ENDPOINT, params=params)
    if response.status_code != 200:
        raise Exception(f"[data_fetcher] HTTP Error (MonthlyRevenue): {response.status_code} {response.text}")
    json_data = response.json()
    if json_data.get("status") != 200 or "data" not in json_data:
        raise ValueError(f"[data_fetcher] FinMind 回傳錯誤 (MonthlyRevenue): {json_data}")
    df = pd.DataFrame(json_data["data"])
    df.rename(columns={"stock_id": "symbol", "revenue": "monthly_revenue"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df = df[["symbol", "date", "monthly_revenue"]]
    return df

def store_monthly_revenue_to_db(df, stock_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    for _, row in df.iterrows():
        date_str = row["date"].strftime("%Y-%m-%d") if hasattr(row["date"], "strftime") else row["date"]
        cursor.execute("""
            INSERT OR REPLACE INTO monthly_revenue (symbol, date, monthly_revenue)
            VALUES (?, ?, ?)
        """, (
            row.get("symbol", stock_id),
            date_str,
            row.get("monthly_revenue")
        ))
    conn.commit()
    conn.close()
    print(f"[data_fetcher] {stock_id} 的月營收資料已存入資料庫。")

def load_monthly_revenue_from_db(stock_id, start_date=None, end_date=None):
    if start_date is None:
        start_date = START_DATE
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT * FROM monthly_revenue
    WHERE symbol = ?
      AND date BETWEEN ? AND ?
    ORDER BY date ASC
    """
    df = pd.read_sql_query(query, conn, params=(stock_id, start_date, end_date))
    conn.close()
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df

# --------------------
# 技術指標計算與存入
def compute_technical_indicators(stock_id, df_daily):
    if "high" not in df_daily.columns and "max" in df_daily.columns:
        df_daily = df_daily.rename(columns={"max": "high"})
    if "low" not in df_daily.columns and "min" in df_daily.columns:
        df_daily = df_daily.rename(columns={"min": "low"})
    if "volume" not in df_daily.columns and "Trading_Volume" in df_daily.columns:
        df_daily = df_daily.rename(columns={"Trading_Volume": "volume"})
    
    df = df_daily.copy()
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # SMA
    df["sma_5"] = df["close"].rolling(window=5).mean()
    df["sma_10"] = df["close"].rolling(window=10).mean()
    df["sma_20"] = df["close"].rolling(window=20).mean()
    df["sma_50"] = df["close"].rolling(window=50).mean()
    df["sma_200"] = df["close"].rolling(window=200).mean()
    
    # EMA
    df["ema_5"] = df["close"].ewm(span=5, adjust=False).mean()
    df["ema_10"] = df["close"].ewm(span=10, adjust=False).mean()
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema_200"] = df["close"].ewm(span=200, adjust=False).mean()
    
    # MACD
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    
    # ADX (簡易版)
    df["prev_close"] = df["close"].shift(1)
    df["tr1"] = df["high"] - df["low"]
    df["tr2"] = abs(df["high"] - df["prev_close"])
    df["tr3"] = abs(df["low"] - df["prev_close"])
    df["tr"] = df[["tr1", "tr2", "tr3"]].max(axis=1)
    df["dm_plus"] = df["high"].diff()
    df["dm_minus"] = -df["low"].diff()
    df["dm_plus"] = df.apply(lambda row: row["dm_plus"] if row["dm_plus"] > row["dm_minus"] and row["dm_plus"] > 0 else 0, axis=1)
    df["dm_minus"] = df.apply(lambda row: row["dm_minus"] if row["dm_minus"] > row["dm_plus"] and row["dm_minus"] > 0 else 0, axis=1)
    period = 14
    df["tr_sum"] = df["tr"].rolling(window=period).sum()
    df["dm_plus_sum"] = df["dm_plus"].rolling(window=period).sum()
    df["dm_minus_sum"] = df["dm_minus"].rolling(window=period).sum()
    df["di_plus"] = 100 * df["dm_plus_sum"] / df["tr_sum"]
    df["di_minus"] = 100 * df["dm_minus_sum"] / df["tr_sum"]
    df["dx"] = (abs(df["di_plus"] - df["di_minus"]) / (df["di_plus"] + df["di_minus"])) * 100
    df["adx"] = df["dx"].rolling(window=period).mean()
    
    # RSI
    def compute_rsi(series, period):
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    df["rsi_14"] = compute_rsi(df["close"], 14)
    df["rsi_7"] = compute_rsi(df["close"], 7)
    
    # CCI
    def compute_cci(df, period):
        tp = (df["high"] + df["low"] + df["close"]) / 3
        sma = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
        return (tp - sma) / (0.015 * mad)
    df["cci_14"] = compute_cci(df, 14)
    df["cci_20"] = compute_cci(df, 20)
    
    # Stochastic Oscillator
    low_min = df["low"].rolling(window=14).min()
    high_max = df["high"].rolling(window=14).max()
    df["stoch_k"] = 100 * ((df["close"] - low_min) / (high_max - low_min))
    df["stoch_d"] = df["stoch_k"].rolling(window=3).mean()
    
    # ROC
    df["roc_10"] = df["close"].pct_change(periods=10) * 100
    df["roc_20"] = df["close"].pct_change(periods=20) * 100
    
    # MOM
    df["mom_10"] = df["close"] - df["close"].shift(10)
    df["mom_20"] = df["close"] - df["close"].shift(20)
    
    # Williams %R
    df["williams_r"] = -100 * ((high_max - df["close"]) / (high_max - low_min))
    
    # ATR
    df["atr_14"] = df["tr"].rolling(window=14).mean()
    
    # Bollinger Bands (20日)
    df["bollinger_middle"] = df["close"].rolling(window=20).mean()
    std_20 = df["close"].rolling(window=20).std()
    df["bollinger_upper"] = df["bollinger_middle"] + 2 * std_20
    df["bollinger_lower"] = df["bollinger_middle"] - 2 * std_20
    
    # OBV
    df["obv"] = (df["volume"] * ((df["close"] - df["close"].shift(1)).apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0)))).cumsum()
    def obv_slope(series, window):
        return series.rolling(window=window).apply(lambda x: pd.Series(x).diff().mean())
    df["obv_slope"] = obv_slope(df["obv"], 5)
    
    # MFI (簡易版)
    tp = (df["high"] + df["low"] + df["close"]) / 3
    mf = tp * df["volume"]
    pos_mf = mf.where(df["close"] > df["close"].shift(1), 0)
    neg_mf = mf.where(df["close"] < df["close"].shift(1), 0)
    df["mfi_14"] = 100 - (100 / (1 + (pos_mf.rolling(window=14).sum() / neg_mf.rolling(window=14).sum())))
    
    # Volume SMA
    df["volume_sma_5"] = df["volume"].rolling(window=5).mean()
    df["volume_sma_10"] = df["volume"].rolling(window=10).mean()
    df["volume_sma_20"] = df["volume"].rolling(window=20).mean()
    
    indicators = df[["date", "sma_5", "sma_10", "sma_20", "sma_50", "sma_200",
                      "ema_5", "ema_10", "ema_20", "ema_50", "ema_200",
                      "macd", "macd_signal", "macd_hist", "adx",
                      "rsi_14", "rsi_7", "cci_14", "cci_20",
                      "stoch_k", "stoch_d", "roc_10", "roc_20",
                      "mom_10", "mom_20", "williams_r", "atr_14",
                      "bollinger_upper", "bollinger_middle", "bollinger_lower",
                      "obv", "obv_slope", "mfi_14", "volume_sma_5", "volume_sma_10", "volume_sma_20"]].copy()
    indicators["market"] = "TW"
    indicators["symbol"] = stock_id
    indicators = indicators[["market", "symbol", "date", "sma_5", "sma_10", "sma_20", "sma_50", "sma_200",
                               "ema_5", "ema_10", "ema_20", "ema_50", "ema_200",
                               "macd", "macd_signal", "macd_hist", "adx",
                               "rsi_14", "rsi_7", "cci_14", "cci_20",
                               "stoch_k", "stoch_d", "roc_10", "roc_20",
                               "mom_10", "mom_20", "williams_r", "atr_14",
                               "bollinger_upper", "bollinger_middle", "bollinger_lower",
                               "obv", "obv_slope", "mfi_14", "volume_sma_5", "volume_sma_10", "volume_sma_20"]]
    indicators.sort_values("date", inplace=True)
    indicators.reset_index(drop=True, inplace=True)
    return indicators

def store_technical_indicators_to_db(df, stock_id):
    import numpy as np
    expected_cols = [
        "market", "symbol", "date",
        "sma_5", "sma_10", "sma_20", "sma_50", "sma_200",
        "ema_5", "ema_10", "ema_20", "ema_50", "ema_200",
        "macd", "macd_signal", "macd_hist", "adx",
        "rsi_14", "rsi_7", "cci_14", "cci_20",
        "stoch_k", "stoch_d", "roc_10", "roc_20",
        "mom_10", "mom_20", "williams_r", "atr_14",
        "bollinger_upper", "bollinger_middle", "bollinger_lower",
        "obv", "obv_slope", "mfi_14", "volume_sma_5", "volume_sma_10", "volume_sma_20"
    ]
    df = df.reindex(columns=expected_cols, fill_value=None)
    df = df.where(pd.notnull(df), None)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    for _, row in df.iterrows():
        row_date = row["date"]
        if hasattr(row_date, "strftime"):
            row_date = row_date.strftime("%Y-%m-%d")
        values = (
            row["market"],
            row["symbol"],
            row_date,
            row["sma_5"],
            row["sma_10"],
            row["sma_20"],
            row["sma_50"],
            row["sma_200"],
            row["ema_5"],
            row["ema_10"],
            row["ema_20"],
            row["ema_50"],
            row["ema_200"],
            row["macd"],
            row["macd_signal"],
            row["macd_hist"],
            row["adx"],
            row["rsi_14"],
            row["rsi_7"],
            row["cci_14"],
            row["cci_20"],
            row["stoch_k"],
            row["stoch_d"],
            row["roc_10"],
            row["roc_20"],
            row["mom_10"],
            row["mom_20"],
            row["williams_r"],
            row["atr_14"],
            row["bollinger_upper"],
            row["bollinger_middle"],
            row["bollinger_lower"],
            row["obv"],
            row["obv_slope"],
            row["mfi_14"],
            row["volume_sma_5"],
            row["volume_sma_10"],
            row["volume_sma_20"]
        )
        cursor.execute("""
            INSERT OR REPLACE INTO technical_indicators 
            (market, symbol, date, sma_5, sma_10, sma_20, sma_50, sma_200,
             ema_5, ema_10, ema_20, ema_50, ema_200,
             macd, macd_signal, macd_hist, adx,
             rsi_14, rsi_7, cci_14, cci_20, stoch_k, stoch_d,
             roc_10, roc_20, mom_10, mom_20, williams_r, atr_14,
             bollinger_upper, bollinger_middle, bollinger_lower,
             obv, obv_slope, mfi_14, volume_sma_5, volume_sma_10, volume_sma_20)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ? ,?)
        """, values)
    conn.commit()
    conn.close()
    print(f"[data_fetcher] {stock_id} 的技術指標已存入資料庫。")

def load_technical_indicators_from_db(stock_id, start_date=None, end_date=None):
    if start_date is None:
        start_date = START_DATE
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT * FROM technical_indicators
    WHERE symbol = ?
      AND date BETWEEN ? AND ?
    ORDER BY date ASC
    """
    df = pd.read_sql_query(query, conn, params=(stock_id, start_date, end_date))
    conn.close()
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df

# --------------------
# 融資融券資料抓取與存入
def fetch_margin_purchase_shortsale(stock_id, start_date=None, end_date=None):
    if start_date is None:
        start_date = START_DATE
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")
    
    print(f"[data_fetcher] 正在從 FinMind 下載 {stock_id} 的融資融券資料...")
    
    params = {
        "dataset": MARGINPURCHASESHORTSALE,
        "data_id": stock_id,
        "start_date": start_date,
        "end_date": end_date,
        "token": TOKEN
    }
    response = requests.get(API_ENDPOINT, params=params)
    if response.status_code != 200:
        raise Exception(f"[data_fetcher] HTTP Error (MarginPurchaseShortSale): {response.status_code} {response.text}")
    
    json_data = response.json()
    if json_data.get("status") != 200 or "data" not in json_data:
        raise ValueError(f"[data_fetcher] FinMind 回傳錯誤 (MarginPurchaseShortSale): {json_data}")
    
    df = pd.DataFrame(json_data["data"])
    df.rename(columns={"stock_id": "symbol"}, inplace=True)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df.sort_values("date", inplace=True)
        df.reset_index(drop=True, inplace=True)
        if "symbol" not in df.columns:
            df["symbol"] = stock_id
    return df

def store_margin_purchase_shortsale_to_db(df, stock_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    for _, row in df.iterrows():
        date_str = row["date"].strftime("%Y-%m-%d") if hasattr(row["date"], "strftime") else row["date"]
        cursor.execute("""
            INSERT OR REPLACE INTO margin_purchase_shortsale 
            (symbol, date, MarginPurchaseBuy, MarginPurchaseCashRepayment, MarginPurchaseLimit, 
             MarginPurchaseSell, MarginPurchaseTodayBalance, MarginPurchaseYesterdayBalance, 
             OffsetLoanAndShort, ShortSaleBuy, ShortSaleCashRepayment, ShortSaleLimit, 
             ShortSaleSell, ShortSaleTodayBalance, ShortSaleYesterdayBalance)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            row.get("symbol", stock_id),
            date_str,
            row.get("MarginPurchaseBuy"),
            row.get("MarginPurchaseCashRepayment"),
            row.get("MarginPurchaseLimit"),
            row.get("MarginPurchaseSell"),
            row.get("MarginPurchaseTodayBalance"),
            row.get("MarginPurchaseYesterdayBalance"),
            row.get("OffsetLoanAndShort"),
            row.get("ShortSaleBuy"),
            row.get("ShortSaleCashRepayment"),
            row.get("ShortSaleLimit"),
            row.get("ShortSaleSell"),
            row.get("ShortSaleTodayBalance"),
            row.get("ShortSaleYesterdayBalance")
        ))
    conn.commit()
    conn.close()
    print(f"[data_fetcher] {stock_id} 的融資融券資料已存入資料庫。")

def load_margin_purchase_shortsale_from_db(stock_id, start_date=None, end_date=None):
    if start_date is None:
        start_date = START_DATE
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")
    
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT * FROM margin_purchase_shortsale
        WHERE symbol = ?
          AND date BETWEEN ? AND ?
        ORDER BY date ASC
    """
    df = pd.read_sql_query(query, conn, params=(stock_id, start_date, end_date))
    conn.close()
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df

# --------------------
# 新增：機構投信與其他投資者買賣資料（Institutional Investors Buy/Sell）
def fetch_investors_buy_sell(stock_id, start_date=None, end_date=None):
    if start_date is None:
        start_date = START_DATE
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")
    print(f"[data_fetcher] 正在從 FinMind 下載 {stock_id} 的機構投信買賣資料...")
    params = {
        "dataset": FINANCIAL_INVESTORSBUYSELL,
        "data_id": stock_id,
        "start_date": start_date,
        "token": TOKEN
    }
    response = requests.get(API_ENDPOINT, params=params)
    if response.status_code != 200:
        raise Exception(f"[data_fetcher] HTTP Error (InvestorsBuySell): {response.status_code} {response.text}")
    json_data = response.json()
    if json_data.get("status") != 200 or "data" not in json_data:
        raise ValueError(f"[data_fetcher] FinMind 回傳錯誤 (InvestorsBuySell): {json_data}")
    df = pd.DataFrame(json_data["data"])
    df = df.pivot(index=['date', 'stock_id'], columns='name', values=['buy', 'sell'])
    # 重新命名欄位，使其符合需求
    df.columns = [f"{col[1]}_{col[0]}" for col in df.columns]
    # 重置索引，使其變回普通欄位
    df.reset_index(inplace=True)
    df.rename(columns={"stock_id": "symbol"}, inplace=True)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df.sort_values("date", inplace=True)
        df.reset_index(drop=True, inplace=True)
    return df

def store_investors_buy_sell_to_db(df, stock_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    for _, row in df.iterrows():
        date_str = row["date"].strftime("%Y-%m-%d") if hasattr(row["date"], "strftime") else row["date"]
        cursor.execute("""
            INSERT OR REPLACE INTO institutional_investors_buy_sell 
            (symbol, date, Dealer_Hedging_buy, Dealer_self_buy, Foreign_Dealer_Self_buy, Foreign_Investor_buy, Investment_Trust_buy,
             Dealer_Hedging_sell, Dealer_self_sell, Foreign_Dealer_Self_sell, Foreign_Investor_sell, Investment_Trust_sell)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            row.get("symbol", stock_id),
            date_str,
            row.get("Dealer_Hedging_buy"),
            row.get("Dealer_self_buy"),
            row.get("Foreign_Dealer_Self_buy"),
            row.get("Foreign_Investor_buy"),
            row.get("Investment_Trust_buy"),
            row.get("Dealer_Hedging_sell"),
            row.get("Dealer_self_sell"),
            row.get("Foreign_Dealer_Self_sell"),
            row.get("Foreign_Investor_sell"),
            row.get("Investment_Trust_sell")
        ))
    conn.commit()
    conn.close()
    print(f"[data_fetcher] {stock_id} 的機構投信買賣資料已存入資料庫。")

def load_investors_buy_sell_from_db(stock_id, start_date=None, end_date=None):
    if start_date is None:
        start_date = START_DATE
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT * FROM institutional_investors_buy_sell
        WHERE symbol = ?
          AND date BETWEEN ? AND ?
        ORDER BY date ASC
    """
    df = pd.read_sql_query(query, conn, params=(stock_id, start_date, end_date))
    conn.close()
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df

# --------------------
# 新增：本益比等資料 (PER資料)
def fetch_per_data(stock_id, start_date=None, end_date=None):
    if start_date is None:
        start_date = START_DATE
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")
    print(f"[data_fetcher] 正在從 FinMind 下載 {stock_id} 的本益比資料...")
    params = {
        "dataset": FINANCIAL_PER,
        "data_id": stock_id,
        "start_date": start_date,
        "end_date": end_date,
        "token": TOKEN
    }
    response = requests.get(API_ENDPOINT, params=params)
    if response.status_code != 200:
        raise Exception(f"[data_fetcher] HTTP Error (PER): {response.status_code} {response.text}")
    json_data = response.json()
    if json_data.get("status") != 200 or "data" not in json_data:
        raise ValueError(f"[data_fetcher] FinMind 回傳錯誤 (PER): {json_data}")
    df = pd.DataFrame(json_data["data"])
    df.rename(columns={"stock_id": "symbol"}, inplace=True)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df.sort_values("date", inplace=True)
        df.reset_index(drop=True, inplace=True)
    return df

def store_per_data_to_db(df, stock_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    for _, row in df.iterrows():
        date_str = row["date"].strftime("%Y-%m-%d") if hasattr(row["date"], "strftime") else row["date"]
        cursor.execute("""
            INSERT OR REPLACE INTO financial_per 
            (symbol, date, dividend_yield, PER, PBR)
            VALUES (?, ?, ?, ?, ?)
        """, (
            row.get("symbol", stock_id),
            date_str,
            row.get("dividend_yield"),
            row.get("PER"),
            row.get("PBR")
        ))
    conn.commit()
    conn.close()
    print(f"[data_fetcher] {stock_id} 的本益比資料已存入資料庫。")

def load_per_data_from_db(stock_id, start_date=None, end_date=None):
    if start_date is None:
        start_date = START_DATE
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT * FROM financial_per
        WHERE symbol = ?
          AND date BETWEEN ? AND ?
        ORDER BY date ASC
    """
    df = pd.read_sql_query(query, conn, params=(stock_id, start_date, end_date))
    conn.close()
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df
