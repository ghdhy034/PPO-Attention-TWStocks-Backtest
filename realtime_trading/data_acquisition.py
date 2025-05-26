# realtime_trading/data_acquisition.py
import random
from datetime import datetime

def get_realtime_data(stock_id):
    """
    模擬從交易所或 API 獲取實時數據。
    返回一個包含 'timestamp'、'close' 與 'volume' 的字典。
    實際上線時應根據 shioaji API 進行查價。
    """
    data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "close": round(random.uniform(50, 150), 2),
        "volume": random.randint(1000, 10000)
    }
    return data

if __name__ == "__main__":
    print(get_realtime_data("2330"))
