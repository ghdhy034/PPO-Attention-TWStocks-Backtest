# realtime_trading/trading_engine.py
import time
from realtime_trading.data_acquisition import get_realtime_data
from realtime_trading.order_management import place_order
from utils.config import BACKTEST_STOCK_LIST

def simulate_agent_decision(state):
    """
    模擬代理人的決策，返回一個字典：
    {
        "selected_stocks": [list of stock ids],
        "actions": {stock_id: {"side": "buy"/"sell", "quantity": value, "price": value}}
    }
    這裡僅為示例，實際決策應由PPO及多標的選股模型輸出。
    """
    decision = {
        "selected_stocks": ["2330", "2317"],
        "actions": {
            "2330": {"side": "buy", "quantity": 0.05, "price": None},
            "2317": {"side": "sell", "quantity": 0.03, "price": None}
        }
    }
    return decision

def main():
    while True:
        print("===== 盤中交易決策循環 =====")
        state = {}
        for stock_id in BACKTEST_STOCK_LIST:
            data = get_realtime_data(stock_id)
            state[stock_id] = data
            print(f"實時數據 {stock_id}: {data}")
        
        decision = simulate_agent_decision(state)
        print(f"代理人決策: {decision}")
        
        for stock_id in decision["selected_stocks"]:
            action = decision["actions"].get(stock_id)
            if action:
                price = state.get(stock_id, {}).get("close")
                order_details = {
                    "symbol": stock_id,
                    "side": action["side"],
                    "quantity": action["quantity"],
                    "price": price
                }
                order_id = place_order(order_details)
                print(f"下單成功，訂單 ID: {order_id}")
        
        time.sleep(10)  # 模擬每10秒進行一次決策

if __name__ == "__main__":
    main()
