# strategies/test_strategies.py
import pandas as pd
import numpy as np

class BaseStrategy:
    def __init__(self, name="BaseStrategy"):
        self.name = name
        self.position = 0
        self.cash = None
        self.initial_capital = None
        self.buy_price = None
        self.holding_days = 0

    def reset(self, initial_capital):
        self.position = 0
        self.cash = initial_capital
        self.initial_capital = initial_capital
        self.buy_price = None
        self.holding_days = 0

    def get_action(self, idx, data):
        return 0

    def update(self, action, price, trade_qty=100):
        if action == 1:
            cost = price * trade_qty
            if self.cash >= cost:
                self.cash -= cost
                self.position += trade_qty
                if self.buy_price is None:
                    self.buy_price = price
                self.holding_days = 0
        elif action == 2:
            if self.position >= trade_qty:
                self.cash += price * trade_qty
                self.position -= trade_qty
                self.holding_days = 0
                if self.position == 0:
                    self.buy_price = None
        else:
            self.holding_days += 1

    def get_portfolio_value(self, price):
        return self.cash + self.position * price

class BiasStrategy(BaseStrategy):
    def __init__(self, window=10, threshold=0.05):
        super().__init__(name="BiasStrategy")
        self.window = window
        self.threshold = threshold

    def get_action(self, idx, data):
        if idx < self.window:
            return 0
        window_data = data['close'].iloc[idx - self.window: idx]
        ma = window_data.mean()
        current_price = data['close'].iloc[idx]
        if current_price < ma * (1 - self.threshold):
            return 1
        elif current_price > ma * (1 + self.threshold):
            return 2
        else:
            return 0

def run_strategy(strategy, data, initial_capital=1_000_000, delay=1):
    strategy.reset(initial_capital)
    records = []
    n = len(data)
    for idx in range(delay, n):
        timestamp = data['date'].iloc[idx]
        action = strategy.get_action(idx - delay, data)
        current_price = data['close'].iloc[idx]
        strategy.update(action, current_price)
        portfolio_value = strategy.get_portfolio_value(current_price)
        record = {
            "timestamp": timestamp,
            "action": {0:"HOLD", 1:"BUY", 2:"SELL"}.get(action, "UNKNOWN"),
            "price": current_price,
            "capital": portfolio_value,
            "shares": strategy.position,
            "strategy": strategy.name
        }
        records.append(record)
    if strategy.position > 0:
        final_price = data['close'].iloc[-1]
        strategy.cash += final_price * strategy.position
        strategy.position = 0
        record = {
            "timestamp": data['date'].iloc[-1],
            "action": "LIQUIDATE",
            "price": final_price,
            "capital": strategy.cash,
            "shares": strategy.position,
            "strategy": strategy.name
        }
        records.append(record)
    return records

if __name__ == "__main__":
    data = pd.DataFrame({
        "date": pd.date_range("2021-01-01", periods=100, freq="B"),
        "close": np.linspace(100, 150, 100),
        "high": np.linspace(101, 151, 100),
        "low": np.linspace(99, 149, 100),
        "volume": np.random.randint(1000, 2000, 100)
    })
    strategies = [BiasStrategy()]
    all_records = []
    for strat in strategies:
        recs = run_strategy(strat, data, initial_capital=1_000_000, delay=1)
        all_records.extend(recs)
    df = pd.DataFrame(all_records)
    df.to_csv("test_strategies_detailed.csv", index=False, float_format="%.2f")
    print("測試策略詳細記錄已儲存到 test_strategies_detailed.csv")
