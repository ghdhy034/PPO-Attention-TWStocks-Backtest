# backtesting/backtest_engine.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from strategies.trading_env import TradingEnv

class BacktestEngine:
    def __init__(self, stock_id):
        self.stock_id = stock_id
        from utils.data_fetcher import load_stock_data_from_db
        self.data = load_stock_data_from_db(stock_id)
        self.env = TradingEnv(self.data, delay=1)
        self.history = []

    def run(self, agent):
        state = self.env.reset()
        done = False
        while not done:
            action, _ = agent.select_action(state)
            next_state, reward, done, info = self.env.step(action)
            price = self.env.data.iloc[self.env.current_step - 1]['close']
            trade_action = {0:"HOLD", 1:"BUY", 2:"SELL"}.get(action, "UNKNOWN")
            record = {
                "timestamp": self.env.data.iloc[self.env.current_step - 1]['date'],
                "action": trade_action,
                "price": price,
                "capital": info.get("portfolio_value", np.nan),
                "shares": self.env.shares
            }
            self.history.append(record)
            state = next_state
        return self.history

    def plot_results(self):
        df = pd.DataFrame(self.history)
        plt.figure(figsize=(10, 5))
        plt.plot(df["timestamp"], df["capital"], marker="o", linestyle="-", label="Portfolio Value")
        plt.xlabel("Date")
        plt.ylabel("Capital")
        plt.title(f"Backtest for Stock {self.stock_id}")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    engine = BacktestEngine("2330")
    from strategies.ppo_agent import PPOAgent
    obs_dim = engine.env.observation_space.shape[0]
    action_dim = 3  # 此處假設離散動作測試
    agent = PPOAgent(obs_dim, action_dim)
    history = engine.run(agent)
    engine.plot_results()
