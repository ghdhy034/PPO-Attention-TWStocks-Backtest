# strategies/multi_stock_trading_env.py
import gym
import numpy as np
import pandas as pd
from gym import spaces
from utils.config import INITIAL_CAPITAL, REWARD_SCALE, STOP_LOSS_THRESHOLD, START_DATE, END_DATE

def get_common_dates(data_dict):
    """
    產生從 config.START_DATE 到 config.END_DATE 的交易日期（以工作日為準）
    """
    common_dates = pd.date_range(start=START_DATE, end=END_DATE, freq='B')
    common_dates = [d.date() for d in common_dates]
    return common_dates

class MultiStockTradingEnv(gym.Env):
    """
    多標的聯合交易環境：
    - 從資料字典中讀取候選股票的歷史數據
    - 狀態：所有股票的 T-1 日收盤價向量
    - 動作：連續向量，經 softmax 正規化後代表投資比例
    - 模擬每日全數再平衡，計算組合價值變化作為回報
    - info 包含兩組日誌：
        capital_log: {Date, PrevHoldings, PrevCash, Action, TodayClosingHoldings, TodayClosingCash}
        trade_log: {Date, StockID, PrevHoldings, Action, Price, TradeQuantity, TodayClosingHoldings}
    """
    metadata = {'render.modes': ['human']}
    
    def __init__(self, data_dict, initial_capital=INITIAL_CAPITAL, delay=1, subset=None):
        super(MultiStockTradingEnv, self).__init__()
        if subset is not None:
            self.data_dict = {sid: data_dict[sid] for sid in subset if sid in data_dict}
        else:
            self.data_dict = data_dict
        
        self.initial_capital = initial_capital
        self.delay = delay
        
        self.stock_ids = list(self.data_dict.keys())
        self.num_stocks = len(self.stock_ids)
        
        self.common_dates = get_common_dates(self.data_dict)
        if len(self.common_dates) < self.delay + 1:
            raise ValueError("Not enough trading days to satisfy delay requirement")
        self.current_idx = self.delay
        self.portfolio_value = self.initial_capital
        
        self.prev_shares = np.zeros(self.num_stocks, dtype=int)
        self.prev_cash = self.initial_capital
        
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(self.num_stocks,), dtype=np.float32)
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_stocks,), dtype=np.float32)
    
    def _get_prices(self, idx):
        """
        根據 common_dates 索引取得所有股票當天的收盤價向量。
        若某股票當天無資料，則取該股票之前最新的價格；若均無則返回 0.
        """
        date = self.common_dates[idx]
        prices = []
        for sid in self.stock_ids:
            df = self.data_dict[sid]
            row = df[df['date'].dt.date == date]
            if not row.empty:
                prices.append(row.iloc[0]['close'])
            else:
                df_before = df[df['date'].dt.date < date]
                if not df_before.empty:
                    prices.append(df_before.iloc[-1]['close'])
                else:
                    prices.append(0.0)
        return np.array(prices, dtype=np.float32)
    
    def reset(self):
        """
        重置環境，返回初始狀態：使用 T-1 日的收盤價向量。
        """
        self.current_idx = self.delay
        self.portfolio_value = self.initial_capital
        self.prev_cash = self.initial_capital
        self.prev_shares = np.zeros(self.num_stocks, dtype=int)
        state = self._get_prices(self.current_idx - self.delay)
        return state
    
    def step(self, action):
        """
        執行代理人的動作：
        - 將 action 經 softmax 正規化為投資比例
        - 根據比例計算各股票應持有數量（整數），並計算交易數量
        - 使用當天價格重新平衡，計算新投資組合價值
        - 若跌幅超過停損閥值，則觸發停損，強制清倉
        - 返回下個狀態、縮放後的 reward、done 與 info 日誌
        """
        eps = 1e-8
        action = np.clip(action, -50, 50)
        exp_a = np.exp(action - np.max(action))
        weights = exp_a / np.sum(exp_a)
        
        current_prices = self._get_prices(self.current_idx)
        current_prices_safe = np.clip(current_prices, 1e-2, None)
        
        # 計算新持倉，強制轉為一維陣列後再四捨五入並轉為整數
        new_shares = (self.portfolio_value * weights) / current_prices_safe
        new_shares = np.array(new_shares).flatten()
        new_shares = np.round(new_shares).astype(int)
        
        self.prev_shares = np.array(self.prev_shares).flatten().astype(int)
        trade_quantity = new_shares - self.prev_shares
        
        trade_log = []
        for i, sid in enumerate(self.stock_ids):
            qty = int(trade_quantity[i])
            if qty > 0:
                act_str = "BUY"
            elif qty < 0:
                act_str = "SELL"
            else:
                act_str = "HOLD"
            trade_log.append({
                "Date": self.common_dates[self.current_idx],
                "StockID": sid,
                "PrevHoldings": int(self.prev_shares[i]),
                "Action": act_str,
                "Price": current_prices[i],
                "TradeQuantity": qty,
                "TodayClosingHoldings": int(new_shares[i])
            })
        
        prev_day_prices = self._get_prices(self.current_idx - self.delay)
        prev_holdings_value = np.sum(self.prev_shares * prev_day_prices)
        capital_log = {
            "Date": self.common_dates[self.current_idx],
            "PrevHoldings": prev_holdings_value,
            "PrevCash": self.prev_cash,
            "Action": weights.tolist(),
            "TodayClosingHoldings": None,
            "TodayClosingCash": 0.0
        }
        
        old_value = self.portfolio_value
        
        self.current_idx += 1
        done = self.current_idx >= len(self.common_dates)
        if not done:
            new_prices = self._get_prices(self.current_idx)
            new_prices_safe = np.clip(new_prices, 1e-2, None)
            new_value = np.sum(new_shares * new_prices_safe)
            if not np.isfinite(new_value):
                new_value = old_value
            self.portfolio_value = new_value
            if (old_value - new_value) / (old_value + eps) > STOP_LOSS_THRESHOLD:
                new_value = 0.0
                reward = -old_value
                new_shares = np.zeros_like(new_shares, dtype=int)
            else:
                reward = new_value - old_value
            self.portfolio_value = new_value
            next_state = self._get_prices(self.current_idx - self.delay)
            capital_log["TodayClosingHoldings"] = np.sum(new_shares * new_prices_safe)
        else:
            reward = 0.0
            next_state = np.zeros(self.observation_space.shape[0], dtype=np.float32)
            capital_log["TodayClosingHoldings"] = self.portfolio_value
        
        reward *= REWARD_SCALE
        
        self.prev_shares = new_shares
        self.prev_cash = 0.0
        
        info = {
            "capital_log": capital_log,
            "trade_log": trade_log,
            "portfolio_value": self.portfolio_value
        }
        return next_state, reward, done, info
    
    def render(self, mode='human'):
        date = self.common_dates[self.current_idx - 1] if self.current_idx > 0 else None
        print(f"Date: {date}, Portfolio Value: {self.portfolio_value:.2f}")

if __name__ == "__main__":
    import pandas as pd
    dates = pd.date_range("2020-01-01", periods=10)
    df1 = pd.DataFrame({
        "date": dates,
        "close": np.linspace(100, 110, 10)
    })
    df2 = pd.DataFrame({
        "date": dates,
        "close": np.linspace(200, 210, 10)
    })
    data_dict = {"StockA": df1, "StockB": df2}
    env = MultiStockTradingEnv(data_dict, initial_capital=1000000, delay=1)
    state = env.reset()
    print("Initial State:", state)
    done = False
    while not done:
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        env.render()
        print("Capital Log:", info["capital_log"])
        print("Trade Log:", info["trade_log"])
