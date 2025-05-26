# strategies/trading_env.py
import gym
import numpy as np
import pandas as pd
from gym import spaces
from utils.config import INITIAL_CAPITAL, COMMISSION_RATE, STAMP_TAX_RATE

class TradingEnv(gym.Env):
    """
    基於 Gym 的交易環境：
    - 支持盤中多次決策與 T-1 延遲
    - 狀態由數值型資料構成，後續可整合多標的特徵降維或注意力處理
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, data, delay=1, initial_capital=INITIAL_CAPITAL):
        super(TradingEnv, self).__init__()
        self.data = data.reset_index(drop=True)
        self.delay = delay
        if len(self.data) <= self.delay:
            raise ValueError("資料筆數不足以滿足延遲需求")
        self.initial_capital = initial_capital
        self.reset()
        
        # 以所有數值型欄位構成狀態向量
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(numeric_cols),), dtype=np.float32)
        # 離散動作：0: HOLD, 1: BUY, 2: SELL
        self.action_space = spaces.Discrete(3)
    
    def reset(self):
        self.current_step = self.delay
        self.cash = self.initial_capital
        self.shares = 0
        self.portfolio_value = self.initial_capital
        self.peak_value = self.portfolio_value
        self.done = False
        return self._get_observation()
    
    def _get_observation(self):
        idx = self.current_step - self.delay
        # 只選取數值型欄位
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        obs = self.data.iloc[idx][numeric_cols].values.astype(np.float32)
        return obs
    
    def step(self, action):
        if self.done:
            return self._get_observation(), 0.0, self.done, {}
        
        current_price = self.data.iloc[self.current_step]['close']
        prev_value = self.portfolio_value
        
        if action == 1:  # BUY
            cost = current_price * 1 * (1 + COMMISSION_RATE)
            if self.cash >= cost:
                self.cash -= cost
                self.shares += 1
        elif action == 2:  # SELL
            if self.shares > 0:
                proceeds = current_price * 1 * (1 - (COMMISSION_RATE + STAMP_TAX_RATE))
                self.cash += proceeds
                self.shares -= 1
        
        self.portfolio_value = self.cash + self.shares * current_price
        if self.portfolio_value > self.peak_value:
            self.peak_value = self.portfolio_value
        
        reward = self.portfolio_value - prev_value
        self.current_step += 1
        if self.current_step >= len(self.data):
            self.done = True
        
        obs = self._get_observation() if not self.done else np.zeros(self.observation_space.shape[0], dtype=np.float32)
        info = {"portfolio_value": self.portfolio_value}
        return obs, reward, self.done, info
    
    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Cash: {self.cash:.2f}, Shares: {self.shares}, Portfolio: {self.portfolio_value:.2f}")

if __name__ == "__main__":
    import pandas as pd
    # 測試用模擬資料
    data = pd.DataFrame({
        "date": pd.date_range("2021-01-01", periods=30),
        "market": ["TW"] * 30,
        "open": np.linspace(100, 110, 30),
        "high": np.linspace(101, 111, 30),
        "low": np.linspace(99, 109, 30),
        "close": np.linspace(100, 110, 30),
        "volume": np.random.randint(1000, 2000, 30)
    })
    env = TradingEnv(data, delay=1)
    obs = env.reset()
    print("初始觀察:", obs)
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        env.render()
