# strategies/ppo_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils.config import (
    PPO_LEARNING_RATE, PPO_CLIP_EPSILON, PPO_DISCOUNT_FACTOR,
    PPO_VALUE_LOSS_COEF, PPO_ENTROPY_COEF, PPO_MAX_GRAD_NORM,
    D_MODEL, NHEAD, NUM_LAYERS, DROPOUT,
    USE_GAE, GAE_LAMBDA, NORMALIZE_ADVANTAGE
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PPONetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        """
        Transformer-based PPO network.
        將輸入 (batch_size, state_dim) 視為序列，每個 token 為一支股票的收盤價，
        先通過線性嵌入將其投影至 D_MODEL 維度，再送入 Transformer Encoder，
        最後對序列做平均池化，分別通過 Actor 與 Critic 頭部生成動作均值與狀態價值。
        """
        super(PPONetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 將每個價格 (1D) 投影到 D_MODEL 維度
        self.embedding = nn.Linear(1, D_MODEL)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=NHEAD, dropout=DROPOUT, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)
        # Actor head: 從聚合後的表示映射到動作均值，使用 Tanh 限制範圍
        self.actor = nn.Linear(D_MODEL, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        # Critic head: 從聚合後的表示映射到狀態價值
        self.critic = nn.Linear(D_MODEL, 1)
    
    def forward(self, x):
        # x shape: (batch_size, state_dim)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        # 將 x 轉換為 (batch_size, state_dim, 1)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        # 執行嵌入: (batch_size, state_dim, D_MODEL)
        x = self.embedding(x)
        # Transformer Encoder: (batch_size, state_dim, D_MODEL)
        x = self.transformer_encoder(x)
        # 平均池化: 得到 (batch_size, D_MODEL)
        x = x.mean(dim=1)
        # Actor head: (batch_size, action_dim)
        action_mean = torch.tanh(self.actor(x))
        # Critic head: (batch_size, 1)
        value = self.critic(x)
        # 將 actor_log_std 擴展至與 action_mean 同形狀
        action_log_std = self.actor_log_std.expand_as(action_mean)
        return action_mean, action_log_std, value

class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.policy = PPONetwork(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=PPO_LEARNING_RATE)
    
    def select_action(self, state):
        state = torch.FloatTensor(state).to(device)
        action_mean, action_log_std, _ = self.policy(state)
        std = torch.exp(action_log_std)
        normal = torch.distributions.Normal(action_mean, std)
        action = normal.sample()
        action_log_prob = normal.log_prob(action).sum(dim=-1)
        return action.detach().cpu().numpy(), action_log_prob.detach().cpu().numpy()
    
    def evaluate_actions(self, states, actions):
        action_mean, action_log_std, state_values = self.policy(states)
        std = torch.exp(action_log_std)
        normal = torch.distributions.Normal(action_mean, std)
        action_log_probs = normal.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = normal.entropy().sum(dim=-1)
        return action_log_probs, entropy, state_values
    
    def update(self, trajectories):
        # 將 trajectories 中資料轉換成 tensor
        states = torch.from_numpy(np.array(trajectories['states'])).float().to(device)
        actions = torch.from_numpy(np.array(trajectories['actions'])).float().to(device)
        old_log_probs = torch.from_numpy(np.array(trajectories['log_probs'])).float().to(device)
        
        # 如果使用 GAE，則根據 GAE 計算優勢與 returns
        T = len(trajectories['rewards'])
        if USE_GAE:
            with torch.no_grad():
                # 計算所有 state 的 value 預測 (shape: T, 1)
                _, _, state_values = self.policy(states)
            state_values = state_values.squeeze(1).cpu().numpy()  # 轉為 1D numpy array
            advantages = np.zeros(T, dtype=np.float32)
            gae = 0
            for t in reversed(range(T)):
                # 如果不是最後一步，則 next_value = V(s[t+1])，否則 0
                next_value = state_values[t+1] if t < T - 1 else 0
                delta = trajectories['rewards'][t] + PPO_DISCOUNT_FACTOR * next_value * (1 - trajectories['dones'][t]) - state_values[t]
                gae = delta + PPO_DISCOUNT_FACTOR * GAE_LAMBDA * (1 - trajectories['dones'][t]) * gae
                advantages[t] = gae
            returns = advantages + state_values
            # 如果需要標準化優勢
            if NORMALIZE_ADVANTAGE:
                adv_mean = advantages.mean()
                adv_std = advantages.std() + 1e-8
                advantages = (advantages - adv_mean) / adv_std
        else:
            # 傳統回報計算
            returns = []
            G = 0
            for r, d in zip(reversed(trajectories['rewards']), reversed(trajectories['dones'])):
                G = r + PPO_DISCOUNT_FACTOR * G * (1 - d)
                returns.insert(0, G)
            returns = np.array(returns)
            advantages = returns.copy()
        
        # 轉換 returns 與 advantages 為 tensor (shape: T, 1)
        returns = torch.from_numpy(returns).float().unsqueeze(1).to(device)
        advantages = torch.from_numpy(advantages).float().unsqueeze(1).to(device)
        
        new_log_probs, entropy, state_values = self.evaluate_actions(states, actions)
        ratios = torch.exp(new_log_probs - old_log_probs)
        
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - PPO_CLIP_EPSILON, 1 + PPO_CLIP_EPSILON) * advantages
        
        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = ((returns - state_values) ** 2).mean()
        entropy_loss = -entropy.mean()
        
        loss = actor_loss + PPO_VALUE_LOSS_COEF * critic_loss + PPO_ENTROPY_COEF * entropy_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), PPO_MAX_GRAD_NORM)
        self.optimizer.step()
        torch.cuda.empty_cache()
        return loss.item()
