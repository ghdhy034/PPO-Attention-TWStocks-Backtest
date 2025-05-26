# strategies/multi_stock_selector.py
import torch
import torch.nn as nn
import torch.optim as optim
from utils.config import STATE_DIM

class MultiStockSelector(nn.Module):
    def __init__(self, candidate_num, state_dim):
        super(MultiStockSelector, self).__init__()
        self.candidate_num = candidate_num
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, stock_states):
        # stock_states: [batch_size, candidate_num, state_dim]
        scores = self.fc(stock_states).squeeze(-1)  # [batch_size, candidate_num]
        sorted_indices = torch.argsort(scores, descending=True, dim=1)
        return sorted_indices, scores

def select_top_stocks(stock_states, top_k):
    """
    stock_states: numpy array，形狀 [candidate_num, state_dim]
    返回 top_k 支股票的索引。
    """
    import torch
    stock_states_tensor = torch.FloatTensor(stock_states).unsqueeze(0)  # [1, candidate_num, state_dim]
    model = MultiStockSelector(candidate_num=stock_states_tensor.shape[1], state_dim=stock_states_tensor.shape[2])
    model.eval()
    with torch.no_grad():
        sorted_indices, scores = model(stock_states_tensor)
    sorted_indices = sorted_indices.squeeze(0).cpu().numpy()
    return sorted_indices[:top_k]
