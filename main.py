# main.py
import os
import time
import torch
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import argparse
from utils.config import (
    STOCK_IDS, DEPLOYMENT_STOCK_IDS, START_DATE, MODEL_SAVE_PATH, INITIAL_CAPITAL,
    TOTAL_TRAINING_STEPS, UPDATE_INTERVAL, PPO_BATCH_SIZE, PPO_EPOCHS, PPO_DISCOUNT_FACTOR,
    REWARD_SCALE, STOP_LOSS_THRESHOLD
)
from utils.data_fetcher import load_stock_data_from_db

def update_data():
    print("=== 資料更新開始 ===")
    import fetch_data.fetch_historical_data as fd
    fd.main()
    print("=== 資料更新完成 ===")

def train_joint_model():
    print("=== 端到端聯合訓練開始 ===")
    from strategies.multi_stock_trading_env import MultiStockTradingEnv
    data_dict = {}
    for sid in STOCK_IDS:
        df = load_stock_data_from_db(sid, start_date="2020-01-01")
        if df.empty:
            print(f"股票 {sid} 資料不足，跳過")
        else:
            data_dict[sid] = df
    if len(data_dict) == 0:
        print("無足夠候選股票資料，結束訓練")
        return
    env = MultiStockTradingEnv(data_dict, initial_capital=INITIAL_CAPITAL, delay=1)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print(f"使用 {action_dim} 支股票進行聯合訓練, 狀態向量維度: {obs_dim}")
    
    from strategies.ppo_agent import PPOAgent
    agent = PPOAgent(obs_dim, action_dim)
    
    local_update_interval = 1024
    total_steps = 0
    episode = 0
    trajectory = {"states": [], "actions": [], "log_probs": [], "rewards": [], "dones": []}
    training_log = []
    
    while total_steps < TOTAL_TRAINING_STEPS:
        state = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action, log_prob = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            trajectory["states"].append(state)
            trajectory["actions"].append(action)
            trajectory["log_probs"].append(log_prob)
            trajectory["rewards"].append(reward)
            trajectory["dones"].append(float(done))
            state = next_state
            episode_reward += reward
            total_steps += 1
            
            if total_steps % local_update_interval == 0:
                returns = []
                G = 0
                for r, d in zip(reversed(trajectory["rewards"]), reversed(trajectory["dones"])):
                    G = r + PPO_DISCOUNT_FACTOR * G * (1 - d)
                    returns.insert(0, G)
                returns = np.array(returns)
                advantages = returns.copy()
                trajectory["returns"] = returns
                trajectory["advantages"] = advantages
                loss = agent.update(trajectory)
                print(f"更新 {total_steps} 步, Loss: {loss:.4f}")
                training_log.append({
                    "Step": total_steps,
                    "Loss": loss,
                    "EpisodeReward": episode_reward
                })
                trajectory = {"states": [], "actions": [], "log_probs": [], "rewards": [], "dones": []}
        episode += 1
        print(f"第 {episode} 集, 累計步數: {total_steps}, 本集獎勵: {episode_reward:.2f}")
    
    df_train_log = pd.DataFrame(training_log)
    df_train_log.to_csv("training_log.csv", index=False, float_format="%.2f", quoting=csv.QUOTE_ALL)
    
    model_path = os.path.join(MODEL_SAVE_PATH, "joint_ppo_model.pth")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(agent.policy.state_dict(), model_path)
    print(f"聯合模型已儲存到 {model_path}")
    print("=== 端到端聯合訓練完成 ===")

def backtest_joint_model(output_dir):
    print("=== Joint Model Backtesting Start ===")
    from strategies.multi_stock_trading_env import MultiStockTradingEnv
    data_dict = {}
    for sid in DEPLOYMENT_STOCK_IDS:
        df = load_stock_data_from_db(sid, start_date="2020-01-01")
        if df.empty:
            print(f"Stock {sid} data insufficient, skipped")
        else:
            data_dict[sid] = df
    if len(data_dict) == 0:
        print("No sufficient candidate stock data for backtesting")
        return
    env = MultiStockTradingEnv(data_dict, initial_capital=INITIAL_CAPITAL, delay=1)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    from strategies.ppo_agent import PPOAgent
    agent = PPOAgent(obs_dim, action_dim)
    model_path = os.path.join(MODEL_SAVE_PATH, "joint_ppo_model.pth")
    if os.path.exists(model_path):
        agent.policy.load_state_dict(torch.load(model_path, map_location=torch.device("cpu"), weights_only=True))
        print("Successfully loaded joint model weights.")
    else:
        print("Joint model weights not found, backtesting aborted.")
        return

    capital_logs = []
    trade_logs = []
    state = env.reset()
    done = False
    while not done:
        action, _ = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        capital_logs.append(info["capital_log"])
        trade_logs.append(info["trade_log"])  # 此處可根據需求進一步整合交易記錄
        state = next_state

    df_capital = pd.DataFrame(capital_logs)
    df_trade = pd.DataFrame(trade_logs)
    
    initial_cash = df_capital.loc[0, "PrevCash"]
    df_capital["PrevHoldings_Pct"] = (df_capital["PrevHoldings"] / initial_cash).round(4)
    df_capital = df_capital[["Date", "PrevHoldings", "PrevHoldings_Pct", "PrevCash", "TodayClosingHoldings", "TodayClosingCash"]]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    capital_csv_path = os.path.join(output_dir, "capital_log.csv")
    trade_csv_path = os.path.join(output_dir, "trade_log.csv")
    chart_path = os.path.join(output_dir, "prev_holdings_pct.png")
    
    df_capital.to_csv(capital_csv_path, index=False, float_format="%.2f", quoting=csv.QUOTE_ALL)
    df_trade.to_csv(trade_csv_path, index=False, float_format="%.2f", quoting=csv.QUOTE_ALL)
    
    plt.close("all")
    plt.figure(figsize=(12, 6))
    plt.plot(df_capital["Date"], df_capital["PrevHoldings_Pct"], marker="o", linestyle="-", label="PrevHoldings_Pct")
    plt.xlabel("Date")
    plt.ylabel("PrevHoldings_Pct (Ratio)")
    plt.title("PrevHoldings_Pct Over Time")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid()
    plt.savefig(chart_path, dpi=300, bbox_inches="tight")
    
    print(f"Capital log saved to {capital_csv_path}")
    print(f"Trade log saved to {trade_csv_path}")
    print(f"Chart saved to {chart_path}")
    print("=== Joint Model Backtesting Complete ===")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true", help="更新資料")
    parser.add_argument("--train", action="store_true", help="訓練模型")
    parser.add_argument("--backtest", action="store_true", help="執行回測")
    args = parser.parse_args()

    print("========== 主流程開始 ==========")

    if args.update:
        update_data()
    if args.train:
        train_joint_model()
    if args.backtest:
        output = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "PROJECT_ROOT/results/result")
        backtest_joint_model(output)

    print("========== 主流程結束 ==========")

    
    # for i in range(1):
    #     # 指定輸出結果的資料夾 (例如 PROJECT_ROOT/results)
    #     output = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", f"PROJECT_ROOT/results/results_{i+1}")
    #     if not os.path.exists(output):
    #         os.makedirs(output)
    #     backtest_joint_model(output)
