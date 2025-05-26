# PPO-Attention-TWStocks-Backtest

本專案運用強化學習中的 Proximal Policy Optimization（PPO）演算法，結合注意力機制（Attention），開發並回測適用於台灣股票市場的交易策略。透過模組化設計，整合資料擷取、模型訓練、策略回測與即時交易等功能，提供一個完整的量化交易研究框架。

## 專案亮點

* **強化學習應用**：利用 PPO 演算法訓練交易代理，學習在台灣股市中獲利的策略。
* **注意力機制整合**：引入 Attention 模型，提升對關鍵市場訊號的捕捉能力。
* **模組化架構**：清晰的目錄結構，便於擴展與維護。
* **回測成果豐富**：提供多次回測結果，涵蓋與訓練集相同及不同的股票，報酬率介於五年內 300% 至 500%。

## 專案結構

```bash
.
├── fetch_data/             # 資料擷取模組
│   └── fetch_historical_data.py
├── models/                 # 模型定義與訓練
├── strategies/             # 交易策略實作
├── backtesting/            # 策略回測模組
├── realtime_trading/       # 即時交易模組
├── results/                # 回測結果與圖表
├── utils/                  # 輔助工具函式
├── main.py                 # 主程式入口
└── requirements.txt        # 相依套件清單
```

## 快速開始

1. **環境建置**

   建議使用虛擬環境（如 `venv` 或 `conda`）：

   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows 使用者請執行 venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **資料擷取**

   執行以下指令以擷取台灣股票歷史資料：

   ```bash
   python fetch_data/fetch_historical_data.py
   ```

3. **執行主流程**

   透過主程式進行資料更新、模型訓練與策略回測：

   ```bash
   python main.py
   ```

## 訓練技術詳解

### Proximal Policy Optimization（PPO）

PPO 是一種先進的策略梯度方法，透過限制策略更新的幅度，避免策略偏離過大，提升訓練的穩定性與效率。

### 注意力機制（Attention）

引入 Attention 模型，使得代理能夠專注於關鍵的市場特徵，提高對重要訊號的辨識能力，進而提升交易決策的品質。

### 模型架構

模型結合 LSTM 與 Attention 結構，處理時間序列資料，捕捉長期依賴關係與關鍵特徵，提升預測與決策能力。

## 回測成果 (注意!尚未加入手續費及證交稅)

專案提供多次回測結果，涵蓋與訓練集相同及不同的股票。每次回測包含以下三個檔案：

* `trade_log.csv`：交易紀錄
* `capital_log.csv`：資本變化紀錄
* `prev_holdings_pct.png`：持股比例變化圖

回測期間約為五年，報酬率介於 300% 至 500%，顯示模型在不同市場條件下的穩健性與高報酬潛力。


