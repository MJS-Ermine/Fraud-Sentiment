# Fraud-Sentiment

## 專案定位與分工

本倉庫為 [scam-bot](https://github.com/Hina-Lin/scam-bot) 專案的「模型分析與串接」子模組，**專責於金融詐騙對話的中文斷詞、關鍵字標註與理論階段分類模型的微調、推論與批次測試**。  
本倉庫**不包含 API/Linebot 前端與伺服器整合**，僅聚焦於模型本身的訓練、推論與分析，並提供明確的串接介面與資料格式，供主系統（如 scam-bot）呼叫。

---

## 專案目標

- 提供高精度的中文金融詐騙對話斷詞與關鍵字標註模型
- 基於學術理論自動分類詐騙對話階段
- 支援批次資料測試與自動化報告產生
- 明確定義與主系統串接的資料格式與呼叫方式

---

## 主要功能

- **中文斷詞與關鍵字標註**：自動辨識金融詐騙高風險詞彙，支援 BIO 格式資料微調。
- **理論階段分類**：依據詐騙七階段/五階段理論，自動分類對話所屬詐騙流程。
- **批次推論與報告**：可對大量對話資料自動批次分析與分類，產生報告。
- **單元測試**：高覆蓋率 pytest 測試，確保斷詞與關鍵字偵測品質。

---

## 與 scam-bot 的串接關係

- **scam-bot**：負責 LINE Bot 前端、API 伺服器、Webhook、用戶互動
- **Fraud-Sentiment（本倉庫）**：負責模型微調、推論、批次分析，並提供標準化的分析腳本與資料格式
- **串接方式**：主系統將對話資料（JSON 格式）傳遞給本倉庫的推論腳本，取得斷詞、關鍵字標註與理論階段分類結果

---

## 串接資料格式（建議）

主系統應傳遞如下 JSON 給模型分析腳本：

| 欄位名稱         | 型別            | 說明                         |
|------------------|-----------------|------------------------------|
| current_message  | string          | 使用者此輪傳送的訊息         |
| chat_history     | list of strings | 此使用者過去的對話紀錄，依序儲存為陣列 |

### 範例 JSON

```json
{
  "current_message": "最近我對投資有點興趣",
  "chat_history": [
    "你好呀！",
    "你平常都做什麼工作？",
    "你看起來很專業欸",
    "我最近對理財有點好奇",
    "最近我對投資有點興趣"
  ]
}
```

---

## 目錄結構

```plaintext
Fraud-Sentiment/
├── infer_ws.py                  # 單句斷詞與關鍵字標註推論
├── batch_infer.py               # 批次推論與理論階段分類
├── theory_stage_classifier.py   # 理論階段分類模組
├── finetune_ws.py               # 斷詞模型微調腳本
├── word_segmentation_eval.py    # 斷詞評估腳本
├── line_dialog_eval.py          # 模擬對話資料分析腳本
├── finetuned_ws/                # 微調後模型與 tokenizer
├── data/                        # 測試與微調資料
├── tests/                       # 單元測試
├── requirements.txt             # 依賴管理
├── .gitignore
├── README.md
├── LICENSE
```

---

## 安裝與環境建議

建議使用 Python 3.10+，可用 venv/conda 建立虛擬環境。

```bash
# 建立虛擬環境
python -m venv venv
# 啟動虛擬環境
source venv/bin/activate  # Windows: venv\Scripts\activate
# 安裝依賴
pip install -r requirements.txt
```

必要套件（部分範例）：
- ckip-transformers>=0.3.2
- transformers>=4.0.0
- torch>=1.7.0
- datasets>=2.0.0

---

## 使用方式

### 1. 斷詞與關鍵字推論
```bash
python infer_ws.py --text "請輸入待分析對話"
```
- 輸出每句話的斷詞與關鍵字標註。

### 2. 批次推論與理論階段分類
```bash
python batch_infer.py --input data/dialogs.json --output results/report.csv
```
- 批次分析對話檔案，並自動分類詐騙階段。

### 3. 斷詞模型微調
```bash
python finetune_ws.py --config configs/finetune.yaml
```
- 需先準備 BIO 格式資料於 `data/ws_finetune_sample.txt`。

### 4. 斷詞與關鍵字自動評估
```bash
python word_segmentation_eval.py
```
- 自動統計關鍵字命中率，給出微調建議。

### 5. 模擬對話資料分析
```bash
python line_dialog_eval.py --input data/sample_dialog.json
```
- 分析模擬對話資料，產生標註與分類結果。

### 6. 單元測試
```bash
pytest tests/
```

---

## 資料格式說明

`data/ws_finetune_sample.txt` 範例（BIO 格式）：

```text
寶 O
貝 O
匯 B-KEYWORD
款 I-KEYWORD
...
```

空行分隔句子，B-KEYWORD/I-KEYWORD 為關鍵字標註。

---

## 串接建議

- 建議主系統以 JSON 格式呼叫本倉庫的推論腳本，並解析回傳的標註與分類結果
- 如需 API 介面，請於主系統自行包裝（本倉庫僅提供模型與分析腳本）

---

## 技術與工具

- Python 3.10+
- Hugging Face Transformers
- PyTorch
- pandas
- pytest
- ruff

---

## 注意事項

- 本倉庫僅聚焦於模型與分析流程，**不包含 API/Linebot 整合**
- 大型模型檔案請勿直接上傳 GitHub，建議使用 git-lfs 或雲端連結

---

## 授權

本專案採用 MIT License，詳見 LICENSE 檔案。

---

如需與 scam-bot 進行串接或有其他協作需求，請參考主倉庫說明或聯絡專案負責人。