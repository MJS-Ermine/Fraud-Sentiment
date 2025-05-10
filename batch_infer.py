from transformers import BertTokenizerFast, BertForTokenClassification
import torch
from pathlib import Path
from theory_stage_classifier import classify_stage
from typing import List

LABELS = ["B-KEYWORD", "I-KEYWORD", "O"]
MODEL_DIR = "finetuned_ws"

model = BertForTokenClassification.from_pretrained(MODEL_DIR)
tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")
model.eval()

def predict(sentence: str) -> List[str]:
    tokens = tokenizer(sentence, return_tensors="pt", return_offsets_mapping=True, truncation=True)
    with torch.no_grad():
        outputs = model(**{k: v for k, v in tokens.items() if k in ["input_ids", "attention_mask"]})
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1).squeeze().tolist()
    offset_mapping = tokens["offset_mapping"].squeeze().tolist()
    result = []
    for idx, (start, end) in enumerate(offset_mapping):
        if start == 0 and end == 0:
            continue
        label = LABELS[preds[idx]] if preds[idx] < len(LABELS) else "O"
        word = sentence[start:end]
        result.append((word, label))
    return result

def batch_infer(filepath: Path):
    with filepath.open(encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    for i, line in enumerate(lines):
        pred = predict(line)
        keywords = {w for w, l in pred if l.startswith("B") or l.startswith("I")}
        stage = classify_stage(keywords)
        print(f"[{i+1}] 原句: {line}")
        print("斷詞標註:", " ".join([f"{w}({l})" for w, l in pred]))
        print(f"理論階段分類: {stage}")
        print("-" * 40)

if __name__ == "__main__":
    # 範例：請將路徑改為你的測試檔案
    batch_infer(Path("../data/[LINE]簡單版對話.txt")) 