"""
BERT 中文斷詞模型微調腳本
- 讀取 BIO 格式資料
- 使用 transformers 進行 token classification 微調
- 適用於金融詐騙關鍵字強化
"""
import logging
from pathlib import Path
from typing import List, Tuple
from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch

def read_bio_data(filepath: Path) -> Tuple[List[List[str]], List[List[str]]]:
    """讀取 BIO 格式資料，回傳字元序列與標籤序列。"""
    sentences, labels = [], []
    with filepath.open(encoding="utf-8") as f:
        chars, tags = [], []
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                if chars:
                    sentences.append(chars)
                    labels.append(tags)
                    chars, tags = [], []
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            c, t = parts
            chars.append(c)
            tags.append(t)
        if chars:
            sentences.append(chars)
            labels.append(tags)
    return sentences, labels

def bio_to_ids(labels: List[List[str]], label2id: dict) -> List[List[int]]:
    return [[label2id[tag] for tag in seq] for seq in labels]

def main():
    logging.basicConfig(level=logging.INFO)
    data_path = Path("../data/ws_finetune_sample.txt")
    sentences, tags = read_bio_data(data_path)
    label_list = sorted({t for seq in tags for t in seq})
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")
    model = BertForTokenClassification.from_pretrained(
        "ckiplab/bert-base-chinese-ws",
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )
    # 編碼資料
    encodings = tokenizer(["".join(seq) for seq in sentences], is_split_into_words=False, return_offsets_mapping=True, padding=True, truncation=True)
    # labels 對齊 tokenizer word_ids
    aligned_labels = []
    for i, seq in enumerate(sentences):
        word_ids = list(range(len(seq)))  # 每個字一個 word_id
        tokens = tokenizer.tokenize("".join(seq))
        encoding = tokenizer("".join(seq), return_offsets_mapping=True)
        offsets = encoding["offset_mapping"]
        label_ids = []
        label_seq = tags[i]
        char_idx = 0
        for offset in offsets:
            if offset == (0, 0):  # special token
                label_ids.append(-100)
            else:
                if char_idx < len(label_seq):
                    label_ids.append(label2id[label_seq[char_idx]])
                else:
                    label_ids.append(-100)
                char_idx += 1
        aligned_labels.append(label_ids)
    # padding labels
    max_len = max(len(e) for e in encodings["input_ids"])
    for i in range(len(aligned_labels)):
        aligned_labels[i] = aligned_labels[i] + [-100] * (max_len - len(aligned_labels[i]))
    dataset = Dataset.from_dict({
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": aligned_labels
    })
    # 微調參數
    args = TrainingArguments(
        output_dir="finetuned_ws",
        per_device_train_batch_size=4,
        num_train_epochs=10,
        learning_rate=5e-5,
        logging_steps=5,
        save_strategy="epoch",
        report_to=[],
        disable_tqdm=False
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        tokenizer=tokenizer
    )
    trainer.train()
    trainer.save_model("finetuned_ws")
    logging.info("微調完成，模型已儲存於 finetuned_ws/")

if __name__ == "__main__":
    main() 