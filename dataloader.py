# ---- FILE: dataloader.py ----
import json
import random
import torch
from torch.utils.data import Dataset

class TrackADataset(Dataset):
    def __init__(self, samples, tokenizer, max_len=256):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def encode_pair(self, anchor, choice):
        return self.tokenizer(
            anchor,
            choice,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        enc_a = self.encode_pair(s["anchor_text"], s["text_a"])
        enc_b = self.encode_pair(s["anchor_text"], s["text_b"])
        label = torch.tensor(int(s["text_a_is_closer"]), dtype=torch.long)
        return enc_a, enc_b, label

def load_and_split_data(path, train_ratio=0.8):
    with open(path, "r", encoding="utf8") as f:
        samples = [json.loads(line) for line in f]
    random.shuffle(samples)
    split_idx = int(train_ratio * len(samples))
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]
    print(f"Train: {len(train_samples)} Val: {len(val_samples)}")
    return train_samples, val_samples
