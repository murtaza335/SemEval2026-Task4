# ---- FILE: run_experiments.py ----

import os
import argparse
import random
import json
from collections import Counter
from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score

from model_proposed import PromptTunedBERT

# ---------------- Dataset ----------------
class TrackADataset(torch.utils.data.Dataset):
    def __init__(self, samples, tokenizer, max_len=128):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def _safe_text(self, value):
        if value is None or str(value).strip() == "":
            return "[NO TEXT]"
        return str(value).strip()

    def __getitem__(self, idx):
        sample = self.samples[idx]
        text_a = self._safe_text(sample.get("text_a"))
        text_b = self._safe_text(sample.get("text_b"))

        enc_a = self.tokenizer(
            text=text_a,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        enc_b = self.tokenizer(
            text=text_b,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        label = int(sample.get("text_a_is_closer", 0))
        enc_a = {k: v.squeeze(0) for k, v in enc_a.items()}
        enc_b = {k: v.squeeze(0) for k, v in enc_b.items()}

        return enc_a, enc_b, torch.tensor(label, dtype=torch.long)

# ---------------- Helpers ----------------
def load_and_split_data(file_path, split_ratio=0.8, seed=42):
    with open(file_path, "r", encoding="utf8") as f:
        samples = [json.loads(line) for line in f]

    random.seed(seed)
    random.shuffle(samples)
    split = int(len(samples) * split_ratio)
    return samples[:split], samples[split:]

def save_history(history, outdir):
    df = pd.DataFrame(history)
    df.to_csv(os.path.join(outdir, "train_history.csv"), index=False)

# ---------------- Training ----------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    set_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--outdir", type=str, default="results/bert_proposed")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--early_stop", type=int, default=3)
    parser.add_argument("--prompt_len", type=int, default=25)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    train_samples, val_samples = load_and_split_data(args.data, split_ratio=0.8)

    print(f"Train: {len(train_samples)} | Val: {len(val_samples)}")
    print("Label counts (train):", Counter(int(s.get("text_a_is_closer", 0)) for s in train_samples))

    train_dataset = TrackADataset(train_samples, tokenizer, args.max_len)
    val_dataset = TrackADataset(val_samples, tokenizer, args.max_len)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = PromptTunedBERT(
        model_name="bert-base-uncased",
        prompt_len=args.prompt_len,
        freeze_backbone=False,
        dropout=0.3
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = args.epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    scaler = GradScaler(enabled=device.type == "cuda")
    criterion = nn.CrossEntropyLoss()

    best_val = 0.0
    no_improve = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for step, (enc_a, enc_b, label) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}"), 1):
            enc_a = {k: v.to(device) for k, v in enc_a.items()}
            enc_b = {k: v.to(device) for k, v in enc_b.items()}
            label = label.to(device)

            with autocast(device_type=device.type):
                logits = model(enc_a, enc_b)
                loss = criterion(logits, label) / args.grad_accum

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if step % args.grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            running_loss += loss.item() * args.grad_accum

        avg_train_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        y_true, y_pred = [], []
        val_loss_total = 0.0
        with torch.no_grad():
            for enc_a, enc_b, label in val_loader:
                enc_a = {k: v.to(device) for k, v in enc_a.items()}
                enc_b = {k: v.to(device) for k, v in enc_b.items()}
                label = label.to(device)

                logits = model(enc_a, enc_b)
                val_loss_total += float(criterion(logits, label))
                pred = torch.argmax(logits, dim=1)

                y_true.append(label.item())
                y_pred.append(pred.item())

        val_acc = accuracy_score(y_true, y_pred)
        val_loss = val_loss_total / len(val_loader)
        print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc
        })
        save_history(history, args.outdir)

        # Save best model
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), os.path.join(args.outdir, "best_model.pt"))
            no_improve = 0
            print(f"Saved best model with Val Acc: {best_val:.4f}")
        else:
            no_improve += 1

        if no_improve >= args.early_stop:
            print("Early stopping triggered.")
            break

    print("Training complete. Best Val Acc:", best_val)
    print("Results & history saved at:", args.outdir)

if __name__ == "__main__":
    main()