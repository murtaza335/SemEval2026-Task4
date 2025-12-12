# ---- FILE: run_experiments.py ----

import os
import argparse
import random
import json
from collections import Counter
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score
import pandas as pd
from dataset import TrackADataset


from model_proposed import PromptTunedBERT

# ---------------- Dataset ----------------
class TrackADataset(torch.utils.data.Dataset):
    def __init__(self, samples, tokenizer, max_len=128):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        enc_a = self.tokenizer(
            sample['text_a'],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        enc_b = self.tokenizer(
            sample['text_b'],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )

        label = int(sample['text_a_is_closer'])
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
    train_samples = samples[:split]
    val_samples = samples[split:]
    return train_samples, val_samples

def save_history(history, outdir):
    pd.DataFrame(history).to_csv(os.path.join(outdir, "train_history.csv"), index=False)

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
    print(f"Train: {len(train_samples)} Val: {len(val_samples)}")
    print("Label counts (train):", Counter(int(s['text_a_is_closer']) for s in train_samples))

    train_dataset = TrackADataset(train_samples, tokenizer, max_len=args.max_len)
    val_dataset = TrackADataset(val_samples, tokenizer, max_len=args.max_len)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = PromptTunedBERT(model_name="bert-base-uncased", prompt_len=args.prompt_len, freeze_backbone=False, dropout=0.3)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = args.epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps
    )

    scaler = GradScaler(enabled=device.type=="cuda")
    criterion = nn.CrossEntropyLoss()

    best_val = 0.0
    no_improve = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(enumerate(train_loader, 1), total=len(train_loader), desc=f"Epoch {epoch}")

        for step, (enc_a, enc_b, label) in pbar:
            enc_a = {k: v.to(device) for k, v in enc_a.items()}
            enc_b = {k: v.to(device) for k, v in enc_b.items()}
            label = label.to(device)

            with autocast(device_type=device.type):
                logits = model(enc_a, enc_b)
                loss = criterion(logits, label) / args.grad_accum

            scaler.scale(loss).backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            if step % args.grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            running_loss += float(loss.item() * args.grad_accum)
            pbar.set_postfix({"loss": running_loss / step})

        avg_train_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        all_y, all_pred = [], []
        val_running_loss = 0.0
        with torch.no_grad():
            for enc_a, enc_b, label in val_loader:
                enc_a = {k: v.to(device) for k, v in enc_a.items()}
                enc_b = {k: v.to(device) for k, v in enc_b.items()}
                label = label.to(device)

                logits = model(enc_a, enc_b)
                loss = criterion(logits, label)
                val_running_loss += float(loss.item())

                pred = torch.argmax(logits, dim=1)
                all_y.append(int(label.item()))
                all_pred.append(int(pred.item()))

        val_acc = accuracy_score(all_y, all_pred)
        val_loss = val_running_loss / len(val_loader)

        print(f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
        history.append({"epoch": epoch, "train_loss": avg_train_loss, "val_loss": val_loss, "val_acc": val_acc})
        save_history(history, args.outdir)

        # Save best
        if val_acc > best_val:
            best_val = val_acc
            ckpt_path = os.path.join(args.outdir, "best_model.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved best model: {ckpt_path} with val_acc={best_val:.4f}")
            no_improve = 0
        else:
            no_improve += 1

        # Early stopping
        if no_improve >= args.early_stop:
            print(f"No improvement for {args.early_stop} epochs. Early stopping.")
            break

    print("Final validation accuracy (best):", best_val)
    print("Done. Results at:", args.outdir)

if __name__ == "__main__":
    main()
