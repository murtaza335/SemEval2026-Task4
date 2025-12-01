# main.py
import csv
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn as nn
from transformers import AutoTokenizer
from data_loader import load_and_split_data, TrackADataset
from model_baseline_A import CrossEncoderBERT
import matplotlib.pyplot as plt
import os

# Paths
DATA_FILE = "/content/drive/MyDrive/ai project/dev_track_a.jsonl"
MODEL_SAVE_PATH = "results/best_model.pt"
METRICS_CSV = "results/metrics.csv"
PLOTS_DIR = "plots/"

os.makedirs("results", exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Hyperparameters
MODEL_NAME = "bert-base-uncased"
BATCH_SIZE = 4
EPOCHS = 10
LR = 2e-5
MAX_LEN = 256

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
train_samples, val_samples = load_and_split_data(DATA_FILE, split_ratio=0.9)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Datasets
train_dataset = TrackADataset(train_samples, tokenizer, MAX_LEN)
val_dataset = TrackADataset(val_samples, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# Model
model = CrossEncoderBERT(MODEL_NAME).to(DEVICE)
optimizer = AdamW(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# Metrics storage
metrics = []

best_acc = 0

# Training loop
for epoch in range(1, EPOCHS+1):
    model.train()
    total_loss = 0

    for enc_a, enc_b, label in tqdm(train_loader, desc=f"Epoch {epoch}"):
        enc_a = {k: v.to(DEVICE) for k, v in enc_a.items()}
        enc_b = {k: v.to(DEVICE) for k, v in enc_b.items()}
        label = label.to(DEVICE)

        logits = model(enc_a, enc_b)
        loss = criterion(logits, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)

    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for enc_a, enc_b, label in val_loader:
            enc_a = {k: v.to(DEVICE) for k, v in enc_a.items()}
            enc_b = {k: v.to(DEVICE) for k, v in enc_b.items()}
            label = label.to(DEVICE)

            logits = model(enc_a, enc_b)
            pred = torch.argmax(logits, dim=1)
            correct += int(pred == label)
            total += 1

    val_acc = correct / total
    print(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Validation Accuracy={val_acc:.4f}")

    # Save metrics
    metrics.append({
        "epoch": epoch,
        "train_loss": avg_train_loss,
        "val_acc": val_acc
    })

    # Save best model
    if val_acc > best_acc:
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        best_acc = val_acc
        print(f"Saved best model at epoch {epoch} with val_acc={val_acc:.4f}")

# Save metrics to CSV
with open(METRICS_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_acc"])
    writer.writeheader()
    for row in metrics:
        writer.writerow(row)

print(f"Metrics saved to {METRICS_CSV}")

# Plot metrics
epochs = [m["epoch"] for m in metrics]
train_loss = [m["train_loss"] for m in metrics]
val_acc = [m["val_acc"] for m in metrics]

plt.figure(figsize=(10,5))
plt.plot(epochs, train_loss, label="Train Loss", marker='o')
plt.plot(epochs, val_acc, label="Validation Accuracy", marker='x')
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("Training Metrics")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(PLOTS_DIR, "training_metrics.pdf"))
plt.show()
