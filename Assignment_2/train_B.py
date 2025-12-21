
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import torch.nn.functional as F
from model_baseline_B import CrossEncoderBERT
from data_loader import load_and_split

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
EPOCHS = 3
LR = 2e-5
MODEL_NAME = "bert-base-uncased"


# -------------------------
# Dataset
# -------------------------
class PairDataset(Dataset):
    def __init__(self, texts_a, texts_b, labels, tokenizer, max_len=128):
        self.texts_a = texts_a
        self.texts_b = texts_b
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def encode(self, text):
        return self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

    def __getitem__(self, idx):
        enc_a = self.encode(self.texts_a[idx])
        enc_b = self.encode(self.texts_b[idx])

        return {
            "enc_a": {k: v.squeeze(0) for k, v in enc_a.items()},
            "enc_b": {k: v.squeeze(0) for k, v in enc_b.items()},
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }


# -------------------------
# Training Loop
# -------------------------
def train_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0

    for batch in loader:
        enc_a = {k: v.to(DEVICE) for k, v in batch["enc_a"].items()}
        enc_b = {k: v.to(DEVICE) for k, v in batch["enc_b"].items()}
        labels = batch["label"].to(DEVICE)

        optimizer.zero_grad()
        logits = model(enc_a, enc_b)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def eval_epoch(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            enc_a = {k: v.to(DEVICE) for k, v in batch["enc_a"].items()}
            enc_b = {k: v.to(DEVICE) for k, v in batch["enc_b"].items()}
            labels = batch["label"].to(DEVICE)

            logits = model(enc_a, enc_b)
            preds = torch.argmax(logits, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    print("Loading dataset...")
    train_anchor, val_anchor, train_A, val_A, train_B, val_B, train_y, val_y = \
        load_and_split("dev_track_a.jsonl")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = PairDataset(train_A, train_B, train_y, tokenizer)
    val_ds   = PairDataset(val_A, val_B, val_y, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = CrossEncoderBERT(MODEL_NAME).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    print("Training model...")
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer)
        val_acc = eval_epoch(model, val_loader)

        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Acc: {val_acc:.4f}")

    print("Saving model...")
    torch.save(model.state_dict(), "results/model_baseline_B.pt")
