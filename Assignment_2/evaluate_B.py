
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from model_baseline_B import CrossEncoderBERT
from data_loader import load_and_split

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "bert-base-uncased"
BATCH_SIZE = 8


# -------------------------
# Dataset (same as training)
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
# Evaluation
# -------------------------
def evaluate(model, loader):
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
    print("Loading validation data...")
    _, val_anchor, _, val_A, _, val_B, _, val_y = \
        load_and_split("dev_track_a.jsonl")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    val_ds = PairDataset(val_A, val_B, val_y, tokenizer)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = CrossEncoderBERT(MODEL_NAME).to(DEVICE)
    model.load_state_dict(torch.load("results/model_baseline_B.pt", map_location=DEVICE))

    val_acc = evaluate(model, val_loader)
    print(f"Baseline-B Validation Accuracy: {val_acc:.4f}")
