import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# ==========================================
# 1. LOAD DATASET
# ==========================================
print("Loading dataset...")
df = pd.read_json("dev_track_a.jsonl", lines=True)

# Ensure correct column names exist
df = df.rename(columns={"text_a": "text_a", "text_b": "text_b", "label": "label"})

# 80/20 split
train_df, val_df = train_test_split(
    df, test_size=0.2, random_state=42, shuffle=True
)

print(f"Train size: {len(train_df)} | Val size: {len(val_df)}")

# ==========================================
# 2. DATASET CLASS
# ==========================================
class SimilarityDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=256):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        encoding = self.tokenizer(
            row["text_a"],
            row["text_b"],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(int(row["label"]), dtype=torch.long)
        }

# ==========================================
# 3. INITIALIZE TOKENIZER + DATASETS
# ==========================================
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

train_dataset = SimilarityDataset(train_df, tokenizer)
val_dataset = SimilarityDataset(val_df, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# ==========================================
# 4. MODEL
# ==========================================
print("Loading BERT model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)
model = model.to(device)

# ==========================================
# 5. OPTIMIZER + SCHEDULER
# ==========================================
epochs = 3
optimizer = AdamW(model.parameters(), lr=2e-5)

total_steps = len(train_loader) * epochs

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

# ==========================================
# 6. VALIDATION FUNCTION
# ==========================================
def evaluate(model, data_loader):
    model.eval()
    preds = []
    trues = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            logits = outputs.logits
            pred = torch.argmax(logits, axis=-1)

            preds.extend(pred.cpu().numpy())
            trues.extend(labels.cpu().numpy())

    acc = accuracy_score(trues, preds)
    f1 = f1_score(trues, preds)

    return acc, f1

# ==========================================
# 7. TRAINING LOOP
# ==========================================
print("Starting training...")

for epoch in range(epochs):
    model.train()
    epoch_losses = []

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        epoch_losses.append(loss.item())

    # Validation
    val_acc, val_f1 = evaluate(model, val_loader)

    print("\n========== Epoch Summary ==========")
    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Train Loss: {sum(epoch_losses)/len(epoch_losses):.4f}")
    print(f"Val Accuracy: {val_acc:.4f}")
    print(f"Val F1 Score: {val_f1:.4f}")
    print("===================================\n")

# ==========================================
# 8. SAVE MODEL
# ==========================================
print("Saving model...")
model.save_pretrained("./bert_similarity_model")
tokenizer.save_pretrained("./bert_similarity_model")

print("\nTraining complete. Model saved in ./bert_similarity_model")
