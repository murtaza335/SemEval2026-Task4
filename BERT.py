import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
from tqdm import tqdm

# -----------------------------
# 1. Load dataset
# -----------------------------
df = pd.read_json("dev_track_a.jsonl", lines=True)
df = df.rename(columns={"text_a": "text_a", "text_b": "text_b", "label": "label"})

# -----------------------------
# 2. Tokenizer
# -----------------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

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
            padding='max_length',
            max_length=self.max_len,
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(int(row["label"]), dtype=torch.long)
        }

train_dataset = SimilarityDataset(df, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# -----------------------------
# 3. Model
# -----------------------------
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)
model = model.cuda()

# -----------------------------
# 4. Optimizer + Scheduler
# -----------------------------
epochs = 3
optimizer = AdamW(model.parameters(), lr=2e-5)

total_steps = len(train_loader) * epochs

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0.1 * total_steps,
    num_training_steps=total_steps
)

# -----------------------------
# 5. Training loop
# -----------------------------
for epoch in range(epochs):
    model.train()
    losses = []

    for batch in tqdm(train_loader):
        input_ids = batch["input_ids"].cuda()
        attention_mask = batch["attention_mask"].cuda()
        labels = batch["labels"].cuda()

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

        losses.append(loss.item())

    print(f"Epoch {epoch+1} Loss: {sum(losses)/len(losses):.4f}")

# -----------------------------
# 6. Save model
# -----------------------------
model.save_pretrained("./bert_similarity_model")
tokenizer.save_pretrained("./bert_similarity_model")

print("Model saved!")
