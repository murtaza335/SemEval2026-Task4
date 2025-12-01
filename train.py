# train.py
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer
from data_loader import load_and_split_data, TrackADataset
from model_baseline_B import CrossEncoderBERT

def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for enc_a, enc_b, label in loader:
            enc_a = {k: v.to(device) for k, v in enc_a.items()}
            enc_b = {k: v.to(device) for k, v in enc_b.items()}
            label = label.to(device)

            logits = model(enc_a, enc_b)
            pred = torch.argmax(logits, dim=1)
            correct += int(pred == label)
            total += 1
    acc = correct / total
    print(f"Validation Accuracy: {correct}/{total} = {acc:.4f}")
    return acc


def train_model(train_samples, val_samples, model_name="bert-base-uncased",
                batch_size=4, epochs=10, lr=2e-5, save_path="model.pt"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset = TrackADataset(train_samples, tokenizer)
    val_dataset = TrackADataset(val_samples, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = CrossEncoderBERT(model_name).to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for enc_a, enc_b, label in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            enc_a = {k: v.to(device) for k, v in enc_a.items()}
            enc_b = {k: v.to(device) for k, v in enc_b.items()}
            label = label.to(device)

            logits = model(enc_a, enc_b)
            loss = criterion(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1} — Train Loss: {total_loss/len(train_loader):.4f}")
        acc = evaluate(model, val_loader, device)

        # Save best model
        if acc > best_acc:
            torch.save(model.state_dict(), save_path)
            best_acc = acc
            print(f"Saved best model with accuracy: {best_acc:.4f}")

    print("Training done.")
    return model, tokenizer
