# evaluate.py
import torch
from torch.utils.data import DataLoader
from data_loader import TrackADataset
from model_baseline_B import CrossEncoderBERT
from transformers import AutoTokenizer

def evaluate_model(model_path, val_samples, model_name="bert-base-uncased"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    val_dataset = TrackADataset(val_samples, tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = CrossEncoderBERT(model_name).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    correct, total = 0, 0
    with torch.no_grad():
        for enc_a, enc_b, label in val_loader:
            enc_a = {k: v.to(device) for k, v in enc_a.items()}
            enc_b = {k: v.to(device) for k, v in enc_b.items()}
            label = label.to(device)

            logits = model(enc_a, enc_b)
            pred = torch.argmax(logits, dim=1)
            correct += int(pred == label)
            total += 1

    acc = correct / total
    print(f"Final Validation Accuracy: {correct}/{total} = {acc:.4f}")
