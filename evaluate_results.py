import os
import json
import torch
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from model_proposed import PromptTunedBERT, TrackADataset  # import your model and dataset
from dataset import TrackADataset

def load_data(file_path, tokenizer, max_len=128):
    with open(file_path, "r", encoding="utf8") as f:
        samples = [json.loads(line) for line in f]

    dataset = TrackADataset(samples, tokenizer, max_len=max_len)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    return loader

def evaluate(model, data_loader, device):
    model.eval()
    all_labels, all_preds = [], []

    with torch.no_grad():
        for enc_a, enc_b, label in data_loader:
            enc_a = {k: v.to(device) for k, v in enc_a.items()}
            enc_b = {k: v.to(device) for k, v in enc_b.items()}
            label = label.to(device)

            logits = model(enc_a, enc_b)
            pred = torch.argmax(logits, dim=1)

            all_labels.append(label.item())
            all_preds.append(pred.item())

    return np.array(all_labels), np.array(all_preds)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to dataset file (jsonl)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved model checkpoint")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--prompt_len", type=int, default=25)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Load dataset
    loader = load_data(args.data, tokenizer, max_len=args.max_len)

    # Load model
    model = PromptTunedBERT(model_name="bert-base-uncased", prompt_len=args.prompt_len)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)

    # Evaluate
    labels, preds = evaluate(model, loader, device)

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average=None, labels=[0,1])
    macro_f1 = precision_recall_fscore_support(labels, preds, average='macro')[2]
    cm = confusion_matrix(labels, preds)

    print("\n=== Evaluation Results ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision per class: {precision}")
    print(f"Recall per class:    {recall}")
    print(f"F1 per class:        {f1}")
    print(f"Macro F1:            {macro_f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)

    # Save results
    results_df = pd.DataFrame({
        "labels": labels,
        "preds": preds
    })
    results_df.to_csv("evaluation_results.csv", index=False)
    print("\nSaved predictions to evaluation_results.csv")

if __name__ == "__main__":
    main()
