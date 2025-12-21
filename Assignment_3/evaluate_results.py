# ---- FILE: evaluate_results.py ----

import os
import json
import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from model_proposed import PromptTunedBERT, TrackADataset  # import your model & dataset


# ---------------- Helpers ----------------
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


# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to dataset file (jsonl)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved model checkpoint")
    parser.add_argument("--history_csv", type=str, required=True, help="Path to train_history.csv")
    parser.add_argument("--outdir", type=str, default="results/bert_eval")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--prompt_len", type=int, default=25)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
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
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average=None, labels=[0, 1])
    macro_f1 = precision_recall_fscore_support(labels, preds, average='macro')[2]
    cm = confusion_matrix(labels, preds)

    print("\n=== Evaluation Results ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision per class: {precision}")
    print(f"Recall per class:    {recall}")
    print(f"F1 per class:        {f1}")
    print(f"Macro F1:            {macro_f1:.4f}")
    print("\nConfusion Matrix:\n", cm)

    # Save confusion matrix plot
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap=plt.cm.Blues)
    cm_path = os.path.join(args.outdir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Saved confusion matrix plot: {cm_path}")

    # Load train history
    history = pd.read_csv(args.history_csv)

    # Plot Loss vs Epoch
    plt.figure()
    plt.plot(history["epoch"], history["train_loss"], label="Train Loss")
    plt.plot(history["epoch"], history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs Epoch")
    plt.legend()
    loss_path = os.path.join(args.outdir, "loss_vs_epoch.png")
    plt.savefig(loss_path)
    plt.close()
    print(f"Saved Loss vs Epoch plot: {loss_path}")

    # Plot Accuracy vs Epoch
    plt.figure()
    plt.plot(history["epoch"], history["val_acc"], label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy vs Epoch")
    plt.legend()
    acc_path = os.path.join(args.outdir, "val_acc_vs_epoch.png")
    plt.savefig(acc_path)
    plt.close()
    print(f"Saved Accuracy vs Epoch plot: {acc_path}")

    # Save predictions CSV
    results_df = pd.DataFrame({"labels": labels, "preds": preds})
    results_csv = os.path.join(args.outdir, "evaluation_results.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"Saved predictions CSV: {results_csv}")


if __name__ == "__main__":
    main()
