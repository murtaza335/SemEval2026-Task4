# model_baseline_A.py
import torch
import torch.nn as nn
from transformers import AutoModel

class CrossEncoderBERT(nn.Module):
    def __init__(self, model_name="bert-base-uncased"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        # Combine CLS_A, CLS_B, and their difference
        self.classifier = nn.Linear(hidden*3, 2)

    def forward(self, enc_a, enc_b):
        # Encode A
        out_a = self.encoder(
            input_ids=enc_a["input_ids"].squeeze(1),
            attention_mask=enc_a["attention_mask"].squeeze(1)
        )
        cls_a = out_a.last_hidden_state[:, 0]

        # Encode B
        out_b = self.encoder(
            input_ids=enc_b["input_ids"].squeeze(1),
            attention_mask=enc_b["attention_mask"].squeeze(1)
        )
        cls_b = out_b.last_hidden_state[:, 0]

        diff = cls_a - cls_b
        combined = torch.cat([cls_a, cls_b, diff], dim=1)

        return self.classifier(combined)
