
import torch
import torch.nn as nn
from transformers import AutoModel

class CrossEncoderBERT(nn.Module):
    """
    Baseline-B: Dual BERT encoder with CLS comparison
    """
    def __init__(self, model_name="bert-base-uncased"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size

        # CLS_A, CLS_B, CLS_A - CLS_B
        self.classifier = nn.Linear(hidden * 3, 2)

    def encode(self, enc):
        out = self.encoder(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"]
        )
        return out.last_hidden_state[:, 0]  # CLS token

    def forward(self, enc_a, enc_b):
        cls_a = self.encode(enc_a)
        cls_b = self.encode(enc_b)

        diff = cls_a - cls_b
        combined = torch.cat([cls_a, cls_b, diff], dim=1)

        return self.classifier(combined)
