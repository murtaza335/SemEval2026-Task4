# ---- FILE: model_proposed.py ----

import torch
import torch.nn as nn
from transformers import AutoModel
from dataset import TrackADataset

class PromptTunedBERT(nn.Module):
    """
    Cross-encoder with small learnable prompt embeddings inserted after the [CLS] token.
    Takes enc_a and enc_b dicts (with 'input_ids' and 'attention_mask' tensors).
    Outputs logits for 2 classes.
    """

    def __init__(self, model_name="bert-base-uncased", prompt_len=25, freeze_backbone=False, dropout=0.2):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.hidden = self.backbone.config.hidden_size
        self.prompt_len = prompt_len

        # Prompt embeddings (learnable)
        self.prompt_embeddings = nn.Parameter(torch.randn(prompt_len, self.hidden) * 0.02)

        # classifier on [cls_a, cls_b, cls_a - cls_b]
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.hidden * 3, 2)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def _apply_prompts(self, input_ids, attention_mask):
        """
        Build inputs_embeds by taking the backbone word embeddings and inserting prompt embeddings
        after the CLS token. Expects input_ids shape (batch, seq_len).
        Returns: inputs_embeds (batch, seq_len + prompt_len, hidden),
                 new_attention_mask (batch, seq_len + prompt_len)
        """
        embeds = self.backbone.get_input_embeddings()(input_ids)
        bsz = embeds.size(0)

        prompt = self.prompt_embeddings.unsqueeze(0).expand(bsz, -1, -1)

        cls_token = embeds[:, :1, :]
        rest = embeds[:, 1:, :]

        inputs_embeds = torch.cat([cls_token, prompt, rest], dim=1)

        prompt_mask = torch.ones(bsz, self.prompt_len, dtype=attention_mask.dtype, device=attention_mask.device)
        new_attention_mask = torch.cat([attention_mask[:, :1], prompt_mask, attention_mask[:, 1:]], dim=1)

        return inputs_embeds, new_attention_mask

    def encode_with_prompts(self, enc):
        input_ids = enc["input_ids"].squeeze(1)
        attention_mask = enc["attention_mask"].squeeze(1)

        inputs_embeds, new_attention_mask = self._apply_prompts(input_ids, attention_mask)

        out = self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=new_attention_mask,
            return_dict=True
        )

        cls_emb = out.last_hidden_state[:, 0, :]  # CLS token embedding
        return cls_emb

    def forward(self, enc_a, enc_b):
        cls_a = self.encode_with_prompts(enc_a)
        cls_b = self.encode_with_prompts(enc_b)

        diff = cls_a - cls_b
        combined = torch.cat([cls_a, cls_b, diff], dim=1)
        combined = self.dropout(combined)

        logits = self.classifier(combined)
        return logits