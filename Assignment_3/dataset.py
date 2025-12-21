# ---- FILE: dataset.py ----
import torch

class TrackADataset(torch.utils.data.Dataset):
    def __init__(self, samples, tokenizer, max_len=128):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        enc_a = self.tokenizer(
            sample['text_a'],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        enc_b = self.tokenizer(
            sample['text_b'],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )

        # Convert label to int
        label = int(sample['text_a_is_closer'])

        enc_a = {k: v.squeeze(0) for k, v in enc_a.items()}
        enc_b = {k: v.squeeze(0) for k, v in enc_b.items()}

        return enc_a, enc_b, torch.tensor(label, dtype=torch.long)
