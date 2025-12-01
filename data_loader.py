import json
import numpy as np
from sklearn.model_selection import train_test_split

def load_jsonl(path):
    anchors, As, Bs, labels = [], [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            anchors.append(obj["anchor_text"])
            As.append(obj["text_a"])
            Bs.append(obj["text_b"])
            labels.append(int(obj["text_a_is_closer"]))
    return anchors, As, Bs, np.array(labels)


def load_and_split(path, test_size=0.1, seed=42):
    anchors, As, Bs, labels = load_jsonl(path)
    return train_test_split(
        anchors, As, Bs, labels,
        test_size=test_size,
        random_state=seed
    )
