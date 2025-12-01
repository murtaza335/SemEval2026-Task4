import json

def load_data(path):
    texts = []
    labels = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            # use 'text_a' and 'text_b' as input
            texts.append(item["text_a"] + " " + item["text_b"])
            labels.append(int(item["text_a_is_closer"]))
    return texts, labels
