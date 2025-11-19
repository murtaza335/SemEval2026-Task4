from transformers import BertTokenizer, BertForSequenceClassification
import torch

model = BertForSequenceClassification.from_pretrained("./bert_similarity_model").cuda()
tokenizer = BertTokenizer.from_pretrained("./bert_similarity_model")

def predict(text_a, text_b):
    encoding = tokenizer(text_a, text_b, return_tensors="pt", padding=True, truncation=True, max_length=256)
    encoding = {k: v.cuda() for k,v in encoding.items()}

    outputs = model(**encoding)
    logits = outputs.logits
    pred = torch.argmax(logits, axis=-1).item()
    return pred

print(predict("A boy plays in the park", "A kid is playing outside"))
