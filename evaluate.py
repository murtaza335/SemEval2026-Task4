import numpy as np
import tensorflow as tf

from data_loader import load_jsonl
from preprocess import create_vectorizer
from model_baseline_ayanre import build_model 

TEST_PATH = "dev_track_a.jsonl"
MODEL_PATH = "results/model_ayanre.h5"
VOCAB_PATH = "results/vectorizer_vocab.npy"
MAX_LEN = 300 

anchors, As, Bs, labels = load_jsonl(TEST_PATH)

vectorizer = create_vectorizer([], max_len=MAX_LEN)
vocab = np.load(VOCAB_PATH, allow_pickle=True)
vectorizer.set_vocabulary(vocab)

enc_anchors = vectorizer(tf.constant(anchors))
enc_As      = vectorizer(tf.constant(As))
enc_Bs      = vectorizer(tf.constant(Bs))

model = tf.keras.models.load_model(MODEL_PATH)

preds = model.predict([enc_anchors, enc_As, enc_Bs], batch_size=32)
preds = (preds > 0.5).astype(int).flatten()

acc = (preds == labels).mean()
print(f"Accuracy: {acc:.4f}")