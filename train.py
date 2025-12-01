from data_loader import load_and_split
from preprocess import create_vectorizer, make_dataset
from model_baseline_ayanre import build_model     # choose baseline here

print("Loading dataset...")
train_anchor, val_anchor, train_A, val_A, train_B, val_B, train_y, val_y = load_and_split("dev_track_a.jsonl")

all_texts = train_anchor + train_A + train_B
vectorizer = create_vectorizer(all_texts)

vocab_size = len(vectorizer.get_vocabulary())
model = build_model(vocab_size)
model.summary()

train_ds = make_dataset(train_anchor, train_A, train_B, train_y, vectorizer)
val_ds   = make_dataset(val_anchor, val_A, val_B, val_y, vectorizer)

print("\nTraining model...")
history = model.fit(train_ds, validation_data=val_ds, epochs=5)

import numpy as np
vocab = np.array(vectorizer.get_vocabulary())
np.save("results/vectorizer_vocab.npy", vocab)
model.save("results/model_ayanre.h5")
