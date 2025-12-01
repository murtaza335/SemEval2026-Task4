import json
import numpy as np
import tensorflow as tf
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


anchors, As, Bs, labels = load_jsonl("dev_track_a.jsonl")

(
    _,
    val_anchor,
    _,
    val_A,
    _,
    val_B,
    _,
    val_y
) = train_test_split(
    anchors, As, Bs, labels,
    test_size=0.1,
    random_state=42     
)

vocab = np.load("results/vectorizer_vocab.npy", allow_pickle=True)

vectorize_layer = tf.keras.layers.TextVectorization(
    output_mode="int",
    output_sequence_length=300
)

vectorize_layer.set_vocabulary(vocab)

def make_dataset(vectorizer, anchor, A, B, y, batch_size=16):
    anchor_vec = vectorizer(anchor)
    A_vec = vectorizer(A)
    B_vec = vectorizer(B)

    ds = tf.data.Dataset.from_tensor_slices((
        (anchor_vec, A_vec, B_vec),
        y
    ))

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

val_ds = make_dataset(vectorize_layer, val_anchor, val_A, val_B, val_y)

model = tf.keras.models.load_model("results/model_ayanre.h5")

loss, acc = model.evaluate(val_ds)
print("Validation accuracy:", acc)
print("Validation loss:", loss)
