import tensorflow as tf

def create_vectorizer(texts, max_len=300):
    vectorizer = tf.keras.layers.TextVectorization(
        output_mode="int",
        output_sequence_length=max_len
    )
    vectorizer.adapt(texts)
    return vectorizer


def make_dataset(anchor, A, B, labels, vectorizer, batch_size=16):
    def encode(a, b, c, y):
        return (
            (
                vectorizer(a),
                vectorizer(b),
                vectorizer(c)
            ),
            y
        )

    ds = tf.data.Dataset.from_tensor_slices((anchor, A, B, labels))
    ds = (
        ds.shuffle(10_000)
          .map(encode, num_parallel_calls=tf.data.AUTOTUNE)
          .batch(batch_size)
          .prefetch(tf.data.AUTOTUNE)
    )
    return ds