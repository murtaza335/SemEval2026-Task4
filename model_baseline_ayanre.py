import tensorflow as tf

def build_encoder(vocab_size, embed_dim=128, lstm_units=64):
    return tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embed_dim, mask_zero=True),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units)),
        tf.keras.layers.Dense(128, activation="relu")
    ])


def build_model(vocab_size):
    encoder = build_encoder(vocab_size)

    inp_anchor = tf.keras.Input(shape=(None,), dtype=tf.int32)
    inp_A = tf.keras.Input(shape=(None,), dtype=tf.int32)
    inp_B = tf.keras.Input(shape=(None,), dtype=tf.int32)

    enc_anchor = encoder(inp_anchor)
    enc_A = encoder(inp_A)
    enc_B = encoder(inp_B)

    # Cosine similarity
    cos = tf.keras.layers.Dot(axes=1, normalize=True)
    sim_A = cos([enc_anchor, enc_A])
    sim_B = cos([enc_anchor, enc_B])

    diff = tf.keras.layers.Subtract()([sim_A, sim_B])
    out = tf.keras.layers.Dense(1, activation="sigmoid")(diff)

    model = tf.keras.Model(
        inputs=[inp_anchor, inp_A, inp_B],
        outputs=out
    )

    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(1e-3),
        metrics=["accuracy"]
    )
    
    return model
