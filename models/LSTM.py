import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Concatenate, GlobalAveragePooling1D
from tensorflow.keras.models import Model

# Updated parameters based on actual data shapes
seq_length = 30
max_title_length = 10
n_genres = 18
vocab_size = 1000
title_vocab_size = 5000


def create_fixed_enhanced_model(seq_length=30, return_lstm_output=False):
    movie_input = Input(shape=(seq_length,), name='movie_sequence')
    title_input = Input(shape=(seq_length, max_title_length), name='title_sequence')
    genre_input = Input(shape=(seq_length, n_genres), name='genre_features')

    # Embeddings and processing (same as before)
    movie_embedded = Embedding(vocab_size, 128, name="movie_embedding")(movie_input)

    title_embedded_list = []
    for i in range(seq_length):
        title_slice = tf.keras.layers.Lambda(lambda x: x[:, i, :])(title_input)
        title_emb = Embedding(title_vocab_size, 64)(title_slice)
        title_pooled = GlobalAveragePooling1D()(title_emb)
        title_embedded_list.append(title_pooled)

    title_embedded = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=1))(title_embedded_list)
    genre_dense = Dense(64, activation='relu')(genre_input)

    combined_features = Concatenate()([movie_embedded, title_embedded, genre_dense])

    # LSTM layers
    x = LSTM(256, return_sequences=True, dropout=0.3)(combined_features)
    lstm_output = LSTM(128, dropout=0.3)(x)

    if return_lstm_output:
        return Model(inputs=[movie_input, title_input, genre_input], outputs=lstm_output)

    # Otherwise, continue to final output
    x = Dense(256, activation='relu')(lstm_output)
    x = Dropout(0.3)(x)
    output = Dense(vocab_size, activation='softmax')(x)

    return Model(inputs=[movie_input, title_input, genre_input], outputs=output)
