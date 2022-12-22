"""
----------------------------
Train model
----------------------------
"""

import os

import pandas as pd

import numpy as np

import tensorflow as tf

from src.utils import ML


if __name__ == '__main__':
    input_file = conf.raw_data_file
    if not os.path.exists(input_file):
        raise RuntimeError(f'No input file: {input_file}')
    df = pd.read_csv(input_file)
    train_df = df[df['subset'] == 'train']
    test_df = df[df['subset'] == 'test']
    logger.info('num rows for train: %d', train_df.shape[0])

    X_train = train_df['msg'].values
    y_train = train_df['label']

    X_test = test_df['msg'].values
    y_test = test_df['label']

    # preprocessing
    X_train = ML.preprocessing(X_train)
    X_test = ML.preprocessing(X_test)

    # fit 
    VOCAB_SIZE = 14000
    encoder = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE)
    encoder.adapt(X_train)

    vocab = encoder.get_vocabulary()

    model = tf.keras.Sequential([
    tf.keras.Input(shape=(1,), dtype=tf.string),
    
    tf.keras.layers.TextVectorization(
        max_tokens=VOCAB_SIZE,output_mode='int',
        vocabulary=np.delete(np.delete(np.array(vocab), 0),0)),
    tf.keras.layers.Embedding(
        input_dim=len(vocab),
        output_dim=32,
        mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=[
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall')
    ])

    history = model.fit(X_train, y_train,validation_data=(X_test, y_test), epochs=20)

    # predict
    val_precision = history.history['val_precision'][-1]
    val_recall = history.history['val_recall'][-1]
    f1_score = (2 * val_recall * val_precision) / (val_recall + val_precision)

    logger.info('best_score %.5f', f1_score)

    # safe better model

    model_path = conf.model_path
    weights_path = conf.weights_path

    # serialize model to JSON
    model_json = model.to_json()
    with open(model_path, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to tf
    model.save_weights(weights_path, save_format = 'tf')
    logger.info("Saved model to disk")
