---
title : "[TensorFlow Cert] Q4-Text Data (Tokenizer, LSTM)"
categories : 
    - Deep Learning
tag : [Python, deep learning, tensorflow, certificate]
toc : true
---

## Sarcasm Text Data
- Layers : Embedding, LSTM, Bidirectional, Dense

```python
import json
import tensorflow as tf
import numpy as np
import urllib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint

def solution_model():
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/sarcasm.json'
    urllib.request.urlretrieve(url, 'sarcasm.json')

    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type='post'
    padding_type='post'
    oov_tok = "<OOV>"
    training_size = 20000

    with open('sarcasm.json') as f:
        datas = json.load(f)

    sentences = []
    labels = []

    for data in datas:
        sentences.append(data['headline'])
        labels.append(data['is_sarcastic'])
    
    # sentence, label -> train, validation split

    train_sentences = sentences[:training_size]
    train_labels = labels[:training_size]

    validation_sentences = sentences[training_size:]
    validation_labels = labels[training_size:]

    # sentence -> tokenizer
    tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(train_sentences)

    # tokenizer -> sequence
    train_sequences = tokenizer.texts_to_sequences(train_sentences)
    validation_sequences = tokenizer.texts_to_sequences(validation_sentences)

    # sequence -> padded
    train_padded = pad_sequences(train_sequences, maxlen = max_length, truncating = trunc_type, padding = padding_type)
    validation_padded = pad_sequences(validation_sequences, maxlen = max_length, truncating = trunc_type, padding = padding_type)

    # label -> numpy array
    train_labels = np.array(train_labels)
    validation_labels = np.array(validation_labels)

    # Model
    model = Sequential([
                        Embedding(vocab_size, embedding_dim, input_length=max_length),
                        Bidirectional(LSTM(64, return_sequences = True)),
                        Bidirectional(LSTM(64, return_sequences = True)),
                        Bidirectional(LSTM(64)),
                        Dense(32, activation='relu'),
                        # Dense(32, activation='relu'),
                        Dense(16, activation='relu'),
                        # Dense(4, activation='relu'),
                        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

    checkpoint_path = 'my_checkpoint_ckpt'
    checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                                 save_weights_only = True,
                                 save_best_only = True,
                                 monitor = 'val_loss',
                                 verbose = 1)
    
    model.fit(train_padded, train_labels,
              validation_data = (validation_padded, validation_labels),
              callbacks = [checkpoint],
              epochs = 10)
    
    model.load_weights(checkpoint_path)

    return model
    

if __name__ == '__main__':
    model = solution_model()
    model.save("TF4-sarcasm.h5")

```