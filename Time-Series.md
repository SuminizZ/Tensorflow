---
title : "[TensorFlow Cert] Q5-Time Series Data (Conv1D, LSTM)"
categories : 
    - Deep Learning
tag : [Python, deep learning, tensorflow, certificate]
toc : true
---

## Sunspot Dataset
- Lambda used / Normalized ver.
- optimizer : SGD(learning_rate=1e-5, momentum=0.9)
- loss : Huber
- window 함수 확인 (w[:-1], w[1:] )

```python
import csv
import tensorflow as tf
import numpy as np
import urllib

from tensorflow.keras.layers import Dense, LSTM, Lambda, Conv1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import Huber

def normalization(series):
    min = np.min(series)      # 1. Normalization 
    max = np.max(series)
    series -= min
    series /= max
    return series

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift = 1, drop_remainder=True)
    ds = ds.flat_map(lambda w : w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    return ds.batch(batch_size).prefetch(1)

def solution_model():
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/Sunspots.csv'
    urllib.request.urlretrieve(url, 'sunspots.csv')

    time_step = []
    sunspots = []

    with open('sunspots.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter = ',')
        next(reader)
        for row in reader:
            sunspots.append(float(row[2]))
            time_step.append(int(row[0]))

    series = np.array(sunspots)
    time = np.array(time_step)

    series = normalization(series)          # normalized

    split_time = 3000

    time_train = time[:split_time]
    time_valid = time[split_time:]

    x_train = series[:split_time]
    x_valid = series[split_time:]

    window_size = 30
    batch_size = 32
    shuffle_size = 1000

    train_set = windowed_dataset(x_train,
                                 window_size = window_size,
                                 batch_size = batch_size,
                                 shuffle_buffer = shuffle_size)
    
    validation_set = windowed_dataset(x_valid,
                                      window_size = window_size,
                                      batch_size = batch_size,
                                      shuffle_buffer = shuffle_size)
    
    model = Sequential([
                        Conv1D(70, kernel_size = 5,
                                padding = 'causal',
                                activation = 'relu',
                                input_shape = [None, 1]),
                        LSTM(64, return_sequences=True),
                        LSTM(64, return_sequences=True),
                        Dense(30, activation='relu'),
                        Dense(10, activation='relu'),
                        Dense(1),
                        Lambda(lambda x: x*400)         # 2. Lambda used
    ]) 

    optimizer = SGD(learning_rate=1e-5, momentum=0.9)
    loss = Huber()

    model.compile(optimizer = optimizer, loss = loss, metrics = ['mae'])

    checkpoint_path = 'my_checkpoint.ckpt'
    checkpoint = ModelCheckpoint(checkpoint_path,
                                 save_weights_only = True,
                                 save_best_only = True,
                                 monitor = 'val_mae',
                                 verbose = 1)
    
    model.fit(train_set,
              validation_data = (validation_set),
              epochs = 100,
              callbacks = [checkpoint])
    
    model.load_weights(checkpoint_path)

    return model

if __name__ == '__main__':
    model = solution_model()
    model.save("model.h5")
```


## Disel Dataset
- Layers : Conv1D, Bidirectional, LSTM
- optimizer : Adam
- loss : mae
- window 함수 확인 (w[:n_past], w[n_past:])

```python
import urllib
import os
import zipfile
import pandas as pd
import tensorflow as tf

from tensorflow.keras.layers import Dense, Conv1D, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint

def normalize_series(data, min, max):
    data = data - min
    data = data / max
    return data

def windowed_dataset(series, batch_size, n_past=10, n_future=10, shift=1):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(size=n_past + n_future, shift=shift, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(n_past + n_future))
    ds = ds.map(lambda w: (w[:n_past], w[n_past:]))
    return ds.batch(batch_size).prefetch(1)


def solution_model():
    df = pd.read_csv('Weekly_U.S.Diesel_Retail_Prices.csv',
                     infer_datetime_format=True, index_col='Week of', header=0)

    N_FEATURES = len(df.columns) 
    data = df.values
    data = normalize_series(data, data.min(axis=0), data.max(axis=0))

    SPLIT_TIME = int(len(data) * 0.8) 
    x_train = data[:SPLIT_TIME]
    x_valid = data[SPLIT_TIME:]

    tf.keras.backend.clear_session()
    tf.random.set_seed(42)

    BATCH_SIZE = 32  
    N_PAST = 10  
    N_FUTURE = 10  
    SHIFT = 1 

    train_set = windowed_dataset(series=x_train, batch_size=BATCH_SIZE,
                                 n_past=N_PAST, n_future=N_FUTURE,
                                 shift=SHIFT)
    valid_set = windowed_dataset(series=x_valid, batch_size=BATCH_SIZE,
                                 n_past=N_PAST, n_future=N_FUTURE,
                                 shift=SHIFT)

    model = Sequential([
                        Conv1D(filters=32, kernel_size=3, padding='causal', activation='relu', input_shape=[N_PAST, 1]),
                        Bidirectional(LSTM(32, return_sequences=True)),
                        Bidirectional(LSTM(32, return_sequences=True)),
                        Dense(32, activation='relu'),
                        Dense(16, activation='relu'),
                        Dense(N_FEATURES)
    ])

    checkpoint_path = 'model/my_checkpoint.ckpt'
    checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                                 save_weights_only=True,
                                 save_best_only=True,
                                 monitor='val_loss',
                                 verbose=1)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss='mae', metrics=['mae'])

    model.fit(train_set,
              validation_data=(valid_set),
              epochs=20,
              callbacks=[checkpoint])
    model.load_weights(checkpoint_path)

    return model

if __name__ == '__main__':
    model = solution_model()
    model.save("model.h5")

```

