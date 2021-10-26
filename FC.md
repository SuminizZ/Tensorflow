---
title : "[TensorFlow Cert] Q2-Fully Connected Layer"
categories : 
    - Deep Learning
tag : [Python, deep learning, tensorflow, certificate]
toc : true
---
## Image Data
### 1. Fashion_Mnist
```python
import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint

def solution_model():
    fashion_mnist = tf.keras.datasets.fashion_mnist
    
    (x_train, y_train), (x_valid, y_valid) = fashion_mnist.load_data()

    x_train = x_train / 255.0
    x_valid = x_valid / 255.0

    # 모델 정의
    model = Sequential([
                        # Flatten(input_shape=(28,28)),
                        # Dense(512, activation='relu'),
                        # Dropout(0.5),
                        # Dense(128, activation='relu'),
                        # Dropout(0.5),
                        # Dense(10, activation='softmax'),

                        # Flatten(input_shape=(28,28)),
                        # # Dropout(0.5),
                        # Dense(512, activation='relu'),
                        # Dense(256, activation='relu'),
                        # Dense(128, activation='relu'),
                        # Dense(64, activation='relu'),
                        # Dense(32, activation='relu'),
                        # Dense(10, activation='softmax'),

                        Flatten(input_shape=(28,28)),
                        Dropout(0.3),
                        Dense(1024, activation='relu'),
                        Dense(512, activation='relu'),
                        Dense(256, activation='relu'),
                        Dense(128, activation='relu'),
                        Dense(64, activation='relu'),
                        Dense(10, activation='softmax'),

                        # Flatten-Dense(128)-Dropout-Dense(64)-Dropout-Dense(10)
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])

    checkpoint_path = 'my_checkpoint.ckpt'
    checkpoint = ModelCheckpoint(filepath = checkpoint_path,
                                 save_weights_only = True,
                                 save_best_only = True,
                                 monitor='val_loss',
                                 verbose=1)

    model.fit(x_train, y_train,
              validation_data = (x_valid, y_valid),
              epochs = 20,
              callbacks = [checkpoint],
              )
    model.load_weights(checkpoint_path)

    return model

if __name__ == '__main__':
    model = solution_model()
    model.save("TF2-fashion-mnist.h5")
```

### 2. Mnist
```python
import tensorflow as tf

import numpy as np
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint

def solution_model():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_valid, y_valid) = mnist.load_data()

    x_train = x_train / 255.0
    x_valid = x_valid / 255.0
    
    model = Sequential([
                        Flatten(input_shape = (28,28)),
                        Dense(1024, activation='relu'),
                        Dense(512, activation='relu'),
                        Dense(256, activation='relu'),
                        Dense(128, activation='relu'),
                        Dense(64, activation='relu'),
                        Dense(32, activation='relu'),
                        Dense(10, activation='softmax'),
    ])

    model.compile(optimizer = 'adam', loss='sparse_categorical_crossentropy', metrics=['acc'])

    checkpoint_path = 'my_checkpoint.ckpt'
    checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                                 save_weights_only = True,
                                 save_best_only = True,
                                 monitor = 'val_loss',
                                 verbose = 1)
    
    model.fit(x_train, y_train,
              validation_data = (x_valid, y_valid),
              epochs = 20,
              callbacks = [checkpoint])
    
    model.load_weights(checkpoint_path)

    return model

model = solution_model()    
model.save("TF2-mnist.h5")

```

## Iris Data
```python
import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint

train_dataset = tfds.load('iris', split='train[:80%]')
valid_dataset = tfds.load('iris', split='train[80%:]')

def preprocess(data):
    x = data['features']
    y = data['label']
    y = tf.one_hot(y,3)
    return x, y

def solution_model():
    batch_size = 10

    train_data = train_dataset.map(preprocess).batch(batch_size)
    valid_data = valid_dataset.map(preprocess).batch(batch_size)

    # modeling
    model = Sequential([
                        Dense(512, activation='relu', input_shape=(4,)),
                        Dense(256, activation='relu'),
                        Dense(128, activation='relu'),
                        Dense(64, activation='relu'),
                        Dense(32, activation='relu'),
                        Dense(3, activation='softmax')
    ])

    # compile
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

    # checkpoint (appropriate fitting)
    checkpoint_path = 'my_checkpoint.ckpt'
    checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                                 save_weights_only=True,
                                 save_best_only=True,
                                 monitor='val_loss',
                                 verbose=1)
    # fitting
    model.fit(train_data,
              validation_data=(valid_data),
              epochs=20,
              callbacks=[checkpoint])
    
    # apply best weight
    model.load_weights(checkpoint_path)

    return model

if __name__ == '__main__':
    model = solution_model()
    model.save("TF2-iris.h5")
```