import pandas as pd
import numpy as np
import os
import pickle
import tensorflow.keras as k
from sklearn.preprocessing import MinMaxScaler
with open('db_info.pkl', 'rb') as f:
    save_db_info = pickle.load(f)
model = k.Sequential([
    k.layers.InputLayer(input_shape=len(save_db_info['want_para'])),
    k.layers.Dense(64, activation='relu'),
    k.layers.Dense(64, activation='relu'),
    k.layers.Dense(4, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
print(model.summary())

model.fit(save_db_info['train_x'], save_db_info['train_y'], epochs=50)
model.save_weights('model.h5')