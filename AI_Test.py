import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
import tensorflow.keras as k
from sklearn.preprocessing import MinMaxScaler
with open('Test_DB_info.pkl', 'rb') as f:
    save_db_info = pickle.load(f)
print(save_db_info.keys())
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
model.load_weights('model.h5')

result = model.predict(save_db_info['Test_DB_x_se']['(15, 35, 40).csv'])
plt.plot(result)
plt.legend(['Normal', 'LOCA', 'SGTR', 'MSLB'])
plt.show()
