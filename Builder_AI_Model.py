import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
import tensorflow.keras as k
from sklearn.preprocessing import MinMaxScaler
with open('All_DB_info.pkl', 'rb') as f:
    save_db_info = pickle.load(f)

print(save_db_info.keys())

get_first_key = list(save_db_info['DB_x_se'].keys())[0]
sh = np.shape(save_db_info['DB_x_se'][get_first_key])
get_shape = len(sh)

if get_shape == 3:
    get_time_seq = sh[1]
    print(f'[Seq Model] We have {sh} {get_time_seq} time seq')
    # Seq Model
    model = k.Sequential([
        # k.layers.RNN(k.layers.SimpleRNNCell(32), input_shape=(get_time_seq, len(save_db_info['want_para']))),
        k.layers.LSTM(32, input_shape=(get_time_seq, len(save_db_info['want_para']))),
        # k.layers.Bidirectional(k.layers.LSTM(32, input_shape=(get_time_seq, len(save_db_info['want_para'])))),
        k.layers.Flatten(),
        k.layers.Dense(128, activation='relu'),
        k.layers.LayerNormalization(),
        k.layers.Dense(256, activation='relu'),
        k.layers.Dense(518, activation='relu'),
        k.layers.Dense(256, activation='relu'),
        k.layers.Dense(128, activation='relu'),
        k.layers.Dense(5, activation='softmax')
    ])
else:
    print(f'[DNN Model] We have {len(save_db_info["want_para"])} inputs')
    # DNN Model
    model = k.Sequential([
        k.layers.InputLayer(input_shape=len(save_db_info['want_para'])),
        k.layers.Dense(64, activation='relu'),
        k.layers.Dense(128, activation='relu'),
        k.layers.Dense(64, activation='relu'),
        k.layers.Dense(5, activation='softmax')
    ])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
print(model.summary())

model.fit(save_db_info['DB_x'], save_db_info['DB_y'], epochs=1)
model.save_weights('model.h5')
# model.load_weights('model.h5')

for _ in save_db_info['Test_DB_x_se'].keys():
    result = model.predict(save_db_info['Test_DB_x_se'][_])
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)
    ax1.plot(result)
    ax2.plot(save_db_info['Test_DB_y_se'][_])
    plt.title(_)
    ax1.legend(['Normal', 'LOCA', 'SGTR', 'MSLB', 'MSLB-non'])
    plt.show()