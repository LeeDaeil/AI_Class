import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import MinMaxScaler


train_x, train_y = np.array([]), np.array([])
want_para = ['ZINST58', 'ZINST72', 'ZINST71', 'ZINST70',
             'WFWLN1', 'WFWLN2', 'WFWLN3']
scaler = MinMaxScaler()

for file in os.listdir('./DB'):
    if '.csv' in file:
        csv_db = pd.read_csv(f'./DB/{file}', index_col=0)
        get_xdb = csv_db[want_para]
        get_ydb = csv_db.loc[:, 'Normal_0'].to_numpy()
        accident_nub = {
            '12': 1,     # LOCA
            '13': 2,     # SGTR
            '15': 1,     # PORV Open (LOCA)
            '17': 1,     # Feedwater Line break (LOCA)
            '18': 3,     # Steam Line break (MSLB)
            '52': 3,     # Steam Line break - non isoable (MSLB)
        }
        get_mal_nub = file.split(',')[0][1:] # '(12, ....)' -> 12
        get_y = np.where(get_ydb != 0, accident_nub[get_mal_nub], get_ydb)
        train_x = get_xdb if train_x.shape[0] == 0 else np.concatenate((train_x, get_xdb), axis=0)
        train_y = np.append(train_y, get_y, axis=0)
        scaler.partial_fit(train_x)
        print(f'Read {file} | train_x_shape {train_x.shape} | train_y_shape {train_y.shape}')

# minmax scale
train_x = scaler.transform(train_x)

save_db_info = {
    'scaler': scaler,
    'want_para': want_para,
    'train_x': train_x,
    'train_y': train_y,
}
with open('db_info.pkl', 'wb') as f:
    pickle.dump(save_db_info, f)