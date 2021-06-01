import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import MinMaxScaler


train_x, train_y = np.array([]), np.array([])
train_x_se, train_y_se = {}, {}

want_para = ['ZINST58', 'ZINST72', 'ZINST71', 'ZINST70',
             'WFWLN1', 'WFWLN2', 'WFWLN3']
scaler = MinMaxScaler()

for type_db, db_path in zip(['DB', 'Test_DB'], ['./DB', './Test_DB']):
    for file in os.listdir(db_path):
        if '.csv' in file:
            csv_db = pd.read_csv(f'./{type_db}/{file}', index_col=0)
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

            train_x_se[file] = get_xdb
            train_y_se[file] = get_ydb

            train_x = get_xdb if train_x.shape[0] == 0 else np.concatenate((train_x, get_xdb), axis=0)
            train_y = np.append(train_y, get_y, axis=0)
            scaler.partial_fit(train_x)
            print(f'Read {file} | train_x_shape {train_x.shape} | train_y_shape {train_y.shape}')

    # minmax scale
    train_x = scaler.transform(train_x)
    for file_ in train_x_se.keys():
        train_x_se[file_] = scaler.transform(train_x_se[file_])

    save_db_info = {
        'scaler': scaler,
        'want_para': want_para,
        f'{type_db}_x': train_x,
        f'{type_db}_y': train_y,

        f'{type_db}_x_se': train_x_se,
        f'{type_db}_y_se': train_y_se,
    }
    with open(f'{type_db}_info.pkl', 'wb') as f:
        pickle.dump(save_db_info, f)