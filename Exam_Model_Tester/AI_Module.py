import numpy as np
import pickle
import os
from collections import deque
from sklearn.preprocessing import MinMaxScaler


class AllAIModule:
    def __init__(self):
        self.models_db = {}
        self.models, self.scalers, self.paras = self._make_models()

    def predict(self, mem=None):
        model_outs = {}
        for i in range(len(self.models)): # , (m, s, ps) in enumerate(zip(self.models, self.scalers, self.paras)):
            get_vals = [mem[p]['Val'] for p in self.paras[i]]
            get_normal_val = self.scalers[i].transform([get_vals])

            self.models_db[i].append(get_normal_val[0])

            if self.models_db[i].maxlen == len(self.models_db[i]):
                o_ = self.models[i].predict(np.array([self.models_db[i]]))
                model_outs[i] = o_.reshape(5).tolist()
            else:
                model_outs[i] = [None for i in range(5)]
        return model_outs

    def _make_models(self):
        import tensorflow.keras as k
        models = {}
        scalers = {}
        paras = {}
        for file in os.listdir('./Model'):
            if 'pkl' in file:
                get_model_nub = int(file.split('_')[3])

                with open(f'./Model/{file}', 'rb') as f:
                    save_db_info = pickle.load(f)

                get_first_key = list(save_db_info['DB_x_se'].keys())[0]
                sh = np.shape(save_db_info['DB_x_se'][get_first_key])
                get_shape = len(sh)

                if get_model_nub in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:# , 1, 2, 3, 4, 5, 6]:
                    if get_model_nub == 0:  # 승윤2
                        model_ = k.Sequential([
                            k.layers.LSTM(32, input_shape=(sh[1], len(save_db_info['want_para']))),
                            k.layers.Flatten(),
                            k.layers.Dense(300, activation='relu'),
                            k.layers.Dense(200, activation='relu'),
                            k.layers.Dense(128, activation='relu'),
                            k.layers.Dense(5, activation='softmax')
                        ])
                        self.models_db[get_model_nub] = deque(maxlen=sh[1])
                    elif get_model_nub == 1:  # 상원
                        model_ = k.Sequential([
                            k.layers.InputLayer(input_shape=len(save_db_info['want_para'])),
                            k.layers.Dense(256, activation='relu'),
                            k.layers.Dense(128, activation='relu'),
                            k.layers.Dense(64, activation='relu'),
                            k.layers.Dense(5, activation='softmax')
                        ])
                        self.models_db[get_model_nub] = deque(maxlen=1)
                    elif get_model_nub == 2:  # 두헌
                        model_ = k.Sequential([
                            k.layers.InputLayer(input_shape=len(save_db_info['want_para'])),
                            k.layers.Dense(64, activation='relu'),
                            k.layers.Dense(128, activation='relu'),
                            k.layers.Dense(64, activation='relu'),
                            k.layers.Dense(5, activation='softmax')
                        ])
                        self.models_db[get_model_nub] = deque(maxlen=1)
                    elif get_model_nub == 3:    # 상현
                        model_ = k.Sequential([
                            k.layers.LSTM(32, input_shape=(sh[1], len(save_db_info['want_para'])),
                                          return_sequences=True),
                            k.layers.LSTM(32),
                            k.layers.Dense(5, activation='softmax')
                            # model.add(Activation('tanh'))
                        ])
                        self.models_db[get_model_nub] = deque(maxlen=sh[1])
                    elif get_model_nub == 4:  # 창상
                        model_ = k.Sequential([
                            k.layers.InputLayer(input_shape=len(save_db_info['want_para'])),
                            k.layers.Dense(128, activation='relu'),
                            k.layers.Dense(256, activation='relu'),
                            k.layers.Dense(128, activation='relu'),
                            k.layers.Dense(5, activation='softmax')
                        ])
                        self.models_db[get_model_nub] = deque(maxlen=1)
                    elif get_model_nub == 5:    # 창주
                        model_ = k.Sequential([
                            k.layers.InputLayer(input_shape=len(save_db_info['want_para'])),
                            k.layers.Dense(518, activation='relu'),
                            k.layers.Dense(256, activation='relu'),
                            k.layers.Dense(128, activation='relu'),
                            k.layers.Dense(64, activation='relu'),
                            k.layers.Dense(32, activation='relu'),
                            k.layers.Dense(16, activation='relu'),
                            k.layers.Dense(5, activation='softmax')
                        ])
                        self.models_db[get_model_nub] = deque(maxlen=1)
                    elif get_model_nub == 6:        # 선준 2
                        model_ = k.Sequential([
                            k.layers.InputLayer(input_shape=len(save_db_info['want_para'])),
                            k.layers.Dense(64, activation='relu'),
                            k.layers.Dense(128, activation='relu'),
                            k.layers.Dense(64, activation='relu'),
                            k.layers.Dense(5, activation='softmax')
                        ])
                        self.models_db[get_model_nub] = deque(maxlen=1)
                    elif get_model_nub == 7:    # 김수창
                        model_ = k.Sequential([
                            k.layers.LSTM(32, input_shape=(sh[1], len(save_db_info['want_para']))),
                            k.layers.Flatten(),
                            k.layers.Dense(128, activation='relu'),
                            k.layers.Dense(256, activation='relu'),
                            k.layers.Dense(512, activation='relu'),
                            k.layers.Dense(256, activation='relu'),
                            k.layers.Dense(128, activation='relu'),
                            k.layers.Dense(5, activation='softmax')
                        ])
                        self.models_db[get_model_nub] = deque(maxlen=sh[1])
                    elif get_model_nub == 8:    # 임지영 (Copy 김수창?)
                        model_ = k.Sequential([
                            k.layers.LSTM(32, input_shape=(sh[1], len(save_db_info['want_para']))),
                            k.layers.Flatten(),
                            k.layers.Dense(128, activation='relu'),
                            k.layers.Dense(256, activation='relu'),
                            k.layers.Dense(512, activation='relu'),
                            k.layers.Dense(256, activation='relu'),
                            k.layers.Dense(128, activation='relu'),
                            k.layers.Dense(5, activation='softmax')
                        ])
                        self.models_db[get_model_nub] = deque(maxlen=sh[1])
                    elif get_model_nub == 9:    # 임아현 (Copy 김수창?)
                        model_ = k.Sequential([
                            k.layers.LSTM(32, input_shape=(sh[1], len(save_db_info['want_para']))),
                            k.layers.Flatten(),
                            k.layers.Dense(128, activation='relu'),
                            k.layers.Dense(256, activation='relu'),
                            k.layers.Dense(512, activation='relu'),
                            k.layers.Dense(256, activation='relu'),
                            k.layers.Dense(128, activation='relu'),
                            k.layers.Dense(5, activation='softmax')
                        ])
                        self.models_db[get_model_nub] = deque(maxlen=sh[1])
                    else:
                        raise ValueError('ALLAIModule : Error Cannot find Model.')

                    paras[get_model_nub] = save_db_info['want_para']
                    scalers[get_model_nub] = save_db_info['scaler']
                    try:
                        model_.load_weights(f'./Model/model_{get_model_nub}_.h5')
                    except:
                        print(f'Model {get_model_nub} is error.')
                        raise
                    models[get_model_nub] = model_
                else:
                    raise ValueError('ALLAIModule : Error Cannot find Model.')
            else:
                print(f'No read {file}')

        return models, scalers, paras


if __name__ == '__main__':
    A_ = AllAIModule()
    A_.predict()