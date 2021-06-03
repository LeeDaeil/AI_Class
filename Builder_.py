"""
Title : AI_DB 생성부터 Model 빌드까지 One process
Developer : Deali Lee
Date : 21-06-02

Revision : Ver0
"""
import time
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import MinMaxScaler

from multiprocessing.managers import BaseManager
from multiprocessing import Process

# fold setting


# SH mem
class SHmem:
    def __init__(self, want_para, time_seq=1):
        self.want_para = want_para
        self.time_seq = time_seq
        self.p = 1 if self.time_seq != 1 else 0
        self.save_db_info = {
            'scaler': None,
            'want_para': want_para,
            'DB_x': np.array([]) if self.time_seq == 1 else np.array([[]]),
            'Test_DB_x': np.array([]) if self.time_seq == 1 else np.array([[]]),
            'DB_y': np.array([]),           'Test_DB_y': np.array([]),
            'DB_x_se': {},  'Test_DB_x_se': {},
            'DB_y_se': {},  'Test_DB_y_se': {},
        }

    def get_want_para(self):
        return self.want_para

    def get_time_seq(self):
        return self.time_seq

    def upload_data(self, db_x_se, db_y_se, test_db_x_se, test_db_y_se):
        for se_name in db_x_se.keys():
            self.save_db_info['DB_x_se'][se_name] = db_x_se[se_name]
            self.save_db_info['DB_y_se'][se_name] = db_y_se[se_name]
        for se_name in test_db_x_se.keys():
            self.save_db_info['Test_DB_x_se'][se_name] = test_db_x_se[se_name]
            self.save_db_info['Test_DB_y_se'][se_name] = test_db_y_se[se_name]

    def accumulate_db(self):
        for key in self.save_db_info['DB_x_se'].keys():
            __x = self.save_db_info['DB_x_se'][key]
            __y = self.save_db_info['DB_y_se'][key]

            self.save_db_info['DB_x'] = __x if self.save_db_info['DB_x'].shape[self.p] == 0 else \
                np.concatenate((self.save_db_info['DB_x'], __x), axis=0)
            self.save_db_info['DB_y'] = np.append(self.save_db_info['DB_y'], __y)

        for key in self.save_db_info['Test_DB_x_se'].keys():
            __x = self.save_db_info['Test_DB_x_se'][key]
            __y = self.save_db_info['Test_DB_y_se'][key]

            self.save_db_info['Test_DB_x'] = __x if self.save_db_info['Test_DB_x'].shape[self.p] == 0 else \
                np.concatenate((self.save_db_info['Test_DB_x'], __x), axis=0)
            self.save_db_info['Test_DB_y'] = np.append(self.save_db_info['Test_DB_y'], __y)

    def check_shape(self, title):
        print(f'[{"ALL":3}] File: {title:20} |'
              f'db_x:{str(np.shape(self.save_db_info["DB_x"])):15} '
              f'test_x:{str(np.shape(self.save_db_info["Test_DB_x"])):15} '
              f'db_y:{str(np.shape(self.save_db_info["DB_y"])):15} '
              f'test_y:{str(np.shape(self.save_db_info["Test_DB_y"])):15} ')

    def update_minmax(self):
        scaler = MinMaxScaler()
        # DB 차원 2 dim 으로 교체
        _db_x = self.save_db_info['DB_x'].reshape(-1, len(self.want_para))
        _test_db_x = self.save_db_info['Test_DB_x'].reshape(-1, len(self.want_para))
        # run Scaler
        scaler.partial_fit(_db_x)
        scaler.partial_fit(_test_db_x)
        #
        if self.time_seq == 1:
            # 전체 데이터 Scaling
            self.save_db_info['DB_x'] = scaler.transform(self.save_db_info['DB_x'])
            self.save_db_info['Test_DB_x'] = scaler.transform(self.save_db_info['Test_DB_x'])
            # 시나리오 별 데이터 Scaling
            for key in self.save_db_info['DB_x_se'].keys():
                self.save_db_info['DB_x_se'][key] = scaler.transform(self.save_db_info['DB_x_se'][key])
            for key in self.save_db_info['Test_DB_x_se'].keys():
                self.save_db_info['Test_DB_x_se'][key] = scaler.transform(self.save_db_info['Test_DB_x_se'][key])
        else:
            # 전체 데이터 Scaling
            _tr_db_x = np.array([scaler.transform(_) for _ in self.save_db_info['DB_x']])
            _tr_te_db_x = np.array([scaler.transform(_) for _ in self.save_db_info['Test_DB_x']])
            self.save_db_info['DB_x'] = _tr_db_x
            self.save_db_info['Test_DB_x'] = _tr_te_db_x
            # 시나리오 별 데이터 Scaling
            for key in self.save_db_info['DB_x_se'].keys():
                _tr_db_x = np.array([scaler.transform(_) for _ in self.save_db_info['DB_x_se'][key]])
                self.save_db_info['DB_x_se'][key] = _tr_db_x
            for key in self.save_db_info['Test_DB_x_se'].keys():
                _tr_db_x = np.array([scaler.transform(_) for _ in self.save_db_info['Test_DB_x_se'][key]])
                self.save_db_info['Test_DB_x_se'][key] = _tr_db_x

        self.save_db_info['scaler'] = scaler

    def dump_to_pkl(self):
        """ 처리된 훈련데이터를 덤프 """
        print('Dump all db info to .pkl file')
        with open(f'All_DB_info.pkl', 'wb') as f:
            pickle.dump(self.save_db_info, f)


class DataProcessor(Process):
    def __init__(self, nub, max_nub, shmem):
        Process.__init__(self)
        self.p_nub = nub
        self.max_p_nub = max_nub
        self.shmem = shmem
        self.want_para = self.shmem.get_want_para()
        self.time_seq = self.shmem.get_time_seq()

        self.work_file_list = []

        self.db_x = np.array([]) if self.time_seq == 1 else np.array([[]])
        self.test_db_x = np.array([]) if self.time_seq == 1 else np.array([[]])
        self.db_y, self.test_db_y = np.array([]), np.array([])
        self.db_x_se, self.db_y_se = {}, {}
        self.test_db_x_se, self.test_db_y_se = {}, {}

    def run(self):
        print(f'[{self.p_nub:3}] Start')
        # get job list
        job_file_list = self.get_job_list()
        # processing
        [self.read_csv_to_array(file_path) for file_path in job_file_list]
        # update mem
        self.shmem.upload_data(self.db_x_se, self.db_y_se,
                               self.test_db_x_se, self.test_db_y_se)

        print(f'[{self.p_nub:3}] End')
        pass

    def get_job_list(self):
        """ 전체 파일 목록을 읽고 그중 프로세스에 할당해줄 파일 명 나눔 """
        all_file = []
        for _ in ['DB', 'Test_DB']:
            for file in os.listdir(_):
                all_file.append(f'./{_}/{file}')

        allocate = int(len(all_file) / self.max_p_nub)

        st = allocate * self.p_nub
        end = allocate * (self.p_nub + 1) if not self.p_nub == self.max_p_nub - 1 else len(all_file)
        print(f'[{self.p_nub:3}][Total file:{len(all_file)}|{st} -> {end}|]')
        return all_file[st:end]

    def read_csv_to_array(self, file_path):
        """
        csv 파일 명을 읽고 처리된 데이터를 메모리로 dump
        :param file_path:
        :param time_seq:
        :return:
        """
        # 1. read csv
        csv_db = pd.read_csv(file_path, index_col=0)
        # 2. 파일 정보 추출
        file_name = file_path.split('/')[-1]
        file_type = file_path.split('/')[-2]
        # 3. 해당하는 변수 추출 및 x, y 데이터 생성
        get_xdb = csv_db[self.want_para].to_numpy()
        get_ydb = csv_db.loc[:, 'Normal_0'].to_numpy()
        accident_nub = {
            '12': 1,  # LOCA
            '13': 2,  # SGTR
            '15': 1,  # PORV Open (LOCA)
            '17': 1,  # Feedwater Line break (LOCA)
            '18': 3,  # Steam Line break (MSLB)
            '52': 4,  # Steam Line break - non isoable (MSLB)
        }

        get_mal_nub = file_name.split(',')[0][1:]  # '(12, ....)' -> 12
        # 3.1 y 데이터 생성
        get_y = np.where(get_ydb != 0, accident_nub[get_mal_nub], get_ydb)

        # 3.2 x 데이터 생성 및 x, y추가
        if file_type == 'DB':
            self._process(get_y, get_xdb, file_name, target_x_se=self.db_x_se, target_y_se=self.db_y_se)
        else:
            self._process(get_y, get_xdb, file_name, target_x_se=self.test_db_x_se, target_y_se=self.test_db_y_se)
        # ---------------------------------------------------------------------------------------
        print(f'[{self.p_nub:3}] File: {file_name:20}')
              # f'db_x:{str(np.shape(self.db_x)):15} test_x:{str(np.shape(self.test_db_x)):15} '
              # f'db_y:{str(np.shape(self.db_y)):15} test_y:{str(np.shape(self.test_db_y)):15} ')

        # ---------------------------------------------------------------------------------------

    def _process(self, get_y, get_xdb, file_name, target_x_se, target_y_se):
        # y, x 데이터 축적
        if self.time_seq == 1:
            target_x_se[file_name] = get_xdb
            target_y_se[file_name] = get_y
        else:
            get_x__se, get_y__se = np.array([[]]), np.array([])
            for i in range(len(get_xdb) - self.time_seq - 1):
                # print(get_xdb[i:i + self.time_seq], get_y[i + self.time_seq + 1])
                x__ = np.array([get_xdb[i:i + self.time_seq]])
                y__ = np.array([get_y[i + self.time_seq + 1]])

                # 시나리오 별 데이터
                get_x__se = x__ if get_x__se.shape[1] == 0 else np.concatenate((get_x__se, x__), axis=0)
                get_y__se = np.append(get_y__se, y__, axis=0)

            target_x_se[file_name] = get_x__se
            target_y_se[file_name] = get_y__se


if __name__ == '__main__':
    t_ = time.time()
    workers = 6

    time_seq = 2  # Default : 1
    want_para = ['ZINST66',                                     # PZR spray
                 'ZINST65', 'ZINST63',                          # PZR press & level
                 'UHOLEG1', 'UHOLEG2', 'UHOLEG3',               # Hot leg
                 'UCOLEG1', 'UCOLEG2', 'UCOLEG3',               # Cold leg
                 'ZINST72', 'ZINST71', 'ZINST70',               # S/G Wide Level
                 'WFWLN1', 'WFWLN2', 'WFWLN3',                  # Feed Line
                 'ZINST87', 'ZINST86', 'ZINST85',               # Steam Line
                 'ZINST75', 'ZINST74', 'ZINST73',               # S/G Press
                 'ZINST22', 'UCTMT',                            # CTMT rad, temp
                 'ZINST102',                                    # Second rad
                 'ZREAC',                                       # Core Level
                 ]

    # info -------------------------------------------------------------------------------------------------------------
    print(f'{"=" * 50}\nNub Worker : {workers}')
    _ = f'(batch, {len(want_para)})' if time_seq == 1 else f'(batch, {time_seq}, {len(want_para)})'
    print(f'Data Shape : {_}\n{"=" * 50}')
    # ------------------------------------------------------------------------------------------------------------------

    # Call shared mem
    BaseManager.register('SHmem', SHmem)
    manager = BaseManager()
    manager.start()
    shmem = manager.SHmem(want_para, time_seq)

    # Build Process ----------------------------------------------------------------------------------------------------
    p_list = [DataProcessor(i, workers, shmem) for i in range(workers)]
    # ------------------------------------------------------------------------------------------------------------------
    [p_.start() for p_ in p_list]
    [p_.join() for p_ in p_list]  # finished at the same time
    # ------------------------------------------------------------------------------------------------------------------
    shmem.accumulate_db()
    print(time.time() - t_)
    shmem.check_shape('Raw')
    shmem.update_minmax()
    shmem.check_shape('MinMax')
    shmem.dump_to_pkl()
    # File End ---------------------------------------------------------------------------------------------------------

