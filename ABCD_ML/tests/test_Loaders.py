from nose.tools import *
from unittest import TestCase
from ABCD_ML import ABCD_ML

import os
import numpy as np
from numpy.random import random
import pandas as pd

TEST_FILE_DR = 'test_data/'

def get_file_path(name):

    file_path = os.path.join(os.path.dirname(__file__), TEST_FILE_DR, name)
    return file_path


class Test_Loaders(TestCase):

    def __init__(self, *args, **kwargs):
        super(Test_Loaders, self).__init__(*args, **kwargs)

        self.ML = ABCD_ML(log_dr=None)

        # Make sure fake files are there
        self.surf_data = get_file_path('fake_surf_data')
        os.makedirs(self.surf_data, exist_ok=True)
        X = random(size = (10, 10242))

        for x in range(len(X)):
            np.save(self.surf_data + '/' + str(x) + '_lh', X[x])
        for x in range(len(X)):
            np.save(self.surf_data + '/' + str(x) + '_rh', X[x])

        self.time_data = get_file_path('fake_time_data')
        os.makedirs(self.time_data, exist_ok=True)
        X = random(size = (10, 5, 10242))

        for x in range(len(X)):
            np.save(self.time_data + '/' + str(x) + '_lh', X[x])
        for x in range(len(X)):
            np.save(self.time_data + '/' + str(x) + '_rh', X[x])

    def test_integration(self):

        files = os.listdir(self.time_data)
        lh_timeseries = [self.time_data + f for f in files if '_lh' in f]
        rh_timeseries = [self.time_data + f for f in files if '_rh' in f]

        files = os.listdir(self.surf_data)
        lh_surf = [self.surf_data + f for f in files if '_lh' in f]
        rh_surf = [self.surf_data + f for f in files if '_rh' in f]

        subjects = [str(i) for i in range(len(lh_surf))]

        df = pd.DataFrame()

        df['lh_time'] = lh_timeseries
        df['rh_time'] = rh_timeseries

        df['lh_surf'] = lh_surf
        df['rh_surf'] = rh_surf

        df['src_subject_id'] = subjects
        df['target'] = np.random.randint(2, size=len(lh_surf))

        self.ML.Load_Data_Files(df = df,
                                load_func = np.load,
                                drop_keys = ['target'],
                                in_memory = False)

        self.assertTrue(len(self.ML.file_mapping) == 40)

        self.ML.Load_Targets(df = df, col_name='target', data_type='b')
        self.ML.Train_Test_Split(test_size=0)

