from unittest import TestCase
from BPt import BPt_ML
import numpy as np
import pandas as pd


class Test_Data(TestCase):

    def __init__(self, *args, **kwargs):
        super(Test_Data, self).__init__(*args, **kwargs)
        self.ML = BPt_ML(log_dr=None)

        fake_data = pd.DataFrame()
        fake_data['index'] = list(range(60))
        fake_data['1'] = 1
        fake_data['2'] = 2
        fake_data['3'] = 3
        fake_data['4'] = 4
        fake_data['5'] = 5

        self.ML.Set_Default_Load_Params(subject_id='index')

        self.ML.Load_Data(df=fake_data, drop_keys=['1'])
        self.ML.Load_Targets(df=fake_data,
                             col_name='1',
                             data_type='a')

        self.ML.Prepare_All_Data()

    def test_auto_data_type(self):

        self.assertTrue(self.ML.targets.shape == (60, 1))
        self.assertTrue(self.ML.targets_encoders['1'] is None)


test = Test_Data()
test.test_auto_data_type()
