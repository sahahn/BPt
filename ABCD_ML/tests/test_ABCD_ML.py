from nose.tools import *
from unittest import TestCase
from ABCD_ML.ABCD_ML import ABCD_ML

import os
TEST_FILE_DR = 'test_files/'


def get_file_path(name):

    file_path = os.path.join(os.path.dirname(__file__), TEST_FILE_DR, name)
    return file_path


class Test_ABCD_ML(TestCase):

    def __init__(self, *args, **kwargs):
        super(Test_ABCD_ML, self).__init__(*args, **kwargs)

        self.ML = ABCD_ML()

    def test_init(self):

        should_be_empty = [self.ML.data, self.ML.covars, self.ML.targets,
                           self.ML.strat]

        for df in should_be_empty:
            self.assertTrue(len(df) == 0)

    def test_load_name_map1(self):

        test_name_map_loc = get_file_path('name_map1.csv')

        self.ML.load_name_map(loc=test_name_map_loc,
                              dataset_type='explorer',
                              source_name_col='src',
                              target_name_col='target')

        self.assertTrue(type(self.ML.name_map) == dict)
        self.assertTrue(len(self.ML.name_map) == 3)

    def test_load_name_map2(self):

        test_name_map_loc = get_file_path('name_map1.csv')

        self.ML.load_name_map(loc=test_name_map_loc,
                              dataset_type='explorer',
                              source_name_col='fake_src')

        self.assertTrue(len(self.ML.name_map) == 0)

    def test_load_data1(self):

        locs = [get_file_path('basic_data1.txt'),
                get_file_path('explorer_data1.csv'),
                get_file_path('custom_data1.csv')]
        dataset_types = ['basic', 'explorer', 'custom']

        for loc, dataset_type in zip(locs, dataset_types):

            # Should drop second subject for eventname
            self.ML.load_data(loc=loc, dataset_type=dataset_type)
            self.assertTrue(self.ML.data.shape == (2, 3))
            self.assertTrue('NDAR_1' in self.ML.data.index)

            # Should clear data
            self.ML.clear_data()
            self.assertTrue(len(self.ML.data) == 0)

            # Should load all 3 subjects
            self.ML.eventname = None
            self.ML.load_data(loc=loc, dataset_type=dataset_type)
            self.assertTrue(self.ML.data.shape == (3, 3))
            self.assertTrue('NDAR_2' in self.ML.data.index)

            # Reset
            self.ML.clear_data()
            self.ML.eventname = 'baseline_year_1_arm_1'

        self.ML.load_data(loc=locs, dataset_type=dataset_types)
        self.assertTrue(self.ML.data.shape == (2, 9))
        self.assertTrue('NDAR_1' in self.ML.data.index)
        self.assertTrue('NDAR_3' in self.ML.data.index)
        self.ML.clear_data()

        # Test name map and drop keys
        self.ML.eventname = None
        self.test_load_name_map1()
        self.ML.load_data(loc=locs[0], dataset_type=dataset_types[0],
                          drop_keys=['oname2', 'cname3'])

        self.assertTrue('oname1' not in self.ML.data)
        self.assertTrue('cname1' in self.ML.data)
        self.assertTrue('cname2' in self.ML.data)
        self.assertTrue('cname3' not in self.ML.data)
        self.ML.clear_data()

    def test_load_data2(self):
        # Test filter outlier percent and winsorize

        self.ML.eventname = None
        loc = get_file_path('custom_data2.csv')

        self.ML.load_data(loc=loc, dataset_type='custom',
                          filter_outlier_percent=10)
        self.assertTrue(self.ML.data.shape == (8, 3))
        self.ML.clear_data()

        self.ML.load_data(loc=loc, dataset_type='custom',
                          filter_outlier_percent=(0, 90))
        self.assertTrue(self.ML.data.shape == (9, 3))
        self.ML.clear_data()

        self.ML.load_data(loc=loc, dataset_type='custom',
                          filter_outlier_percent=(.1))
        self.assertTrue(self.ML.data.shape == (8, 3))
        self.ML.clear_data()

        self.ML.load_data(loc=loc, dataset_type='custom',
                          filter_outlier_percent=.2)
        self.assertTrue(self.ML.data.shape == (6, 3))
        self.ML.clear_data()

        self.ML.load_data(loc=loc, dataset_type='custom',
                          winsorize_val=.2)
        self.assertTrue(self.ML.data.loc['NDAR_1']['oname1'] == 3)
        self.assertTrue(self.ML.data.loc['NDAR_10']['oname1'] == 8)
        self.ML.clear_data()

        self.ML.load_data(loc=loc, dataset_type='custom',
                          winsorize_val=(.2, 0))
        self.assertTrue(self.ML.data.loc['NDAR_1']['oname1'] == 3)
        self.assertTrue(self.ML.data.loc['NDAR_10']['oname1'] == 10)
        self.ML.clear_data()

    def test_common_load(self):
        pass

    def test_load_covars(self):

        pass

        #loc = get_file_path('custom_data2.csv')

        #self.ML.load_covars(loc, col_names, data_types, dataset_type='default',
        #                    dummy_code_categorical=True, filter_float_outlier_percent=None,
        #                    standardize=True, normalize=False)
