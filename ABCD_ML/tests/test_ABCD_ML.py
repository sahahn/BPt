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
        self.assertTrue(len(self.ML.name_map) == 2)

    def test_load_name_map2(self):

        test_name_map_loc = get_file_path('name_map1.csv')

        self.ML.load_name_map(loc=test_name_map_loc,
                              dataset_type='explorer',
                              source_name_col='fake_src')

        self.assertTrue(len(self.ML.name_map) == 0)
