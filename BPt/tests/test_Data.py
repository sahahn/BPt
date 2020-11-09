from nose.tools import raises
from unittest import TestCase
from BPt import BPt_ML

import os
import numpy as np
TEST_FILE_DR = 'test_files/'


def get_file_path(name):

    file_path = os.path.join(os.path.dirname(__file__), TEST_FILE_DR, name)
    return file_path


class Test_Data(TestCase):

    def __init__(self, *args, **kwargs):
        super(Test_Data, self).__init__(*args, **kwargs)

        self.ML = BPt_ML(log_dr=None, use_abcd_subject_ids=True)

    def test_init(self):

        should_be_empty = [self.ML.data, self.ML.covars, self.ML.targets,
                           self.ML.strat]

        for df in should_be_empty:
            self.assertTrue(len(df) == 0)

    def test_Load_Name_Map1(self):

        test_name_map_loc = get_file_path('name_map1.csv')

        self.ML.Load_Name_Map(loc=test_name_map_loc,
                              dataset_type='explorer',
                              source_name_col='src',
                              target_name_col='target')

        self.assertTrue(type(self.ML.name_map) == dict)
        self.assertTrue(len(self.ML.name_map) == 3)

    def test_Load_Name_Map2(self):

        test_name_map_loc = get_file_path('name_map1.csv')

        self.ML.Load_Name_Map(loc=test_name_map_loc,
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
            self.ML.Load_Data(loc=loc, dataset_type=dataset_type,
                              drop_col_duplicates=False,
                              eventname="baseline_year_1_arm_1")

            self.assertTrue(self.ML.data.shape == (2, 3))
            self.assertTrue('NDAR_1' in self.ML.data.index)

            # Should clear data
            self.ML.Clear_Data()
            self.assertTrue(len(self.ML.data) == 0)

            # Should load all 3 subjects
            self.ML.Load_Data(loc=loc, dataset_type=dataset_type)
            self.assertTrue(self.ML.data.shape == (3, 3))
            self.assertTrue('NDAR_2' in self.ML.data.index)

            # Reset
            self.ML.Clear_Data()

        self.ML.Load_Data(loc=locs, dataset_type=dataset_types,
                          eventname="baseline_year_1_arm_1")
        self.assertTrue(self.ML.data.shape == (2, 9))
        self.assertTrue('NDAR_1' in self.ML.data.index)
        self.assertTrue('NDAR_3' in self.ML.data.index)
        self.ML.Clear_Data()

        # Test name map and drop keys
        self.ML.eventname = None
        self.test_Load_Name_Map1()
        self.ML.Load_Data(loc=locs[0], dataset_type=dataset_types[0],
                          drop_keys=['oname2', 'cname3'])

        self.assertTrue('oname1' not in self.ML.data)
        self.assertTrue('cname1' in self.ML.data)
        self.assertTrue('cname2' in self.ML.data)
        self.assertTrue('cname3' not in self.ML.data)
        self.ML.Clear_Data()

    def test_load_data2(self):

        self.ML.eventname = None
        loc = get_file_path('custom_data2.csv')

        self.ML.Load_Data(loc=loc, dataset_type='custom',
                          filter_outlier_percent=10)

        self.assertTrue(self.ML.data.shape == (8, 3))
        self.ML.Clear_Data()

        self.ML.Load_Data(loc=loc, dataset_type='custom',
                          filter_outlier_percent=(None, 10))

        self.assertTrue(self.ML.data.shape == (9, 3))
        self.ML.Clear_Data()

        self.ML.Load_Data(loc=loc, dataset_type='custom',
                          filter_outlier_percent=(.1))
        self.assertTrue(self.ML.data.shape == (8, 3))
        self.ML.Clear_Data()

        self.ML.Load_Data(loc=loc, dataset_type='custom',
                          filter_outlier_percent=20)
        self.assertTrue(self.ML.data.shape == (6, 3))
        self.ML.Clear_Data()

    def test_load_data3(self):

        self.ML.eventname = None
        loc = get_file_path('custom_data1.csv')

        self.ML.Load_Data(loc=loc, dataset_type='custom',
                          drop_col_duplicates=True)
        self.assertTrue(self.ML.data.shape == (3, 1))

    @raises(AssertionError)
    def test_load_target1(self):

        loc = get_file_path('basic_data1.txt')
        self.ML.Load_Targets(loc, 'gender', 'b')

    def test_load_target2(self):

        loc = get_file_path('custom_covars1.csv')
        self.ML.Load_Targets(loc=loc, col_name='sex', data_type='b',
                             dataset_type='custom')
        self.assertTrue(self.ML.targets.shape == (5, 1))

        self.ML.Clear_Targets()
        self.assertTrue(len(self.ML.targets) == 0)

        self.ML.Load_Targets(loc=loc, col_name='age', data_type='f',
                             dataset_type='custom',
                             filter_outlier_percent=5)
        self.assertTrue(self.ML.targets.shape == (4, 1))
        self.ML.Clear_Targets()

        self.ML.Load_Targets(loc=loc, col_name='sex', data_type='c',
                             dataset_type='custom')
        self.assertTrue(self.ML.targets.shape == (6, 1))

    @raises(KeyError)
    def test_load_target3(self):

        loc = get_file_path('custom_covars1.csv')
        self.ML.Load_Targets(loc=loc, col_name='fake_key', data_type='b',
                             dataset_type='custom')

    def test_validation1(self):

        loc = get_file_path('custom_data2.csv')
        self.ML.Load_Data(loc=loc, dataset_type='custom')

        loc = get_file_path('custom_covars1.csv')

        self.ML.Load_Targets(loc=loc, col_name='sex', data_type='b',
                             dataset_type='custom')

        self.ML.Load_Strat(loc=loc, col_name='education',
                           dataset_type='custom')
        self.assertTrue(self.ML.strat.shape == (6, 1))
        ed_size =\
            len(np.unique(self.ML.strat['education' + self.ML.strat_u_name]))
        self.assertTrue(ed_size == 3)

        self.ML.Define_Validation_Strategy(groups='education')
        self.assertTrue(len(self.ML.cv.groups == 6))

    def test_load_exclusions1(self):

        self.ML.eventname = None
        loc = get_file_path('custom_data2.csv')

        self.ML.Load_Data(loc=loc, dataset_type='custom')
        self.assertTrue(self.ML.data.shape == (10, 3))

        loc = get_file_path('exclusions1.txt')
        self.ML.Load_Exclusions(loc=loc)
        self.assertTrue(self.ML.data.shape == (7, 3))

        self.ML.Load_Exclusions(subjects='4')
        self.ML.Load_Exclusions(subjects=['Ndar_notreeal', 'NDAR_5'])
        self.assertTrue(self.ML.data.shape == (5, 3))

        self.ML.Clear_Exclusions()
        self.assertTrue(len(self.ML.exclusions) == 0)

        # Data shouldn't change
        self.assertTrue(self.ML.data.shape == (5, 3))

    def test_load_inclusions1(self):

        self.ML.eventname = None
        loc = get_file_path('custom_data2.csv')

        self.ML.Load_Inclusions(subjects=['1'])
        self.ML.Load_Data(loc=loc, dataset_type='custom')
        self.assertTrue(self.ML.data.shape == (1, 3))

    def test_load_all1(self):

        loc = get_file_path('custom_data2.csv')
        self.ML.Load_Data(loc=loc, dataset_type='custom')

        loc = get_file_path('custom_covars1.csv')
        self.ML.Load_Targets(loc=loc, col_name='sex', data_type='b',
                             dataset_type='custom')
        self.ML.Load_Covars(loc=loc, col_name='education', data_type='c',
                            dataset_type='custom')

        # With low memory mode false, data should be same
        self.assertTrue(self.ML.data.shape == (10, 3))
        self.assertTrue(self.ML.targets.shape == (5, 1))

        self.ML.Load_Strat(loc=loc, col_name='education',
                           dataset_type='custom',
                           overlap_subjects=True)
        self.assertTrue(self.ML.strat.shape == (5, 1))

        self.ML.Define_Validation_Strategy(groups='education')
        self.assertTrue(len(self.ML.cv.groups == 5))

        rs = 1
        for x in range(50):
            self.ML.Train_Test_Split(test_size=.2,
                                     random_state=rs)
            rs += 1

            train_groups = set(self.ML.strat.loc[self.ML.train_subjects])
            test_groups = set(self.ML.strat.loc[self.ML.test_subjects])

            self.assertTrue(len(train_groups.intersection(test_groups)))

        self.assertTrue(self.ML.all_data.shape == (5, 6))

        self.ML.Define_Validation_Strategy(stratify=self.ML.targets_keys[0])
        self.assertTrue(len(self.ML.cv.stratify) == 5)
        self.assertTrue(len(np.unique(self.ML.cv.stratify)) == 2)

        self.ML.Define_Validation_Strategy(stratify=[self.ML.targets_keys[0],
                                                     'education'])
        self.assertTrue(len(self.ML.cv.stratify) == 5)
        self.assertTrue(len(np.unique(self.ML.cv.stratify)) == 5)

        # Load target as categorical
        self.ML.Load_Targets(loc=loc, col_name='sex', data_type='c',
                             dataset_type='custom', clear_existing=True)
        self.ML.Define_Validation_Strategy(
            stratify=self.ML.targets_keys[0])
        self.assertTrue(len(self.ML.cv.stratify) == 5)

        self.ML.Train_Test_Split(test_subjects=['NDAR_1'])
        self.assertTrue(len(self.ML.test_subjects) == 1)
