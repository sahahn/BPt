from nose.tools import *
from unittest import TestCase
from ABCD_ML.ABCD_ML import ABCD_ML

import os
import numpy as np
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

    def test_load_covars1(self):

        loc = get_file_path('basic_covars1.txt')

        self.ML.load_covars(loc, 'gender', 'b')
        self.assertTrue(self.ML.covars.shape == (2, 1))
        self.assertTrue(len(np.unique(self.ML.covars['gender'])) == 2)

        self.ML.clear_covars()
        self.assertTrue(len(self.ML.covars) == 0)

        loc = get_file_path('custom_covars1.csv')
        self.ML.load_covars(loc, ['sex', 'age', 'education'], ['b', 'f', 'o'],
                            dataset_type='custom', dummy_code_categorical=True,
                            standardize=False, normalize=False
                            )

        # Check correct binary load
        self.assertTrue(len(np.unique(self.ML.covars['sex']) == 2))

        # Should filter out NDAR_4, with sex = 5
        self.assertTrue(self.ML.covars.shape == (5, 3))
        self.assertTrue('NDAR_4' not in self.ML.covars)
        self.ML.clear_covars()

        self.ML.load_covars(loc, 'education', 'c',
                            dataset_type='custom', dummy_code_categorical=True,
                            standardize=False, normalize=False
                            )
        self.assertTrue(self.ML.covars.shape == (6, 2))
        self.ML.clear_covars()

        self.ML.load_covars(loc, 'age', 'f',
                            dataset_type='custom', dummy_code_categorical=True,
                            filter_float_outlier_percent=.01,
                            standardize=True, normalize=True
                            )

        self.assertTrue(self.ML.covars.shape == (4, 1))
        self.assertTrue(min(self.ML.covars['age']) == 0)
        self.assertTrue(max(self.ML.covars['age']) == 1)

    @raises(AssertionError)
    def test_load_target1(self):

        loc = get_file_path('basic_data1.txt')
        self.ML.load_targets(loc, 'gender', 'b')

    def test_load_target2(self):

        loc = get_file_path('custom_covars1.csv')
        self.ML.load_targets(loc, 'sex', 'b', dataset_type='custom')
        self.assertTrue(self.ML.targets.shape == (5, 1))
        self.assertTrue(len(np.unique(
            self.ML.targets[self.ML.targets_key])) == 2)

        self.ML.clear_targets()
        self.assertTrue(len(self.ML.targets) == 0)

        self.ML.load_targets(loc, 'age', 'f', dataset_type='custom',
                             filter_outlier_percent=5)
        self.assertTrue(self.ML.targets.shape == (4, 1))
        self.ML.clear_targets()

        self.ML.load_targets(loc, 'sex', 'c', dataset_type='custom')
        self.assertTrue(self.ML.targets.shape == (6, 3))

    @raises(KeyError)
    def test_load_target3(self):

        loc = get_file_path('custom_covars1.csv')
        self.ML.load_targets(loc, 'fake_key', 'b', dataset_type='custom')

    def test_validation1(self):

        loc = get_file_path('custom_covars1.csv')

        self.ML.load_strat(loc, 'education', dataset_type='custom')
        self.assertTrue(self.ML.strat.shape == (6, 1))
        self.assertTrue(len(np.unique(self.ML.strat['education'])) == 3)

        self.ML.define_validation_strategy(groups='education')
        self.assertTrue(len(self.ML.CV.groups == 6))

        self.ML.load_strat(loc, 'sex', dataset_type='custom',
                           binary_col_inds=0)
        self.assertTrue(self.ML.strat.shape == (5, 2))

        self.ML.define_validation_strategy(groups=['education', 'sex'])
        self.assertTrue(len(self.ML.CV.groups == 5))

        self.ML.define_validation_strategy(stratify=['education', 'sex'])
        self.assertTrue(len(self.ML.CV.stratify == 5))
        self.assertTrue(len(np.unique(self.ML.CV.stratify)) == 5)

        self.ML.define_validation_strategy(stratify='sex')
        self.assertTrue(len(self.ML.CV.stratify == 5))

        self.ML.clear_strat()
        self.assertTrue(len(self.ML.strat) == 0)

    @raises(AttributeError)
    def test_validation2(self):

        loc = get_file_path('custom_covars1.csv')

        self.ML.load_strat(loc, 'education', dataset_type='custom')
        self.ML.define_validation_strategy(
            stratify=self.ML.original_targets_key)

    def test_load_exclusions1(self):

        self.ML.eventname = None
        loc = get_file_path('custom_data2.csv')

        self.ML.load_data(loc=loc, dataset_type='custom')
        self.assertTrue(self.ML.data.shape == (10, 3))

        loc = get_file_path('exclusions1.txt')
        self.ML.load_exclusions(loc=loc)
        self.assertTrue(self.ML.data.shape == (7, 3))

        self.ML.load_exclusions(exclusions='4')
        self.ML.load_exclusions(exclusions=['Ndar_notreeal', 'NDAR_5'])
        self.assertTrue(self.ML.data.shape == (5, 3))

        self.ML.clear_exclusions()
        self.assertTrue(len(self.ML.exclusions) == 0)

        # Data shouldn't change
        self.assertTrue(self.ML.data.shape == (5, 3))

    def test_load_all1(self):

        self.ML.eventname = None
        loc = get_file_path('custom_data2.csv')
        self.ML.load_data(loc=loc, dataset_type='custom')

        loc = get_file_path('custom_covars1.csv')
        self.ML.load_targets(loc, 'sex', 'b', dataset_type='custom')

        self.ML.load_covars(loc, 'education', 'c', dataset_type='custom')

        # With low memory mode false, data should be same
        self.assertTrue(self.ML.data.shape == (10, 3))
        self.assertTrue(self.ML.targets.shape == (5, 1))

        self.ML.load_strat(loc, 'education', dataset_type='custom')
        self.assertTrue(self.ML.strat.shape == (5, 1))

        self.ML.define_validation_strategy(groups='education')
        self.assertTrue(len(self.ML.CV.groups == 5))

        rs = 1
        for x in range(50):
            self.ML.train_test_split(test_size=.2,
                                     random_state=rs)
            rs += 1

            train_groups = set(self.ML.strat.loc[self.ML.train_subjects])
            test_groups = set(self.ML.strat.loc[self.ML.test_subjects])

            self.assertTrue(len(train_groups.intersection(test_groups)))

        self.assertTrue(self.ML.all_data.shape == (5, 6))

        self.ML.define_validation_strategy(stratify=self.ML.targets_key)
        self.assertTrue(len(self.ML.CV.stratify) == 5)
        self.assertTrue(len(np.unique(self.ML.CV.stratify)) == 2)

        self.ML.define_validation_strategy(stratify=[self.ML.targets_key,
                                                     'education'])
        self.assertTrue(len(self.ML.CV.stratify) == 5)
        self.assertTrue(len(np.unique(self.ML.CV.stratify)) == 5)

        # Load target as categorical
        self.ML.load_targets(loc, 'sex', 'c', dataset_type='custom')
        self.ML.define_validation_strategy(
            stratify=self.ML.original_targets_key)
        self.assertTrue(len(self.ML.CV.stratify) == 5)

        self.ML.train_test_split(test_subjects='NDAR_1')
        self.assertTrue(len(self.ML.test_subjects) == 1)

    def test_low_memory_mode1(self):

        ML = self.ML = ABCD_ML(low_memory_mode=True, eventname=None)

        loc = get_file_path('custom_data2.csv')
        ML.load_data(loc=loc, dataset_type='custom')

        loc = get_file_path('custom_covars1.csv')
        ML.load_targets(loc, 'sex', 'b', dataset_type='custom')

        # With low memory mode false, data should be removed in place
        self.assertTrue(self.ML.data.shape == (5, 3))
        self.assertTrue(self.ML.targets.shape == (5, 1))

        self.ML._prepare_data()
        self.assertTrue(len(self.ML.data) == 0)
        self.assertTrue(len(self.ML.targets) == 0)
