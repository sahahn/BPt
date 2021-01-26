import numpy as np
from ..Dataset import Dataset
from nose.tools import assert_raises
import os
import tempfile
from ...main.Params_Classes import CV_Strategy


def get_fake_dataset():

    fake = Dataset()
    fake['1'] = [1, 1, 1]
    fake['2'] = [1, 1, 1]
    fake['3'] = ['2', '2', '2']
    fake['4'] = ['2', '2', '2']
    fake['5'] = ['2', '1', '2']

    fake.set_roles({'1': 'data',
                    '2': 'target',
                    '3': 'non input',
                    '4': 'non input',
                    '5': 'non input'})

    fake.ordinalize(scope='all')

    return fake


def test_proc_cv_strategy_base():

    df = get_fake_dataset()

    cv_params = CV_Strategy()

    cv = df._proc_cv_strategy(cv_params)
    assert cv.groups is None
    assert cv.stratify is None
    assert cv.train_only is None

    cv = df._proc_cv_strategy(cv_params=None)
    assert cv.groups is None
    assert cv.stratify is None
    assert cv.train_only is None


def test_proc_cv_strategy_train_only():

    df = get_fake_dataset()

    cv_params = CV_Strategy(train_only_subjects=[0, 1])
    cv = df._proc_cv_strategy(cv_params)

    assert cv.groups is None
    assert cv.stratify is None
    assert np.array_equal(cv.train_only, np.array([0, 1]))

    # Make sure sorts for repeatable behavior
    cv_params = CV_Strategy(train_only_subjects=[0, 1])
    cv = df._proc_cv_strategy(cv_params)

    assert cv.groups is None
    assert cv.stratify is None
    assert np.array_equal(cv.train_only, np.array([0, 1]))


def test_proc_cv_strategy_groups():

    df = get_fake_dataset()

    with assert_raises(RuntimeError):
        cv_params = CV_Strategy(groups=['1', '2'])

    cv_params = CV_Strategy(groups='1')
    with assert_raises(RuntimeError):
        cv = df._proc_cv_strategy(cv_params)

    cv_params = CV_Strategy(groups='doesnt exist')
    with assert_raises(KeyError):
        cv = df._proc_cv_strategy(cv_params)

    cv_params = CV_Strategy(groups='2')
    with assert_raises(RuntimeError):
        cv = df._proc_cv_strategy(cv_params)

    cv_params = CV_Strategy(groups='3')
    cv = df._proc_cv_strategy(cv_params)
    assert len(cv.groups) == 3
    assert cv.groups.nunique() == 1


def test_proc_cv_strategy_stratify():

    df = get_fake_dataset()

    with assert_raises(RuntimeError):
        cv_params = CV_Strategy(stratify=['1', '2'])

    cv_params = CV_Strategy(stratify='1')
    with assert_raises(RuntimeError):
        cv = df._proc_cv_strategy(cv_params)

    cv_params = CV_Strategy(stratify='doesnt exist')
    with assert_raises(KeyError):
        cv = df._proc_cv_strategy(cv_params)

    cv_params = CV_Strategy(stratify='2')
    cv = df._proc_cv_strategy(cv_params)
    assert len(cv.stratify) == 3
    assert cv.stratify.nunique() == 1

    cv_params = CV_Strategy(stratify='3')
    cv = df._proc_cv_strategy(cv_params)
    assert len(cv.stratify) == 3
    assert cv.stratify.nunique() == 1


def test_set_test_split():

    df = get_fake_dataset()

    with assert_raises(TypeError):
        df.set_test_split()

    with assert_raises(TypeError):
        df.set_test_split(size=.2, subjects=[1, 2])

    df.set_test_split(size=1, cv_strategy=None, random_state=None)

    assert len(df.test_subjects) == 1
    assert len(df.train_subjects) == 2

    df.set_test_split(size=.3, cv_strategy=None, random_state=None)

    assert len(df.test_subjects) == 1
    assert len(df.train_subjects) == 2

    df.set_test_split(size=.5, cv_strategy=CV_Strategy(train_only_subjects=[0]),
                      random_state=1)

    assert len(df.test_subjects) == 1
    assert len(df.train_subjects) == 2
    assert 0 in df.train_subjects

    df.set_test_split(size=1, cv_strategy=CV_Strategy(train_only_subjects=[0]),
                      random_state=1)

    assert len(df.test_subjects) == 1
    assert len(df.train_subjects) == 2
    assert 0 in df.train_subjects

    temp_loc = os.path.join(tempfile.gettempdir(), 'temp.txt')
    df.save_test_subjects(temp_loc)
    df.set_test_split(subjects=temp_loc)
    os.remove(temp_loc)

    assert len(df.test_subjects) == 1
    assert len(df.train_subjects) == 2
    assert 0 in df.train_subjects


def test_set_train_split():

    df = get_fake_dataset()

    with assert_raises(TypeError):
        df.set_train_split()

    with assert_raises(TypeError):
        df.set_train_split(size=.2, subjects=[1, 2])

    df.set_train_split(size=1, cv_strategy=None, random_state=None)
    assert len(df.test_subjects) == 2
    assert len(df.train_subjects) == 1

    df.set_train_split(size=.4, cv_strategy=None, random_state=None)
    assert len(df.test_subjects) == 2
    assert len(df.train_subjects) == 1

    df.set_train_split(size=.5, cv_strategy=CV_Strategy(train_only_subjects=[0]),
                       random_state=1)
    assert len(df.test_subjects) == 1
    assert len(df.train_subjects) == 2
    assert 0 in df.train_subjects

    df.set_train_split(size=1, cv_strategy=CV_Strategy(train_only_subjects=[0]),
                       random_state=1)
    assert len(df.test_subjects) == 1
    assert len(df.train_subjects) == 2
    assert 0 in df.train_subjects

    # Test save and load
    temp_loc = os.path.join(tempfile.gettempdir(), 'temp.txt')
    df.save_train_subjects(temp_loc)
    df.set_train_split(subjects=temp_loc)
    os.remove(temp_loc)

    assert len(df.test_subjects) == 1
    assert len(df.train_subjects) == 2
    assert 0 in df.train_subjects

    with assert_raises(ValueError):
        df.set_train_split(size=1, cv_strategy=CV_Strategy(train_only_subjects=[0, 1]),
                           random_state=1)

    with assert_raises(RuntimeError):
        df.set_train_split(size=0)


def get_fake_multi_index_dataset():

    fake = Dataset()
    fake['0'] = [1, 2, 3, 4, 5, 6]
    fake['1'] = [1, 2, 3, 4, 5, 6]
    fake['2'] = [1, 2, 3, 4, 5, np.nan]
    fake['subj'] = ['s1', 's2', 's3', 's1', 's2', 's3']
    fake['event'] = ['e1', 'e1', 'e1', 'e2', 'e2', 'e2']
    fake.set_index(['subj', 'event'], inplace=True)

    return fake


def test_multi_index_proc_cv_strategy():

    df = get_fake_multi_index_dataset()

    cv_params = CV_Strategy(train_only_subjects=['s1'])
    cv = df._proc_cv_strategy(cv_params)

    assert len(cv.train_only) == 2
    assert df.loc[cv.train_only].shape == (2, 3)

    subjects = df.get_subjects('all', return_as='flat index')
    assert len(subjects) == 6

    # Make sure flat index works for train only internally
    _, subjects, train_only = cv.get_train_only(subjects)

    assert len(subjects) == 4
    assert len(train_only) == 2


def test_multi_index_proc_cv_strategy_groups():

    df = get_fake_multi_index_dataset()
    df.copy_as_non_input('0', 'zero')

    cv_params = CV_Strategy(groups='zero')
    cv = df._proc_cv_strategy(cv_params)

    assert len(cv.groups) == 6
    assert cv.groups.nunique() == 6


def test_multi_index_set_test_split():

    df = get_fake_multi_index_dataset()

    df.set_test_split(size=1, cv_strategy=None, random_state=None)
    assert len(df.test_subjects) == 1
    assert len(df.train_subjects) == 5

    df.set_test_split(size=0, cv_strategy=None, random_state=None)
    assert len(df.test_subjects) == 0
    assert len(df.train_subjects) == 6

    df.set_test_split(size=.3, cv_strategy=None, random_state=None)
    assert len(df.test_subjects) == 2
    assert len(df.train_subjects) == 4

    df.set_test_split(size=.5, cv_strategy=CV_Strategy(train_only_subjects=['s1']),
                      random_state=1)

    assert len(df.test_subjects) == 2
    assert len(df.train_subjects) == 4
    assert df.loc[df.train_subjects].shape == (4, 3)
    assert ('s1', 'e1') in df.train_subjects

    temp_loc = os.path.join(tempfile.gettempdir(), 'temp.txt')
    df.save_test_subjects(temp_loc)
    df.set_test_split(subjects=temp_loc)
    os.remove(temp_loc)

    assert len(df.test_subjects) == 2
    assert len(df.train_subjects) == 4
    assert df.loc[df.train_subjects].shape == (4, 3)
    assert ('s1', 'e1') in df.train_subjects


def test_multi_index_set_train_split():

    df = get_fake_multi_index_dataset()

    df.set_train_split(size=1, cv_strategy=None, random_state=None)
    assert len(df.test_subjects) == 5
    assert len(df.train_subjects) == 1

    df.set_train_split(size=.2, cv_strategy=None, random_state=None)
    assert len(df.test_subjects) == 5
    assert len(df.train_subjects) == 1

    df.set_train_split(size=2,
                       cv_strategy=CV_Strategy(train_only_subjects=['s1']),
                       random_state=1)

    assert len(df.test_subjects) == 2
    assert len(df.train_subjects) == 4
    assert df.loc[df.train_subjects].shape == (4, 3)
    assert ('s1', 'e1') in df.train_subjects

    temp_loc = os.path.join(tempfile.gettempdir(), 'temp.txt')
    df.save_train_subjects(temp_loc)
    df.set_train_split(subjects=temp_loc)
    os.remove(temp_loc)

    assert len(df.test_subjects) == 2
    assert len(df.train_subjects) == 4
    assert df.loc[df.train_subjects].shape == (4, 3)
    assert ('s1', 'e1') in df.train_subjects
