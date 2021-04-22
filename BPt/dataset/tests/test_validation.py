import numpy as np
from ..Dataset import Dataset
import pytest
import os
import tempfile
from ...main.input import CVStrategy


def get_fake_dataset():
    '''
    fake = Dataset()
    fake['1'] = [1, 1, 1]
    fake['2'] = [1, 1, 1]
    fake['3'] = ['2', '2', '2']
    fake['4'] = ['2', '2', '2']
    fake['5'] = ['2', '1', '2']

    fake = fake.set_role('1', 'data')
    fake = fake.set_role('2', 'target')
    fake = fake.set_role(['3', '4', '5'], 'non input')

    fake.ordinalize(scope='all', inplace=True)
    '''

    fake = Dataset()
    fake['1'] = [1, 1, 1]
    fake['2'] = [1, 1, 1]
    fake['3'] = ['2', '2', '2']
    fake['4'] = ['2', '2', '2']
    fake['5'] = ['2', '1', '2']

    fake = fake.set_role('1', 'data')
    fake = fake.set_role('2', 'target')
    fake = fake.set_role(['3', '4', '5'], 'non input')

    fake.ordinalize(scope='all', inplace=True)

    return fake


def test_proc_cv_strategy_base():

    df = get_fake_dataset()

    cv_params = CVStrategy()

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

    cv_params = CVStrategy(train_only_subjects=[0, 1])
    cv = df._proc_cv_strategy(cv_params)

    assert cv.groups is None
    assert cv.stratify is None
    assert np.array_equal(cv.train_only, np.array([0, 1]))

    # Make sure sorts for repeatable behavior
    cv_params = CVStrategy(train_only_subjects=[0, 1])
    cv = df._proc_cv_strategy(cv_params)

    assert cv.groups is None
    assert cv.stratify is None
    assert np.array_equal(cv.train_only, np.array([0, 1]))


def test_proc_cv_strategy_groups():

    df = get_fake_dataset()

    with pytest.raises(RuntimeError):
        cv_params = CVStrategy(groups=['1', '2'])

    cv_params = CVStrategy(groups='1')
    with pytest.raises(RuntimeError):
        cv = df._proc_cv_strategy(cv_params)

    cv_params = CVStrategy(groups="doesn't exist")
    with pytest.raises(KeyError):
        cv = df._proc_cv_strategy(cv_params)

    cv_params = CVStrategy(groups='2')
    with pytest.raises(RuntimeError):
        cv = df._proc_cv_strategy(cv_params)

    cv_params = CVStrategy(groups='3')
    cv = df._proc_cv_strategy(cv_params)
    assert len(cv.groups) == 3
    assert cv.groups.nunique() == 1


def test_proc_cv_strategy_stratify():

    df = get_fake_dataset()

    with pytest.raises(RuntimeError):
        cv_params = CVStrategy(stratify=['1', '2'])

    cv_params = CVStrategy(stratify='1')
    with pytest.raises(RuntimeError):
        cv = df._proc_cv_strategy(cv_params)

    cv_params = CVStrategy(stratify="doesn't exist")
    with pytest.raises(KeyError):
        cv = df._proc_cv_strategy(cv_params)

    cv_params = CVStrategy(stratify='2')
    cv = df._proc_cv_strategy(cv_params)
    assert len(cv.stratify) == 3
    assert cv.stratify.nunique() == 1

    cv_params = CVStrategy(stratify='3')
    cv = df._proc_cv_strategy(cv_params)
    assert len(cv.stratify) == 3
    assert cv.stratify.nunique() == 1


def test_set_test_split():

    df = get_fake_dataset()
    df.verbose = -1

    with pytest.raises(TypeError):
        df = df.set_test_split()

    with pytest.raises(TypeError):
        df = df.set_test_split(size=.2, subjects=[1, 2])

    df = df.set_test_split(size=1, cv_strategy=None, random_state=None)

    assert len(df.test_subjects) == 1
    assert len(df.train_subjects) == 2

    df = df.set_test_split(size=.3, cv_strategy=None, random_state=None)

    assert len(df.test_subjects) == 1
    assert len(df.train_subjects) == 2

    df.set_test_split(size=.5,
                      cv_strategy=CVStrategy(train_only_subjects=[0]),
                      random_state=1, inplace=True)

    assert len(df.test_subjects) == 1
    assert len(df.train_subjects) == 2
    assert 0 in df.train_subjects

    df.set_test_split(size=1, cv_strategy=CVStrategy(train_only_subjects=[0]),
                      random_state=1, inplace=True)

    assert len(df.test_subjects) == 1
    assert len(df.train_subjects) == 2
    assert 0 in df.train_subjects

    temp_loc = os.path.join(tempfile.gettempdir(), 'temp.txt')
    df.save_test_split(temp_loc)
    df = df.set_test_split(subjects=temp_loc)
    os.remove(temp_loc)

    assert len(df.test_subjects) == 1
    assert len(df.train_subjects) == 2
    assert 0 in df.train_subjects


def test_set_train_split():

    df = get_fake_dataset()

    with pytest.raises(TypeError):
        df = df.set_train_split()

    with pytest.raises(TypeError):
        df = df.set_train_split(size=.2, subjects=[1, 2])

    df = df.set_train_split(size=1, cv_strategy=None, random_state=None)
    assert len(df.test_subjects) == 2
    assert len(df.train_subjects) == 1

    df = df.set_train_split(size=.4, cv_strategy=None, random_state=None)
    assert len(df.test_subjects) == 2
    assert len(df.train_subjects) == 1

    df.set_train_split(size=.5,
                       cv_strategy=CVStrategy(train_only_subjects=[0]),
                       random_state=1, inplace=True)
    assert len(df.test_subjects) == 1
    assert len(df.train_subjects) == 2
    assert 0 in df.train_subjects

    df.set_train_split(size=1,
                       cv_strategy=CVStrategy(train_only_subjects=[0]),
                       random_state=1, inplace=True)
    assert len(df.test_subjects) == 1
    assert len(df.train_subjects) == 2
    assert 0 in df.train_subjects

    # Test save and load
    temp_loc = os.path.join(tempfile.gettempdir(), 'temp.txt')
    df.save_train_split(temp_loc)
    df = df.set_train_split(subjects=temp_loc)
    os.remove(temp_loc)

    assert len(df.test_subjects) == 1
    assert len(df.train_subjects) == 2
    assert 0 in df.train_subjects

    with pytest.raises(ValueError):
        df.set_train_split(size=1,
                           cv_strategy=CVStrategy(train_only_subjects=[0, 1]),
                           random_state=1, inplace=True)

    with pytest.raises(RuntimeError):
        df = df.set_train_split(size=0)


def test_test_split():

    df = get_fake_dataset()
    df.add_scope('1', 'test_scope', inplace=True)

    tr_df, test_df = df.test_split(subjects=[2])

    # Check copy data behavior
    original = tr_df.loc[0, '5']
    tr_df.loc[0, '5'] = 0
    assert tr_df.loc[0, '5'] != df.loc[0, '5']
    assert original == df.loc[0, '5']

    # Check some assumptions
    assert tr_df.shape == (2, 5)
    assert 0 in tr_df.index
    assert 1 in tr_df.index

    assert test_df.shape == (1, 5)
    assert 2 in test_df.index

    for d in [tr_df, test_df]:
        assert isinstance(d, Dataset)
        assert d.test_subjects is None
        assert d.train_subjects is None

        scopes = d.get_scopes()
        assert 'test_scope' in scopes['1']

    # Check meta data behavior
    tr_df.add_scope('1', 'train', inplace=True)
    scopes = df.get_scopes()
    assert 'train' not in scopes['1']

    test_scopes = test_df.get_scopes()
    assert 'train' not in test_scopes['1']


def test_train_split():

    df = get_fake_dataset()
    tr_df, test_df = df.train_split(subjects=[0])
    assert tr_df.shape == (1, 5)
    assert 0 in tr_df.index
    assert test_df.shape == (2, 5)
    assert 1 in test_df.index
    assert 2 in test_df.index


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

    cv_params = CVStrategy(train_only_subjects=['s1'])
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
    df = df.copy_as_non_input('0', 'zero')

    cv_params = CVStrategy(groups='zero')
    cv = df._proc_cv_strategy(cv_params)

    assert len(cv.groups) == 6
    assert cv.groups.nunique() == 6


def test_multi_index_set_test_split():

    df = get_fake_multi_index_dataset()
    df.verbose = -1

    df = df.set_test_split(size=1, cv_strategy=None, random_state=None)
    assert len(df.test_subjects) == 1
    assert len(df.train_subjects) == 5

    df = df.set_test_split(size=0, cv_strategy=None, random_state=None)
    assert len(df.test_subjects) == 0
    assert len(df.train_subjects) == 6

    df = df.set_test_split(size=.3, cv_strategy=None, random_state=None)
    assert len(df.test_subjects) == 2
    assert len(df.train_subjects) == 4

    df.set_test_split(size=.5,
                      cv_strategy=CVStrategy(train_only_subjects=['s1']),
                      random_state=1, inplace=True)

    assert len(df.test_subjects) == 2
    assert len(df.train_subjects) == 4
    assert df.loc[df.train_subjects].shape == (4, 3)
    assert ('s1', 'e1') in df.train_subjects

    temp_loc = os.path.join(tempfile.gettempdir(), 'temp.txt')
    df.save_test_split(temp_loc)
    df = df.set_test_split(subjects=temp_loc)
    os.remove(temp_loc)

    assert len(df.test_subjects) == 2
    assert len(df.train_subjects) == 4
    assert df.loc[df.train_subjects].shape == (4, 3)
    assert ('s1', 'e1') in df.train_subjects


def test_multi_index_set_train_split():

    df = get_fake_multi_index_dataset()

    df = df.set_train_split(size=1, cv_strategy=None, random_state=None)
    assert len(df.test_subjects) == 5
    assert len(df.train_subjects) == 1

    df = df.set_train_split(size=.2, cv_strategy=None, random_state=None)
    assert len(df.test_subjects) == 5
    assert len(df.train_subjects) == 1

    df.set_train_split(size=2,
                       cv_strategy=CVStrategy(train_only_subjects=['s1']),
                       random_state=1, inplace=True)

    assert len(df.test_subjects) == 2
    assert len(df.train_subjects) == 4
    assert df.loc[df.train_subjects].shape == (4, 3)
    assert ('s1', 'e1') in df.train_subjects

    temp_loc = os.path.join(tempfile.gettempdir(), 'temp.txt')
    df.save_train_split(temp_loc)
    df = df.set_train_split(subjects=temp_loc)
    os.remove(temp_loc)

    assert len(df.test_subjects) == 2
    assert len(df.train_subjects) == 4
    assert df.loc[df.train_subjects].shape == (4, 3)
    assert ('s1', 'e1') in df.train_subjects
