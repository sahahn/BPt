import numpy as np
import pandas as pd
import tempfile
import os
import pytest
from ...main.input_operations import Intersection
from ...main.input import ProblemSpec
from ..helpers import base_load_subjects, save_subjects
from .datasets import (get_fake_dataset,
                       get_fake_multi_index_dataset,
                       get_fake_dataset5, get_full_dataset,
                       get_full_int_index_dataset)
from ..Dataset import Dataset
import pickle


def test_check_cols():

    data = Dataset()
    data.verbose = -1
    data[1] = [1, 2, 3]
    data[2] = [1, 2, 3]

    data._check_cols()
    assert 1 not in data
    assert 2 not in data
    assert '1' in data
    assert '2' in data


def test_get_roles():

    df = get_fake_dataset()
    df.set_role('1', 'target', inplace=True)
    roles = df.get_roles()
    assert roles['1'] == 'target'
    assert roles['2'] == 'data'


def test_set_role_fail():

    df = get_fake_dataset()

    with pytest.raises(AttributeError):
        df.set_role('1', 'not real role')


def test_set_role_int_col():

    df = get_fake_dataset()
    df.set_role(1, 'target', inplace=True)
    roles = df.get_roles()
    assert roles['1'] == 'target'


def test_pickle():

    df = get_full_dataset()

    temp_dr = tempfile.gettempdir()
    loc = os.path.join(temp_dr, 'temp.pkl')

    with open(loc, 'wb') as f:
        pickle.dump(df, f)

    with open(loc, 'rb') as f:
        df2 = pickle.load(f)

    assert len(df.test_subjects) == len(df2.test_subjects)

    # Clean up when done
    os.remove(loc)

    # Test pandas to_pickle
    df.to_pickle(loc)
    q = pd.read_pickle(loc)
    assert len(df.test_subjects) == len(q.test_subjects)
    assert isinstance(q, Dataset)


def test_indexing():
    '''Want to make sure when we grab a sub-index of the dataframe,
    the attributes gets set in the sub-frame as a copy of the original.'''

    data = Dataset()
    data['1'] = [1, 2, 3]
    data['2'] = [4, 5, 6]

    data = data.set_role('2', 'target')
    assert len(data.roles) == 2

    copy = data[['1']]
    copy.add_scope('1', 'something', inplace=True)

    o_scopes = data.get_scopes()
    c_scopes = copy.get_scopes()

    assert 'something' in c_scopes['1']
    assert 'something' not in o_scopes['1']


def test_add_scope():

    df = get_fake_dataset()

    df.add_scope(scope='1', scope_val='a')
    df._check_scopes()
    assert df.scopes['1'] != set(['a'])

    df = df.add_scope(scope='1', scope_val='a')
    assert df.scopes['1'] == set(['a'])

    df = df.add_scope(scope='1', scope_val='b')
    assert df.scopes['1'] != set(['a'])
    assert df.scopes['1'] == set(['a', 'b'])

    # Test some get cols from scope
    assert(set(df.get_cols('a')) == set(['1']))
    assert(set(df.get_cols('b')) == set(['1']))
    assert(set(df.get_cols(['a', 'b'])) == set(['1']))

    df = get_fake_dataset()
    df = df.add_scope(scope='1', scope_val='category')
    assert(df['1'].dtype.name == 'category')


def test_copy():

    df = get_fake_dataset()
    df_copy = df.copy(deep=True)

    assert df_copy.loc[0, '1'] == 1
    df_copy.loc[0, '1'] = 6
    assert df_copy.loc[0, '1'] == 6
    assert df.loc[0, '1'] == 1


def test_copy_shallow():

    df = get_fake_dataset()
    df_copy = df.copy(deep=False)

    df_copy['4'] = [6, 6, 6]

    assert '4' in list(df_copy)
    assert '4' not in list(df)


def test_scope():

    dataset = Dataset()
    dataset['float col'] = [1, 2, 3, 4, 5, 6]
    dataset['cat'] = [1, 1, 2, 2, 3, 3]
    dataset['cat missing'] = [1, 1, np.nan, 2, np.nan, 3]
    dataset.add_scope(['cat', 'cat missing'], 'category', inplace=True)

    assert dataset.get_cols('category') == ['cat', 'cat missing']


def test_copy_behavior():

    df = get_fake_dataset()

    # Make copy via add scope
    df_copy = df.add_scope(scope='1', scope_val='a')
    df_copy.loc[0, '1'] = 8

    # Setting like
    # this won't effect the original
    df_copy['2'] = [10, 10, 10]
    assert df.loc[0, '2'] != 10

    # Setting like this will effect the original
    assert df_copy.loc[0, '1'] == 8
    assert df.loc[0, '1'] == 8


def test_remove_scope():

    df = get_fake_dataset()

    df = df.add_scope(scope='1', scope_val='a')
    df = df.remove_scope(scope='1', scope_val='a')
    assert df.scopes['1'] != set(['a'])


def test_add_scope_inplace():

    df = get_fake_dataset()

    df.add_scope(scope='1', scope_val='a', inplace=True)
    assert df.scopes['1'] == set(['a'])


def test_remove_scope_inplace():

    df = get_fake_dataset()
    df.add_scope(scope='1', scope_val='a', inplace=True)
    df.remove_scope(scope='1', scope_val='a', inplace=True)
    assert df.scopes['1'] != set(['a'])


def test_to_category_from_bool():

    df = Dataset()
    df['1'] = [True, False]
    df['1'] = df['1'].astype('bool')

    assert df['1'].dtype.name == 'bool'

    df.add_scope('1', 'category', inplace=True)
    assert df['1'].dtype.name == 'category'


def test_add_multiple_scopes():

    df = get_fake_dataset()
    df = df.add_scope('1', ['a', 'b', 'c'])

    scopes = df.get_scopes()
    assert 'a' in scopes['1']
    assert 'b' in scopes['1']
    assert 'c' in scopes['1']


def test_check_scopes():

    df = get_fake_dataset()

    df._check_scopes()
    assert 'category' in df.scopes['2']
    assert df._is_category('2')

    df['2'] = df['2'].astype(float)
    df._check_scopes()
    assert 'category' not in df.scopes['2']
    assert not df._is_category('2')


def test_is_category():

    df = get_fake_dataset()
    assert df._is_category('2')
    assert not df._is_category('all')


def test_set_target_inplace():

    df = get_fake_dataset()
    df.set_target('1', inplace=True)
    assert(set(df.get_cols('target')) == set(['1']))


def test_set_target():

    df = get_fake_dataset()
    df = df.set_target('1')
    assert(set(df.get_cols('target')) == set(['1']))


def test_set_non_input():

    df = get_fake_dataset()
    df = df.set_non_input('2')
    assert(set(df.get_cols('non input')) == set(['2']))


def test_set_non_input_inplace():

    df = get_fake_dataset()
    df.set_non_input('2', inplace=True)
    assert(set(df.get_cols('non input')) == set(['2']))

    df = df.set_non_input('2', inplace=True)
    assert df is None


def test_set_role_inplace():

    df = get_fake_dataset()
    df.set_role('1', 'target', inplace=True)
    df.set_role('2', 'non input', inplace=True)

    assert(set(df.get_cols('target')) == set(['1']))
    assert(set(df.get_cols('non input')) == set(['2']))


def test_set_role():

    df = get_fake_dataset()
    df.set_role('1', 'target')
    df.set_role('2', 'non input')

    assert(set(df.get_cols('target')) != set(['1']))
    assert(set(df.get_cols('non input')) != set(['2']))

    df = df.set_role('1', 'target')
    df = df.set_role('2', 'non input')

    assert(set(df.get_cols('target')) == set(['1']))
    assert(set(df.get_cols('non input')) == set(['2']))


def test_set_roles():

    df = get_fake_dataset()
    df = df.set_roles({'1': 'target', '2': 'non input'})

    assert(set(df.get_cols('target')) == set(['1']))
    assert(set(df.get_cols('non input')) == set(['2']))


def test_set_roles_inplace():

    df = get_fake_dataset()
    df.set_roles({'1': 'target', '2': 'non input'}, inplace=True)

    assert(set(df.get_cols('target')) == set(['1']))
    assert(set(df.get_cols('non input')) == set(['2']))


def test_get_cols():

    df = get_fake_dataset()
    assert(set(df.get_cols('all')) == set(['1', '2', '3']))
    assert(set(df.get_cols('data')) == set(['1', '2', '3']))
    assert(set(df.get_cols('1')) == set(['1']))
    assert(set(df.get_cols('category')) == set(['2']))


def test_get_cols_as_index():

    df = get_fake_dataset()
    assert len(df['all']) == 3
    assert(set(df['data']) == set(['1', '2', '3']))
    assert(set(df['category']) == set(['2']))
    assert isinstance(df['1'], pd.Series)


def test_auto_detect_categorical_inplace():

    df = get_fake_dataset5()
    df.auto_detect_categorical(scope='all', obj_thresh=4,
                               all_thresh=None, inplace=True)
    cat_cols = df.get_cols(scope='category')
    assert cat_cols == ['2', '3', '4']

    df = get_fake_dataset5()
    df.auto_detect_categorical(scope='all', obj_thresh=3,
                               all_thresh=None, inplace=True)
    cat_cols = df.get_cols(scope='category')
    assert cat_cols == ['2', '3']

    df = get_fake_dataset5()
    df.auto_detect_categorical(scope='all', obj_thresh=None,
                               all_thresh=None, inplace=True)
    cat_cols = df.get_cols(scope='category')
    assert cat_cols == ['2', '3']

    df = get_fake_dataset5()
    df.auto_detect_categorical(scope='all', obj_thresh=None,
                               all_thresh=10, inplace=True)
    cat_cols = df.get_cols(scope='category')
    assert cat_cols == ['1', '2', '3', '4']


def test_multi_index_load_save():

    df = get_fake_multi_index_dataset()

    loc = os.path.join(tempfile.gettempdir(), 'temp.txt')
    save_subjects(loc, df.index)
    subjs = base_load_subjects(loc)

    for ind in df.index:
        assert ind in subjs


def test_get_Xy_base():

    df = get_full_dataset()
    ps = ProblemSpec()

    X, y = df.get_Xy(ps)
    X, y = np.array(X), np.array(y)
    assert X.shape == (5, 3)
    assert np.array_equal(X[:, 0], np.array([1, 2, 3, 4, 5]).astype('float32'))
    assert np.array_equal(X[:, 2],
                          np.array([11, 12, 13, 14, 15]).astype('float32'))
    assert np.array_equal(y, np.array([.1, .2, .3, .4, .5]).astype('float64'))

    X, y = df.get_Xy(ps, subjects='train')
    X, y = np.array(X), np.array(y)
    assert X.shape == (3, 3)
    assert np.array_equal(X[:, 0], np.array([1, 2, 3]).astype('float32'))
    assert np.array_equal(X[:, 2],
                          np.array([11, 12, 13]).astype('float32'))
    assert np.array_equal(y, np.array([.1, .2, .3]).astype('float64'))

    X, y = df.get_Xy(ps, subjects='test')
    X, y = np.array(X), np.array(y)
    assert X.shape == (2, 3)
    assert np.array_equal(X[:, 0], np.array([4, 5]).astype('float32'))
    assert np.array_equal(X[:, 2],
                          np.array([14, 15]).astype('float32'))
    assert np.array_equal(y, np.array([.4, .5]).astype('float64'))


def test_get_Xy_base_int_index():

    df = get_full_int_index_dataset()
    ps = ProblemSpec()

    X, y = df.get_Xy(ps)
    X, y = np.array(X), np.array(y)
    assert X.shape == (5, 3)
    assert np.array_equal(X[:, 0], np.array([1, 2, 3, 4, 5]).astype('float32'))
    assert np.array_equal(X[:, 2],
                          np.array([11, 12, 13, 14, 15]).astype('float32'))
    assert np.array_equal(y, np.array([.1, .2, .3, .4, .5]).astype('float64'))

    X, y = df.get_Xy(ps, subjects='train')
    X, y = np.array(X), np.array(y)
    assert X.shape == (3, 3)
    assert np.array_equal(X[:, 0], np.array([1, 2, 3]).astype('float32'))
    assert np.array_equal(X[:, 2],
                          np.array([11, 12, 13]).astype('float32'))
    assert np.array_equal(y, np.array([.1, .2, .3]).astype('float64'))

    X, y = df.get_Xy(ps, subjects='test')
    X, y = np.array(X), np.array(y)
    assert X.shape == (2, 3)
    assert np.array_equal(X[:, 0], np.array([4, 5]).astype('float32'))
    assert np.array_equal(X[:, 2],
                          np.array([14, 15]).astype('float32'))
    assert np.array_equal(y, np.array([.4, .5]).astype('float64'))


def test_get_Xy_alt():

    df = get_full_dataset()
    ps = ProblemSpec(subjects=['s2', 's3', 's4'], scope=['1', '3'])

    X, y = df.get_Xy(ps)
    X, y = np.array(X), np.array(y)

    assert X.shape == (3, 2)
    assert np.array_equal(X[:, 0], np.array([2, 3, 4]).astype('float32'))
    assert np.array_equal(X[:, 1],
                          np.array([12, 13, 14]).astype('float32'))
    assert np.array_equal(y, np.array([.2, .3, .4]).astype('float64'))

    X, y = df.get_Xy(ps, subjects=Intersection(['train', ['s2', 's3', 's4']]),
                     ignore_me='ignore ignore')
    X, y = np.array(X), np.array(y)
    assert X.shape == (2, 2)
    assert np.array_equal(X[:, 0], np.array([2, 3]).astype('float32'))
    assert np.array_equal(X[:, 1],
                          np.array([12, 13]).astype('float32'))
    assert np.array_equal(y, np.array([.2, .3]).astype('float64'))


def test_get_Xy_alt_int_index():

    df = get_full_int_index_dataset()
    ps = ProblemSpec(subjects=[2, 3, 4], scope=['1', '3'])

    X, y = df.get_Xy(ps)
    X, y = np.array(X), np.array(y)

    assert X.shape == (3, 2)
    assert np.array_equal(X[:, 0], np.array([2, 3, 4]).astype('float32'))
    assert np.array_equal(X[:, 1],
                          np.array([12, 13, 14]).astype('float32'))
    assert np.array_equal(y, np.array([.2, .3, .4]).astype('float64'))

    X, y = df.get_Xy(ps, subjects=Intersection(['train', [2, 3, 4]]),
                     ignore_me='ignore ignore')
    X, y = np.array(X), np.array(y)
    assert X.shape == (2, 2)
    assert np.array_equal(X[:, 0], np.array([2, 3]).astype('float32'))
    assert np.array_equal(X[:, 1],
                          np.array([12, 13]).astype('float32'))
    assert np.array_equal(y, np.array([.2, .3]).astype('float64'))


def test_invalid_names():

    df = Dataset()
    df['target'] = ['1']

    with pytest.raises(RuntimeError):
        df._check_cols()

    df = Dataset()
    df['data'] = ['1']

    with pytest.raises(RuntimeError):
        df._check_cols()


def get_data_inds_df():

    df = Dataset(columns=['d1', 'd2', 'd3', 'n1',
                          'n2', 'c1', 'c2', 't1', 't2'])

    df.add_scope('c', 'category', inplace=True)
    df.set_target('t', inplace=True)
    df.set_non_input('n', inplace=True)
    df._check_sr()

    return df


def test_get_data_inds_empty():

    df = get_data_inds_df()

    # All of these should return nothing
    inds = df._get_data_inds(ps_scope='float', scope='category')
    assert len(inds) == 0

    inds = df._get_data_inds(ps_scope='float', scope='non input')
    assert len(inds) == 0

    inds = df._get_data_inds(ps_scope='float', scope=['c1'])
    assert len(inds) == 0

    inds = df._get_data_inds(ps_scope='float', scope=['c1', 'c2'])
    assert len(inds) == 0


def test_get_data_inds():

    df = get_data_inds_df()

    # Based on sorting, should be:
    inds = df._get_data_inds(ps_scope='all', scope='category')
    assert inds == [0, 1]

    inds = df._get_data_inds(ps_scope='all', scope='float')
    assert inds == [2, 3, 4]


def test_get_data_cols():

    df = get_data_inds_df()

    cols = df._get_data_cols('all')
    assert len(cols) == 5

    cols = df._get_data_cols('category')
    assert cols == ['c1', 'c2']

    cols = df._get_data_cols('float')
    assert cols == ['d1', 'd2', 'd3']

    cols = df._get_data_cols(['d1', 'd2'])
    assert cols == ['d1', 'd2']

    # Sort should work
    cols = df._get_data_cols(['d2', 'd1'])
    assert cols == ['d1', 'd2']

    # Should remove repeats
    cols = df._get_data_cols(['d1', 'd2', 'd2', 'd2'])
    assert cols == ['d1', 'd2']


def test_is_data_cat():

    df = get_data_inds_df()

    all_cat = df._is_data_cat(ps_scope='all', scope='category')
    assert all_cat

    all_cat = df._is_data_cat(ps_scope='all', scope=['c1', 'd2'])
    assert not all_cat

    all_cat = df._is_data_cat(ps_scope='all', scope=['d2'])
    assert not all_cat

    # In the case of an invalid scope, should return False
    all_cat = df._is_data_cat(ps_scope='category', scope='d2')
    assert not all_cat


def test_get_cols_limit_to():

    df = get_data_inds_df()

    cols = df._get_cols('category', limit_to=['c1'])
    assert cols == ['c1']

    cols = df._get_cols('category', limit_to=['data file'])
    assert len(cols) == 0


def test_get_problem_type_binary():

    data = Dataset()
    data['1'] = [0, 1, 1, 1]

    assert data._get_problem_type('1') == 'binary'


def test_get_problem_type_binary_error():

    data = Dataset()
    data['1'] = [1, 2, 2, 2]

    with pytest.raises(RuntimeError):
        data._get_problem_type('1')


def test_get_problem_type_categorical():

    data = Dataset()
    data['1'] = [0, 1, 2, 3]
    data['1'] = data['1'].astype('category')

    assert data._get_problem_type('1') == 'categorical'


def test_get_problem_type_regression():

    data = Dataset()
    data['1'] = [0, 1, 2, 3]

    assert data._get_problem_type('1') == 'regression'


def test_default_test_subjects():

    data = Dataset()
    data._check_test_subjects()
    assert data.test_subjects is None


def test_rename_cols1():

    data = Dataset()
    data['1'] = [1, 2, 3]
    data['2'] = [1, 2, 3]

    data = data.add_scope('1', 'some_scope')
    data = data.rename({'1': '3'}, axis=1)

    assert 'some_scope' in data.get_scopes()['3']


def test_rename_cols2():

    data = Dataset()
    data['1'] = [1, 2, 3]
    data['2'] = [1, 2, 3]

    data = data.add_scope('1', 'some_scope')
    data = data.rename(columns={'1': '3'})

    assert 'some_scope' in data.get_scopes()['3']


def test_rename_cols_duplicate_cols():

    data = Dataset()
    data['1'] = [1, 2, 3]
    data['2'] = [1, 2, 3]

    with pytest.raises(RuntimeError):
        data.rename({'1': '2'}, axis=1)


def test_rename_cols_inplace():

    data = Dataset()
    data['1'] = [1, 2, 3]
    data['2'] = [1, 2, 3]

    data = data.add_scope('1', 'some_scope')
    data.rename({'1': '3'}, axis=1)

    assert 'some_scope' in data.get_scopes()['1']
    assert '1' in list(data)


def test_rename_cols_func():

    data = Dataset()
    data['HI'] = [1, 2, 3]
    data['BYE'] = [1, 2, 3]
    data = data.add_scope('all', '1')

    data = data.rename(columns=str.lower)
    assert '1' in data.scopes['hi']
    assert '1' in data.scopes['bye']

    assert data.roles['hi'] == 'data'
    assert data.roles['bye'] == 'data'


def test_rename_index():

    data = Dataset()
    data['HI'] = [1, 2, 3]
    data = data.set_train_split(subjects=[0, 1])

    data = data.rename(index={0: '00', 1: '11', 2: '22'})

    assert '00' in data.train_subjects
    assert '11' in data.train_subjects
    assert '22' in data.test_subjects

    assert 0 not in data.index


def test_rename_index_no_tr_test():

    data = Dataset()
    data['HI'] = [1, 2, 3]

    data = data.rename(index={0: '00', 1: '11', 2: '22'})
    assert 0 not in data.index


def test_rename_index_inplace_test():

    data = Dataset()
    data['HI'] = [1, 2, 3]
    data = data.set_train_split(subjects=[0, 1])

    data.rename(index={0: '00', 1: '11', 2: '22'})

    assert '00' not in data.train_subjects
    assert '11' not in data.train_subjects
    assert '22' not in data.test_subjects

    assert 0 in data.index


def test_rename_index_inplace_test_mapper():

    data = Dataset()
    data['HI'] = [1, 2, 3]
    data = data.set_train_split(subjects=[0, 1])

    data.rename(mapper={0: '00', 1: '11', 2: '22'}, axis=0)

    assert '00' not in data.train_subjects
    assert '11' not in data.train_subjects
    assert '22' not in data.test_subjects

    assert 0 in data.index


def test_extra_constructor_scopes():

    data = Dataset([1, 2, 3], columns=['1'],
                   scopes={'1': 'a'})

    assert 'a' in data.scopes['1']


def test_extra_constructor_roles():

    data = Dataset([1, 2, 3], columns=['1'],
                   roles={'1': 'target'})

    assert data.roles['1'] == 'target'


def test_extra_constructor_targets():

    data = Dataset([1, 2, 3], columns=['1'],
                   targets='1')

    assert data.roles['1'] == 'target'


def test_extra_constructor_non_inputs():

    data = Dataset([1, 2, 3], columns=['1'],
                   non_inputs='1')

    assert data.roles['1'] == 'non input'


def test_repr_html():

    # Just make sure throws no errors
    data = get_full_int_index_dataset()
    data._repr_html_()


def test_get_defaults():

    data = Dataset()
    assert len(data._get_encoders()) == 0
    assert data._get_test_subjects() is None
    assert data._get_train_subjects() is None
