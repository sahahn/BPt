import numpy as np
import pandas as pd
import tempfile
import os
from nose.tools import assert_raises
from ...main.input_operations import Value_Subset, Intersection
from ...main.input import ProblemSpec
from ..helpers import base_load_subjects, save_subjects
from .datasets import (get_fake_dataset, get_fake_dataset2,
                       get_fake_multi_index_dataset, get_fake_dataset4,
                       get_fake_dataset5, get_full_dataset)
from ..Dataset import Dataset


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


def test_check_scopes():

    df = get_fake_dataset()

    df._check_scopes()
    assert 'category' in df.scopes['2']
    assert df._is_category('2')

    df['2'] = df['2'].astype(float)
    df._check_scopes()
    assert 'category' not in df.scopes['2']
    assert not df._is_category('2')


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


def test_add_data_files():

    df = get_fake_dataset()

    def file_to_subject(i):
        return int(i.split('_')[-1])

    files = {'q': ['a_0', 'b_1', 'c_2']}

    df = df.add_data_files(files=files,
                           file_to_subject=file_to_subject,
                           load_func=np.load)

    assert len(df['q']) == 3
    assert df.loc[2, 'q'] == 2

    assert 'a_0' in df.file_mapping[0].loc
    assert 'b_1' in df.file_mapping[1].loc
    assert 'c_2' in df.file_mapping[2].loc

    # Test in place
    df = get_fake_dataset()
    df.add_data_files(files=files,
                      file_to_subject=file_to_subject,
                      load_func=np.load, inplace=True)
    assert len(df['q']) == 3
    assert df.loc[2, 'q'] == 2
    assert 'a_0' in df.file_mapping[0].loc
    assert 'b_1' in df.file_mapping[1].loc
    assert 'c_2' in df.file_mapping[2].loc


def test_data_files_integration():

    df = get_fake_dataset()

    def file_to_subject(i):
        return int(i.split('/')[-1].replace('.npy', ''))

    temp_dr = tempfile.gettempdir()
    locs = []
    for i in range(3):
        loc = os.path.join(temp_dr, str(i) + '.npy')
        locs.append(loc)

        data = np.zeros(shape=2)
        data[:] = i
        np.save(loc, data)

    files = {'data_files': locs}

    # Test add data files
    df = df.add_data_files(files=files,
                           file_to_subject=file_to_subject,
                           load_func=np.load)
    assert len(df['data_files']) == 3
    assert df.loc[0, 'data_files'] == 0
    assert df.loc[2, 'data_files'] == 2

    # Test get values
    values = df.get_values(col='data_files', reduce_func=np.mean)
    assert values.loc[0] == 0
    assert values.loc[1] == 1
    assert values.loc[2] == 2

    values = df.get_values(col='data_files', reduce_func=np.max, n_jobs=2)
    assert values.loc[0] == 0
    assert values.loc[1] == 1
    assert values.loc[2] == 2

    # Test auto detect categorical
    df = df.auto_detect_categorical()
    cat_cols = df.get_cols(scope='category')
    assert cat_cols == ['2', '3']

    # This also tests copy
    df_copy = df.copy(deep=True)

    # Drop outliers by std, should drop all by 1
    df_copy.filter_outliers_by_std(n_std=.5, scope='data file',
                                   drop=True, reduce_func=np.mean,
                                   n_jobs=1, inplace=True)

    # Make sure self._check_file_mapping works
    assert len(df_copy.file_mapping) == 1
    assert 1 in df_copy.file_mapping
    assert len(df_copy) == 1
    assert len(df.file_mapping) == 3

    # Try drop = False
    df_copy = df.copy(deep=True)
    df_copy = df_copy.filter_outliers_by_std(n_std=.5, scope='data file',
                                             drop=False, reduce_func=np.mean,
                                             n_jobs=1)

    assert np.nan in df_copy.file_mapping
    assert len(df_copy.file_mapping) == 2
    assert 1 in df_copy.file_mapping
    assert len(df_copy) == 3


def test_get_subjects_None():

    df = get_fake_dataset()

    subjects = df.get_subjects(None, return_as='set')
    assert len(subjects) == 0

    subjects = df.get_subjects(None, return_as='index')
    assert len(subjects) == 0

    subjects = df.get_subjects(None, return_as='flat index')
    assert len(subjects) == 0


def test_get_subjects_intersection():

    df = get_fake_dataset4()
    assert len(df.get_subjects('all')) == 6
    assert len(df.get_subjects([1, 2])) == 2

    subjs = df.get_subjects(Intersection([[1, 2]]))
    assert len(subjs) == 2

    subjs = df.get_subjects(Intersection([[1, 2], [2, 3]]))
    assert len(subjs) == 1

    subjs = df.get_subjects(Intersection([[1, 2], [2, 3], [4, 5]]))
    assert len(subjs) == 0

    subjs = df.get_subjects(Intersection([[1, 2], [2, 3], [2, 3]]))
    assert len(subjs) == 1


def test_get_subjects_nan():

    # Subject id == 0 is only one with NaN
    df = get_fake_dataset()
    index_dtype = df.index.dtype.name

    subjects = df.get_subjects('nan', return_as='set')
    assert len(subjects) == 1
    assert df.loc[subjects].shape == (1, 3)
    subj = subjects.pop()
    assert subj == 0
    assert isinstance(subj, int)

    subjects = df.get_subjects('nan', return_as='flat index')
    assert len(subjects) == 1
    assert df.loc[subjects].shape == (1, 3)
    assert subjects[0] == 0
    assert subjects[0].dtype == index_dtype

    subjects = df.get_subjects('nan', return_as='index')
    assert len(subjects) == 1
    assert df.loc[subjects].shape == (1, 3)
    assert subjects[0] == 0
    assert subjects[0].dtype == index_dtype

    # Dataset with no Nan's
    df = get_fake_dataset2()
    subjects = df.get_subjects('nan', return_as='set')
    assert len(subjects) == 0


def test_get_subjects_value_subset():

    df = get_fake_dataset()

    # Int column test 1 value
    vs = Value_Subset(name='1', values=1, decode_values=False)
    subjects = df.get_subjects(vs, return_as='set')
    assert len(subjects) == 1
    assert subjects.pop() == 0

    # Int column test 2 values
    vs = Value_Subset(name='1', values=[1, 2], decode_values=False)
    subjects = df.get_subjects(vs, return_as='set')
    assert len(subjects) == 2

    # Str column test 1 value
    vs = Value_Subset(name='2', values=['6'], decode_values=False)
    subjects = df.get_subjects(vs, return_as='set')
    assert len(subjects) == 1
    assert subjects.pop() == 0

    # Str column test 2 values
    vs = Value_Subset(name='2', values=['6', '7'], decode_values=False)
    subjects = df.get_subjects(vs, return_as='set')
    assert len(subjects) == 2

    # Str / cat column test extra values
    vs = Value_Subset(name='2', values=['6', '7', '9'], decode_values=False)
    subjects = df.get_subjects(vs, return_as='set')
    assert len(subjects) == 2

    # Column w/ nan
    vs = Value_Subset(name='3', values=np.nan, decode_values=False)
    subjects = df.get_subjects(vs, return_as='set')
    assert len(subjects) == 1
    assert subjects.pop() == 0

    # Column w/ nan 2 values
    vs = Value_Subset(name='3', values=[np.nan, 2], decode_values=False)
    subjects = df.get_subjects(vs, return_as='set')
    assert len(subjects) == 2
    assert df.loc[subjects].shape == (2, 3)

    # Bad name col
    vs = Value_Subset(name=1, values=[np.nan, 2], decode_values=False)
    with assert_raises(KeyError):
        subjects = df.get_subjects(vs, return_as='set')

    # Bad input
    with assert_raises(ValueError):
        vs = Value_Subset(name=[1, 2], values=2)


def test_get_subjects_base():

    df = get_fake_dataset()

    subjs = [0]
    subjects = df.get_subjects(subjs, return_as='set')
    assert len(subjects) == 1
    assert df.loc[subjects].shape == (1, 3)

    subjs = ['0']
    subjects = df.get_subjects(subjs, return_as='set')
    assert len(subjects) == 1
    assert df.loc[subjects].shape == (1, 3)

    subjs = np.array([0])
    subjects = df.get_subjects(subjs, return_as='set')
    assert len(subjects) == 1
    assert df.loc[subjects].shape == (1, 3)

    subjs = pd.Index(data=[0], name=df.index.name)
    subjects = df.get_subjects(subjs, return_as='set')
    assert len(subjects) == 1
    assert df.loc[subjects].shape == (1, 3)

    subjs = pd.Index(data=np.array([0, 2]), name=df.index.name)
    subjects = df.get_subjects(subjs, return_as='index')
    assert len(subjects) == 2
    assert df.loc[subjects].shape == (2, 3)
    assert np.array_equal(np.array([0, 2]), subjects)

    subjects = df.get_subjects(subjs, return_as='flat index')
    assert len(subjects) == 2
    assert df.loc[subjects].shape == (2, 3)
    assert np.array_equal(np.array([0, 2]), subjects)


def test_get_subjects_base_file():

    df = get_fake_dataset()

    temp_loc = os.path.join(tempfile.gettempdir(), 'temp.txt')
    with open(temp_loc, 'w') as f:
        f.write('0\n')

    subjects = df.get_subjects(temp_loc, return_as='set')
    assert len(subjects) == 1
    assert df.loc[subjects].shape == (1, 3)

    temp_loc = os.path.join(tempfile.gettempdir(), 'temp.txt')
    with open(temp_loc, 'w') as f:
        f.write('0\n')
        f.write('1\n')

    subjects = df.get_subjects(temp_loc, return_as='set')
    assert len(subjects) == 2
    assert df.loc[subjects].shape == (2, 3)


def test_multi_index_load_save():

    df = get_fake_multi_index_dataset()

    loc = os.path.join(tempfile.gettempdir(), 'temp.txt')
    save_subjects(loc, df.index)
    subjs = base_load_subjects(loc)

    for ind in df.index:
        assert ind in subjs


def test_multi_index_get_subjects():

    df = get_fake_multi_index_dataset()

    subjs = df.get_subjects(subjects=None)
    assert len(subjs) == 0

    subjs = df.get_subjects(subjects=['s1'])
    assert len(subjs) == 2
    assert ('s1', 'e1') in subjs
    assert ('s1', 'e2') in subjs

    subjs = df.get_subjects(subjects=['s1'], only_level=0)
    assert len(subjs) == 1
    assert 's1' in subjs

    subjs = df.get_subjects(subjects=['s1'], return_as='index',
                            only_level=0)
    assert 's1' in subjs
    assert subjs.name == 'subj'

    subjs = df.get_subjects(subjects=['s1'], return_as='index',
                            only_level=None)
    assert ('s1', 'e1') in subjs
    assert ('s1', 'e2') in subjs
    assert subjs.names == ['subj', 'event']

    subjs = df.get_subjects(subjects=(['s1', 's2'], ['e1']),
                            return_as='set')

    assert len(subjs) == 2
    assert ('s1', 'e1') in subjs
    assert ('s2', 'e1') in subjs

    subjs = df.get_subjects(subjects=('all', ['e1']),
                            return_as='set')
    assert len(subjs) == 3
    assert df.loc[subjs].shape == (3, 3)


def test_multi_index_add_data_files():

    df = get_fake_multi_index_dataset()

    def file_to_subject(i):

        subj = i.split('_')[1]
        event = i.split('_')[2]

        return (subj, event)

    # Leave c_s3_e2' as NaN
    files = {'files': ['a_s1_e1', 'a_s1_e2',
                       'b_s2_e1', 'b_s2_e2', 'c_s3_e1']}

    df = df.add_data_files(files=files,
                           file_to_subject=file_to_subject,
                           load_func=np.load)

    assert len(df['files']) == 6
    assert 'a_s1_e1' in df.file_mapping[0].loc


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

    X, y = df.get_train_Xy(ps)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)

    X, y = np.array(X), np.array(y)
    assert X.shape == (3, 3)
    assert np.array_equal(X[:, 0], np.array([1, 2, 3]).astype('float32'))
    assert np.array_equal(X[:, 2],
                          np.array([11, 12, 13]).astype('float32'))
    assert np.array_equal(y, np.array([.1, .2, .3]).astype('float64'))

    X, y = df.get_test_Xy(ps)
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


def test_invalid_names():

    df = Dataset()
    df['target'] = ['1']

    with assert_raises(RuntimeError):
        df._check_cols()

    df = Dataset()
    df['data'] = ['1']

    with assert_raises(RuntimeError):
        df._check_cols()
