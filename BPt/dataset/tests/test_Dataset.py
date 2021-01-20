import numpy as np
from ..Dataset import Dataset
import pandas as pd
import tempfile
import os
from nose.tools import assert_raises
from ...main.Input_Tools import Value_Subset
from ..helpers import base_load_subjects, save_subjects


def get_fake_dataset():

    fake = Dataset()
    fake['1'] = [1, 2, 3]
    fake['2'] = ['6', '7', '8']
    fake['2'] = fake['2'].astype('category')
    fake['3'] = [np.nan, 2, 3]

    return fake


def get_fake_dataset2():

    fake = Dataset()
    fake['1'] = [1, 1, 1]
    fake['2'] = [1, 2, 3]
    fake['3'] = ['1', '2', '3']

    return fake


def get_fake_dataset3():

    fake = Dataset()
    fake['1'] = [1, 1, 1]
    fake['2'] = [1, 1, 1]
    fake['3'] = ['2', '2', '2']
    fake['4'] = ['2', '2', '2']
    fake['5'] = ['2', 1, '2']

    return fake


def get_fake_dataset4():

    fake = Dataset()
    fake['1'] = [1, 2, 3, 4, 5, 6]
    fake['2'] = [1, 2, 3, 4, 5, np.nan]

    return fake


def get_fake_dataset5():

    df = Dataset()
    df['1'] = [1, 2, 3, 4, 5]
    df['2'] = [1, 1, 1, 1, 2]
    df['3'] = [np.nan, np.nan, 1, 1, 2]
    df['4'] = [1, 2, 2, 3, 3]
    df['4'] = df['4'].astype('object')
    return df


def get_fake_dataset6():

    df = Dataset()
    df['1'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    df['2'] = [0, 0, 0, 0, 0, 0, 0, 0, np.nan, 1]
    return df


def test_add_scope():

    df = get_fake_dataset()

    df.add_scope(col='1', scope_val='a')
    assert df.scopes['1'] == set(['a'])

    df.add_scopes(scope='1', scope_val='a')
    assert df.scopes['1'] == set(['a'])

    df.add_scope(col='1', scope_val='b')
    assert df.scopes['1'] != set(['a'])
    assert df.scopes['1'] == set(['a', 'b'])

    df.add_scopes(scope='1', scope_val='b')
    assert df.scopes['1'] != set(['a'])
    assert df.scopes['1'] == set(['a', 'b'])

    # Test some get cols from scope
    assert(set(df.get_cols('a')) == set(['1']))
    assert(set(df.get_cols('b')) == set(['1']))
    assert(set(df.get_cols(['a', 'b'])) == set(['1']))

    df = get_fake_dataset()
    df.add_scope(col='1', scope_val='category')
    assert(df['1'].dtype.name == 'category')

    df = get_fake_dataset()
    df.add_scopes(scope='1', scope_val='category')
    assert(df['1'].dtype.name == 'category')


def test_set_roles():

    df = get_fake_dataset()
    df.set_role('1', 'target')
    df.set_role('2', 'non input')

    assert(set(df.get_cols('target')) == set(['1']))
    assert(set(df.get_cols('non input')) == set(['2']))


def test_get_cols():

    df = get_fake_dataset()
    assert(set(df.get_cols('all')) == set(['1', '2', '3']))
    assert(set(df.get_cols('data')) == set(['1', '2', '3']))
    assert(set(df.get_cols('1')) == set(['1']))
    assert(set(df.get_cols('category')) == set(['2']))


def test_filter_outliers():

    df = get_fake_dataset()
    df.filter_outliers_by_percent(20, scope='3', drop=False)
    assert pd.isnull(df['3']).all()


def test_filter_outliers_by_percent():

    df = get_fake_dataset4()
    df.filter_outliers_by_percent(20, scope='1', drop=True)
    assert len(df) == 4

    # Make sure works with NaNs
    df = get_fake_dataset4()
    df.filter_outliers_by_percent(20, scope='2', drop=True)
    assert len(df) == 4
    assert pd.isnull(df.loc[5, '2'])

    # Make sure range works
    df = get_fake_dataset4()
    df.filter_outliers_by_percent((20, None), scope='2', drop=True)
    assert len(df) == 5
    assert pd.isnull(df.loc[5, '2'])

    # Make sure drop false works
    df = get_fake_dataset4()
    df.filter_outliers_by_percent((20, None), scope='2', drop=False)
    assert len(df) == 6
    assert pd.isnull(df.loc[0, '2'])
    assert pd.isnull(df.loc[5, '2'])


def test_filter_outliers_by_std():

    df = get_fake_dataset4()

    # Mean is 3.5, std of 1 is ~1.7 for col 1
    df.filter_outliers_by_std(1, scope='1', drop=True)
    assert len(df) == 4

    # Make sure works with NaNs - mean is 3, std is ~1.4
    df = get_fake_dataset4()
    df.filter_outliers_by_std(1, scope='2', drop=True)
    assert len(df) == 4
    assert pd.isnull(df.loc[5, '2'])

    # Make sure range works
    df = get_fake_dataset4()
    df.filter_outliers_by_std((1, None), scope='2', drop=True)
    assert len(df) == 5
    assert pd.isnull(df.loc[5, '2'])

    # Make sure drop false works
    df = get_fake_dataset4()
    df.filter_outliers_by_std((1, None), scope='2', drop=False)
    assert len(df) == 6
    assert pd.isnull(df.loc[0, '2'])
    assert pd.isnull(df.loc[5, '2'])


def test_drop_cols_by_unique_val():

    df = get_fake_dataset2()
    df.drop_cols_by_unique_val()

    assert '1' not in df
    assert '2' in df
    assert '3' in df

    df = get_fake_dataset2()
    df.drop_cols_by_unique_val(threshold=3)
    assert '1' not in df
    assert '2' not in df
    assert '3' not in df


def test_drop_id_cols():

    df = get_fake_dataset2()
    df.drop_id_cols()

    assert '1' in df
    assert '2' in df
    assert '3' not in df


def test_drop_duplicate_cols():

    df = get_fake_dataset3()
    df.drop_duplicate_cols()

    assert '5' in df
    assert df.shape == (3, 3)


def test_apply_inclusions():

    df = get_fake_dataset3()
    df.apply_inclusions([0])
    assert len(df) == 1


def test_apply_exclusions():

    df = get_fake_dataset()
    df.apply_exclusions([0, 1])
    assert len(df) == 1

    df = get_fake_dataset()
    df.apply_exclusions([0])
    assert len(df) == 2


def test_drop_cols_inclusions():

    df = get_fake_dataset()
    df.drop_cols(inclusions='1')
    assert '1' in df
    assert df.shape[1] == 1

    df = get_fake_dataset()
    df.drop_cols(inclusions='category')
    assert '2' in df

    df = get_fake_dataset()
    df.drop_cols(inclusions=['1', '2'])
    assert df.shape[1] == 2

    df = Dataset(columns=['xxx1', 'xxx2', 'xxx3', '4'])
    df.drop_cols(inclusions=['xxx'])
    assert '4' not in df
    assert df.shape[1] == 3


def test_drop_cols_exclusions():

    df = get_fake_dataset()
    df.drop_cols(exclusions='1')
    assert '1' not in df
    assert df.shape[1] == 2

    df = get_fake_dataset()
    df.drop_cols(exclusions=['1', '2'])
    assert '3' in df
    assert df.shape[1] == 1

    df = Dataset(columns=['xxx1', 'xxx2', 'xxx3', '4'])
    df.drop_cols(exclusions=['xxx'])
    assert '4' in df
    assert df.shape[1] == 1


def test_auto_detect_categorical():

    df = get_fake_dataset5()
    df.auto_detect_categorical(scope='all', obj_thresh=4, all_thresh=None)
    cat_cols = df.get_cols(scope='category')
    assert cat_cols == ['2', '3', '4']

    df = get_fake_dataset5()
    df.auto_detect_categorical(scope='all', obj_thresh=3, all_thresh=None)
    cat_cols = df.get_cols(scope='category')
    assert cat_cols == ['2', '3']

    df = get_fake_dataset5()
    df.auto_detect_categorical(scope='all', obj_thresh=None, all_thresh=None)
    cat_cols = df.get_cols(scope='category')
    assert cat_cols == ['2', '3']

    df = get_fake_dataset5()
    df.auto_detect_categorical(scope='all', obj_thresh=None, all_thresh=10)
    cat_cols = df.get_cols(scope='category')
    assert cat_cols == ['1', '2', '3', '4']


def test_filter_categorical_by_percent():

    df = get_fake_dataset6()
    df.filter_categorical_by_percent(drop_percent=1, scope='all',
                                     drop=True)
    assert len(df) == 10
    assert len(df['2'].unique()) == 3
    assert 'category' in df.scopes['1']
    assert 'category' in df.scopes['2']

    df = get_fake_dataset6()
    df.filter_categorical_by_percent(drop_percent=11, scope='all',
                                     drop=True)
    assert len(df) == 9
    assert len(df['2'].unique()) == 2

    df = get_fake_dataset6()
    df.filter_categorical_by_percent(drop_percent=11, scope='2',
                                     drop=True)
    assert len(df) == 10

    df = get_fake_dataset6()
    df.filter_categorical_by_percent(drop_percent=20, scope='2',
                                     drop=True)
    assert len(df) == 9
    assert len(df['2'].unique()) == 2

    df = get_fake_dataset6()
    df.filter_categorical_by_percent(drop_percent=20, scope='2',
                                     drop=False)
    assert len(df) == 10
    assert len(df['2'].unique()) == 2
    assert 'category' not in df.scopes['1']
    assert 'category' in df.scopes['2']


def test_add_data_files():

    df = get_fake_dataset()

    def file_to_subject(i):
        return int(i.split('_')[-1])

    files = {'q': ['a_0', 'b_1', 'c_2']}

    df.add_data_files(files=files,
                      file_to_subject=file_to_subject,
                      load_func=np.load)

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
    df.add_data_files(files=files,
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
    df.auto_detect_categorical()
    cat_cols = df.get_cols(scope='category')
    assert cat_cols == ['2', '3']

    # This also tests copy
    df_copy = df.copy(deep=True)

    # Drop outliers by std, should drop all by 1
    df_copy.filter_outliers_by_std(n_std=.5, scope='data file',
                                   drop=True, reduce_func=np.mean,
                                   n_jobs=1)

    # Make sure self._check_file_mapping works
    assert len(df_copy.file_mapping) == 1
    assert 1 in df_copy.file_mapping
    assert len(df_copy) == 1
    assert len(df.file_mapping) == 3

    # Try drop = False
    df_copy = df.copy(deep=True)
    df_copy.filter_outliers_by_std(n_std=.5, scope='data file',
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
    vs = Value_Subset(name='1', values=1, encoded_values=False)
    subjects = df.get_subjects(vs, return_as='set')
    assert len(subjects) == 1
    assert subjects.pop() == 0

    # Int column test 2 values
    vs = Value_Subset(name='1', values=[1, 2], encoded_values=False)
    subjects = df.get_subjects(vs, return_as='set')
    assert len(subjects) == 2

    # Str column test 1 value
    vs = Value_Subset(name='2', values=['6'], encoded_values=False)
    subjects = df.get_subjects(vs, return_as='set')
    assert len(subjects) == 1
    assert subjects.pop() == 0

    # Str column test 2 values
    vs = Value_Subset(name='2', values=['6', '7'], encoded_values=False)
    subjects = df.get_subjects(vs, return_as='set')
    assert len(subjects) == 2

    # Str / cat column test extra values
    vs = Value_Subset(name='2', values=['6', '7', '9'], encoded_values=False)
    subjects = df.get_subjects(vs, return_as='set')
    assert len(subjects) == 2

    # Column w/ nan
    vs = Value_Subset(name='3', values=np.nan, encoded_values=False)
    subjects = df.get_subjects(vs, return_as='set')
    assert len(subjects) == 1
    assert subjects.pop() == 0

    # Column w/ nan 2 values
    vs = Value_Subset(name='3', values=[np.nan, 2], encoded_values=False)
    subjects = df.get_subjects(vs, return_as='set')
    assert len(subjects) == 2
    assert df.loc[subjects].shape == (2, 3)

    # Bad name col
    vs = Value_Subset(name=1, values=[np.nan, 2], encoded_values=False)
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


def get_fake_multi_index_dataset():

    fake = Dataset()
    fake['0'] = [1, 2, 3, 4, 5, 6]
    fake['1'] = [1, 2, 3, 4, 5, 6]
    fake['2'] = [1, 2, 3, 4, 5, np.nan]
    fake['subj'] = ['s1', 's2', 's3', 's1', 's2', 's3']
    fake['event'] = ['e1', 'e1', 'e1', 'e2', 'e2', 'e2']
    fake.set_index(['subj', 'event'], inplace=True)

    return fake


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


def test_multi_index_apply_inclusions():

    df = get_fake_multi_index_dataset()
    df.apply_inclusions(subjects=['s1'])
    assert df.shape == (2, 3)

    df = get_fake_multi_index_dataset()
    df.apply_inclusions(subjects='all')
    assert df.shape == (6, 3)

    df = get_fake_multi_index_dataset()
    df.apply_inclusions(subjects=(['s1', 's2'], ['e1']))
    assert df.shape == (2, 3)


def test_multi_index_apply_exclusions():

    df = get_fake_multi_index_dataset()
    df.apply_exclusions(subjects=['s1'])
    assert df.shape == (4, 3)

    df = get_fake_multi_index_dataset()
    df.apply_exclusions(subjects='all')
    assert df.shape == (0, 3)

    df = get_fake_multi_index_dataset()
    df.apply_exclusions(subjects=(['s1', 's2'], ['e1']))
    assert df.shape == (4, 3)


def test_multi_index_add_unique_overlap():

    df = get_fake_multi_index_dataset()
    df.add_unique_overlap(cols=['0', '1'],
                          new_col='new',
                          encoded_values=True)
    assert df['new'].nunique() == 6


def test_multi_index_add_data_files():

    df = get_fake_multi_index_dataset()

    def file_to_subject(i):

        subj = i.split('_')[1]
        event = i.split('_')[2]

        return (subj, event)

    # Leave c_s3_e2' as NaN
    files = {'files': ['a_s1_e1', 'a_s1_e2',
                       'b_s2_e1', 'b_s2_e2', 'c_s3_e1']}

    df.add_data_files(files=files,
                      file_to_subject=file_to_subject,
                      load_func=np.load)

    assert len(df['files']) == 6
    assert 'a_s1_e1' in df.file_mapping[0].loc


def get_nans_dataset():

    fake = Dataset()
    fake['1'] = [np.nan, np.nan, np.nan, 1]
    fake['2'] = [np.nan, np.nan, 1, 1]
    fake['3'] = [np.nan, 1, 1, 1]
    fake['4'] = [1, 1, 1, 1]
    return fake


def test_drop_subjects_by_nan():

    df = get_nans_dataset()
    df.drop_subjects_by_nan(threshold=1, scope='all')
    assert df.shape == (1, 4)

    df = get_nans_dataset()
    df.drop_subjects_by_nan(threshold=.25, scope='all')
    assert df.shape == (1, 4)

    df = get_nans_dataset()
    df.drop_subjects_by_nan(threshold=2, scope='all')
    assert df.shape == (2, 4)

    df = get_nans_dataset()
    df.drop_subjects_by_nan(threshold=.5, scope='all')
    assert df.shape == (2, 4)

    df = get_nans_dataset()
    df.drop_subjects_by_nan(threshold=3, scope='all')
    assert df.shape == (3, 4)

    df = get_nans_dataset()
    df.drop_subjects_by_nan(threshold=.75, scope='all')
    assert df.shape == (3, 4)

    df = get_nans_dataset()
    df.drop_subjects_by_nan(threshold=4, scope='all')
    assert df.shape == (4, 4)

    df = get_nans_dataset()
    df.drop_subjects_by_nan(threshold=.9, scope='all')
    assert df.shape == (4, 4)


def test_drop_cols_by_nan():

    df = get_nans_dataset()
    df.drop_cols_by_nan(threshold=1, scope='all')
    assert df.shape == (4, 1)

    df = get_nans_dataset()
    df.drop_cols_by_nan(threshold=.25, scope='all')
    assert df.shape == (4, 1)

    df = get_nans_dataset()
    df.drop_cols_by_nan(threshold=2, scope='all')
    assert df.shape == (4, 2)

    df = get_nans_dataset()
    df.drop_cols_by_nan(threshold=.5, scope='all')
    assert df.shape == (4, 2)

    df = get_nans_dataset()
    df.drop_cols_by_nan(threshold=3, scope='all')
    assert df.shape == (4, 3)

    df = get_nans_dataset()
    df.drop_cols_by_nan(threshold=4, scope='all')
    assert df.shape == (4, 4)

    df = get_nans_dataset()
    df.drop_cols_by_nan(threshold=.9, scope='all')
    assert df.shape == (4, 4)
