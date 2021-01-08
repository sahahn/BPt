import numpy as np
from ..Dataset import Dataset
import pandas as pd


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

    df.add_scope(col='1', scope='a')
    assert df.scopes['1'] == set(['a'])

    df.add_scope(col='1', scope='b')
    assert df.scopes['1'] != set(['a'])
    assert df.scopes['1'] == set(['a', 'b'])

    # Test some get cols from scope
    assert(set(df._get_cols_from_scope('a')) == set(['1']))
    assert(set(df._get_cols_from_scope('b')) == set(['1']))
    assert(set(df._get_cols_from_scope(['a', 'b'])) == set(['1']))

    df = get_fake_dataset()
    df.add_scope(col='1', scope='category')
    assert(df['1'].dtype.name == 'category')


def test_set_roles():

    df = get_fake_dataset()
    df.set_role('1', 'target')
    df.set_role('2', 'non input')

    assert(set(df._get_cols_from_scope('target')) == set(['1']))
    assert(set(df._get_cols_from_scope('non input')) == set(['2']))


def test_get_cols_from_scope():

    df = get_fake_dataset()
    assert(set(df._get_cols_from_scope('all')) == set(['1', '2', '3']))
    assert(set(df._get_cols_from_scope('data')) == set(['1', '2', '3']))
    assert(set(df._get_cols_from_scope('1')) == set(['1']))
    assert(set(df._get_cols_from_scope('category')) == set(['2']))


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


def test_drop_non_unique():

    df = get_fake_dataset2()
    df.drop_non_unique()

    assert '1' not in df
    assert '2' in df
    assert '3' in df


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


def get_fake_dataset7():

    df = Dataset()
    df['1'] = [0, 1, 1, 1, 2, 2, 2]
    df['1'] = df['1'].astype('category')

    df['2'] = [0, 1, 1, 1, 2, 2, 2]

    return df


def test_binarize_base():

    df = get_fake_dataset7()
    assert len(df) == 7

    df.binarize(scope='1', base=True)
    assert len(df) == 6
    assert 0 in df['1'].unique()
    assert 1 in df['1'].unique()
    assert len(df['1'].unique()) == 2

    df = get_fake_dataset7()
    df.binarize(scope='2', base=True, drop=False)
    assert df['2'].dtype.name == 'category'
    assert pd.isnull(df.loc[0, '2'])
    assert len(df) == 7
    assert df.encoders['2'][0] == 1
    assert df.encoders['2'][1] == 2


def test_nan_to_class():

    df = get_fake_dataset7()
    df.binarize(scope='1', base=True, drop=False)
    assert len(df) == 7
    assert pd.isnull(df.loc[0, '1'])

    df.nan_to_class(scope='1')
    assert df.loc[0, '1'] == 2

    df = get_fake_dataset7()
    df.loc[6, '2'] = np.nan
    df.binarize(scope='2', base=True, drop=False)
    df.nan_to_class(scope='2')

    assert df.loc[6, '2'] == 2
    assert df.loc[0, '2'] == 2
    assert pd.isnull(df.encoders['2'][2])


def test_binarize_threshold():

    df = get_fake_dataset()
    df.binarize('1', threshold=1.5)

    assert df.loc[0, '1'] == 0
    assert df.loc[1, '1'] == 1
    assert 'category' in df.scopes['1']
    assert df.encoders['1'] == {0: '<1.5', 1: '>=1.5'}


def test_get_non_nan_values():

    df = get_fake_dataset()
    values, nan_subjects = df._get_non_nan_values('3')

    assert len(values) == 2
    assert len(nan_subjects) == 1
    assert nan_subjects[0] == 0


def test_binarize_with_nans():

    df = get_fake_dataset()
    df.binarize('3', threshold=2.5)

    assert pd.isnull(df.loc[0, '3'])
    assert df.loc[1, '3'] == 0
    assert df.loc[2, '3'] == 1


def test_binarize_upper_lower():

    df = get_fake_dataset()
    df.binarize('1', lower=2, upper=2)

    assert len(df) == 2
    assert df.loc[0, '1'] == 0
    assert df.loc[2, '1'] == 1
    assert 'category' in df.scopes['1']
    assert df.encoders['1'] == {0: '<2', 1: '>2'}


def test_binarize_upper_lower_drop():

    # Test with drop True
    df = get_fake_dataset()
    df.binarize('1', lower=1.1, upper=2.2, drop=True)
    assert len(df) == 2
    assert pd.isnull(df.loc[0, '3'])
    assert df.loc[0, '1'] == 0
    assert df.loc[2, '1'] == 1

    # With drop False
    df = get_fake_dataset()
    df.binarize('1', lower=1.1, upper=2.2, drop=False)

    assert len(df) == 3
    assert df.loc[0, '1'] == 0
    assert pd.isnull(df.loc[1, '1'])
    assert df.loc[2, '1'] == 1


def test_get_category_cols():

    df = get_fake_dataset()
    cat_cols = df.get_category_cols()
    assert '2' in cat_cols
    assert len(cat_cols) == 1


def test_auto_detect_categorical():

    df = get_fake_dataset5()
    df.auto_detect_categorical(scope='all', obj_thresh=4, all_thresh=None)
    cat_cols = df.get_category_cols()
    assert cat_cols == ['2', '3', '4']

    df = get_fake_dataset5()
    df.auto_detect_categorical(scope='all', obj_thresh=3, all_thresh=None)
    cat_cols = df.get_category_cols()
    assert cat_cols == ['2', '3']

    df = get_fake_dataset5()
    df.auto_detect_categorical(scope='all', obj_thresh=None, all_thresh=None)
    cat_cols = df.get_category_cols()
    assert cat_cols == ['2', '3']

    df = get_fake_dataset5()
    df.auto_detect_categorical(scope='all', obj_thresh=None, all_thresh=10)
    cat_cols = df.get_category_cols()
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
