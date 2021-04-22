
from ..Dataset import Dataset
import pandas as pd
from .datasets import (get_fake_dataset, get_fake_dataset2, get_fake_dataset3,
                       get_fake_dataset4, get_fake_dataset6,
                       get_nans_dataset, get_fake_multi_index_dataset)


def test_filter_outliers():

    df = get_fake_dataset()
    df = df.filter_outliers_by_percent(fop=20, scope='3', drop=False)
    assert pd.isnull(df['3']).all()


def test_filter_outliers_inplace():

    df = get_fake_dataset()
    df.filter_outliers_by_percent(fop=20, scope='3',
                                  drop=False, inplace=True)
    assert pd.isnull(df['3']).all()


def test_filter_outliers_by_percent():

    df = get_fake_dataset4()
    df = df.filter_outliers_by_percent(fop=20, scope='1', drop=True)
    assert len(df) == 4

    # Make sure works with NaNs
    df = get_fake_dataset4()
    df = df.filter_outliers_by_percent(fop=20, scope='2', drop=True)
    assert len(df) == 4
    assert pd.isnull(df.loc[5, '2'])

    # Make sure range works
    df = get_fake_dataset4()
    df = df.filter_outliers_by_percent(fop=(20, None), scope='2', drop=True)
    assert len(df) == 5
    assert pd.isnull(df.loc[5, '2'])

    # Make sure drop false works
    df = get_fake_dataset4()
    df = df.filter_outliers_by_percent(fop=(20, None), scope='2', drop=False)
    assert len(df) == 6
    assert pd.isnull(df.loc[0, '2'])
    assert pd.isnull(df.loc[5, '2'])


def test_filter_outliers_by_percent_inplace():

    df = get_fake_dataset4()
    df.filter_outliers_by_percent(fop=20, scope='1', drop=True, inplace=True)
    assert len(df) == 4

    # Make sure works with NaNs
    df = get_fake_dataset4()
    df.filter_outliers_by_percent(fop=20, scope='2', drop=True, inplace=True)
    assert len(df) == 4
    assert pd.isnull(df.loc[5, '2'])

    # Make sure range works
    df = get_fake_dataset4()
    df.filter_outliers_by_percent(fop=(20, None), scope='2',
                                  drop=True, inplace=True)
    assert len(df) == 5
    assert pd.isnull(df.loc[5, '2'])

    # Make sure drop false works
    df = get_fake_dataset4()
    df.filter_outliers_by_percent(fop=(20, None), scope='2',
                                  drop=False, inplace=True)
    assert len(df) == 6
    assert pd.isnull(df.loc[0, '2'])
    assert pd.isnull(df.loc[5, '2'])


def test_filter_outliers_by_std():

    df = get_fake_dataset4()

    # Mean is 3.5, std of 1 is ~1.7 for col 1
    df = df.filter_outliers_by_std(n_std=1, scope='1', drop=True)
    assert len(df) == 4

    # Make sure works with NaNs - mean is 3, std is ~1.4
    df = get_fake_dataset4()
    df = df.filter_outliers_by_std(n_std=1, scope='2', drop=True)
    assert len(df) == 4
    assert pd.isnull(df.loc[5, '2'])

    # Make sure range works
    df = get_fake_dataset4()
    df = df.filter_outliers_by_std(n_std=(1, None), scope='2', drop=True)
    assert len(df) == 5
    assert pd.isnull(df.loc[5, '2'])

    # Make sure drop false works
    df = get_fake_dataset4()
    df = df.filter_outliers_by_std(n_std=(1, None), scope='2', drop=False)
    assert len(df) == 6
    assert pd.isnull(df.loc[0, '2'])
    assert pd.isnull(df.loc[5, '2'])


def test_drop_cols_by_unique_val():

    df = get_fake_dataset2()
    df = df.drop_cols_by_unique_val()

    assert '1' not in df
    assert '2' in df
    assert '3' in df

    df = get_fake_dataset2()
    df = df.drop_cols_by_unique_val(threshold=3)
    assert '1' not in df
    assert '2' not in df
    assert '3' not in df


def test_drop_id_cols():

    df = get_fake_dataset2()
    df = df.drop_id_cols(scope='all')

    assert '1' in df
    assert '2' in df
    assert '3' not in df

    # Inplace = false case
    df = get_fake_dataset2()
    df.drop_id_cols(scope='all', inplace=True)

    assert '1' in df
    assert '2' in df
    assert '3' not in df


def test_drop_duplicate_cols():

    df = get_fake_dataset3()
    df = df.drop_duplicate_cols(scope='all')
    assert '5' in df
    assert df.shape == (3, 3)

    df = get_fake_dataset3()
    df.drop_duplicate_cols(scope='all', inplace=True)
    assert '5' in df
    assert df.shape == (3, 3)


def test_apply_inclusions():

    df = get_fake_dataset3()
    df = df.apply_inclusions([0])
    assert len(df) == 1


def test_apply_exclusions():

    df = get_fake_dataset()
    df = df.apply_exclusions([0, 1])
    assert len(df) == 1

    df = get_fake_dataset()
    df = df.apply_exclusions([0])
    assert len(df) == 2


def test_drop_cols_inclusions():

    df = get_fake_dataset()
    df = df.drop_cols(inclusions='1')
    assert '1' in df
    assert df.shape[1] == 1

    df = get_fake_dataset()
    df = df.drop_cols(inclusions='category')
    assert '2' in df

    df = get_fake_dataset()
    df = df.drop_cols(inclusions=['1', '2'])
    assert df.shape[1] == 2

    df = Dataset(columns=['xxx1', 'xxx2', 'xxx3', '4'])
    df = df.drop_cols(inclusions=['xxx'])
    assert '4' not in df
    assert df.shape[1] == 3


def test_drop_cols_exclusions():

    df = get_fake_dataset()
    df = df.drop_cols(exclusions='1')
    assert '1' not in df
    assert df.shape[1] == 2

    df = get_fake_dataset()
    df = df.drop_cols(exclusions=['1', '2'])
    assert '3' in df
    assert df.shape[1] == 1

    df = Dataset(columns=['xxx1', 'xxx2', 'xxx3', '4'])
    df = df.drop_cols(exclusions=['xxx'])
    assert '4' in df
    assert df.shape[1] == 1


def test_filter_categorical_by_percent():

    df = get_fake_dataset6()
    df.filter_categorical_by_percent(drop_percent=1, scope='all',
                                     drop=True, inplace=True)
    assert len(df) == 10
    assert len(df['2'].unique()) == 3
    assert 'category' in df.scopes['1']
    assert 'category' in df.scopes['2']

    df = get_fake_dataset6()
    df = df.filter_categorical_by_percent(drop_percent=11, scope='all',
                                          drop=True)
    assert len(df) == 9
    assert len(df['2'].unique()) == 2

    df = get_fake_dataset6()
    df = df.filter_categorical_by_percent(drop_percent=11, scope='2',
                                          drop=True)
    assert len(df) == 10

    df = get_fake_dataset6()
    df.filter_categorical_by_percent(drop_percent=20, scope='2',
                                     drop=True, inplace=True)
    assert len(df) == 9
    assert len(df['2'].unique()) == 2

    df = get_fake_dataset6()
    df.filter_categorical_by_percent(drop_percent=20, scope='2',
                                     drop=False, inplace=True)
    assert len(df) == 10
    assert len(df['2'].unique()) == 2
    assert 'category' not in df.scopes['1']
    assert 'category' in df.scopes['2']


def test_drop_subjects_by_nan():

    df = get_nans_dataset()
    df = df.drop_subjects_by_nan(threshold=1, scope='all')
    assert df.shape == (1, 4)

    df = get_nans_dataset()
    df = df.drop_subjects_by_nan(threshold=.25, scope='all')
    assert df.shape == (1, 4)

    df = get_nans_dataset()
    df = df.drop_subjects_by_nan(threshold=2, scope='all')
    assert df.shape == (2, 4)

    df = get_nans_dataset()
    df = df.drop_subjects_by_nan(threshold=.5, scope='all')
    assert df.shape == (2, 4)

    df = get_nans_dataset()
    df = df.drop_subjects_by_nan(threshold=3, scope='all')
    assert df.shape == (3, 4)

    df = get_nans_dataset()
    df = df.drop_subjects_by_nan(threshold=.75, scope='all')
    assert df.shape == (3, 4)

    df = get_nans_dataset()
    df = df.drop_subjects_by_nan(threshold=4, scope='all')
    assert df.shape == (4, 4)

    df = get_nans_dataset()
    df = df.drop_subjects_by_nan(threshold=.9, scope='all')
    assert df.shape == (4, 4)


def test_drop_cols_by_nan():

    df = get_nans_dataset()
    df = df.drop_cols_by_nan(threshold=1, scope='all')
    assert df.shape == (4, 1)

    df = get_nans_dataset()
    df = df.drop_cols_by_nan(threshold=.25, scope='all')
    assert df.shape == (4, 1)

    df = get_nans_dataset()
    df = df.drop_cols_by_nan(threshold=2, scope='all')
    assert df.shape == (4, 2)

    df = get_nans_dataset()
    df = df.drop_cols_by_nan(threshold=.5, scope='all')
    assert df.shape == (4, 2)

    df = get_nans_dataset()
    df.drop_cols_by_nan(threshold=3, scope='all', inplace=True)
    assert df.shape == (4, 3)

    df = get_nans_dataset()
    df.drop_cols_by_nan(threshold=4, scope='all', inplace=True)
    assert df.shape == (4, 4)

    df = get_nans_dataset()
    df.drop_cols_by_nan(threshold=.9, scope='all', inplace=True)
    assert df.shape == (4, 4)


def test_multi_index_apply_inclusions():

    df = get_fake_multi_index_dataset()
    df = df.apply_inclusions(subjects=['s1'])
    assert df.shape == (2, 3)

    df = get_fake_multi_index_dataset()
    df = df.apply_inclusions(subjects='all')
    assert df.shape == (6, 3)

    df = get_fake_multi_index_dataset()
    df = df.apply_inclusions(subjects=(['s1', 's2'], ['e1']))
    assert df.shape == (2, 3)


def test_multi_index_apply_exclusions():

    df = get_fake_multi_index_dataset()
    df = df.apply_exclusions(subjects=['s1'])
    assert df.shape == (4, 3)

    df = get_fake_multi_index_dataset()
    df = df.apply_exclusions(subjects='all')
    assert df.shape == (0, 3)

    df = get_fake_multi_index_dataset()
    df = df.apply_exclusions(subjects=(['s1', 's2'], ['e1']))
    assert df.shape == (4, 3)
