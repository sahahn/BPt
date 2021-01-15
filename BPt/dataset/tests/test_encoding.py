import numpy as np
from ..Dataset import Dataset
import pandas as pd
from nose.tools import assert_raises


def get_fake_dataset():

    fake = Dataset()
    fake['1'] = [1, 2, 3]
    fake['2'] = ['6', '7', '8']
    fake['2'] = fake['2'].astype('category')
    fake['3'] = [np.nan, 2, 3]

    return fake


def get_fake_dataset7():

    df = Dataset()
    df['1'] = [0, 1, 1, 1, 2, 2, 2]
    df['1'] = df['1'].astype('category')

    df['2'] = [0, 1, 1, 1, 2, 2, 2]

    return df


def test_add_unique_overlap():

    df = get_fake_dataset()
    df.add_scope('1', 'q')
    df.add_scope('1', 'b')
    df.add_scope('2', 'q')
    df.ordinalize(scope='2')
    df.add_unique_overlap(cols=['1', '2'], new_col='combo',
                          encoded_values=True)
    assert df['combo'].nunique() == 3
    assert 'category' in df.scopes['combo']
    assert 'q' in df.scopes['combo']
    assert 'b' not in df.scopes['combo']
    assert df.roles['combo'] == 'data'

    with assert_raises(RuntimeError):
        df.add_unique_overlap(cols='1', new_col='combo')

    with assert_raises(RuntimeError):
        df.add_unique_overlap(cols=['1'], new_col='combo')

    with assert_raises(KeyError):
        df.add_unique_overlap(cols=['does not exist', '1'], new_col='combo')

    with assert_raises(KeyError):
        df.add_unique_overlap(cols=['1', '2'], new_col='1')


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


def test_k_bin():

    df = get_fake_dataset7()
    df.k_bin(scope='1', n_bins=2, strategy='uniform')
    assert df['1'].nunique() == 2

    df.k_bin(scope='2', n_bins=2, strategy='uniform')
    assert df['2'].nunique() == 2

    df._check_scopes()
    assert 'category' in df.scopes['2']
    assert len(df.encoders['1']) == 2

    # Test with nans
    df = get_fake_dataset7()
    df.loc[1, '1'] = np.nan
    df.k_bin(scope='1', n_bins=2, strategy='uniform')
    assert df['1'].nunique(dropna=True) == 2

    # Test comap with nan to class
    df.nan_to_class(scope='1')
    assert df['1'].nunique(dropna=True) == 3
    assert len(df.encoders['1']) == 3


def test_binarize_threshold():

    df = get_fake_dataset()
    df.binarize('1', threshold=1.5)

    assert df.loc[0, '1'] == 0
    assert df.loc[1, '1'] == 1
    assert 'category' in df.scopes['1']
    assert df.encoders['1'] == {0: '<1.5', 1: '>=1.5'}


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


def test_copy_as_non_input():

    df = get_fake_dataset()
    df.add_scope('1', 'bleh')
    df.copy_as_non_input(col='1', new_col='1_copy', copy_scopes=False)

    assert df.shape == ((3, 4))
    assert df.roles['1_copy'] == 'non input'
    assert df.roles['1'] != 'non input'
    assert np.max(np.array(df['1_copy'])) == 2
    assert 'bleh' not in df.scopes['1_copy']

    # Make sure copy scopes works
    df = get_fake_dataset()
    df.add_scope('1', 'bleh')
    df.copy_as_non_input(col='1', new_col='1_copy', copy_scopes=True)
    assert df.shape == ((3, 4))
    assert 'bleh' in df.scopes['1_copy']
