import numpy as np
import pandas as pd
from nose.tools import assert_raises
from .datasets import (get_fake_dataset, get_fake_dataset7,
                       get_fake_multi_index_dataset)


def test_add_unique_overlap():

    df = get_fake_dataset()
    df.add_scope('1', 'q')
    df.add_scope('1', 'b')
    df.add_scope('2', 'q')
    df.ordinalize(scope='2')
    df.add_unique_overlap(cols=['1', '2'], new_col='combo',
                          decode_values=True)
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


def test_multi_index_add_unique_overlap():

    df = get_fake_multi_index_dataset()
    df.add_unique_overlap(cols=['0', '1'],
                          new_col='new',
                          decode_values=True)
    assert df['new'].nunique() == 6


def test_to_binary_object():

    df = get_fake_dataset()
    df.to_binary('2')
    assert len(df['2'].unique() == 2)

    df = get_fake_dataset7()
    df['2'] = [' ', 1, 1, 1, 2, 2, 2]
    df.to_binary('2')
    assert len(df['2'].unique() == 2)
    assert 0 not in df.index
    assert 1 in df.index
    assert df.shape == (6, 2)


def test_to_binary():

    df = get_fake_dataset7()
    assert len(df) == 7

    df.to_binary(scope='1')
    assert len(df) == 6
    assert 0 in df['1'].unique()
    assert 1 in df['1'].unique()
    assert len(df['1'].unique()) == 2

    df = get_fake_dataset7()
    df.to_binary(scope='2', drop=False)
    assert df['2'].dtype.name == 'category'
    assert pd.isnull(df.loc[0, '2'])
    assert len(df) == 7
    assert df.encoders['2'][0] == 1
    assert df.encoders['2'][1] == 2

    df = get_fake_dataset7()
    df['2'] = [1, 1, 1, 1, 1, 1, 1]
    df.to_binary(scope='2')
    assert len(df['2'].unique() == 1)


def test_to_binary_inplace():

    df = get_fake_dataset7()

    df_copy = df.to_binary(scope='1', inplace=False)
    assert len(df) == 7
    assert len(df_copy) == 6


def test_nan_to_class():

    df = get_fake_dataset7()

    df.to_binary(scope='1', drop=False)
    assert len(df) == 7
    assert pd.isnull(df.loc[0, '1'])

    df.nan_to_class(scope='1')
    assert df.loc[0, '1'] == 2

    df = get_fake_dataset7()
    df.loc[6, '2'] = np.nan
    df.to_binary(scope='2', drop=False)
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
    df.binarize('1', threshold=(2, 2))

    assert len(df) == 2
    assert df.loc[0, '1'] == 0
    assert df.loc[2, '1'] == 1
    assert 'category' in df.scopes['1']
    assert df.encoders['1'] == {0: '<2', 1: '>2'}


def test_binarize_upper_lower_drop():

    # Test with drop True
    df = get_fake_dataset()
    df.binarize('1', threshold=(1.1, 2.2), drop=True)
    assert len(df) == 2
    assert pd.isnull(df.loc[0, '3'])
    assert df.loc[0, '1'] == 0
    assert df.loc[2, '1'] == 1

    # With drop False
    df = get_fake_dataset()
    df.binarize('1', threshold=(1.1, 2.2), drop=False)

    assert len(df) == 3
    assert df.loc[0, '1'] == 0
    assert pd.isnull(df.loc[1, '1'])
    assert df.loc[2, '1'] == 1


def test_copy_as_non_input():

    df = get_fake_dataset()
    df.add_scope('1', 'bleh')
    df.copy_as_non_input(col='1', new_col='1_copy', copy_scopes=False)
    df._check_scopes()

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


def test_copy_as_non_input_inplace():

    df = get_fake_dataset()
    df.add_scope('1', 'bleh')
    df_copy = df.copy_as_non_input(col='1', new_col='1_copy',
                                   copy_scopes=True, inplace=False)

    df._check_scopes()
    df_copy._check_scopes()

    assert df.shape == ((3, 3))
    assert df_copy.shape == ((3, 4))
    assert '1_copy' not in df
    assert '1_copy' in df_copy
    assert 'bleh' in df_copy.scopes['1_copy']
    assert '1_copy' not in df.scopes
