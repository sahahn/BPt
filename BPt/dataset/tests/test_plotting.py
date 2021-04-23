from ..Dataset import Dataset
import numpy as np
import tempfile
import os
import warnings


def get_fake_dataset():

    fake = Dataset()
    fake['class'] = [1, 1, 1, 2, 2, 2]
    fake['class2'] = [2, 2, 2, 1, 1, 1]
    fake['val1'] = [.1, .5, 1, 4, 1, np.nan]
    fake['val2'] = [2, 1, .1, 5, 2, 7]

    fake = fake.add_scope('class', 'category')
    fake = fake.add_scope('class2', 'category')

    return fake


def test_for_no_errors():

    df = get_fake_dataset()
    df.plot('all', show=False)
    df.plot_bivar('val2', 'val1', show=False)
    df.plot_bivar('class', 'class2', show=False)
    df.plot_bivar('class', 'val1', show=False)


def test_summary_cat_col():

    df = get_fake_dataset()

    df1, df2 =\
        df.summary(scope='class',
                   cat_measures=['count', 'freq', 'nan count'],
                   decode_values=True)

    assert len(df1) == 0

    assert df2.loc['class', 'count'] == 6
    assert df2.loc['class=1', 'count'] == 3
    assert df2.loc['class=2', 'count'] == 3

    assert df2.loc['class', 'freq'] == 1
    assert df2.loc['class=1', 'freq'] == .5
    assert df2.loc['class=2', 'freq'] == .5

    assert df2.loc['class', 'nan count'] == 0
    assert df2.loc['class=1', 'nan count'] == 0
    assert df2.loc['class=2', 'nan count'] == 0


def test_summary_float_col_mean_std():

    df = get_fake_dataset()

    df1, df2 =\
        df.summary(scope='val1',
                   measures=['mean +- std'])

    assert len(df2) == 0
    print(df1)
    assert df1.loc['val1', 'mean ± std'] == '1.32 ± 1.545'


def test_summary_float_col():

    df = get_fake_dataset()

    df1, df2 =\
        df.summary(scope='val1',
                   measures=['count', 'nan count',
                             'mean', 'max',
                             'min', 'std', 'var',
                             'skew', 'kurtosis'])

    assert len(df2) == 0

    # Loose tests, more want to make sure results are there.
    assert df1.loc['val1', 'count'] == 5
    assert df1.loc['val1', 'nan count'] == 1
    assert df1.loc['val1', 'mean'] > 1.31
    assert df1.loc['val1', 'mean'] < 1.33
    assert df1.loc['val1', 'max'] == 4
    assert df1.loc['val1', 'min'] == .1
    assert df1.loc['val1', 'std'] > 1.5
    assert df1.loc['val1', 'std'] < 1.6
    assert df1.loc['val1', 'skew'] < 1.9
    assert df1.loc['val1', 'skew'] > 1.88
    assert df1.loc['val1', 'kurtosis'] > 3.9
    assert df1.loc['val1', 'kurtosis'] < 3.91
    assert df1.loc['val1', 'var'] < 2.388
    assert df1.loc['val1', 'var'] > 2.386


def test_summary_save():

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        df = get_fake_dataset()

        temp = os.path.join(tempfile.gettempdir(), 't1.docx')
        if os.path.exists(temp):
            os.remove(temp)

        df.summary(scope=['class', 'val1'],
                   save_file=temp)

        assert os.path.exists(temp)
        os.remove(temp)
