from ..Dataset import Dataset
import numpy as np


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


def get_fake_dataset7():

    df = Dataset()
    df['1'] = [0, 1, 1, 1, 2, 2, 2]
    df['1'] = df['1'].astype('category')

    df['2'] = [0, 1, 1, 1, 2, 2, 2]

    return df


def get_nans_dataset():

    fake = Dataset()
    fake['1'] = [np.nan, np.nan, np.nan, 1]
    fake['2'] = [np.nan, np.nan, 1, 1]
    fake['3'] = [np.nan, 1, 1, 1]
    fake['4'] = [1, 1, 1, 1]
    return fake


def get_full_dataset():

    fake = Dataset()
    fake['1'] = [1, 2, 3, 4, 5]
    fake['2'] = [6, 7, 8, 9, 10]
    fake['3'] = [11, 12, 13, 14, 15]
    fake.add_scope('3', 'category')

    fake['subj'] = ['s1', 's2', 's3', 's4', 's5']
    fake.set_index('subj', inplace=True)

    fake['target'] = [.1, .2, .3, .4, .5]
    fake.set_role('target', 'target')

    fake.set_test_split(subjects=['s4', 's5'])

    return fake


def get_fake_multi_index_dataset():

    fake = Dataset()
    fake['0'] = [1, 2, 3, 4, 5, 6]
    fake['1'] = [1, 2, 3, 4, 5, 6]
    fake['2'] = [1, 2, 3, 4, 5, np.nan]
    fake['subj'] = ['s1', 's2', 's3', 's1', 's2', 's3']
    fake['event'] = ['e1', 'e1', 'e1', 'e2', 'e2', 'e2']
    fake.set_index(['subj', 'event'], inplace=True)

    return fake