from ..Dataset import Dataset
import numpy as np


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
