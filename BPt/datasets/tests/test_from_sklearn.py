from ..from_sklearn import load_boston
from ...dataset.Dataset import Dataset


def test_load_boston():

    data = load_boston()

    assert data.shape == (506, 14)
    assert data.train_subjects is not None
    assert isinstance(data, Dataset)
