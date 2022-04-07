from ..from_sklearn import load_cali
from ...dataset.dataset import Dataset


def test_load_cali():

    data = load_cali()

    assert data.shape == (20640, 9)
    assert data.train_subjects is not None
    assert isinstance(data, Dataset)
