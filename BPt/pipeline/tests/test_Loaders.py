from ..Loaders import BPtLoader
from ...extensions import Identity
import numpy as np
from .helpers import get_fake_mapping, clean_fake_mapping


def test_BPtLoader():

    mapping = get_fake_mapping(10)

    # Test base behavior
    loader = BPtLoader(estimator=Identity(),
                       inds=[0, 1],
                       file_mapping=mapping,
                       n_jobs=1,
                       fix_n_jobs=False,
                       cache_loc=None)
    assert loader._n_jobs == 1

    X = np.arange(10).reshape((5, 2))
    X_trans = loader.fit_transform(X)

    assert X_trans.shape == (5, 8)
    assert X_trans[0][0] == 0
    assert X_trans[0][1] == 0
    assert X_trans[-1][-1] == 9

    # Test to see if passing limited inds works
    loader = BPtLoader(estimator=Identity(),
                       inds=[0],
                       file_mapping=mapping,
                       n_jobs=1,
                       fix_n_jobs=False,
                       cache_loc=None)
    X_trans = loader.fit_transform(X)

    assert X_trans.shape == (5, 5)
    assert np.all(X_trans[:, -1] == np.array([1, 3, 5, 7, 9]))

    # Test passing mapping
    fit_mapping = {0: 1, 1: 0}
    X_trans = loader.fit_transform(X, mapping=fit_mapping)

    assert X_trans.shape == (5, 5)
    assert np.all(X_trans[:, -1] == np.array([0, 2, 4, 6, 8]))

    # Make sure passed on fit mapping is correct
    assert fit_mapping[0] == 4
    assert fit_mapping[1] == [0, 1, 2, 3]

    # Test with mapping to both 0 and 1
    fit_mapping = {0: [0, 1], 1: None}
    X_trans = loader.fit_transform(X, mapping=fit_mapping)
    assert X_trans.shape == (5, 8)
    assert X_trans[0][0] == 0
    assert X_trans[0][1] == 0
    assert X_trans[-1][-1] == 9

    # Make sure passed on fit mapping is correct
    assert fit_mapping[0] == [0, 1, 2, 3, 4, 5, 6, 7]
    assert fit_mapping[1] is None

    # Clean up
    clean_fake_mapping(10)
