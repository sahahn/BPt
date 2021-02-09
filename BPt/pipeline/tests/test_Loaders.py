from ..BPtLoader import BPtLoader, CompatArray, BPtListLoader
from ...extensions import Identity
import numpy as np
from .helpers import get_fake_mapping, clean_fake_mapping, IdentityListLoader


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


def test_compat_array():

    x = np.reshape(np.arange(10), (2, 5)).astype('float')
    c = CompatArray(x)

    assert c.shape == (2, 5)
    assert np.array_equal(c[0], x[:, 0])
    assert np.array_equal(c[-1], x[:, -1])

    assert np.array_equal(np.array([], dtype='float').reshape((2, 0)),
                          c.conv_rest_back(rest_inds=[]))
    assert np.array_equal(x[:, 0].reshape((2, 1)),
                          c.conv_rest_back(rest_inds=[0]))
    assert np.array_equal(x, c.conv_rest_back(rest_inds=[0, 1, 2, 3, 4]))


def testBPtListLoader():

    mapping = get_fake_mapping(6)

    loader = BPtListLoader(estimator=IdentityListLoader(),
                           inds=[0],
                           file_mapping=mapping,
                           n_jobs=1,
                           fix_n_jobs=False,
                           cache_loc=None)

    m = {0: 0, 1: 1}
    m_copy = m.copy()
    X = np.arange(6).reshape((3, 2)).astype('float')
    X_trans = loader.fit_transform(X, mapping=m)

    assert X_trans.shape == (3, 5)
    assert X_trans.dtype == 'float'
    assert np.array_equal(X_trans[:, 0], np.array([0, 2, 4]))
    assert np.array_equal(X_trans[:, 1], np.array([0, 2, 4]))
    assert np.array_equal(X_trans[:, 2], np.array([0, 2, 4]))
    assert np.array_equal(X_trans[:, 3], np.array([0, 2, 4]))

    assert loader.n_trans_feats_ == 4
    assert len(loader.X_trans_inds_) == 1
    assert loader.X_trans_inds_[0] == [0, 1, 2, 3]
    assert loader.mapping_ == m_copy
    assert loader.inds_ == [0]
    assert m == loader.out_mapping_
    assert m[0] == [0, 1, 2, 3]
