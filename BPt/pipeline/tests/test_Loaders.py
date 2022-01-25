from ..BPtLoader import BPtLoader, CompatArray, BPtListLoader
from ...extensions import Identity
import numpy as np
from .helpers import get_fake_mapping, clean_fake_mapping, IdentityListLoader
from ..helpers import proc_mapping


def test_BPtLoader():

    mapping = get_fake_mapping(10)
    X = np.arange(10).reshape((5, 2))

    # Test base behavior
    loader = BPtLoader(estimator=Identity(),
                       inds=[0, 1],
                       file_mapping=mapping,
                       n_jobs=1,
                       fix_n_jobs=False,
                       cache_loc=None)
    assert loader._n_jobs == 1

    X_trans = loader.fit_transform(X)
    

    assert X_trans.shape == (5, 8)
    assert X_trans[0][0] == 0
    assert X_trans[0][1] == 0
    assert X_trans[-1][-1] == 9

    # Clean up
    clean_fake_mapping(10)

def test_BPtLoader_fit():

    mapping = get_fake_mapping(10)
    X = np.arange(10).reshape((5, 2))

    # Test base behavior
    loader = BPtLoader(estimator=Identity(),
                       inds=[0, 1],
                       file_mapping=mapping,
                       n_jobs=1,
                       fix_n_jobs=False,
                       cache_loc=None)

    X_trans = loader.fit(X)

def test_BPtLoader2():

    mapping = get_fake_mapping(10)
    X = np.arange(10).reshape((5, 2))

    # Test to see if passing limited inds works
    loader = BPtLoader(estimator=Identity(),
                       inds=[0],
                       file_mapping=mapping,
                       n_jobs=1,
                       fix_n_jobs=False,
                       cache_loc=None)
     
    X_trans = loader.fit_transform(X)

    assert np.all(X_trans[:, 0] == X[:, 0])
    assert X_trans.shape == (5, 5)
    assert np.all(X_trans[:, -1] == np.array([1, 3, 5, 7, 9]))

    # Clean up
    clean_fake_mapping(10)

def test_BPtLoader_custom_mapping():

    mapping = get_fake_mapping(10)
    X = np.arange(10).reshape((5, 2))

    loader = BPtLoader(estimator=Identity(),
                       inds=[0],
                       file_mapping=mapping,
                       n_jobs=1,
                       fix_n_jobs=False,
                       cache_loc=None)

    # Test passing mapping
    fit_mapping = {0: 0, 1: 1}

    X_trans = loader.fit_transform(X, mapping=fit_mapping)

    assert np.all(X_trans[:, 0] == X[:, 0])
    assert X_trans.shape == (5, 5)
    assert np.all(X_trans[:, -1] == np.array([1, 3, 5, 7, 9]))

    # Fit mapping should look like this
    assert fit_mapping[0] == [0, 1, 2, 3]
    assert fit_mapping[1] == 4

    clean_fake_mapping(10)

def test_BPtLoader_custom_mapping_flip():

    mapping = get_fake_mapping(10)

    loader = BPtLoader(estimator=Identity(),
                       inds=[0],
                       file_mapping=mapping,
                       n_jobs=1,
                       fix_n_jobs=False,
                       cache_loc=None)

    # Test passing mapping
    fit_mapping = {1: 0,
                   0: 1}

    X_ref = np.arange(10).reshape((5, 2))
    X = X_ref.copy()
    X[:, 0] = X_ref[:, 1]
    X[:, 1] = X_ref[:, 0]

    X_trans = loader.fit_transform(X, mapping=fit_mapping)

    # First 4 cols should be the inds of 1 in X
    assert np.all(X_trans[:, 0] == X[:, 1])
    assert np.all(X_trans[:, 1] == X[:, 1])
    assert np.all(X_trans[:, 2] == X[:, 1])
    assert np.all(X_trans[:, 3] == X[:, 1])

    # Or same as 0 in X ref
    assert np.all(X_trans[:, 0] == X_ref[:, 0])

    # Last col should be inds of 0
    assert np.all(X_trans[:, 4] == X[:, 0])
    assert np.all(X_trans[:, 4] == X_ref[:, 1])

    # Shape
    assert X_trans.shape == (5, 5)

    # Fit mapping should reflect that the original
    # inds for 1 are now in ind 4
    assert fit_mapping[1] == 4

    # And that for 0 is now 0-3
    assert fit_mapping[0] == [0, 1, 2, 3]

    clean_fake_mapping(10)


def test_loader_mapping():

    mapping = get_fake_mapping(12)

    loader = BPtLoader(estimator=Identity(),
                       inds=[0],
                       file_mapping=mapping,
                       n_jobs=1,
                       fix_n_jobs=False,
                       cache_loc=None)

    # Test passing mapping
    fit_mapping = {0: 1, 1: 2, 2: 0}

    # Make X match mapping
    X_ref = np.arange(12).reshape((4, 3))
    X = X_ref.copy()
    
    X[:, 0] = X_ref[:, 2]
    X[:, 1] = X_ref[:, 0]
    X[:, 2] = X_ref[:, 1]

    X_trans = loader.fit_transform(X, mapping=fit_mapping)
    
    # Shape
    assert X_trans.shape == (4, 6)

    # First 4 cols should be same as ref 0
    assert np.all(X_trans[:, 0] == X_ref[:, 0])
    assert np.all(X_trans[:, 1] == X_ref[:, 0])
    assert np.all(X_trans[:, 2] == X_ref[:, 0])
    assert np.all(X_trans[:, 3] == X_ref[:, 0])

    # Make sure cols 1 and 2 match  to original
    assert np.all(X_trans[:, proc_mapping([1], fit_mapping)[0]] == X_ref[:, 1])
    assert np.all(X_trans[:, proc_mapping([2], fit_mapping)[0]] == X_ref[:, 2])

    # Okay, now for the fit mapping, should reflect
    # the idea that if we wanted to access col 0
    # we would get the inds of the transformed cols,
    # so...
    fit_mapping[0] == [0, 1, 2, 3]

    clean_fake_mapping(12)

def test_loader_mapping_alt():

    mapping = get_fake_mapping(12)

    # Idea is that this was the starting X
    X_ref = np.arange(12).reshape((4, 3))
    
    # Then it went through a step that swapped
    # cols 0 and 2
    X = X_ref.copy()
    X[:, 0] = X_ref[:, 2]
    X[:, 2] = X_ref[:, 0]

    # The new mapping that BPtLoader will
    # receive at this step reflects that swap
    # with 0 to 2.
    fit_mapping = {0: 2,
                   1: 1,
                   2: 0}

    # Now we specify in the loader, that we
    # want it to transform the data originally in
    # column 0, which is now in column 2
    loader = BPtLoader(estimator=Identity(),
                       inds=[0],
                       file_mapping=mapping,
                       n_jobs=1,
                       fix_n_jobs=False,
                       cache_loc=None)

    X_trans = loader.fit_transform(X, mapping=fit_mapping)

    # The values in original col 0 should have been transformed
    assert np.all(X_trans[:, 0] == X_ref[:, 0])
    assert np.all(X_trans[:, 1] == X_ref[:, 0])


    # Now for the current mapping it should reflect
    # that if say we now passed the current state to
    # a model with inds == 0, the proc mapping should be 0-3
    zero_cols = proc_mapping([0], fit_mapping)
    for col in zero_cols:
        assert np.all(X_trans[:, col] == X_ref[:, 0])

    assert zero_cols == [0, 1, 2, 3]
    
    # Make sure cols 1 and 2 match  to original
    assert np.all(X_trans[:, proc_mapping([1], fit_mapping)[0]] == X_ref[:, 1])
    assert np.all(X_trans[:, proc_mapping([2], fit_mapping)[0]] == X_ref[:, 2])

    clean_fake_mapping(12)


def test_loader_mapping_None():

    mapping = get_fake_mapping(10)
    X = np.arange(10).reshape((5, 2))

    loader = BPtLoader(estimator=Identity(),
                       inds=[0],
                       file_mapping=mapping,
                       n_jobs=1,
                       fix_n_jobs=False,
                       cache_loc=None)


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
