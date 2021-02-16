import numpy as np
from ..BPtFeatureSelector import BPtFeatureSelector
from .helpers import FakeSelector


def test_bptfeature_selector():

    estimator = FakeSelector(mask=np.array([True, False, True]))
    fs = BPtFeatureSelector(estimator=estimator, inds=[0, 1, 2],
                            cache_loc=None)

    # Test base behavior
    X = np.ones((3, 3))
    X[:, 0] = 0
    X[:, 2] = 2

    X_trans = fs.fit_transform(X)
    assert X_trans.shape == (3, 2)
    assert X_trans[0][0] == 0
    assert X_trans[0][1] == 2

    # Try with scope
    X = np.ones((3, 4))
    X[:, 0] = 0
    X[:, 2] = 2
    X[:, 3] = 3

    X_trans = fs.fit_transform(X)
    fs = BPtFeatureSelector(estimator=estimator, inds=[0, 1, 3],
                            cache_loc=None)

    X_trans = fs.fit_transform(X)
    assert X_trans.shape == (3, 3)
    assert X_trans[0][0] == 0
    assert X_trans[0][1] == 3
    assert X_trans[0][2] == 2

    # Try adding fit mapping, just flip 0 and 2
    fit_mapping = {0: 2, 1: 1, 2: 0, 3: 3}
    X_trans = fs.fit_transform(X, mapping=fit_mapping)
    assert X_trans.shape == (3, 3)
    assert X_trans[0][0] == 1
    assert X_trans[0][1] == 3
    assert X_trans[0][2] == 0

    X_rev = fs.inverse_transform(X_trans)
    assert X_rev[0][0] == 0
    assert X_rev[0][1] == 1
    assert X_rev[0][2] == 0
    assert X_rev[0][3] == 3
