import numpy as np
from ..FeatSelectors import FeatureSelector


def basic_test():

    mask = np.ones(10)
    mask[0] = .4
    X = np.random.random((20, 10))

    fs = FeatureSelector(mask=mask)

    X_trans = fs.fit_transform(X)

    assert X_trans.shape == (20, 9)
    assert np.array_equal(X_trans[:, 0], X[:, 1])
