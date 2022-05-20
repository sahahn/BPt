from ..loaders import Identity
import numpy as np
import tempfile
import os
import pytest
import warnings


def test_identity():

    i = Identity()
    x = np.ones((10, 10))
    assert i.fit_transform(x).shape == (100, )


def test_single_connectivity_measure():

    try:
        from ..loaders import SingleConnectivityMeasure
    except ImportError:
        return

    scm = SingleConnectivityMeasure()
    X_trans = scm.fit_transform(np.ones((5, 10)))
    assert X_trans.shape == (10, 10)

    scm = SingleConnectivityMeasure(vectorize=True, discard_diagonal=True)
    X_trans = scm.fit_transform(np.ones((5, 10)))
    assert X_trans.shape == (45, )

    scm = SingleConnectivityMeasure(vectorize=True)
    X_trans = scm.fit_transform(np.ones((5, 10)))
    assert X_trans.shape == (55, )

