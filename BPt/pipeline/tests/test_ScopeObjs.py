from ..ScopeObjs import ScopeTransformer
import numpy as np
from .helpers import ToFixedTransformer


def test_scope_transformer_1():

    to_ones = ToFixedTransformer(to=1)
    st = ScopeTransformer(estimator=to_ones,
                          inds=[0, 1, 2, 3, 4],
                          cache_loc=None)
    X = np.zeros((10, 5))
    X_trans = st.fit_transform(X)
    assert np.all(X_trans == 1)
    assert len(st.inds) == 5
    assert len(st.mapping_) == 0


def test_scope_transformer_2():

    to_ones = ToFixedTransformer(to=1)
    st = ScopeTransformer(estimator=to_ones,
                          inds=[0, 3, 4],
                          cache_loc=None)
    X = np.zeros((10, 5))
    X_trans = st.fit_transform(X)

    assert np.all(X_trans[:, [0, 1, 2]] == 1)
    assert len(st.inds) == 3
    assert len(st.mapping_) == 0
    assert st.rest_inds_ == [1, 2]
