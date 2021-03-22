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


def test_scope_transformer_passthrough():

    to_ones = ToFixedTransformer(to=1)
    st = ScopeTransformer(estimator=to_ones,
                          inds=[0, 1, 2, 3, 4],
                          passthrough=True,
                          cache_loc=None)
    X = np.zeros((10, 5))
    X_trans = st.fit_transform(X)
    assert np.all(X_trans[:, :5] == 1)
    assert np.all(X_trans[:, 5:] == 0)

    assert len(st.inds) == 5
    assert len(st.mapping_) == 0


def test_scope_transformer_passthrough_mapping():

    to_ones = ToFixedTransformer(to=1)
    st = ScopeTransformer(estimator=to_ones,
                          inds=[0, 1, 2, 3, 4],
                          passthrough=True,
                          cache_loc=None)

    mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
    X = np.zeros((10, 5))
    X_trans = st.fit_transform(X, mapping=mapping)
    assert np.all(X_trans[:, :5] == 1)
    assert np.all(X_trans[:, 5:] == 0)
    assert len(st.inds) == 5

    assert len(mapping) == 5
    assert 0 in mapping[0]
    assert 5 in mapping[0]
    assert 4 in mapping[4]
    assert 9 in mapping[4]

    assert st.out_mapping_ == mapping


def test_scope_transformer_passthrough_mapping2():

    to_ones = ToFixedTransformer(to=1)
    st = ScopeTransformer(estimator=to_ones,
                          inds=[0, 1, 2, 3, 4],
                          passthrough=True,
                          cache_loc=None)

    mapping = {0: 4, 1: 1, 2: 2, 3: 3, 4: 0}
    X = np.zeros((10, 5))
    X_trans = st.fit_transform(X, mapping=mapping)
    assert np.all(X_trans[:, :5] == 1)
    assert np.all(X_trans[:, 5:] == 0)
    assert len(st.inds) == 5

    assert len(mapping) == 5
    assert 0 in mapping[4]
    assert 5 in mapping[4]
    assert 4 in mapping[0]
    assert 9 in mapping[0]


def test_scope_transformer_passthrough_mapping3():

    to_ones = ToFixedTransformer(to=1)
    st = ScopeTransformer(estimator=to_ones,
                          inds=[0],
                          passthrough=True,
                          cache_loc=None)

    mapping = {0: [0, 1, 2], 1: [0, 1, 2], 2: [0, 1, 2],
               3: 3, 4: 4}
    X = np.zeros((10, 5))
    X[:, 3] = 3
    X[:, 4] = 4

    X_trans = st.fit_transform(X, mapping=mapping)
    assert np.all(X_trans[:, :3] == 1)
    assert np.all(X_trans[:, 5:] == 0)
    assert np.all(X_trans[:, 3] == 3)
    assert np.all(X_trans[:, 4] == 4)
    assert len(st.inds) == 1
    assert len(st.inds_) == 3

    assert len(mapping[0]) == 6
    assert len(mapping[1]) == 6
    assert len(mapping[2]) == 6

    assert mapping[3] == 3
    assert mapping[4] == 4

    assert 0 in st.out_mapping_[0]
    assert 5 in st.out_mapping_[0]
    assert 2 in st.out_mapping_[2]
    assert 7 in st.out_mapping_[2]


def test_scope_transformer_pass_val_index():

    class Trans(ToFixedTransformer):
        _needs_fit_index = True
        _needs_transform_index = True

        def fit(self, X, y, fit_index=None):
            self.fit_index = fit_index
            self.n_features_in_ = X.shape[1]
            return self

        def transform(self, X, transform_index=None):
            self.transform_index = transform_index
            X_trans = np.zeros(X.shape)
            X_trans[:] = self.to
            return X_trans

    to_ones = Trans(to=1)
    st = ScopeTransformer(estimator=to_ones,
                          inds=[0, 3, 4],
                          cache_loc=None)

    assert st._needs_transform_index

    X = np.zeros((10, 5))
    st.fit_transform(X)
    assert st.estimator_.fit_index is None
    assert st.estimator_.transform_index is None

    st.fit(X, fit_index=np.arange(10))
    assert len(st.estimator_.fit_index) == 10

    to_ones = Trans(to=1)
    st = ScopeTransformer(estimator=to_ones,
                          inds=[0, 3, 4],
                          cache_loc=None)

    st.fit_transform(X, fit_index=np.arange(10))

    assert len(st.estimator_.fit_index) == 10
    assert len(st.estimator_.transform_index) == 10

    st.transform(X, transform_index=np.arange(5))
    assert len(st.estimator_.fit_index) == 10
    assert len(st.estimator_.transform_index) == 5
