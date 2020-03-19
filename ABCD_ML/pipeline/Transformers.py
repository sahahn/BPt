from ..helpers.ML_Helpers import get_obj_and_params, proc_mapping, update_mapping
import numpy as np

from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator


class Transformer_Wrapper(BaseEstimator):

    def __init__(self, transformer, inds, **params):

        self.transformer = transformer
        self.inds = inds

        # Set any remaining params to transformer
        self.transformer.set_params(**params)

    def _proc_mapping(self, mapping):

        try:
            self._mapping
            return

        except AttributeError:
            self._mapping = mapping.copy()

        if len(mapping) > 0:
            self.inds = proc_mapping(self.inds, mapping)

        return

    def fit(self, X, y=None, mapping={}):

        # Need to call fit_transform to figure out change to mapping
        self.fit_transform(X, y, mapping=mapping)
        return self

    def fit_transform(self, X, y=None, mapping={}):

        self._proc_mapping(mapping)

        inds = self.inds
        rest_inds = [i for i in range(X.shape[1]) if i not in inds]

        # Fit transform just inds of X
        X_trans = self.transformer.fit_transform(X[:, inds])
        X_trans_inds = [i for i in range(X_trans.shape[1])]
        self._n_trans = len(X_trans_inds)

        new_mapping = {}

        # Many to Many case
        for i in inds:
            new_mapping[i] = X_trans_inds

        for cnt in range(len(rest_inds)):
            new_mapping[rest_inds[cnt]] = len(X_trans_inds) + cnt

        # Update mapping
        update_mapping(mapping, new_mapping)

        return np.hstack([X_trans, X[:, rest_inds]])

    def transform(self, X):

        # Fit transform just inds of X
        rest_inds = [i for i in range(X.shape[1]) if i not in self.inds]
        X_trans = self.transformer.transform(X[:, self.inds])

        return np.hstack([X_trans, X[:, rest_inds]])

    def set_params(self, **params):

        if 'transformer' in params:
            self.transformer = params.pop('transformer')
        if 'inds' in params:
            self.inds = params.pop('inds')

        self.transformer.set_params(**params)

    def get_params(self, deep=False):

        params = {'transformer': self.transformer,
                  'inds': self.inds}

        params.update(self.transformer.get_params())

        return params


TRANSFORMERS = {
    'pca': (PCA, ['default', 'pca var search']),
}


def get_transformer_and_params(transformer_str, extra_params, params, search_type,
                               inds, random_state=None, n_jobs=1):

    base_trans_obj, extra_trans_params, transformer_params =\
        get_obj_and_params(transformer_str, TRANSFORMERS, extra_params, params,
                           search_type)

    base_trans = base_trans_obj(**extra_trans_params)

    # Try to set attributes
    try:
        base_trans.random_state = random_state
    except AttributeError:
        pass

    if search_type is None:
        try:
            base_trans.n_jobs = n_jobs
        except AttributeError:
            pass

    transformer = Transformer_Wrapper(base_trans, inds)
    return transformer, transformer_params
