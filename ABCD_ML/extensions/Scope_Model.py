from sklearn.base import BaseEstimator, clone
from ..helpers.ML_Helpers import proc_mapping
from sklearn.utils.metaestimators import if_delegate_has_method
import numpy as np
from copy import deepcopy


class Scope_Model(BaseEstimator):

    def __init__(self, wrapper_model, wrapper_inds, **params):
        self.wrapper_model = wrapper_model
        self.wrapper_inds = wrapper_inds

        # Set remaining params to base model
        self.wrapper_model.set_params(**params)

    def set_params(self, **params):

        if 'wrapper_model' in params:
            self.wrapper_model = params.pop('wrapper_model')
        if 'wrapper_inds' in params:
            self.wrapper_inds = params.pop('wrapper_inds')

        self.wrapper_model.set_params(**params)

    def get_params(self, deep=False):

        if deep:
            params = {'wrapper_model': clone(self.wrapper_model),
                      'wrapper_inds': deepcopy(self.wrapper_inds)}
        else:
            params = {'wrapper_model': self.wrapper_model,
                      'wrapper_inds': self.wrapper_inds}

        params.update(self.wrapper_model.get_params(deep=deep))
        return params

    def _proc_mapping(self, mapping):

        try:
            self._mapping
            return
        except AttributeError:
            self._mapping = mapping.copy()

        if len(mapping) > 0:
            self.wrapper_inds_ = proc_mapping(self.wrapper_inds_, mapping)

        return

    def fit(self, X, y=None, mapping=None, **kwargs):

        # Clear any previous fits
        self.wrapper_model_ = clone(self.wrapper_model)
        self.wrapper_inds_ = np.copy(self.wrapper_inds)

        if mapping is None:
            mapping = {}

        self._proc_mapping(mapping)

        try:
            self.wrapper_model_.fit(X[:, self.wrapper_inds_],
                                    y=y, mapping=mapping, **kwargs)
        except TypeError:
            self.wrapper_model_.fit(X[:, self.wrapper_inds_],
                                    y=y, **kwargs)

        return self

    @if_delegate_has_method(delegate='wrapper_model_')
    def predict(self, X, *args, **kwargs):
        return self.wrapper_model_.predict(X[:, self.wrapper_inds_],
                                           *args, **kwargs)

    @if_delegate_has_method(delegate='wrapper_model_')
    def predict_proba(self, X, *args, **kwargs):
        return self.wrapper_model_.predict_proba(X[:, self.wrapper_inds_],
                                                 *args, **kwargs)

    @if_delegate_has_method(delegate='wrapper_model_')
    def decision_function(self, X, *args, **kwargs):
        return self.wrapper_model_.decision_function(X[:, self.wrapper_inds_],
                                                     *args, **kwargs)

    @if_delegate_has_method(delegate='wrapper_model_')
    def predict_log_proba(self, X, *args, **kwargs):
        return self.wrapper_model_.predict_log_proba(X[:, self.wrapper_inds_],
                                                     *args, **kwargs)

    @if_delegate_has_method(delegate='wrapper_model_')
    def score(self, X, *args, **kwargs):
        return self.wrapper_model_.score(X[:, self.wrapper_inds_],
                                         *args, **kwargs)