from sklearn.base import BaseEstimator, clone
from ..helpers.ML_Helpers import proc_mapping, update_mapping
from sklearn.utils.metaestimators import if_delegate_has_method
import numpy as np
from copy import deepcopy


class Scope_Model(BaseEstimator):

    def __init__(self, wrapper_model, wrapper_inds, **params):
        self.wrapper_model = wrapper_model
        self.wrapper_inds = wrapper_inds

        # Set remaining params to base model
        self.wrapper_model.set_params(**params)

        if hasattr(self.wrapper_model, '_estimator_type'):
            self._estimator_type = self.wrapper_model._estimator_type

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

        # Set n_features in 
        self.n_features_in_ = X.shape[1]

        # Clear any previous fits
        self.wrapper_model_ = clone(self.wrapper_model)
        self.wrapper_inds_ = np.copy(self.wrapper_inds)

        # Proc mapping
        if mapping is None:
            mapping = {}

        self._proc_mapping(mapping)

        # Okay now want to create the new_mapping based on wrapper_inds
        new_mapping = {}
        for i in range(len(self.wrapper_inds_)):
            new_mapping[self.wrapper_inds_[i]] = i

        # Now, we only want to pass along the updated mapping
        # and specifically not change the originally passed mapping
        pass_on_mapping = mapping.copy()
        update_mapping(pass_on_mapping, new_mapping)

        # Try to fit
        try:
            self.wrapper_model_.fit(X[:, self.wrapper_inds_],
                                    y=y, mapping=pass_on_mapping, **kwargs)
        except TypeError:
            self.wrapper_model_.fit(X[:, self.wrapper_inds_],
                                    y=y, **kwargs)

        # Check for feat importances
        try:
            self.coef_ = self.wrapper_model_.coef_
        except AttributeError:
            pass

        try:
            self.feature_importances_ =\
                self.wrapper_model_.feature_importances_
        except AttributeError:
            pass

        return self

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
