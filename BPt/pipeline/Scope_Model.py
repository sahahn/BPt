from sklearn.base import BaseEstimator, clone
from ..helpers.ML_Helpers import proc_mapping, update_mapping
from sklearn.utils.metaestimators import if_delegate_has_method
import numpy as np
from copy import deepcopy
from .base import _get_est_fit_params


class Scope_Model(BaseEstimator):

    needs_mapping = True
    needs_train_data_index = True

    def __init__(self, wrapper_model, wrapper_inds, **params):
        self.wrapper_model = wrapper_model
        self.wrapper_inds = wrapper_inds

        # Set remaining params to base model
        self.wrapper_model.set_params(**params)

    @property
    def _estimator_type(self):
        return self.wrapper_model._estimator_type

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

    def fit(self, X, y=None, mapping=None, train_data_index=None, **kwargs):

        # Set n_features in
        self.n_features_in_ = X.shape[1]

        # Set is fitted
        self.is_fitted_ = True

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
        for i in mapping:
            if i not in self.wrapper_inds_:
                new_mapping[i] = None

        # Now, we only want to pass along the updated mapping
        # and specifically not change the originally passed mapping
        pass_on_mapping = mapping.copy()
        update_mapping(pass_on_mapping, new_mapping)

        # Set the necc. fit params for thisa base estimator
        fit_params = _get_est_fit_params(estimator=self.wrapper_model_,
                                         mapping=pass_on_mapping,
                                         train_data_index=train_data_index,
                                         other_params=kwargs)

        # print('Fit Scope Model, len(train_data_index) == ',
        #      len(train_data_index),
        #      'len wrapper_inds == ', len(self.wrapper_inds),
        #      'len wrapper_inds_ == ', len(self.wrapper_inds_),
        #      'mapping in fit_params == ', 'mapping' in fit_params,
        #      'train_data_index in fit_params ==',
        #      'train_data_index' in fit_params,
        #      'X.shape ==', X.shape)

        # Fit the base model
        self.wrapper_model_.fit(X=X[:, self.wrapper_inds_], y=y, **fit_params)

        return self

    @property
    def coef_(self):
        return self.wrapper_model_.coef_

    @property
    def feature_importances_(self):
        return self.wrapper_model_.feature_importances_

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
