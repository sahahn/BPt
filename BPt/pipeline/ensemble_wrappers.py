
from copy import deepcopy
from sklearn.model_selection import train_test_split

from sklearn.ensemble import (StackingRegressor, StackingClassifier,
                              VotingClassifier, VotingRegressor)

from joblib import Parallel, delayed
from sklearn.base import clone, is_classifier
from sklearn.utils import Bunch
from sklearn.model_selection import check_cv, cross_val_predict
import numpy as np
from .base import _fit_single_estimator, _get_est_fit_params
from ..main.CV import BPtCV

from sklearn.utils.multiclass import check_classification_targets
from sklearn.preprocessing import LabelEncoder
from .helpers import get_mean_fis


def _fit_all_estimators(self, X, y, sample_weight=None, mapping=None,
                        train_data_index=None):

    # Validate
    names, all_estimators = self._validate_estimators()

    # Fit all estimators
    self.estimators_ = Parallel(n_jobs=self.n_jobs)(
        delayed(_fit_single_estimator)(clone(est), X, y, sample_weight,
                                       mapping, train_data_index)
        for est in all_estimators if est != 'drop'
    )

    self.named_estimators_ = Bunch()
    est_fitted_idx = 0
    for name_est, org_est in zip(names, all_estimators):
        if org_est != 'drop':
            self.named_estimators_[name_est] = self.estimators_[
                est_fitted_idx]
            est_fitted_idx += 1
        else:
            self.named_estimators_[name_est] = 'drop'

    return names, all_estimators


def voting_fit(self, X, y, sample_weight=None, mapping=None,
               train_data_index=None, **kwargs):

    # Fit self.estimators_ on all data
    self._fit_all_estimators(
        X, y, sample_weight=sample_weight, mapping=mapping,
        train_data_index=train_data_index)

    return self


def stacking_fit(self, X, y, sample_weight=None, mapping=None,
                 train_data_index=None, **kwargs):

    # Validate final estimastor
    self._validate_final_estimator()

    # Fit self.estimators_ on all data
    names, all_estimators = self._fit_all_estimators(
        X, y, sample_weight=sample_weight, mapping=mapping,
        train_data_index=train_data_index)

    # To train the meta-classifier using the most data as possible, we use
    # a cross-validation to obtain the output of the stacked estimators.

    # If BPtCV call get_cv
    if isinstance(self.cv, BPtCV):

        random_state = None
        if hasattr(self, 'random_state'):
            random_state = self.random_state

        cv_inds = self.cv.get_cv(train_data_index,
                                 random_state=random_state,
                                 return_index=True)

    # Otherwise treat as sklearn arg directly
    else:
        cv_inds = self.cv

    # To ensure that the data provided to each estimator are the same, we
    # need to set the random state of the cv if there is one and we need to
    # take a copy.
    cv = check_cv(cv_inds, y=y, classifier=is_classifier(self))
    if hasattr(cv, 'random_state') and cv.random_state is None:
        cv.random_state = np.random.RandomState()

    # Proc stack method
    stack_method = [self.stack_method] * len(all_estimators)

    self.stack_method_ = [
        self._method_name(name, est, meth)
        for name, est, meth in zip(names, all_estimators, stack_method)
    ]

    # Base fit params for sample weight
    sample_weight_params = ({"sample_weight": sample_weight}
                            if sample_weight is not None else None)

    # Get the fit params for each indv estimator
    all_fit_params = [_get_est_fit_params(est, mapping=mapping,
                                          train_data_index=train_data_index,
                                          other_params=sample_weight_params)
                      for est in all_estimators]

    # Make the cross validated internal predictions to train
    # the final_estimator
    predictions = Parallel(n_jobs=self.n_jobs)(
        delayed(cross_val_predict)(clone(est), X, y, cv=deepcopy(cv),
                                   method=meth, n_jobs=self.n_jobs,
                                   fit_params=fit_params,
                                   verbose=self.verbose)
        for est, meth, fit_params in zip(all_estimators,
                                         self.stack_method_,
                                         all_fit_params) if est != 'drop'
    )

    # Only not None or not 'drop' estimators will be used in transform.
    # Remove the None from the method as well.
    self.stack_method_ = [
        meth for (meth, est) in zip(self.stack_method_, all_estimators)
        if est != 'drop'
    ]

    # @TODO make sure train data index is concat'ed correctly
    X_meta = self._concatenate_predictions(X, predictions)
    _fit_single_estimator(self.final_estimator_, X_meta, y,
                          sample_weight=sample_weight,
                          mapping=None,
                          train_data_index=train_data_index)

    return self


def ensemble_classifier_fit(self, X, y,
                            sample_weight=None, mapping=None,
                            train_data_index=None, **kwargs):

    check_classification_targets(y)

    # To make compatible with each Voting and Stacking ...
    self._le = LabelEncoder().fit(y)
    self.le_ = self._le

    self.classes_ = self._le.classes_

    return self.bpt_fit(X, self._le.transform(y),
                        sample_weight=sample_weight,
                        mapping=mapping,
                        train_data_index=train_data_index,
                        **kwargs)


class BPtStackingRegressor(StackingRegressor):
    _needs_mapping = True
    _needs_train_data_index = True
    _fit_all_estimators = _fit_all_estimators
    fit = stacking_fit


class BPtStackingClassifier(StackingClassifier):
    _needs_mapping = True
    _needs_train_data_index = True
    _fit_all_estimators = _fit_all_estimators
    bpt_fit = stacking_fit
    fit = ensemble_classifier_fit


class BPtVotingRegressor(VotingRegressor):
    _needs_mapping = True
    _needs_train_data_index = True
    _fit_all_estimators = _fit_all_estimators
    fit = voting_fit

    @property
    def feature_importances_(self):
        return get_mean_fis(self.estimators_, 'feature_importances_')

    @property
    def coef_(self):
        return get_mean_fis(self.estimators_, 'coef_')


class BPtVotingClassifier(VotingClassifier):
    _needs_mapping = True
    _needs_train_data_index = True
    _fit_all_estimators = _fit_all_estimators
    bpt_fit = voting_fit
    fit = ensemble_classifier_fit

    @property
    def feature_importances_(self):
        return get_mean_fis(self.estimators_, 'feature_importances_')

    @property
    def coef_(self):
        return get_mean_fis(self.estimators_, 'coef_')


class DES_Ensemble(VotingClassifier):

    # @TODO make sure this object works, and re-write ...

    def __init__(self, estimators, ensemble, ensemble_name, ensemble_split,
                 ensemble_params=None, random_state=None, weights=None):

        self.estimators = estimators
        self.ensemble = ensemble
        self.ensemble_name = ensemble_name
        self.ensemble_split = ensemble_split

        if ensemble_params is None:
            ensemble_params = {}
        self.ensemble_params = ensemble_params

        self.random_state = random_state
        self.weights = weights

    def fit(self, X, y, sample_weight=None):
        '''Assume y is multi-class'''

        X_train, X_ensemble, y_train, y_ensemble =\
            train_test_split(X, y, test_size=self.ensemble_split,
                             random_state=self.random_state,
                             stratify=y)

        # Fit estimators
        # See Base Ensemble for why this implementation is bad
        try:
            self.estimators_ = [estimator[1].fit(X_train, y_train,
                                sample_weight=sample_weight)
                                for estimator in self.estimators]
        except TypeError:
            self.estimators_ = [estimator[1].fit(X_train, y_train)
                                for estimator in self.estimators]
        # super().fit(X_train, y_train, sample_weight)

        self.ensemble_ = deepcopy(self.ensemble)
        self.ensemble_.set_params(pool_classifiers=self.estimators_)
        self.ensemble_.set_params(**self.ensemble_params)

        self.ensemble_.fit(X_ensemble, y_ensemble)

        return self

    def predict(self, X):
        return self.ensemble_.predict(X)

    def predict_proba(self, X):
        return self.ensemble_.predict_proba(X)

    def get_params(self, deep=True):
        return self._get_params('estimators', deep=deep)

    def set_params(self, **kwargs):

        ensemble_params = {}

        keys = list(kwargs.keys())
        for param in keys:
            if self.ensemble_name + '__' in param:

                nm = param.split('__')[-1]
                ensemble_params[nm] = kwargs.pop(param)

        self.ensemble_params.update(ensemble_params)

        self._set_params('estimators', **kwargs)
        return self
