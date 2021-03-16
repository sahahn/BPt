
from copy import deepcopy
from .helpers import set_n_jobs, replace_with_in_params

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
                        fit_index=None):

    # Validate
    names, all_estimators = self._validate_estimators()

    # Fit all estimators
    self.estimators_ = Parallel(n_jobs=self.n_jobs)(
        delayed(_fit_single_estimator)(clone(est), X, y, sample_weight,
                                       mapping, fit_index)
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
               fit_index=None, **kwargs):

    # Fit self.estimators_ on all data
    self._fit_all_estimators(
        X, y, sample_weight=sample_weight, mapping=mapping,
        fit_index=fit_index)

    return self


def stacking_fit(self, X, y, sample_weight=None, mapping=None,
                 fit_index=None, **kwargs):

    # Validate final estimator
    self._validate_final_estimator()

    # Fit self.estimators_ on all data
    names, all_estimators = self._fit_all_estimators(
        X, y, sample_weight=sample_weight, mapping=mapping,
        fit_index=fit_index)

    # To train the meta-classifier using the most data as possible, we use
    # a cross-validation to obtain the output of the stacked estimators.

    # If BPtCV call get_cv
    if isinstance(self.cv, BPtCV):

        random_state = None
        if hasattr(self, 'random_state'):
            random_state = self.random_state

        cv_inds = self.cv.get_cv(fit_index,
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
                                          fit_index=fit_index,
                                          other_params=sample_weight_params)
                      for est in all_estimators]

    # Catch rare error - TODO come up with fix
    if X.shape[0] == X.shape[1]:
        raise RuntimeError('Same numbers of data points and ',
                           'features can lead to error.')

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

    # @TODO make sure train data index is concatenated correctly
    X_meta = self._concatenate_predictions(X, predictions)
    _fit_single_estimator(self.final_estimator_, X_meta, y,
                          sample_weight=sample_weight,
                          mapping=None,
                          fit_index=fit_index)

    return self


def ensemble_classifier_fit(self, X, y,
                            sample_weight=None, mapping=None,
                            fit_index=None, **kwargs):

    check_classification_targets(y)

    # To make compatible with each Voting and Stacking ...
    self._le = LabelEncoder().fit(y)
    self.le_ = self._le

    self.classes_ = self._le.classes_

    return self.bpt_fit(X, self._le.transform(y),
                        sample_weight=sample_weight,
                        mapping=mapping,
                        fit_index=fit_index,
                        **kwargs)


class BPtStackingRegressor(StackingRegressor):
    _needs_mapping = True
    _needs_fit_index = True
    _fit_all_estimators = _fit_all_estimators
    fit = stacking_fit


class BPtStackingClassifier(StackingClassifier):
    _needs_mapping = True
    _needs_fit_index = True
    _fit_all_estimators = _fit_all_estimators
    bpt_fit = stacking_fit
    fit = ensemble_classifier_fit


class BPtVotingRegressor(VotingRegressor):
    _needs_mapping = True
    _needs_fit_index = True
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
    _needs_fit_index = True
    _fit_all_estimators = _fit_all_estimators
    bpt_fit = voting_fit
    fit = ensemble_classifier_fit

    @property
    def feature_importances_(self):
        return get_mean_fis(self.estimators_, 'feature_importances_')

    @property
    def coef_(self):
        return get_mean_fis(self.estimators_, 'coef_')


class EnsembleWrapper():

    def __init__(self, model_params, ensemble_params,
                 _get_ensembler, n_jobs, random_state):

        self.model_params = model_params
        self.ensemble_params = ensemble_params
        self._get_ensembler = _get_ensembler
        self.n_jobs = n_jobs
        self.random_state = random_state

    def _update_model_ensemble_params(self, to_add, model=True, ensemble=True):

        if model:
            new_model_params = {}
            for key in self.model_params:
                new_model_params[to_add + '__' + key] =\
                    self.model_params[key]
            self.model_params = new_model_params

        if ensemble:

            new_ensemble_params = {}
            for key in self.ensemble_params:
                new_ensemble_params[to_add + '__' + key] =\
                    self.ensemble_params[key]
            self.ensemble_params = new_ensemble_params

    def _basic_ensemble(self, models, name, ensemble=False):

        if len(models) == 1:
            return models

        else:
            basic_ensemble = self._get_ensembler(models)
            self._update_model_ensemble_params(name, ensemble=ensemble)

            return [(name, basic_ensemble)]

    def get_updated_params(self):

        self.model_params.update(self.ensemble_params)
        return self.model_params

    def wrap_ensemble(self, models, ensemble, ensemble_params,
                      final_estimator=None,
                      final_estimator_params=None):

        # If no ensemble is passed, return either the 1 model,
        # or a voting wrapper
        if ensemble is None or len(ensemble) == 0:
            return self._basic_ensemble(models=models,
                                        name='Default Voting',
                                        ensemble=True)

        # Otherwise special ensembles
        else:

            # If needs a single estimator, but multiple models passed,
            # wrap in ensemble!
            if ensemble_params.single_estimator:
                se_ensemb_name = 'Single-Estimator Compatible Ensemble'
                models = self._basic_ensemble(models,
                                              se_ensemb_name,
                                              ensemble=False)

            # If no split and single estimator
            if ensemble_params.single_estimator:
                return self._wrap_single(models, ensemble,
                                         ensemble_params.n_jobs_type)

            # Last case is, no split/DES ensemble and also
            # not single estimator based
            # e.g., in case of stacking regressor.
            else:
                return self._wrap_multiple(models, ensemble,
                                           final_estimator,
                                           final_estimator_params,
                                           ensemble_params.n_jobs_type,
                                           ensemble_params.cv)

    def _wrap_single(self, models, ensemble_info, n_jobs_type):
        '''If passed single_estimator flag'''

        # Unpack ensemble info
        ensemble_name = ensemble_info[0]
        ensemble_obj = ensemble_info[1][0]
        ensemble_extra_params = ensemble_info[1][1]

        # Models here since single estimator is assumed
        # to be just a list with
        # of one tuple as
        # [(model or ensemble name, model or ensemble)]
        base_estimator = models[0][1]

        # Set n jobs based on passed type
        if n_jobs_type == 'ensemble':
            model_n_jobs = 1
            ensemble_n_jobs = self.n_jobs
        else:
            model_n_jobs = self.n_jobs
            ensemble_n_jobs = 1

        # Set model / base_estimator n_jobs
        set_n_jobs(base_estimator, model_n_jobs)

        # Make sure random_state is set (should be already)
        if hasattr(base_estimator, 'random_state'):
            setattr(base_estimator, 'random_state', self.random_state)

        # Create the ensemble object
        ensemble = ensemble_obj(base_estimator=base_estimator,
                                **ensemble_extra_params)

        # Set ensemble n_jobs
        set_n_jobs(ensemble, ensemble_n_jobs)

        # Set random state
        if hasattr(ensemble, 'random_state'):
            setattr(ensemble, 'random_state', self.random_state)

        # Wrap as object
        new_ensemble = [(ensemble_name, ensemble)]

        # Have to change model name to base_estimator
        self.model_params =\
            replace_with_in_params(self.model_params, models[0][0],
                                   'base_estimator')

        # Append ensemble name to all model params
        self._update_model_ensemble_params(ensemble_name,
                                           ensemble=False)

        return new_ensemble

    def _wrap_multiple(self, models, ensemble_info,
                       final_estimator, final_estimator_params,
                       n_jobs_type, cv):
        '''In case of no split/DES ensemble, and not single estimator based.'''

        # Unpack ensemble info
        ensemble_name = ensemble_info[0]
        ensemble_obj = ensemble_info[1][0]
        ensemble_extra_params = ensemble_info[1][1]

        # Models here just self.models a list of tuple of
        # all models.
        # So, ensemble_extra_params should contain the
        # final estimator + other params

        # Set model_n_jobs and ensemble n_jobs based on type
        if n_jobs_type == 'ensemble':
            model_n_jobs = 1
            ensemble_n_jobs = self.n_jobs
        else:
            model_n_jobs = self.n_jobs
            ensemble_n_jobs = 1

        # Set the model jobs
        set_n_jobs(models, model_n_jobs)

        # Make sure random state is propegated
        for model in models:
            if hasattr(model[1], 'random_state'):
                setattr(model[1], 'random_state', self.random_state)

        # Determine the parameters to init the ensemble
        pass_params = ensemble_extra_params
        pass_params['estimators'] = models

        # Process final_estimator if passed
        if final_estimator is not None:

            # Replace name of final estimator w/ final_estimator in params
            final_estimator_params =\
                replace_with_in_params(params=final_estimator_params,
                                       original=final_estimator[0][0],
                                       replace='final_estimator')

            # Add final estimator params to model_params - once name changed
            # to avoid potential overlap.
            self.model_params.update(final_estimator_params)

            # Unpack actual model obj
            final_estimator_obj = final_estimator[0][1]

            # Set final estimator n_jobs to model n_jobs
            set_n_jobs(final_estimator_obj, model_n_jobs)

            # Redundant random state check
            if hasattr(final_estimator_obj, 'random_state'):
                setattr(final_estimator_obj, 'random_state', self.random_state)

            # Add to pass params
            pass_params['final_estimator'] = final_estimator_obj

        # Check if cv passed
        if cv is not None:
            pass_params['cv'] = cv

        # Init the ensemble object
        ensemble = ensemble_obj(**pass_params)

        # Set ensemble n_jobs
        set_n_jobs(ensemble, ensemble_n_jobs)

        # Set random state
        if hasattr(ensemble, 'random_state'):
            setattr(ensemble, 'random_state', self.random_state)

        # Wrap as pipeline compatible object
        new_ensemble = [(ensemble_name, ensemble)]

        # Append ensemble name to all model params
        self._update_model_ensemble_params(ensemble_name,
                                           ensemble=False)

        return new_ensemble
