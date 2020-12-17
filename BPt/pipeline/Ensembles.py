from ..helpers.ML_Helpers import (replace_with_in_params,
                                  get_obj_and_params, set_n_jobs)

from deslib.dcs.a_posteriori import APosteriori
from deslib.dcs.a_priori import APriori
from deslib.dcs.lca import LCA
from deslib.dcs.mcb import MCB
from deslib.dcs.mla import MLA
from deslib.dcs.ola import OLA
from deslib.dcs.rank import Rank

from deslib.des.meta_des import METADES
from deslib.des.des_clustering import DESClustering
from deslib.des.des_p import DESP
from deslib.des.des_knn import DESKNN
from deslib.des.knop import KNOP
from deslib.des.knora_e import KNORAE
from deslib.des.knora_u import KNORAU
from deslib.des.des_mi import DESMI

from deslib.des.probabilistic import RRC
from deslib.des.probabilistic import DESKL
from deslib.des.probabilistic import MinimumDifference
from deslib.des.probabilistic import Exponential
from deslib.des.probabilistic import Logarithmic

from deslib.static.single_best import SingleBest
from deslib.static.stacked import StackedClassifier

from copy import deepcopy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (BaggingClassifier, BaggingRegressor,
                              AdaBoostRegressor, AdaBoostClassifier)
from sklearn.ensemble import (StackingRegressor, StackingClassifier,
                              VotingClassifier, VotingRegressor)
from joblib import Parallel, delayed
from sklearn.base import clone, is_classifier
from sklearn.utils import Bunch
from sklearn.model_selection import check_cv, cross_val_predict
import numpy as np
from .base import _fit_single_estimator, _get_est_fit_params
from ..main.Params_Classes import CV_Splits

from sklearn.utils.multiclass import check_classification_targets
from sklearn.preprocessing import LabelEncoder


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
    if isinstance(self.cv, CV_Splits):
        _, cv_inds = self.cv.get_splits(train_data_index)
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

    # Whatever the final estimator is might use train data index,
    # but I don't see why it would need mapping - as the final estimator
    # is being trained on only the predictions from the base models...
    # I guess a future @TODO could be pass the right mapping here
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


class BPtVotingClassifier(VotingClassifier):
    _needs_mapping = True
    _needs_train_data_index = True
    _fit_all_estimators = _fit_all_estimators
    bpt_fit = voting_fit
    fit = ensemble_classifier_fit


class DES_Ensemble(VotingClassifier):

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


class Ensemble_Wrapper():

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

        # If no ensembling is passed, return either the 1 model,
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

            # If DES Ensemble,
            if ensemble_params.is_des:
                return self._wrap_des(models, ensemble,
                                      ensemble_params.ensemble_split)

            # If no split and single estimator
            elif ensemble_params.single_estimator:
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
                                           ensemble_params.cv_splits)

    def _wrap_des(self, models, ensemble_info, ensemble_split):

        # Unpack ensemble info
        ensemble_name = ensemble_info[0]
        ensemble_obj = ensemble_info[1][0]
        ensemble_extra_params = ensemble_info[1][1]

        # Init with default params
        ensemble = ensemble_obj()

        # Set ensemble random_state
        if hasattr(ensemble, 'random_state'):
            setattr(ensemble, 'random_state', self.random_state)

        # Regardless of n_jobs_type, go with models, as
        # default des doesn't handle multi-proc well.
        set_n_jobs(ensemble, 1)
        set_n_jobs(models, self.n_jobs)

        # Create pipeline compatible des ensemble
        new_ensemble =\
            [(ensemble_name, DES_Ensemble(models,
                                          ensemble,
                                          ensemble_name,
                                          ensemble_split,
                                          ensemble_extra_params,
                                          self.random_state))]

        # Update the params
        self._update_model_ensemble_params(ensemble_name)

        return new_ensemble

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
                       n_jobs_type, cv_splits):
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
        if cv_splits is not None:
            pass_params['cv'] = cv_splits

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


AVALIABLE = {
        'binary': {
            'basic ensemble': 'voting classifier',
            None: 'voting classifier',
            'voting': 'voting classifier',
            'aposteriori': 'aposteriori',
            'apriori': 'apriori',
            'lca': 'lca',
            'mcb': 'mcb',
            'mla': 'mla',
            'ola': 'ola',
            'rank': 'rank',
            'metades': 'metades',
            'des clustering': 'des clustering',
            'desp': 'desp',
            'des knn': 'des knn',
            'knop': 'knop',
            'knorae': 'knorae',
            'knrau': 'knrau',
            'desmi': 'desmi',
            'rrc': 'rrc',
            'deskl': 'deskl',
            'min dif': 'min dif',
            'exponential': 'exponential',
            'logarithmic': 'logarithmic',
            'single best': 'single best',
            'stacked': 'stacked',
            'bagging': 'bagging classifier',
            'adaboost': 'adaboost classifier',
            'stacking': 'stacking classifier',
        },
        'regression': {
                'basic ensemble': 'voting regressor',
                None: 'voting regressor',
                'voting': 'voting regressor',
                'bagging': 'bagging regressor',
                'stacking': 'stacking regressor',
                'adaboost': 'adaboost regressor',
        },
}


ENSEMBLES = {
    'aposteriori': (APosteriori, ['default']),
    'apriori': (APriori, ['default']),
    'lca': (LCA, ['default']),
    'mcb': (MCB, ['default']),
    'mla': (MLA, ['default']),
    'ola': (OLA, ['default']),
    'rank': (Rank, ['default']),
    'metades': (METADES, ['default']),
    'des clustering': (DESClustering, ['default']),
    'desp': (DESP, ['default']),
    'des knn': (DESKNN, ['default']),
    'knop': (KNOP, ['default']),
    'knorae': (KNORAE, ['default']),
    'knrau': (KNORAU, ['default']),
    'desmi': (DESMI, ['default']),
    'rrc': (RRC, ['default']),
    'deskl': (DESKL, ['default']),
    'min dif': (MinimumDifference, ['default']),
    'exponential': (Exponential, ['default']),
    'logarithmic': (Logarithmic, ['default']),
    'single best': (SingleBest, ['default']),
    'stacked': (StackedClassifier, ['default']),
    'bagging classifier': (BaggingClassifier, ['default']),
    'bagging regressor': (BaggingRegressor, ['default']),
    'adaboost classifier': (AdaBoostClassifier, ['default']),
    'adaboost regressor': (AdaBoostRegressor, ['default']),
    'stacking regressor': (BPtStackingRegressor,
                           ['default']),
    'stacking classifier': (BPtStackingClassifier,
                            ['default']),
    'voting classifier': (BPtVotingClassifier,
                          ['voting classifier']),
    'voting regressor': (BPtVotingRegressor,
                         ['default']),
}

try:

    from imblearn.ensemble import BalancedBaggingClassifier

    AVALIABLE['binary']['balanced bagging'] = 'balanced bagging classifier'

    ENSEMBLES['balanced bagging classifier'] =\
        (BalancedBaggingClassifier, ['default'])

except ImportError:
    pass

# Should be the same
AVALIABLE['categorical'] = AVALIABLE['binary'].copy()


def get_ensemble_and_params(ensemble_str, extra_params, params, search_type,
                            random_state=None, num_feat_keys=None):

    if ensemble_str == 'basic ensemble' or ensemble_str is None:
        return None, {}

    ensemble, extra_ensemble_params, ensemble_params =\
        get_obj_and_params(ensemble_str, ENSEMBLES, extra_params, params)

    # Slight tweak here, return tuple ensemble, extra_params
    return (ensemble, extra_ensemble_params), ensemble_params
