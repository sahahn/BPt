import numpy as np
from numpy.random import RandomState
import nevergrad as ng

from concurrent import futures
import multiprocessing as mp

from sklearn.base import clone
from copy import deepcopy

from .base import _get_est_fit_params
from ..helpers.CV import CV
from os.path import dirname, abspath, exists
from sklearn.base import BaseEstimator
from sklearn.utils.metaestimators import if_delegate_has_method
import warnings

try:
    from loky import get_reusable_executor
except ImportError:
    pass


class ProgressLogger():

    def __init__(self, loc):
        self.loc = loc

    def __call__(self, optimizer=None, candidate=None, value=None):

        # If progress loc parent folder is removed, stop program
        if not exists(dirname(abspath(self.loc))):
            raise SystemExit('Folder where progress is stored was removed!')

        with open(self.loc, 'a') as f:
            f.write('params,')


def ng_cv_score(X, y, estimator, scoring, weight_scorer,
                cv_inds, cv_subjects, mapping, fit_params, **kwargs):

    cv_scores = []
    for i in range(len(cv_inds)):
        tr_inds, test_inds = cv_inds[i]

        # Clone estimator & set search params
        estimator = clone(estimator)
        estimator.set_params(**kwargs)

        # Adds mapping / train data index if needed
        f_params = _get_est_fit_params(
            estimator,
            mapping=mapping,
            train_data_index=cv_subjects[i][0],
            other_params=fit_params)

        # Fit estimator on train
        estimator.fit(X[tr_inds], y[tr_inds], **deepcopy(f_params))

        # Get the score, but scoring return high values as better,
        # so flip sign
        score = -scoring(estimator, X[test_inds], y[test_inds])
        cv_scores.append(score)

    if weight_scorer:
        weights = [len(cv_inds[i][1]) for i
                   in range(len(cv_inds))]
        return np.average(cv_scores, weights=weights)
    else:
        return np.mean(cv_scores)


class NevergradSearchCV(BaseEstimator):

    needs_mapping = True
    needs_train_data_index = True
    name = 'nevergrad'

    def __init__(self, estimator=None, param_search=None,
                 param_distributions=None,
                 progress_loc=None, n_jobs=1,
                 random_state=None,
                 verbose=False):

        self.estimator = estimator
        self.param_search = param_search
        self.param_distributions = param_distributions
        self.progress_loc = progress_loc
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            try:
                value = getattr(self, key)
            except AttributeError:
                warnings.warn('From version 0.24, get_params will raise an '
                              'AttributeError if a parameter cannot be '
                              'retrieved as an instance attribute. Previously '
                              'it would return None.',
                              FutureWarning)
                value = None
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out

    @property
    def _estimator_type(self):
        return self.estimator._estimator_type

    @property
    def n_features_in_(self):
        return self.best_estimator_.n_features_in_

    @property
    def classes_(self):
        return self.best_estimator_.classes_

    def _set_cv(self, train_data_index):

        # If no CV, use random
        if self.param_search._cv is None:
            self.param_search._cv = CV()

        self.cv_subjects, self.cv_inds =\
            self.param_search._cv.get_cv(train_data_index,
                                         self.param_search.splits,
                                         self.param_search.n_repeats,
                                         self.param_search._splits_vals,
                                         self.param_search._random_state,
                                         return_index='both')

    def get_instrumentation(self, X, y, mapping, fit_params, client):

        if client is None:
            instrumentation =\
                ng.p.Instrumentation(X, y, self.estimator,
                                     self.param_search._scorer,
                                     self.param_search.weight_scorer,
                                     self.cv_inds,
                                     self.cv_subjects, mapping,
                                     fit_params, **self.param_distributions)

        # If using dask client, pre-scatter some big memory fixed params
        else:
            X_s = client.scatter(X)
            y_s = client.scatter(y)
            cv_inds_s = client.scatter(self.cv_inds)
            cv_subjects_s = client.scatter(self.cv_subjects)

            instrumentation =\
                ng.p.Instrumentation(X_s, y_s, self.estimator,
                                     self.param_search._scorer,
                                     self.param_search.weight_scorer,
                                     cv_inds_s,
                                     cv_subjects_s, mapping,
                                     fit_params, **self.param_distributions)

        return instrumentation

    def get_optimizer(self, instrumentation):

        try:
            opt = ng.optimizers.registry[self.param_search.search_type]

        # If not found, look for in expirimental variants
        except KeyError:
            import nevergrad.optimization.experimentalvariants
            opt = ng.optimizers.registry[self.param_search.search_type]

        optimizer = opt(parametrization=instrumentation,
                        budget=self.param_search.n_iter,
                        num_workers=self.n_jobs)

        # Set random state is defined
        if isinstance(self.random_state, int):
            optimizer.parametrization.random_state =\
                RandomState(self.random_state)

        elif self.random_state is not None:
            optimizer.parametrization.random_state =\
                self.random_state

        if self.progress_loc is not None:
            logger = ProgressLogger(self.progress_loc)
            optimizer.register_callback('tell', logger)

        return optimizer

    def run_search(self, optimizer, client):

        # n_jobs 1, always local
        if self.n_jobs == 1:
            recommendation = optimizer.minimize(ng_cv_score,
                                                batch_mode=False)

        # If generated client
        elif client is not None:
            recommendation = optimizer.minimize(ng_cv_score,
                                                executor=client,
                                                batch_mode=False)

        # Otherwise use futures pool executor
        else:

            if self.param_search.mp_context == 'loky':

                try:
                    executor = get_reusable_executor(
                        max_workers=self.n_jobs, timeout=120)

                    recommendation = optimizer.minimize(ng_cv_score,
                                                        executor=executor,
                                                        batch_mode=False)
                except NameError:
                    raise(ImportError('Make sure loky is installed'))

            try:
                with futures.ProcessPoolExecutor(
                  max_workers=self.n_jobs,
                  mp_context=mp.get_context(self.param_search.mp_context)) as ex:

                    recommendation = optimizer.minimize(ng_cv_score,
                                                        executor=ex,
                                                        batch_mode=False)
            except RuntimeError:
                raise(RuntimeError('Try changing the mp_context'))

        # Save best search search score
        # "optimistic", "pessimistic", "average"
        # and best params
        self.best_search_score = optimizer.current_bests["pessimistic"].mean
        self.best_params_ = recommendation.kwargs

        return recommendation

    def fit(self, X, y=None, mapping=None,
            train_data_index=None, **fit_params):

        if train_data_index is None:
            raise RuntimeWarning('NevergradSearchCV must be passed a ' +
                                 'train_data_index!')

        if self.verbose:
            print('Fit Nevergrad CV, len(train_data_index) == ',
                  len(train_data_index),
                  'has mapping == ', 'mapping' in fit_params,
                  'X.shape ==', X.shape)

        # Set the search cv passed on passed train_data_index
        self._set_cv(train_data_index)

        # Check if need to make dask client
        # Criteria is greater than 1 job, and passed as dask_ip of non-None
        if self.n_jobs > 1 and self.param_search.dask_ip is not None:
            from dask.distributed import Client
            client = Client(self.param_search.dask_ip)
        else:
            client = None

        # Get the instrumentation
        instrumentation =\
            self.get_instrumentation(X, y, mapping=mapping,
                                     fit_params=fit_params,
                                     client=client)

        # Get the optimizer
        optimizer = self.get_optimizer(instrumentation)

        # Run the search
        recommendation = self.run_search(optimizer, client)

        # Fit best est, w/ best params
        self.fit_best_estimator(recommendation, X, y, mapping,
                                train_data_index, fit_params)

    def fit_best_estimator(self, recommendation,  X, y, mapping,
                           train_data_index, fit_params):

        # Fit best estimator, w/ found best params
        self.best_estimator_ = clone(self.estimator)
        self.best_estimator_.set_params(**recommendation.kwargs)

        # Add in mapping / train data index to fit params if needed
        f_params = _get_est_fit_params(
            self.best_estimator_,
            mapping=mapping,
            train_data_index=train_data_index,
            other_params=fit_params)

        # Fit
        self.best_estimator_.fit(X, y, **f_params)

    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    def predict(self, X):
        return self.best_estimator_.predict(X)
    
    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    def predict_log_proba(self, X):
        return self.best_estimator_.predict_log_proba(X)

    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    def predict_proba(self, X):
        return self.best_estimator_.predict_proba(X)

    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    def decision_function(self, X):
        return self.best_estimator_.decision_function(X)

    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    def transform(self, X):
        return self.best_estimator_.transform(X)

    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    def inverse_transform(self, Xt):
        return self.best_estimator_.inverse_transform(Xt)


def wrap_param_search(param_search, model_obj, model_params):

    if param_search is None:
        return model_obj, model_params

    name = model_obj[0]
    prepend = name + '__'

    # Remove the relevant model params
    # and put in m_params
    m_params = {}
    model_param_names = list(model_params)
    for param in model_param_names:

        if param.startswith(prepend):
            m_params[param.replace(prepend, '', 1)] =\
                model_params.pop(param)

    # Create the wrapper nevergrad CV model
    search_obj = NevergradSearchCV(
        estimator=model_obj[1],
        param_search=param_search,
        param_distributions=m_params,
        n_jobs=param_search._n_jobs,
        random_state=param_search._random_state)

    return (name + '_NGSearchCV', search_obj), model_params
