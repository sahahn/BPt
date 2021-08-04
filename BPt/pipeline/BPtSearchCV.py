from sklearn.base import BaseEstimator
from .base import _get_est_fit_params, _needs, _get_est_trans_params
from sklearn.model_selection import GridSearchCV
from sklearn.utils.metaestimators import if_delegate_has_method
import warnings
from os.path import dirname, abspath, exists

import numpy as np
from numpy.random import RandomState
import nevergrad as ng

from concurrent import futures
import multiprocessing as mp

from sklearn.base import clone
from copy import deepcopy
import os
import pandas as pd

from .helpers import to_memmap, from_memmap, get_grid_params
from loky import get_reusable_executor


class BPtSearchCV(BaseEstimator):

    _needs_mapping = True
    _needs_fit_index = True
    name = 'search'

    def __init__(self, estimator=None, ps=None,
                 param_distributions=None,
                 n_jobs=1,
                 random_state=None):

        self.estimator = estimator
        self.ps = ps
        self.param_distributions = param_distributions
        self.n_jobs = n_jobs
        self.random_state = random_state

    @property
    def _needs_transform_index(self):
        return _needs(self.estimator, '_needs_transform_index',
                      'transform_index', 'transform')

    @property
    def n_jobs(self):
        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, n_jobs):

        # Store in self._n_jobs
        self._n_jobs = n_jobs

        # If passed n_jobs, then propegate
        # n_jobs == 1 to base models.
        if hasattr(self.estimator, 'n_jobs'):
            setattr(self.estimator, 'n_jobs', 1)

    @property
    def random_state(self):
        return self._random_state

    @random_state.setter
    def random_state(self, random_state):

        # Store
        self._random_state = random_state

        # Propegate
        if hasattr(self.estimator, 'random_state'):
            setattr(self.estimator, 'random_state', random_state)

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
    def feature_importances_(self):
        if hasattr(self.best_estimator_, 'feature_importances_'):
            return getattr(self.best_estimator_, 'feature_importances_')
        return None

    @property
    def coef_(self):
        if hasattr(self.best_estimator_, 'coef_'):
            return getattr(self.best_estimator_, 'coef_')
        return None

    @property
    def _final_estimator(self):
        if hasattr(self.best_estimator_, '_final_estimator'):
            return self.best_estimator_._final_estimator
        return None

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
    def transform(self, X, transform_index=None):

        trans_params = _get_est_trans_params(
            self.best_estimator_,
            transform_index=transform_index)

        return self.best_estimator_.transform(X, **trans_params)

    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    def inverse_transform(self, Xt):
        return self.best_estimator_.inverse_transform(Xt)

    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    def score(self, X, y=None, sample_weight=None):
        return self.best_estimator_.score(X=X, y=y,
                                          sample_weight=sample_weight)

    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    def transform_df(self, X_df, encoders=None):
        return self.best_estimator_.transform_df(X_df, encoders=encoders)

    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    def transform_feat_names(self, X_df, encoders=None):
        return self.best_estimator_.transform_feat_names(X_df,
                                                         encoders=encoders)

    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    def inverse_transform_FIs(self, fis):
        return self.best_estimator_.inverse_transform_FIs(fis)

    def _set_cv(self, fit_index):

        # Set cv based on fit_index
        self.cv_subjects, self.cv_inds =\
            self.ps['cv'].get_cv(fit_index,
                                 self.ps['random_state'],
                                 return_index='both')

    def fit(self, X, y=None, mapping=None,
            fit_index=None, **fit_params):

        # Conv from dataframe if dataframe.
        if isinstance(X, pd.DataFrame):
            fit_index = X.index
            X = np.array(X)
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = np.array(y)

        # Make sure train data index is passed
        if fit_index is None:
            raise RuntimeWarning('SearchCV Object must be passed a '
                                 'fit_index! Or passed X as '
                                 'a DataFrame.')

        if self.ps['verbose'] >= 3:
            print('Fit Search CV, len(fit_index) == ',
                  len(fit_index),
                  'has mapping == ', 'mapping' in fit_params,
                  'X.shape ==', X.shape)

        # Set the search cv passed on passed fit_index
        self._set_cv(fit_index)

        # Run different fit depending on type of search
        if self.ps['search_type'] == 'grid':
            self.fit_grid(X=X, y=y, mapping=mapping,
                          fit_index=fit_index,
                          **fit_params)
        else:
            self.fit_nevergrad(X=X, y=y, mapping=mapping,
                               fit_index=fit_index,
                               **fit_params)

        return self


class BPtGridSearchCV(BPtSearchCV):

    def fit_grid(self, X, y=None, mapping=None,
                 fit_index=None, **fit_params):

        # Conv nevergrad to grid compat. param grid
        param_grid = get_grid_params(self.param_distributions)

        # Fit GridSearchCV object
        self.search_obj_ = GridSearchCV(estimator=self.estimator,
                                        param_grid=param_grid,
                                        scoring=self.ps['scorer'],
                                        n_jobs=self.n_jobs,
                                        cv=self.cv_inds,
                                        refit=True,
                                        verbose=self.ps['verbose'])

        # Generate the fit params to pass
        f_params = _get_est_fit_params(
            self.estimator,
            mapping=mapping,
            fit_index=fit_index,
            other_params=fit_params)

        # Hack to support scoring for needs_transform
        if _needs(self.estimator, '_needs_transform_index',
                  'transform_index', 'transform'):
            score_X = pd.DataFrame(X, index=fit_index)
            self.search_obj_.fit(score_X, y, **f_params)

        else:
            self.search_obj_.fit(X, y, **f_params)

        return self

    @property
    def n_features_in_(self):
        return self.search_obj_.n_features_in_

    @property
    def classes_(self):
        return self.search_obj_.classes_

    @property
    def best_estimator_(self):
        return self.search_obj_.best_estimator_

    @property
    def best_score_(self):
        return self.search_obj_.best_score_

    @property
    def best_params_(self):
        return self.search_obj_.best_params_


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
                cv_inds, cv_subjects, mapping, fit_params,
                search_only_params, **search_params):

    # If passing memmap
    if isinstance(X, tuple):
        X = from_memmap(X)

    cv_scores = []
    for i in range(len(cv_inds)):
        tr_inds, test_inds = cv_inds[i]

        # Clone estimator & set search params
        estimator = clone(estimator)
        estimator.set_params(**search_params)

        # For each search only param, try to update, if invalid, just skip
        for key in search_only_params:
            try:
                estimator.set_params(**{key: search_only_params[key]})
            except ValueError:
                pass

        # Adds mapping / train data index if needed
        f_params = _get_est_fit_params(
            estimator,
            mapping=mapping,
            fit_index=cv_subjects[i][0],
            other_params=fit_params)

        # Fit estimator on train
        estimator.fit(X[tr_inds], y[tr_inds], **deepcopy(f_params))

        # Get the score, but scoring return high values as better,
        # so flip sign.

        # Hack to allow passing info on transform_index.
        if _needs(estimator, '_needs_transform_index',
                  'transform_index', 'transform'):
            score_X = pd.DataFrame(X[test_inds], index=cv_subjects[i][1])
            score = -scoring(estimator, score_X, y[test_inds])

        else:
            score = -scoring(estimator, X[test_inds], y[test_inds])

        cv_scores.append(score)

    if weight_scorer:
        weights = [len(cv_inds[i][1]) for i
                   in range(len(cv_inds))]
        return np.average(cv_scores, weights=weights)
    else:
        return np.mean(cv_scores)


class NevergradSearchCV(BPtSearchCV):

    # @TODO add more info from verbose!
    # @TODO add cv_results_, best_index_
    # other features of sklearn style.

    @property
    def n_features_in_(self):
        return self.best_estimator_.n_features_in_

    @property
    def classes_(self):
        return self.best_estimator_.classes_

    def get_instrumentation(self, X, y, mapping, fit_params, client):

        X_file = None

        if client is None:

            # Check for memmap X, only if no client
            if self.ps['memmap_X']:
                X_mem = to_memmap(X)
            else:
                X_mem = X

            # Convert parameters into compatible instrumentation
            instrumentation =\
                ng.p.Instrumentation(X_mem, y, self.estimator,
                                     self.ps['scorer'],
                                     self.ps['weight_scorer'],
                                     self.cv_inds,
                                     self.cv_subjects, mapping,
                                     fit_params,
                                     self.ps['search_only_params'],
                                     **self.param_distributions)

        # If using dask client, pre-scatter some big memory fixed params
        else:

            if self.ps['verbose'] >= 2:
                print('Scattering data to dask nodes.', flush=True)

            X_s = client.scatter(X)
            y_s = client.scatter(y)
            cv_inds_s = client.scatter(self.cv_inds)
            cv_subjects_s = client.scatter(self.cv_subjects)

            # Convert parameters into compatible instrumentation
            instrumentation =\
                ng.p.Instrumentation(X_s, y_s, self.estimator,
                                     self.ps['scorer'],
                                     self.ps['weight_scorer'],
                                     cv_inds_s,
                                     cv_subjects_s, mapping,
                                     fit_params,
                                     self.ps['search_only_params'],
                                     **self.param_distributions)

        return instrumentation, X_file

    def get_optimizer(self, instrumentation):

        try:
            opt = ng.optimizers.registry[self.ps['search_type']]

        # If not found, look for in experimental variants
        except KeyError:
            import nevergrad.optimization.experimentalvariants
            opt = ng.optimizers.registry[self.ps['search_type']]

        optimizer = opt(parametrization=instrumentation,
                        budget=self.ps['n_iter'],
                        num_workers=self.n_jobs)

        # Set random state is defined
        if isinstance(self.random_state, int):
            optimizer.parametrization.random_state =\
                RandomState(self.random_state)

        elif self.random_state is not None:
            optimizer.parametrization.random_state =\
                self.random_state

        if self.ps['progress_loc'] is not None:

            pl = self.ps['progress_loc']
            logger = ProgressLogger(pl)
            optimizer.register_callback('tell', logger)

            if self.ps['verbose'] >= 2:
                print(f'Storing SearchCV progress logs at: {pl}')

        return optimizer

    def run_search(self, optimizer, client):

        if self.ps['verbose'] >= 1:
            print('Starting nevergrad hyper-parameter search.', flush=True)

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

            if self.ps['mp_context'] == 'loky':

                executor = get_reusable_executor(
                    max_workers=self.n_jobs, timeout=120)

                recommendation = optimizer.minimize(ng_cv_score,
                                                    executor=executor,
                                                    batch_mode=False)

            try:
                with futures.ProcessPoolExecutor(
                  max_workers=self.n_jobs,
                  mp_context=mp.get_context(self.ps['mp_context'])) as ex:

                    recommendation = optimizer.minimize(ng_cv_score,
                                                        executor=ex,
                                                        batch_mode=False)
            except RuntimeError:
                raise(RuntimeError('Error with SearchCV!' +
                                   ' try changing the mp_context.'))

        # Save best search search score
        # "optimistic", "pessimistic", "average"
        # and best params
        self.best_score_ = optimizer.current_bests["optimistic"].mean
        self.best_params_ = recommendation.kwargs

        if self.ps['verbose'] >= 1:
            print('Finished nevergrad hyper-parameter search,',
                  'with best internal CV score:', self.best_score_)
        if self.ps['verbose'] >= 2:
            print('Selected hyper-parameters:', self.best_params_)

        return recommendation

    def fit_nevergrad(self, X, y=None, mapping=None,
                      fit_index=None, **fit_params):

        # Check if need to make dask client
        # Criteria is greater than 1 job, and passed as dask_ip of non-None
        if self.n_jobs > 1 and self.ps['dask_ip'] is not None:
            from dask.distributed import Client
            client = Client(self.ps['dask_ip'])
        else:
            client = None

        # Get the instrumentation
        instrumentation, X_file =\
            self.get_instrumentation(X, y, mapping=mapping,
                                     fit_params=fit_params,
                                     client=client)

        # Get the optimizer
        optimizer = self.get_optimizer(instrumentation)

        # Run the search
        recommendation = self.run_search(optimizer, client)

        # If X was mem mapped, unlink here
        if X_file is not None:
            os.unlink(X_file)

        # Fit best est, w/ best params
        self.fit_best_estimator(recommendation, X, y, mapping,
                                fit_index, fit_params)

        return self

    def fit_best_estimator(self, recommendation,  X, y, mapping,
                           fit_index, fit_params):

        if self.ps['verbose'] >= 1:
            print('Fitting SearchCV with best parameters on all train data.')

        # Fit best estimator, w/ found best params
        self.best_estimator_ = clone(self.estimator)
        self.best_estimator_.set_params(**recommendation.kwargs)

        # Add in mapping / train data index to fit params if needed
        f_params = _get_est_fit_params(
            self.best_estimator_,
            mapping=mapping,
            fit_index=fit_index,
            other_params=fit_params)

        # Fit
        self.best_estimator_.fit(X, y, **f_params)


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

    # Get search cv
    search_obj = get_search_cv(
        estimator=model_obj[1],
        param_search=param_search,
        param_distributions=m_params)

    # Create the wrapper nevergrad CV model
    return (name + '_SearchCV', search_obj), model_params


def get_search_cv(estimator, param_search,
                  param_distributions):

    # Determine which CV model to make
    if param_search['search_type'] == 'grid':
        SearchCV = BPtGridSearchCV
    else:
        SearchCV = NevergradSearchCV

    search_obj = SearchCV(
        estimator=estimator,
        ps=param_search,
        param_distributions=param_distributions,
        n_jobs=param_search['n_jobs'],
        random_state=param_search['random_state'])

    return search_obj
