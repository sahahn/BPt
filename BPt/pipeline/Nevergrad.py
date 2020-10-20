import numpy as np
from numpy.random import RandomState
import nevergrad as ng

from concurrent import futures
import multiprocessing as mp

from sklearn.base import clone
from copy import deepcopy

from ..helpers.CV import CV as Base_CV
from ..helpers.ML_Helpers import get_possible_fit_params

from os.path import dirname, abspath, exists
from sklearn.base import BaseEstimator
import warnings


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
                cv_inds, cv_subjects, fit_params, **kwargs):

    cv_scores = []
    for i in range(len(cv_inds)):
        tr_inds, test_inds = cv_inds[i]

        # Clone estimator & set search params
        estimator = clone(estimator)
        estimator.set_params(**kwargs)

        # Add this folds train_data_index to fit_params, if valid
        if 'train_data_index' in get_possible_fit_params(estimator):
            fit_params['train_data_index'] = cv_subjects[i][0]

        # Fit estimator on train
        estimator.fit(X[tr_inds], y[tr_inds], **deepcopy(fit_params))

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
                 scoring=None, weight_scorer=False,
                 random_state=None, executor=None,
                 progress_loc=None, verbose=False):

        self.param_search = param_search
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.scoring = scoring
        self.weight_scorer = weight_scorer
        self.random_state = random_state
        self.executor = executor
        self.progress_loc = progress_loc
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

    def _set_cv(self, train_data_index):

        # If no CV, use random
        if self.param_search.CV is None:
            self.param_search.CV = Base_CV()

        self.cv_subjects, self.cv_inds =\
            self.param_search.CV.get_cv(train_data_index,
                                        self.param_search.splits,
                                        self.param_search.n_repeats,
                                        self.param_search._splits_vals,
                                        self.random_state,
                                        return_index='both')

    def ng_cv_score(self, X, y, fit_params, **kwargs):

        cv_scores = []
        for i in range(len(self.cv_inds)):
            tr_inds, test_inds = self.cv_inds[i]

            # Clone estimator & set search params
            estimator = clone(self.estimator)
            estimator.set_params(**kwargs)

            # Add this folds train_data_index to fit_params, if valid
            if 'train_data_index' in get_possible_fit_params(estimator):
                fit_params['train_data_index'] = self.cv_subjects[i][0]

            # Fit estimator on train
            estimator.fit(X[tr_inds], y[tr_inds], **deepcopy(fit_params))

            # Get the score, but scoring return high values as better,
            # so flip sign
            score = -self.scoring(estimator, X[test_inds], y[test_inds])
            cv_scores.append(score)

        if self.weight_scorer:
            weights = [len(self.cv_inds[i][1]) for i
                       in range(len(self.cv_inds))]
            return np.average(cv_scores, weights=weights)
        else:
            return np.mean(cv_scores)

    def fit(self, X, y=None, train_data_index=None, **fit_params):

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

        # Fit the nevergrad optimizer
        instrumentation =\
            ng.p.Instrumentation(X, y, self.estimator, self.scoring,
                                 self.weight_scorer, self.cv_inds,
                                 self.cv_subjects,
                                 fit_params, **self.param_distributions)

        try:
            opt = ng.optimizers.registry[self.param_search.search_type]

        # If not found, look for in expirimental variants
        except KeyError:
            import nevergrad.optimization.experimentalvariants
            opt = ng.optimizers.registry[self.param_search.search_type]

        optimizer = opt(parametrization=instrumentation,
                        budget=self.param_search.n_iter,
                        num_workers=self.param_search.n_jobs)

        # Set random state is defined
        if isinstance(self.random_state, int):
            optimizer.parametrization.random_state =\
                RandomState(self.random_state)

        elif self.random_state is not None:
            optimizer.parametrization.random_state = self.random_state

        # with warnings.catch_warnings():
        #    warnings.simplefilter("ignore")

        if self.progress_loc is not None:
            logger = ProgressLogger(self.progress_loc)
            optimizer.register_callback('tell', logger)

        if self.param_search.n_jobs == 1:
            recommendation = optimizer.minimize(ng_cv_score,
                                                batch_mode=False)

        elif self.executor is not None:
            from dask.distributed import Client
            client = Client(self.executor)
            recommendation = optimizer.minimize(ng_cv_score,
                                                executor=client,
                                                batch_mode=False)

        else:
            try:
                with futures.ProcessPoolExecutor(
                  max_workers=self.param_search.n_jobs,
                  mp_context=mp.get_context(self.param_search.mp_context)) as ex:

                    recommendation = optimizer.minimize(ng_cv_score,
                                                        executor=ex,
                                                        batch_mode=False)
            except RuntimeError:
                raise(RuntimeError('Try changing the mp_context'))

        # Save best search search score
        self.best_search_score = optimizer.current_bests["pessimistic"].mean
        # "optimistic", "pessimistic", "average"

        # Fit best estimator, w/ found best params
        self.best_estimator_ = clone(self.estimator)
        self.best_estimator_.set_params(**recommendation.kwargs)
        self.best_params_ = recommendation.kwargs

        # Full train index here
        if 'train_data_index' in get_possible_fit_params(self.best_estimator_):
            fit_params['train_data_index'] = train_data_index

        self.best_estimator_.fit(X, y, **fit_params)

    def predict(self, X):
        return self.best_estimator_.predict(X)

    def predict_log_proba(self, X):
        return self.best_estimator_.predict_log_proba(X)

    def predict_proba(self, X):
        return self.best_estimator_.predict_proba(X)

    def decision_function(self, X):
        return self.best_estimator_.decision_function(X)
