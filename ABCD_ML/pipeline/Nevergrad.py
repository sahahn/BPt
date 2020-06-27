import numpy as np
from numpy.random import RandomState
import nevergrad as ng

from concurrent import futures
import multiprocessing as mp

from sklearn.base import clone
from copy import deepcopy

from ..helpers.CV import CV as Base_CV
from ..helpers.ML_Helpers import get_possible_fit_params


class NevergradSearchCV():

    def __init__(self, params, estimator, param_distributions,
                 scoring=None, weight_metric=False, random_state=None):

        self.params = params
        self.estimator = estimator
        self.param_distributions = param_distributions

        # If no CV, use random
        if self.params._CV is None:
            self.params.CV = Base_CV()

        self.scoring = scoring
        self.weight_metric = weight_metric
        self.random_state = random_state

        self.name = 'nevergrad'

    def _set_cv(self, train_data_index):

        self.cv_subjects, self.cv_inds =\
            self.params._CV.get_cv(train_data_index, self.params.splits,
                                   self.params.n_repeats,
                                   self.params._splits_vals, self.random_state,
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

        if self.weight_metric:
            weights = [len(self.cv_inds[i][1]) for i
                       in range(len(self.cv_inds))]
            return np.average(cv_scores, weights=weights)
        else:
            return np.mean(cv_scores)

    def fit(self, X, y=None, train_data_index=None, **fit_params):

        # Set the search cv passed on passed train_data_index
        self._set_cv(train_data_index)

        # Fit the nevergrad optimizer
        instrumentation =\
            ng.p.Instrumentation(X, y, fit_params, **self.param_distributions)

        try:
            opt = ng.optimizers.registry[self.params.search_type]

        # If not found, look for in expirimental variants
        except KeyError:
            import nevergrad.optimization.experimentalvariants
            opt = ng.optimizers.registry[self.params.search_type]

        optimizer = opt(parametrization=instrumentation,
                        budget=self.params.n_iter,
                        num_workers=self.params._n_jobs)

        # Set random state is defined
        if isinstance(self.random_state, int):
            optimizer.parametrization.random_state =\
                RandomState(self.random_state)

        elif self.random_state is not None:
            optimizer.parametrization.random_state = self.random_state

        #with warnings.catch_warnings():
        #    warnings.simplefilter("ignore")

        if self.params._n_jobs == 1:
            recommendation = optimizer.minimize(self.ng_cv_score,
                                                batch_mode=False)

        else:
            try:
                with futures.ProcessPoolExecutor(
                  max_workers=self.params._n_jobs,
                  mp_context=mp.get_context(self.params.mp_context)) as ex:

                    recommendation = optimizer.minimize(self.ng_cv_score,
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

    
