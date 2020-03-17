import warnings
import numpy as np
import nevergrad as ng

from concurrent import futures
import multiprocessing as mp

from sklearn.base import clone
from sklearn.model_selection import cross_val_score


class NevergradSearchCV():

    def __init__(self, optimizer_name, estimator, param_distributions,
                 scoring=None, cv=3, weight_metric=False,
                 n_iter=10, n_jobs=1, random_state=None):

        self.optimizer_name = optimizer_name
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.scoring = scoring
        self.cv = cv
        self.weight_metric = weight_metric,
        self.n_iter = n_iter
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.name = 'nevergrad'

    def ng_cv_score(self, X, y, **kwargs):

        estimator = clone(self.estimator)
        estimator.set_params(**kwargs)

        # All sklearn scorers should return high values as better, so flip sign
        cv_scores = -cross_val_score(estimator, X=X, y=y, scoring=self.scoring,
                                    cv=self.cv)

        if self.weight_metric:
            weights=[len(self.cv[i][1]) for i in range(len(self.cv))]
            return np.average(cv_scores, weights=weights)
        else:
            return np.mean(cv_scores)
            

    def fit(self, X, y=None):

        # Fit the nevergrad optimizer
        instrumentation = ng.Instrumentation(X, y, **self.param_distributions)

        optimizer = ng.optimizers.registry[self.optimizer_name](
            instrumentation=instrumentation,
            budget=self.n_iter,
            num_workers=self.n_jobs)

        # Set random state is defined
        if isinstance(self.random_state, int):
            optimizer.instrumentation.random_state =\
                np.random.RandomState(self.random_state)

        elif self.random_state is not None:
            optimizer.instrumentation.random_state = self.random_state

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            with futures.ProcessPoolExecutor(max_workers=self.n_jobs,
                                             mp_context=mp.get_context('spawn')) as ex:

                recommendation = optimizer.minimize(self.ng_cv_score,
                                                    executor=ex,
                                                    batch_mode=False)

        # Save best search search score
        self.best_search_score = optimizer.current_bests["pessimistic"].mean

        # Fit best estimator
        self.best_estimator_ = clone(self.estimator)
        self.best_estimator_.set_params(**recommendation.kwargs)

        self.best_estimator_.fit(X, y)

    def predict(self, X):
        return self.best_estimator_.predict(X)

    def predict_log_proba(self, X):
        return self.best_estimator_.predict_log_proba(X)

    def predict_proba(self, X):
        return self.best_estimator_.predict_proba(X)

    def decision_function(self, X):
        return self.best_estimator_.decision_function(X)

    
