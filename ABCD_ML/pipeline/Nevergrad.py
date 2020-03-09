import warnings
import numpy as np
import nevergrad as ng

from concurrent import futures
from sklearn.base import clone
from sklearn.model_selection import cross_val_score


class NevergradSearchCV():

    def __init__(self, optimizer_name, estimator, param_distributions,
                 scoring=None, cv=3,
                 n_iter=10, n_jobs=1, random_state=None):

        self.optimizer_name = optimizer_name
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.scoring = scoring
        self.cv = cv
        self.n_iter = n_iter
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.name = 'nevergrad'

    def ng_cv_score(self, X, y, **kwargs):

        estimator = clone(self.estimator)
        estimator.set_params(**kwargs)

        # All sklearn scorers should return high values as better, so flip sign
        cv_score = -cross_val_score(estimator, X=X, y=y, scoring=self.scoring,
                                    cv=self.cv)

        return np.mean(cv_score)

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

            with futures.ProcessPoolExecutor(max_workers=self.n_jobs) as ex:
                recommendation = optimizer.minimize(self.ng_cv_score,
                                                    executor=ex,
                                                    batch_mode=False)

        # Fit best estimator
        self.best_estimator_ = self.estimator
        self.best_estimator_.set_params(**recommendation.kwargs)

        # optimizer.current_bests["pessimistic"].mean,

        self.best_estimator_.fit(X, y)

    def predict(self, X):
        return self.best_estimator_.predict(X)

    def predict_log_proba(self, X):
        return self.best_estimator_.predict_log_proba(X)

    def predict_proba(self, X):
        return self.best_estimator_.predict_proba(X)

    def decision_function(self, X):
        return self.best_estimator_.decision_function(X)

    
