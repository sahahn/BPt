from autogluon import TabularPrediction as task
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator


def get_dataset(X, y=None):

    dataset = task.Dataset(X)

    if y is not None:
        dataset['target'] = y

    return dataset


class AutoGluon(BaseEstimator):

    def __init__(self, presets='best_quality',
                 problem_type=None, verbosity=0,
                 n_jobs=None, random_state=None):

        self.presets = presets
        self.problem_type = problem_type
        self.verbosity = verbosity
        self.n_jobs = n_jobs
        self.random_state = random_state

    def _estimator_type(self):

        if self.problem_type is None or self.problem_type == 'regression':
            return 'regression'
        return 'classifier'

    def fit(self, X, y):

        # Set is_fitted_
        self.is_fitted_ = True

        # Get dataset
        train_data = get_dataset(X, y)

        # Fit autogluon model
        self._model = task.fit(train_data=task.Dataset(train_data),
                               label='target',
                               problem_type=self.problem_type,
                               presets=self.presets,
                               verbosity=self.verbosity,
                               random_seed=self.random_state,
                               nthreads_per_trial=self.n_jobs)

        return self

    def predict(self, X):
        check_is_fitted(self, 'is_fitted_')

        test_data = get_dataset(X)
        return self._model.predict(test_data)

    def predict_proba(self, X):
        check_is_fitted(self, 'is_fitted_')

        test_data = get_dataset(X)
        return self._model.predict_proba(test_data)
