from lightgbm import LGBMRegressor, LGBMClassifier
from ..main.CV import BPtCV
from ..pipeline.helpers import proc_mapping
from sklearn.model_selection import train_test_split
import warnings


class BPtMixIn():

    _needs_mapping = True
    _needs_fit_index = True
    _needs_cat_inds = True

    def _get_categorical_feature(self, mapping):

        if not hasattr(self, 'cat_inds'):
            return 'auto'

        # If passed cat inds
        if self.cat_inds is not None:

            # Check for passed mapping, if found apply
            if mapping is not None:
                return proc_mapping(self.cat_inds, mapping)

            # Otherwise assume as is
            return self.cat_inds

        # Otherwise keep as default
        return 'auto'

    def _get_eval_set(self, X, y, fit_index=None):

        # Make sure there's early stop rounds set!
        if not hasattr(self, 'early_stopping_rounds'):
            return X, y, None
        if self.early_stopping_rounds is None:
            return X, y, None

        # Check for eval split
        if hasattr(self, 'eval_split'):

            # If any eval_split
            if self.eval_split is not None:
                if isinstance(self.eval_split, BPtCV):

                    # Get the cv_inds
                    train_inds, eval_inds =\
                        self.eval_split.get_split(
                            fit_index,
                            random_state=self.random_state)

                    # Index
                    X_train, y_train = X[train_inds], y[train_inds]
                    X_eval, y_eval = X[eval_inds], y[eval_inds]

                # Otherwise interpret as test size
                else:
                    X_train, X_eval, y_train, y_eval =\
                        train_test_split(X, y, test_size=self.eval_split,
                                         random_state=self.random_state)

                # Return
                return X_train, y_train, (X_eval, y_eval)

        return X, y, None

    def fit(self, X, y, mapping=None, fit_index=None, **kwargs):

        # Get rid of "other params"
        self._other_params = {}

        # Get categorical features based on if passed mapping
        categorical_feature = self._get_categorical_feature(mapping)

        # Proc eval set
        X_train, y_train, eval_set =\
            self._get_eval_set(X, y, fit_index=fit_index)

        # Check early stopping rounds:
        if hasattr(self, 'early_stopping_rounds'):
            early_stopping_rounds = self.early_stopping_rounds
        else:
            early_stopping_rounds = None

        # To avoid categorical feature warnings...
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Call super fit
            super().fit(X_train, y_train,
                        eval_set=eval_set,
                        categorical_feature=categorical_feature,
                        early_stopping_rounds=early_stopping_rounds,
                        verbose=False,
                        **kwargs)


class BPtLGBMRegressor(BPtMixIn, LGBMRegressor):
    pass


class BPtLGBMClassifier(BPtMixIn, LGBMClassifier):
    pass
