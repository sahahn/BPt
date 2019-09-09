import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


class Regular_Imputer():

    def __init__(self, imputer, inds, copy=True):

        self.imputer = imputer
        self.inds = inds

        self.valid_mask = None
        self.copy = copy

    def fit(self, X, y=None):

        impute_X = self.get_impute_X(X)
        self.imputer.fit(impute_X, y)

    def get_impute_X(self, X):

        if self.valid_mask is None:
            self.valid_mask = ~pd.isnull(X).any(axis=0)

        valid_extra_X = X[:, self.valid_mask]

        # Just in case this col didnt have NaN in train, but does in test
        valid_extra_X[pd.isnull(valid_extra_X)] = 0

        ind = valid_extra_X.shape[1]
        impute_shape = (np.shape(X)[0], ind + len(self.inds))

        impute_X = np.zeros(impute_shape)
        impute_X[:, :ind] = valid_extra_X

        for i in range(len(self.inds)):
            impute_X[:, [ind+i]] = X[:, [self.inds[i]]]

        return impute_X

    def transform(self, X):

        if self.copy:
            X_copy = X.copy()
        else:
            X_copy = X

        impute_X = self.get_impute_X(X)
        imputed = self.imputer.transform(impute_X)

        ind = np.sum(self.valid_mask)

        for i in range(len(self.inds)):
            X_copy[:, self.inds[i]] = imputed[:, ind+i]

        return X_copy

    def fit_transform(self, X, y=None):

        self.fit(X, y)
        return self.transform(X)

    def set_params(self, **params):
        self.imputer.set_params(**params)


class Categorical_Imputer():

    def __init__(self, imputer, encoder_inds=[], ordinal_inds=[], encoders=[],
                 copy=True):

        self.imputer = imputer
        self.encoders = encoders
        self.encoder_inds = encoder_inds
        self.ordinal_inds = ordinal_inds

        self.valid_mask = None
        self.copy = copy

    def fit(self, X, y=None):

        impute_X = self.get_impute_X(X)
        self.imputer.fit(impute_X)

    def get_impute_X(self, X):

        if self.valid_mask is None:
            self.valid_mask = ~pd.isnull(X).any(axis=0)

        valid_extra_X = X[:, self.valid_mask]

        # Just in case this col didnt have NaN in train, but does in test
        valid_extra_X[pd.isnull(valid_extra_X)] = 0

        ind = valid_extra_X.shape[1]
        impute_shape = (np.shape(X)[0],
                        ind + len(self.encoder_inds) + len(self.ordinal_inds))
        impute_X = np.zeros(impute_shape)

        impute_X[:, :ind] = valid_extra_X

        for i in range(len(self.encoder_inds)):

            ordinal = self.to_ordinal(X, self.encoders[i],
                                      self.encoder_inds[i])
            impute_X[:, [ind+i]] = ordinal

        for i in range(len(self.ordinal_inds)):

            ordinal = X[:, [self.ordinal_inds[i]]]
            impute_X[:, [ind+i+len(self.encoder_inds)]] = ordinal

        return impute_X

    def transform(self, X):

        if self.copy:
            X_copy = X.copy()
        else:
            X_copy = X

        impute_X = self.get_impute_X(X)
        imputed = self.imputer.transform(impute_X)

        ind = np.sum(self.valid_mask)

        for i in range(len(self.encoder_inds)):

            imputed_col = imputed[:, [ind+i]]
            encoded = self.encoders[i][1].transform(imputed_col)

            # If originally dummy coded
            if len(self.encoders[i]) > 2:
                encoded = np.delete(encoded, self.encoders[i][2], axis=1)

            X_copy[:, self.encoder_inds[i]] = encoded

        for i in range(len(self.ordinal_inds)):

            X_copy[:, self.ordinal_inds[i]] =\
                imputed[:, ind+i+len(self.encoder_inds)]

        return X_copy

    def to_ordinal(self, X, encoder, inds):
        '''encoder should be the tuple object here'''

        selection = X[:, inds]
        non_nan_mask = ~pd.isnull(selection).any(axis=1)

        enc = encoder[1]
        non_nan_selection = selection[non_nan_mask]

        if len(encoder) > 2:
            non_nan_selection =\
                self.add_back_dummy_col(non_nan_selection, encoder[2])

        ordinal = enc.inverse_transform(non_nan_selection)

        to_fill = np.full((len(non_nan_mask), 1), np.nan)
        to_fill[non_nan_mask] = ordinal

        return to_fill

    def add_back_dummy_col(self, non_nan_selection, e):

        nn_shape = non_nan_selection.shape
        reverse_dummy_code = np.zeros((nn_shape[0], nn_shape[1]+1))

        # Add around dropped
        reverse_dummy_code[:, 0:e] = non_nan_selection[:, :e]
        reverse_dummy_code[:, e+1:] = non_nan_selection[:, e:]

        # Recover dropped values
        reverse_dummy_code[:, e] =\
            np.where(np.sum(reverse_dummy_code, axis=1) == 1, 0, 1)

        return reverse_dummy_code

    def fit_transform(self, X, y=None):

        self.fit(X, y)
        return self.transform(X)

    def set_params(self, **params):
        self.imputer.set_params(**params)


IMPUTERS = {
    'mean': (SimpleImputer, {'strategy': 'mean'}),
    'median': (SimpleImputer, {'strategy': 'median'}),
    'most frequent': (SimpleImputer, {'strategy': 'most_frequent'}),
    'constant': (SimpleImputer, {'strategy': 'constant'}),
    'iterative': (IterativeImputer, {'initial_strategy': 'mean'}),
}


def get_imputer(imputer_str, inds=[], encoder_inds=[], ordinal_inds=[],
                encoders=[], base_estimator=None):

    # Grab from imputer objs directly if no base estimator
    if base_estimator is None:

        try:
            base_imputer_obj, params = IMPUTERS[imputer_str]
        except KeyError:
            print('Requested:', imputer_str, 'does not exist!')
            print('If attempting to select a model str, make sure that str',
                  'exists for the right problem type, where the problem type',
                  'regression for float + custom scopes,',
                  'and binary/multiclass for binary and categorical!')

    # If base estimator, then using iterative imputer
    else:
        base_imputer_obj, params = IMPUTERS['iterative']
        params['estimator'] = base_estimator

        if len(inds) == 0:
            params['initial_strategy'] = 'median'

    base_imputer = base_imputer_obj(**params)

    # Categorical
    if len(inds) == 0:

        imputer = Categorical_Imputer(base_imputer, encoder_inds,
                                      ordinal_inds, encoders)

    # Float or Binary
    else:
        imputer = Regular_Imputer(base_imputer, inds)

    return imputer
