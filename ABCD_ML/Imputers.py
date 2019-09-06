import numpy as np
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
        self.imputer.fit(impute_X)

    def get_impute_X(self, X):

        if self.valid_mask is None:
            self.valid_mask = ~np.isnan(X).any(axis=0)

        valid_extra_X = X[:, self.valid_mask]

        # Just in case this col didnt have NaN in train, but does in test
        valid_extra_X[np.isnan(valid_extra_X)] = 0

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
            self.valid_mask = ~np.isnan(X).any(axis=0)

        valid_extra_X = X[:, self.valid_mask]

        # Just in case this col didnt have NaN in train, but does in test
        valid_extra_X[np.isnan(valid_extra_X)] = 0

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
            encoded = encoders[i].transform(imputed_col)

            X_copy[:, self.encoder_inds[i]] = encoded

        for i in range(len(self.ordinal_inds)):

            X_copy[:, self.ordinal_inds[i]] =\
                imputed[:, ind+i+len(self.encoder_inds)]

        return X_copy

    def to_ordinal(self, X, encoder, inds):

        selection = X[:, inds]
        non_nan_mask = ~np.isnan(selection).any(axis=1)

        ordinal = encoder.inverse_transform(selection[non_nan_mask])
        to_fill = np.full((len(non_nan_mask), 1), np.nan)

        to_fill[non_nan_mask] = ordinal

        return to_fill

    def fit_transform(self, X, y=None):

        self.fit(X, y)
        return self.transform(X)


IMPUTERS = {
    'mean': (SimpleImputer, {'strategy': 'mean'}),
    'median': (SimpleImputer, {'strategy': 'median'}),
    'most frequent': (SimpleImputer, {'strategy': 'most_frequent'}),
    'constant': (SimpleImputer, {'strategy': 'constant'}),
    'iterative': (IterativeImputer, {}),
}


def get_imputer(imputer_str, inds=[], encoder_inds=[], ordinal_inds=[],
                base_estimator=None):

    base_imputer, params = IMPUTERS[imputer_str]
    base_imputer = base_imputer(**params)



