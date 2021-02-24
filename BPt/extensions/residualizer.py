from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.linear_model import LinearRegression


def get_resid_with_nans(covars, data):

    # Go from pandas df to numpy array
    covars = np.array(covars)

    # Make sure data is numpy array
    data = np.array(data)

    # Init empty resid array of NaN's
    resid = np.empty(shape=data.shape)
    resid[:] = np.nan

    # For each feature seperately
    for i in range(data.shape[1]):

        # Operate on non-nan subjects for this feature
        mask = ~np.isnan(data[:, i])

        # If not at least 2 subjects valid,
        # skip and propegate NaN's for this
        # voxel.
        if len(mask) > 1:

            # Fit model
            model = LinearRegression().fit(covars[mask], data[mask, i])

            # Compute difference of real value - predicted
            dif_i = data[mask, i] - model.predict(covars[mask])

            # Set resid as diff + intercept
            resid_i = model.intercept_ + dif_i

            # Fill in NaN mask
            resid[mask, i] = resid_i

    return resid


def get_resid_base(covars, data):

    # Go from pandas df to numpy array
    covars = np.array(covars)

    # Make sure data is numpy array
    data = np.array(data)

    model = LinearRegression().fit(covars, data)

    # The difference is the real value of the voxel, minus the predicted value
    dif = data - model.predict(covars)

    resid = model.intercept_ + dif

    return resid


class LinearResidualizer(BaseEstimator, TransformerMixin):
    '''This class is used for residualizing data with the LinearRegression
    model from scikit-learn according to a DataFrame with one or
    more columns.

    In the case of feature columns with any NaN's, the linear model
    will be fit just on all subject's with non NaN data, and NaN's
    propagated.

    Warning: you like should not use this transformer to
    transform categorical data.

    Parameters
    ------------
    to_resid_df : pandas DataFrame or BPt Dataset
        Should contain one or more columns as indexed
        by the same index as provided input features, this
        df will be used to residualize the input data.

        There may not be any NaN data here, and there must
        be data for each subject (i.e., any subject
        potentially present in X, must have a value here.)

    demean : bool, optional
        If True, then the variables passed in to_resid_df
        will be de-meaned according to the actual group
        of subjects passed during fit. For transform,
        values will be de-meaned according to the mean
        for each column as calculated during fit.

        If dummy_code is True, then dummy coded variables
        will be de-meaned after dummy coding.

        ::

            default = True

    dummy_code : bool, optional
        If True, then any categorical variables,
        as determined by checking to see if the underlying
        pandas dtype is 'category', will be dummy coded
        with scikit-learn's :class:`OneHotEncoder`.

        If no categorical variables, this option will be skipped.

        If demean is True, then categorical variables will
        first be dummy coded and then de-meaned.

        ::

            default = True
    '''

    _needs_fit_index = True
    _needs_transform_index = True

    def __init__(self, to_resid_df, demean=True, dummy_code=True):

        self.to_resid_df = to_resid_df
        self.demean = demean
        self.dummy_code = dummy_code

    def fit(self, X, y=None):
        pass

    def fit_transform(self, X, y=None, fit_index=None):
        return self.fit(X=X, y=y,
                        fit_index=fit_index).transform(X)

    def transform(self,)