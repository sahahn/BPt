from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
import pandas as pd


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

    fit_intercept : bool, optional
        If True, then the intercept of the
        linear model will be fitted. It will then
        be added onto the residualized data as
        intercept + difference in real - predicted

        ::

            default = False
    '''

    _needs_fit_index = True
    _needs_transform_index = True

    def __init__(self, to_resid_df, demean=True,
                 dummy_code=True, fit_intercept=True):

        self.to_resid_df = to_resid_df
        self.demean = demean
        self.dummy_code = dummy_code
        self.fit_intercept = fit_intercept

        self._check_args()

    def _transform_covars(self, X, index=None, fit=True):

        # Grab covars as copy of indexed df
        covars = self.to_resid_df.loc[index].copy()

        if self.dummy_code:

            # Find which cols are categorical
            cat_cols = [col for col in covars
                        if covars[col].dtype.name == 'category']

            # Must be more than 0 cat columns
            if len(cat_cols) > 0:

                # Cast to np
                cat_trans = np.array(covars[cat_cols])

                # Init and fit if fit
                if fit:
                    self.encoder_ = OneHotEncoder(drop='first', sparse=False)
                    self.encoder_.fit(cat_trans)

                # Transform
                cat_trans = self.encoder_.transform(cat_trans)

                # Stack with non-cat in place
                rest_cols = [col for col in covars if col not in cat_cols]
                covars = np.hstack([np.array(covars[rest_cols]), cat_trans])

        # At this stage make sure np array if not already
        covars = np.array(covars)

        if self.demean:

            # Need to calculate means if fit
            if fit:
                self.means_ = np.mean(covars, axis=0)

            covars = covars - self.means_

        return covars

    def fit(self, X, y=None, fit_index=None):

        if isinstance(X, pd.DataFrame):
            fit_index = X.index
            X = np.array(X)

        # Get covars as proc'ed np array, fitting demean or dummy code.
        covars = self._transform_covars(X, index=fit_index, fit=True)
        return self._fit(X, covars)

    def transform(self, X, transform_index=None):

        if isinstance(X, pd.DataFrame):
            transform_index = X.index
            X = np.array(X)

        # Get covars as proc'ed np array
        covars =\
            self._transform_covars(X, index=transform_index, fit=False)

        # Returned transformed copy
        return self._transform(X, covars)

    def _check_args(self):

        if not isinstance(self.to_resid_df, pd.DataFrame):
            raise RuntimeError('to_resid_df must be a DataFrame or Dataset.')

        if not isinstance(self.demean, bool):
            raise RuntimeError('demean must be True or False.')

        if not isinstance(self.dummy_code, bool):
            raise RuntimeError('dummy code must be True or False.')

        if not isinstance(self.fit_intercept, bool):
            raise RuntimeError('fit_intercept must be True or False.')

    def _fit(self, X, covars):

        self._check_args()

        # If NaN, use NaN fit
        if np.isnan(X).any():
            return self._nan_fit(X, covars)

        # Fit linear regression
        self.estimator_ = LinearRegression(fit_intercept=self.fit_intercept)
        self.estimator_.fit(covars, X)

        # Save score
        self.score_ = self.estimator_.score(covars, X)

        return self

    def _nan_fit(self, X, covars):

        # For each feature seperately
        self.estimators_ = []
        for i in range(X.shape[1]):

            # Operate on non-nan subjects for this feature
            mask = ~np.isnan(X[:, i])

            # If not at least 2 subjects valid,
            # skip and propegate NaN.
            if np.sum(mask) > 1:

                # Fit model
                model = LinearRegression(fit_intercept=self.fit_intercept)
                model.fit(covars[mask], X[mask, i])
                self.estimators_.append(model)

            # During transform, if skipped here, then will
            # transform to all NaNs.
            else:
                self.estimators_.append(None)

        return self

    def _transform(self, X, covars):

        # if fit with NaN's
        if hasattr(self, 'estimators_'):
            return self._nan_transform(X, covars)

        # Otherwise use estimator
        # The difference is the real value
        # minus the predicted value
        dif = X - self.estimator_.predict(covars)

        # Set resid as either diff or diff + intercept
        resid = dif

        if self.fit_intercept:
            resid += self.estimator_.intercept_

        return resid

    def _nan_transform(self, X, covars):

        # Init empty resid array of NaN's
        resid = np.empty(shape=X.shape)
        resid[:] = np.nan

        # For each feature seperately
        for i in range(X.shape[1]):

            # Operate on non-nan subjects for this feature
            mask = ~np.isnan(X[:, i])

            # If not at least 2 subjects valid,
            # skip and propegate NaN.
            if len(mask) > 1:

                # Grab saved estimator
                model = self.estimators_[i]

                # Compute difference of real value - predicted
                dif_i = X[mask, i] - model.predict(covars[mask])

                # Set resid as either diff or diff + intercept
                resid_i = dif_i

                if self.fit_intercept:
                    resid_i += model.intercept_

                # Fill in NaN mask
                resid[mask, i] = resid_i

        return resid

    def fit_transform(self, X, y=None, fit_index=None):

        if isinstance(X, pd.DataFrame):
            fit_index = X.index
            X = np.array(X)

        covars = self._transform_covars(X, index=fit_index, fit=True)
        return self._fit(X, covars)._transform(X, covars)

    def __repr__(self):

        with pd.option_context('display.max_rows', 1,
                               'display.max_columns', 1,
                               'display.max_colwidth', 1):
            rep = super().__repr__()
        return rep
