from sklearn.feature_selection import RFE
from sklearn.base import BaseEstimator
from sklearn.feature_selection._base import SelectorMixin
import numpy as np


class RFE(RFE):
    def fit(self, X, y):
        '''Override the fit function from base
           specifically allow passing in float % to keep.
        '''

        if isinstance(self.n_features_to_select, float):

            if self.n_features_to_select <= 0:
                self.n_features_to_select = 1

            if self.n_features_to_select < 1:
                divide_by = self.n_features_to_select ** -1
                self.n_features_to_select = X.shape[1] // divide_by

        return self._fit(X, y)


class FeatureSelector(SelectorMixin, BaseEstimator):

    def __init__(self, mask):
        self.mask=mask

    def fit(self, X, y=None):
        
        self.mask = np.array(self.mask) > .5
        return self
    
    def _get_support_mask(self):
        return self.mask
