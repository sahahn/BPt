from sklearn.feature_selection import RFE
from sklearn.base import BaseEstimator
from sklearn.feature_selection._base import SelectorMixin
import numpy as np
from ...helpers.ML_Helpers import proc_mapping


class RFE_Wrapper(RFE):
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
        self.mask = mask
        self.name = 'selector'

    def _proc_mapping(self, mapping):

        if hasattr(self, '_mapping'):
            return

        if len(mapping) > 0:

            mask_0 = np.where(self.mask == False)[0]
            mask_1 = np.where(self.mask == True)[0]

            mask_0 = proc_mapping(mask_0, mapping)
            mask_1 = proc_mapping(mask_1, mapping)

            if len(mask_0) > 0:
                mx_0 = np.max(mask_0)
            else:
                mx_0 = 0

            if len(mask_1) > 0:
                mx_1 = np.max(mask_1)
            else:
                mx_1 = 0

            max_ind = np.max([mx_0, mx_1])
            self.mask = np.zeros(max_ind+1, dtype='bool')
            self.mask[mask_1] = True
        return

    def fit(self, X, y=None, mapping=None):

        if mapping == None:
            mapping = {}

        self.mask = np.array(self.mask)

        threshold = .5
        while np.sum(self.mask >= threshold) == 0:
            threshold -= .001

        self.mask = self.mask >= threshold
        self._proc_mapping(mapping)

        return self
    
    def _get_support_mask(self):
        return self.mask
