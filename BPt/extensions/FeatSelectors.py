
from sklearn.base import BaseEstimator
from sklearn.feature_selection._base import SelectorMixin
import numpy as np
from ..pipeline.helpers import proc_mapping


class FeatureSelector(SelectorMixin, BaseEstimator):

    _needs_mappings = True

    def __init__(self, mask='sets as random features'):
        ''' Custom BPt feature selector for
        integrating in feature selection with a hyper-parameter search.

        Parameters
        ----------
        mask : 'sets as random features' or 'sets as hyperparameters'
            - 'sets as random features': Use random features.

            - 'sets as hyperparameters': Each feature is set as a
                hyperparameter, such that the parameter search can
                tune if each feature is included or not.
        '''

        self.mask = mask
        self.name = 'selector'

    def _proc_mapping(self, mapping):

        if hasattr(self, 'mapping_'):
            return
        else:
            self.mapping_ = mapping

        if len(mapping) > 0:

            mask_0 = np.where(~self.mask)[0]
            mask_1 = np.where(self.mask)[0]

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

        if mapping is None:
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
