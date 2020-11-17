from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import FLOAT_DTYPES
import numpy as np


class Winsorizer(TransformerMixin, BaseEstimator):
    '''This Scaler performs winzorization, or clipping by feature.

    Parameters
    ----------
    quantile_range : tuple (q_min, q_max), 0.0 < q_min < q_max < 100.0
        Default: (5.0, 95.0), the lower and upper range in which to clip
        values to.

    copy : boolean, optional, default is True
        Make a copy of the data.
    '''

    def __init__(self, quantile_range=(5, 95), copy=True):
        self.quantile_range = quantile_range
        self.copy = copy

    def fit(self, X, y=None):
        X = check_array(X, accept_sparse='csc', estimator=self,
                        dtype=FLOAT_DTYPES, force_all_finite='allow-nan')

        q_min, q_max = self.quantile_range
        if not 0 <= q_min <= q_max <= 100:
            raise ValueError("Invalid quantile range: %s" %
                             str(self.quantile_range))

        self.data_lb_ = np.nanpercentile(X, q_min, axis=0)
        self.data_ub_ = np.nanpercentile(X, q_max, axis=0)

        return self

    def transform(self, X):
        X = check_array(X, accept_sparse=('csr', 'csc'), copy=self.copy,
                        estimator=self, dtype=FLOAT_DTYPES,
                        force_all_finite='allow-nan')

        X = np.clip(X, self.data_lb_, self.data_ub_)
        return X

    def _more_tags(self):
        return {'allow_nan': True}
