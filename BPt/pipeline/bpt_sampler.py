from .ScopeObjs import ScopeTransformer
from .base import _get_est_trans_params
import numpy as np

class BPtSampler(ScopeTransformer):

    def transform(self, X, transform_index=None):

        # If None, pass along as is
        if self.estimator_ is None:
            return X

        # Get transform params
        trans_params = _get_est_trans_params(self.estimator_,
                                             transform_index=transform_index)

        # Transform - with two cases here
        # either will return just X as is, or
        # a three tuple in the fit_transform case
        transformed =\
            self.estimator_.transform(X=X, **trans_params)

        # Check for resampled case
        if isinstance(transformed, tuple) and len(transformed) == 3:
        
            # Get X_trans, the resamp inds and the new fit index
            X_trans, resamp_inds, new_fit_index = transformed

            # Save number of output features after X_trans
            self.n_trans_feats_ = X_trans.shape[1]

            # Note, no passthrough case possible
            # likewise, no rest inds case possible

            # Expected alternative transform output is
            # the resampled X_trans, the associated resample
            # integer index, and the new corresponding
            # pandas Index new_fit_index
            return X_trans, resamp_inds, new_fit_index

        # Otherwise, return as is
        return transformed
        
        

