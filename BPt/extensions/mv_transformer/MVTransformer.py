from ...pipeline.helpers import proc_mapping
from ...pipeline.BPtTransformer import BPtTransformer
from sklearn.utils.validation import check_memory
import numpy as np


def _fit_estimator(estimator, Xs, y=None, **fit_params):

    estimator.fit(Xs=Xs, y=y, **fit_params)
    return estimator


class MVTransformer(BPtTransformer):

    def _proc_mapping(self, mapping):

        # Save a copy of the passed mapping
        self.mapping_ = mapping.copy()

        self.inds_ = []
        for inds in self.inds:
            self.inds_ += proc_mapping(inds, mapping)
        self.inds_ = sorted(list(set(self.inds_)))

        # Calculate view inds seperately
        self.view_inds_ = [proc_mapping(inds, mapping)
                           for inds in self.inds]

        return self

    def _fit(self, X, y=None, **fit_params):

        # Get correct fit function as either with memory
        # caching, or just as is, if no cache loc passed.
        if self.cache_loc is not None:
            memory = check_memory(self.cache_loc)
            _fit_estimator_c = memory.cache(_fit_estimator)
        else:
            _fit_estimator_c = _fit_estimator

        # Fit the estimator
        self.estimator_ =\
            _fit_estimator_c(estimator=self.estimator_,
                             Xs=[X[:, inds] for inds in self.view_inds_],
                             y=y, **fit_params)

    def transform(self, X, transform_index=None):

        # If None, pass along as is
        if self.estimator_ is None:
            return X

        X_trans = self.estimator_.transform(
            Xs=[X[:, inds] for inds in self.view_inds_])

        # Save number of output features after X_trans
        self.n_trans_feats_ = X_trans.shape[1]

        # Return stacked X_trans with rest inds
        return np.hstack([X_trans, X[:, self.rest_inds_]])