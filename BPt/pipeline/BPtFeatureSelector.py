from sklearn.feature_selection._base import SelectorMixin
from .helpers import update_mapping
from .ScopeObjs import ScopeTransformer
import numpy as np


class BPtFeatureSelector(ScopeTransformer, SelectorMixin):

    def _update_feat_mapping(self, X, mapping):

        # Need to pass along the correct mapping
        # overwrite existing out mapping
        self.out_mapping_ = {}

        # This is the calculated support from the base estimator
        support = self.estimator_.get_support()

        # Set in scope inds by if all case
        if self.inds_ is Ellipsis:
            in_scope_inds = list(range(X.shape[1]))
        else:
            in_scope_inds = self.inds_

        # First half is for updating the index within scope
        cnt = 0
        for i, ind in enumerate(in_scope_inds):

            # If kept by feat selection, add, otherwise set to None
            if support[i]:
                self.out_mapping_[ind] = cnt
                cnt += 1
            else:
                self.out_mapping_[ind] = None

        # Next, need to update the mapping for the remaining wrapper inds
        # essentially setting them where the cnt left off, then sequentially
        # If None, will just  skip
        for rem_ind in range(len(self.rest_inds_)):
            self.out_mapping_[self.rest_inds_[rem_ind]] = cnt
            cnt += 1

        # Update the original mapping, this is the mapping which
        # will be passed to the next piece of the pipeline
        update_mapping(mapping, self.out_mapping_)

        return self

    def fit(self, X, y=None, mapping=None,
            train_data_index=None, **fit_params):

        if mapping is None:
            mapping = {}

        # Call parent fit
        super().fit(X, y=y, mapping=mapping,
                    train_data_index=train_data_index,
                    **fit_params)

        # Need to update mapping
        return self._update_feat_mapping(X, mapping)

    def _proc_new_names(self, feat_names, base_name=None, encoders=None):

        # Get base new names from parent class
        new_names = super()._proc_new_names(feat_names)

        # This feat mask corresponds to the already transformed feats
        feat_mask = self._get_support_mask()

        # Apply the computed mask to get the actually selected features
        return_names = np.array(new_names)[feat_mask]

        return list(return_names)

    def _get_support_mask(self):

        # Create full support as base support + True's for all rest inds
        # i.e., those features originally out of scope
        base_support = self.estimator_.get_support()
        rest_support = np.ones(len(self.rest_inds_), dtype='bool')
        support = np.concatenate([base_support, rest_support])

        return support