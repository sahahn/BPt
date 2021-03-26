from sklearn.base import TransformerMixin, clone
from .helpers import proc_mapping, update_mapping
from sklearn.utils.validation import check_memory
import numpy as np
import warnings

from .base import BPtBase, _get_est_fit_params, _needs, _get_est_trans_params


def _fit_estimator(estimator, X, y=None, **fit_params):

    estimator.fit(X=X, y=y, **fit_params)
    return estimator


class ScopeObj(BPtBase):

    _needs_mapping = True
    _needs_fit_index = True

    # Override
    _required_parameters = ["estimator", "inds"]

    def __init__(self, estimator, inds, passthrough=False, cache_loc=None):

        # Set estimator
        super().__init__(estimator=estimator)

        # These are the index to restrict scope to
        self.inds = inds

        self.passthrough = passthrough

        # This is the optional cache_loc for memory
        self.cache_loc = cache_loc

    def __repr__(self):

        temp_inds = self.inds

        if self.inds is Ellipsis:
            self.inds = 'all'
        else:
            self.inds = 'len(' + str(len(self.inds)) + ')'

        rep = super().__repr__()
        self.inds = temp_inds

        return rep

    def _proc_mapping(self, mapping):

        # Save a copy of the passed mapping
        self.mapping_ = mapping.copy()

        # If any mapping passed, update inds in inds_
        if len(mapping) > 0:
            self.inds_ = proc_mapping(self.inds, mapping)

        # Otherwise, inds are as is
        else:
            self.inds_ = self.inds

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
        self.estimator_ = _fit_estimator_c(estimator=self.estimator_,
                                           X=X[:, self.inds_],
                                           y=y, **fit_params)

    def _all_case_update_mappings(self, mapping):

        # In case where the scope is everything

        # Rest inds is empty
        self.rest_inds_ = []

        # So is out_mapping
        self.out_mapping_ = {}

        # The mapping to pass on is just
        # a copy of the original
        return mapping.copy()

    def _update_mappings(self, X, mapping):

        # Check in case of Ellipsis / all case
        if self.inds_ is Ellipsis:
            return self._all_case_update_mappings(mapping)

        # Create a mapping that maps old to new value
        self.out_mapping_ = {}

        # For each ind, set sequential
        for i in range(len(self.inds_)):
            self.out_mapping_[self.inds_[i]] = i

        # Compute rest of inds
        self.rest_inds_ =\
            list(np.setdiff1d(list(range(X.shape[1])),
                 self.inds_, assume_unique=True))

        # Set every i in rest_inds_ to None, as out of scope
        for i in self.rest_inds_:
            self.out_mapping_[i] = None

        # Use the out mapping to create a mapping to pass along
        # to any nested objects.
        # Make copy so that the original mapping doesn't change.
        pass_on_mapping = mapping.copy()
        update_mapping(pass_on_mapping, self.out_mapping_)

        return pass_on_mapping

    def fit(self, X, y=None, mapping=None,
            fit_index=None, **fit_params):

        # Save base dtype of input, and n_features_in
        self.base_dtype_ = X.dtype
        self.n_features_in_ = X.shape[1]

        # Process the passed mapping, sets self.inds_
        if mapping is None:
            mapping = {}

        self._proc_mapping(mapping)

        # If no inds to map, skip by setting estimator to None
        if self.inds_ is not Ellipsis and len(self.inds_) == 0:
            self.estimator_ = None
            return self

        # Get pass on mapping
        # All saves class attributes:
        # out_mapping_ and rest_inds_,
        pass_on_mapping = self._update_mappings(X, mapping)

        # Clone estimator, clears previous fits
        self.estimator_ = clone(self.estimator)

        # Get the correct fit params to pass along
        pass_fit_params =\
            _get_est_fit_params(estimator=self.estimator_,
                                mapping=pass_on_mapping,
                                fit_index=fit_index,
                                other_params=fit_params)

        # Fit actual estimator with or without caching
        self._fit(X=X, y=y, **pass_fit_params)

        # Set as fitted
        self.is_fitted_ = True

        return self


class ScopeTransformer(ScopeObj, TransformerMixin):

    @property
    def _needs_transform_index(self):
        return _needs(self.estimator, '_needs_transform_index',
                      'transform_index', 'transform')

    def fit(self, X, y=None, mapping=None,
            fit_index=None, **fit_params):

        if mapping is None:
            mapping = {}

        # Call parent fit - base shared fit with BPtModel
        super().fit(X, y=y, mapping=mapping,
                    fit_index=fit_index,
                    **fit_params)

        # If skip
        if self.estimator_ is None:
            return self

        # Also skip if original scope was all
        # which means the out_mapping shouldn't change
        # and also no updates to the mapping should be made.
        if self.inds_ is Ellipsis:
            return self

        # Now need to make changes to the original mapping
        # to reflect that the new order is self.inds_ + self.rest_inds_
        # or will be after a transform.

        # Use same out mapping from before, but override the None's
        # with the real new spot
        for c in range(len(self.rest_inds_)):
            ind = self.rest_inds_[c]
            self.out_mapping_[ind] = len(self.inds_) + c

        # Passthrough case
        if self.passthrough:

            # Need to update out_mapping_ to reflect that
            # each ind in self.inds_, now maps also to
            # len(self.inds_) + len(self.rest_inds_)
            t = len(self.inds_) + len(self.rest_inds_)
            for i in range(len(self.inds_)):

                extra = i + t
                loc = self.inds_[i]
                current = self.out_mapping_[loc]

                if not isinstance(current, list):
                    current = [current]

                self.out_mapping_[loc] = current + [extra]

        # Update the original mapping, this is the mapping which
        # will be passed to the next piece of the pipeline
        update_mapping(mapping, self.out_mapping_)

        return self

    def transform(self, X, transform_index=None):

        # If None, pass along as is
        if self.estimator_ is None:
            return X

        # Get transform params
        trans_params = _get_est_trans_params(self.estimator_,
                                             transform_index=transform_index)

        # Get X_trans
        X_trans = self._est_transform(X, **trans_params)

        # Save number of output features after X_trans
        self.n_trans_feats_ = X_trans.shape[1]

        # Passthrough case
        if self.passthrough:

            # If no rest inds, skip
            if len(self.rest_inds_) == 0:
                return np.hstack([X_trans, X[:, self.inds_]])

            return np.hstack([X_trans, X[:, self.rest_inds_],
                              X[:, self.inds_]])

        # Return stacked X_trans with rest inds
        if len(self.rest_inds_) > 0:
            return np.hstack([X_trans, X[:, self.rest_inds_]])

        # No rest inds, return directly
        return X_trans

    def _est_transform(self, X, **trans_params):

        # if self.inds_ is Ellipsis, just selects all
        return self.estimator_.transform(X=X[:, self.inds_], **trans_params)

    def fit_transform(self, X, y=None, mapping=None,
                      fit_index=None, transform_index=None,
                      **fit_params):
        '''Override fit transfrom to pass fit_index to transform_index'''

        return self.fit(
            X=X, y=y, mapping=mapping, fit_index=fit_index,
            **fit_params).transform(X=X, transform_index=fit_index)

    def transform_df(self, df, base_name=None, encoders=None):

        # Important: Operate on a copy of the data frame
        df_copy = df.copy()

        # If None, pass along as is
        if self.estimator_ is None:
            return df

        # Get transfrom index from df
        transform_index = df_copy.index

        # Prepare as numpy array - make sure same as original passed dtype
        X = np.array(df_copy).astype(self.base_dtype_)

        # Transform data
        X_trans = self.transform(X, transform_index=transform_index)

        # Feat names are as is
        feat_names = list(df_copy)

        # Process new names
        new_names = self._proc_new_names(feat_names, base_name,
                                         encoders=encoders)

        # Fill in the new values directly to the passed df_copy
        for i, feat_name in enumerate(new_names):
            df_copy.loc[:, feat_name] = X_trans[:, i]

        # Return by re-ordering the df so that it matches
        # the order of new_names, and only with those included in new_names
        return df_copy.loc[:, new_names]

    def _proc_new_names(self, feat_names, base_name=None, encoders=None):

        # If skip, return passed names as is
        if self.estimator_ is None:
            return feat_names

        # If all, return as is
        if self.inds_ is Ellipsis:
            return feat_names

        # Compute new feature names
        new_names = [feat_names[i] for i in self.inds_] +\
                    [feat_names[i] for i in self.rest_inds_]

        return new_names

    def _remove_old_names(self, feat_names):
        '''Create new feature names for the transformed features.
        This class is used in child classes'''

        # If all, all original feat_names get removed
        # return empty list
        if self.inds_ is Ellipsis:
            return []

        to_remove = set([feat_names[i] for i in self.inds_])
        feat_names = [name for name in feat_names if name not in to_remove]
        return feat_names

    def inverse_transform(self, X, name='transformer'):

        # If None, pass along as is
        if self.estimator_ is None:
            return X

        # Compute reverse inds - if Ellipsis, returns Ellipsis
        reverse_inds = proc_mapping(self.inds_, self.out_mapping_)

        # If no inverse_transformer in base transformer, set to 0
        try:
            X_trans =\
                self.estimator_.inverse_transform(X[:, reverse_inds])
        except AttributeError:
            X_trans = np.zeros((X.shape[0], len(self.inds_)))
            warnings.warn('Passed obj: "' + name + '" has no '
                          'inverse_transform! '
                          'Setting relevant inverse '
                          'feat importances to 0.')

        # Compute reverse of rest inds
        reverse_rest_inds = proc_mapping(self.rest_inds_, self.out_mapping_)

        # Create empty Xt to fill
        all_inds_len = len(self.inds_) + len(self.rest_inds_)
        Xt = np.zeros((X.shape[0], all_inds_len), dtype=X.dtype)

        # Fill in Xt
        Xt[:, self.inds_] = X_trans
        Xt[:, self.rest_inds_] = X[:, reverse_rest_inds]

        return Xt
