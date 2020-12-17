from sklearn.base import TransformerMixin, clone
from ..helpers.ML_Helpers import proc_mapping, update_mapping
from sklearn.utils.metaestimators import if_delegate_has_method
import numpy as np

from .base import BPtBase, _get_est_fit_params


class ScopeObj(BPtBase):

    _needs_mapping = True
    _needs_train_data_index = True

    # Override
    _required_parameters = ["estimator", "inds"]

    def __init__(self, estimator, inds):

        # Set estimator
        super().__init__(estimator=estimator)

        # These are the index to restrict scope to
        self.inds = inds

    def _proc_mapping(self, mapping):

        # If None, set to empty
        if mapping is None:
            mapping = {}

        # Save a copy of the passed mapping
        self.mapping_ = mapping.copy()

        # If any mapping passed, update inds in inds_
        if len(mapping) > 0:
            self.inds_ = proc_mapping(self.inds, mapping)

        return self

    def fit(self, X, y=None, mapping=None,
            train_data_index=None, **fit_params):

        # Save base dtype of input, and n_features_in
        self.base_dtype_ = X.dtype
        self.n_features_in_ = X.shape[1]

        # Process the passed mapping, sets self.inds_
        self._proc_mapping(mapping)

        # If no inds to map, skip by setting estimator to None
        if len(self.inds_) == 0:
            self.estimator_ = None
            return self

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

        # Use the out mapping to create a mapping to pass along.
        # Make copy so that the original mapping doesn't change.
        pass_on_mapping = mapping.copy()
        update_mapping(pass_on_mapping, self.out_mapping_)

        # Clone estimator, clears previous fits
        self.estimator_ = clone(self.estimator)

        # Get the correct fit params to pass along
        fit_params = _get_est_fit_params(estimator=self.estimator_,
                                         mapping=pass_on_mapping,
                                         train_data_index=train_data_index,
                                         other_params=fit_params)

        # Fit the base estimator
        self.estimator_.fit(X=X[:, self.inds_], y=y, **fit_params)

        # Set as fitted
        self.is_fitted_ = True

        return self


class ScopeTransformer(ScopeObj, TransformerMixin):

    def fit(self, X, y=None, mapping=None,
            train_data_index=None, **fit_params):

        # Call parent fit - base shared fit with ScopeModel
        super().fit(X, y=y, mapping=mapping,
                    train_data_index=train_data_index,
                    **fit_params)

        # If skip
        if self.estimator_ is None:
            return self

        # Now need to make changes to the original mapping
        # to reflect that the new order is self.inds_ + self.rest_inds_
        # or will be after a transform.

        # Use same out mapping from before, but override the None's
        # with the real new spot
        for c in range(len(self.rest_inds_)):
            ind = self.rest_inds_[c]
            self.out_mapping_[ind] = len(self.inds_) + c

        # Update the original mapping, this is the mapping which
        # will be passed to the next piece of the pipeline
        update_mapping(mapping, self.out_mapping_)

        return self

    def transform(self, X):

        # If None, pass along as is
        if self.estimator_ is None:
            return X

        # Get X_trans
        X_trans = self.estimator_.transform(X=X[:, self.inds_])

        # Return stacked X_trans with rest inds
        return np.hstack([X_trans, X[:, self.rest_inds_]])

    def transform_df(self, df):

        # If None, pass along as is
        if self.estimator_ is None:
            return df

        # Prepare as numpy array - make sure same as original passed dtype
        X = np.array(df).astype(self.base_dtype_)

        # Transform data
        X_trans = self.transform(X)

        # Compute new feature names
        new_names = [list(df)[i] for i in self.inds_] +\
                    [list(df)[i] for i in self.rest_inds_]

        # Fill in the new values directly to the passed df
        for i, feat_name in enumerate(new_names):
            df[feat_name] = X_trans[:, i]

        # Return by re-ordering the df so that it matches
        # the order of new_names
        return df[new_names]


class ScopeModel(ScopeObj):

    @property
    def _estimator_type(self):
        return self.estimator._estimator_type

    @property
    def coef_(self):
        return self.estimator_.coef_

    @property
    def feature_importances_(self):
        return self.estimator_.feature_importances_

    def predict(self, X, *args, **kwargs):
        return self.estimator_.predict(X[:, self.inds_], *args, **kwargs)

    @if_delegate_has_method(delegate='estimator_')
    def predict_proba(self, X, *args, **kwargs):
        return self.estimator_.predict_proba(X[:, self.inds_], *args, **kwargs)

    @if_delegate_has_method(delegate='estimator_')
    def decision_function(self, X, *args, **kwargs):
        return self.estimator_.decision_function(X[:, self.inds_],
                                                 *args, **kwargs)

    @if_delegate_has_method(delegate='estimator_')
    def predict_log_proba(self, X, *args, **kwargs):
        return self.estimator_.predict_log_proba(X[:, self.inds_],
                                                 *args, **kwargs)

    @if_delegate_has_method(delegate='estimator_')
    def score(self, X, *args, **kwargs):
        return self.estimator_.score(X[:, self.inds_], *args, **kwargs)
