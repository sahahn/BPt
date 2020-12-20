from ..helpers.ML_Helpers import (get_obj_and_params, proc_mapping,
                                  update_mapping)
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.decomposition import (PCA, FactorAnalysis,
                                   MiniBatchDictionaryLearning,
                                   DictionaryLearning, FastICA,
                                   IncrementalPCA, KernelPCA,
                                   MiniBatchSparsePCA, NMF, SparsePCA,
                                   TruncatedSVD)
import warnings
from sklearn.utils.validation import check_memory
from sklearn.base import clone
from sklearn.preprocessing import OneHotEncoder
from .ScopeObjs import ScopeTransformer


class BPtTransformer(ScopeTransformer):

    def fit(self, X, y=None, mapping=None,
            train_data_index=None, **fit_params):

        # Need the output from a transform to full fit,
        # so when fit is called, call fit_transform instead
        self.fit_transform(X=X, y=y, mapping=mapping,
                           train_data_index=train_data_index,
                           **fit_params)

        return self

    def fit_transform(self, X, y=None, mapping=None,
                      train_data_index=None, **fit_params):

        # Call parent fit
        super().fit(X, y=y, mapping=mapping,
                    train_data_index=train_data_index,
                    **fit_params)

        # If skip
        if self.estimator_ is None:
            return X

        # Transform X
        X_trans = self.transform(X)

        # Need to update the mapping before returning

        # Many to many case for transformer,
        # override existing out_mapping_
        self.out_mapping_ = {}
        X_trans_inds = list(range(self.n_trans_feats_))

        # Many to many case, each ind is mapped
        # to all output'ed X_trans_inds
        for i in self.inds_:
            self.out_mapping_[i] = X_trans_inds

        # Fill the remaining spots sequentially,
        # for each of the rest inds.
        for c in range(len(self.rest_inds_)):
            ind = self.rest_inds_[c]
            self.out_mapping_[ind] = self.n_trans_feats_ + c

        # Update the original mapping, this is the mapping which
        # will be passed to the next piece of the pipeline
        update_mapping(mapping, self.out_mapping_)

        # Now return X_trans
        return X_trans

    def transform_df(self, df, base_name='transformer'):

        # If None, pass along as is
        if self.estimator_ is None:
            return df

        # Prepare as numpy array - make sure same as original passed dtype
        X = np.array(df).astype(self._base_dtype)

        # Transform data
        X_trans = self.transform(X)

        # Feat names are as is
        feat_names = list(df)

        # Get new names
        new_names = self._get_new_df_names(base_name=base_name,
                                           feat_names=feat_names)

        # Remove old names
        df, feat_names = self._remove_old_df_names(df, feat_names)

        # New names come first, then rest of names
        feat_names = new_names + feat_names

        # Replace vals in df with transformed vals and new names
        for i in range(len(feat_names)):
            df[feat_names[i]] = X_trans[:, i]

        # Return by re-ordering the df so that it matches
        # the order of new_names
        return df[feat_names]

    def _get_new_df_names(self, base_name=None, feat_names=None):
        '''Create new feature names for the transformed features'''

        if len(self.inds_) == 1:
            alt_name = feat_names[self.inds_[0]]
        else:
            alt_name = base_name

        try:
            new_names = [alt_name + '_' + str(i)
                         for i in range(self.n_trans_feats_)]
        except IndexError:
            new_names = [base_name + '_' + str(i)
                         for i in range(self.n_trans_feats_)]

        return new_names

    def _remove_old_df_names(self, df, feat_names):
        '''Create new feature names for the transformed features'''

        to_remove = set([feat_names[i] for i in self.inds_])
        feat_names = [name for name in feat_names if name not in to_remove]
        df = df.drop(list(to_remove), axis=1)

        return df, feat_names

    def inverse_transform(self, X, name='base transformer'):

        # If None, pass along as is
        if self.estimator_ is None:
            return X

        # Compute reverse inds
        reverse_inds = proc_mapping(self.inds_, self.out_mapping_)

        # If no inverse_transformer in base transformer, set to 0
        try:
            X_trans =\
                self.estimator_.inverse_transform(X[:, reverse_inds])
        except AttributeError:
            X_trans = np.zeros((X.shape[0], len(self.inds_)))
            warnings.warn('Passed transformer: "' + name + '" has no '
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


def _fit_transform_single_transformer(transformer, X, y):

    transformer.fit(X=X, y=y)
    X_trans = transformer.transform(X=X)
    return transformer, X_trans


class Transformer_Wrapper(BaseEstimator, TransformerMixin):

    _needs_mapping = True

    def __init__(self, wrapper_transformer, wrapper_inds,
                 cache_loc=None, fix_n_wrapper_jobs='default', **params):

        self.wrapper_transformer = wrapper_transformer
        self.wrapper_inds = wrapper_inds
        self.cache_loc = cache_loc

        # For compat. right now unused.
        self.fix_n_wrapper_jobs = fix_n_wrapper_jobs

        # Set any remaining params to wrapper transformer
        self.wrapper_transformer.set_params(**params)

    def _proc_mapping(self, mapping):

        try:
            self.mapping_
            return

        except AttributeError:
            self.mapping_ = mapping.copy()

        if len(mapping) > 0:
            self.wrapper_inds_ = proc_mapping(self.wrapper_inds, mapping)

        return

    def fit(self, X, y=None, mapping=None, **fit_params):

        if mapping is None:
            mapping = {}

        # Need to call fit_transform to figure out change to mapping
        self.fit_transform(X, y, mapping=mapping, **fit_params)
        return self

    def fit_transform(self, X, y=None, mapping=None, **fit_params):

        # Save base dtype of input
        self._base_dtype = X.dtype

        if mapping is None:
            mapping = {}

        self._proc_mapping(mapping)

        inds = self.wrapper_inds_
        self.rest_inds_ = list(np.setdiff1d(list(range(X.shape[1])), inds,
                                            assume_unique=True))

        if len(inds) > 0:

            # Before fit, need to handle annoying categorical encoders case
            # where there is no default setting to set to all cols
            # It shouldn't hurt to set these for other transformers
            # (hopefully...)
            self.wrapper_transformer_ = clone(self.wrapper_transformer)
            self.wrapper_transformer_.cols = list(range(len(inds)))
            self.wrapper_transformer_.return_df = False

            if self.cache_loc is not None:
                memory = check_memory(self.cache_loc)
                _fit_transform_single_transformer_c =\
                    memory.cache(_fit_transform_single_transformer)
            else:
                _fit_transform_single_transformer_c =\
                    _fit_transform_single_transformer

            self.wrapper_transformer_, X_trans =\
                _fit_transform_single_transformer_c(
                    transformer=self.wrapper_transformer_,
                    X=X[:, inds],
                    y=y)

        # If out of scope...
        else:
            self.wrapper_transformer_ = None
            return X

        self._X_trans_inds = list(range(X_trans.shape[1]))

        new_mapping = {}

        # Many to Many case
        for i in inds:
            new_mapping[i] = self._X_trans_inds

        for cnt in range(len(self.rest_inds_)):
            new_mapping[self.rest_inds_[cnt]] = len(self._X_trans_inds) + cnt

        self._out_mapping = new_mapping.copy()

        # Update mapping
        update_mapping(mapping, new_mapping)

        # Return stacked
        return np.hstack([X_trans, X[:, self.rest_inds_]])

    def transform(self, X):

        # If None, pass along as is
        if self.wrapper_transformer_ is None:
            return X

        # Transform just wrapper inds
        X_trans = self.wrapper_transformer_.transform(X[:, self.wrapper_inds_])
        return np.hstack([X_trans, X[:, self.rest_inds_]])

    def transform_df(self, df, base_name='transformer'):

        # If None, pass along as is
        if self.wrapper_transformer_ is None:
            return df

        feat_names = list(df)

        # Prepare as numpy array - make sure same as original passed dtype
        X = np.array(df).astype(self._base_dtype)

        # Transform data
        X_trans = self.transform(X)

        # Get new names
        new_names = self._get_new_df_names(base_name=base_name,
                                           feat_names=feat_names)

        # Remove old names
        df, feat_names = self._remove_old_df_names(df, feat_names)

        # New names come first, then rest of names
        feat_names = new_names + feat_names

        # Replace vals in df with transformed vals and new names
        for i in range(len(feat_names)):
            df[feat_names[i]] = X_trans[:, i]

        return df[feat_names]

    def inverse_transform(self, X, name='base transformer'):

        # If None, pass along as is
        if self.wrapper_transformer_ is None:
            return X

        reverse_inds = proc_mapping(self.wrapper_inds_, self._out_mapping)

        # If no inverse_transformer in base transformer, set to 0
        try:
            X_trans =\
                self.wrapper_transformer_.inverse_transform(X[:, reverse_inds])
        except AttributeError:
            X_trans = np.zeros((X.shape[0], len(self.wrapper_inds_)))
            warnings.warn('Passed transformer: "' + name + '" has no '
                          'inverse_transform! '
                          'Setting relevant inverse '
                          'feat importances to 0.')

        reverse_rest_inds = proc_mapping(self.rest_inds_, self._out_mapping)

        all_inds_len = len(self.wrapper_inds_) + len(self.rest_inds_)
        Xt = np.zeros((X.shape[0], all_inds_len), dtype=X.dtype)

        # Fill in Xt
        Xt[:, self.wrapper_inds_] = X_trans
        Xt[:, self.rest_inds_] = X[:, reverse_rest_inds]

        return Xt

    def _get_new_df_names(self, base_name=None, feat_names=None):
        '''Create new feature names for the transformed features'''

        if len(self.wrapper_inds_) == 1:
            alt_name = feat_names[self.wrapper_inds_[0]]
        else:
            alt_name = base_name

        n_trans = len(self._X_trans_inds)

        try:
            new_names = [alt_name + '_' + str(i) for i in range(n_trans)]
        except IndexError:
            new_names = [base_name + '_' + str(i) for i in range(n_trans)]

        return new_names

    def _remove_old_df_names(self, df, feat_names):
        '''Create new feature names for the transformed features'''

        to_remove = [feat_names[i] for i in self.wrapper_inds_]
        feat_names = [name for name in feat_names if name not in to_remove]
        df = df.drop(to_remove, axis=1)

        return df, feat_names

    def set_params(self, **params):

        if 'wrapper_transformer' in params:
            self.wrapper_transformer = params.pop('wrapper_transformer')
        if 'wrapper_inds' in params:
            self.wrapper_inds = params.pop('wrapper_inds')
        if 'cache_loc' in params:
            self.cache_loc = params.pop('cache_loc')
        if 'fix_n_wrapper_jobs' in params:
            self.fix_n_wrapper_jobs = params.pop('fix_n_wrapper_jobs')

        self.wrapper_transformer.set_params(**params)

    def get_params(self, deep=False):

        params = {'wrapper_transformer': self.wrapper_transformer,
                  'wrapper_inds': self.wrapper_inds,
                  'cache_loc': self.cache_loc,
                  'fix_n_wrapper_jobs': self.fix_n_wrapper_jobs}

        params.update(self.wrapper_transformer.get_params(deep=deep))

        return params


TRANSFORMERS = {
    'pca': (PCA, ['default', 'pca var search']),
    'sparse pca': (SparsePCA, ['default']),
    'mini batch sparse pca': (MiniBatchSparsePCA, ['default']),
    'factor analysis': (FactorAnalysis, ['default']),
    'dictionary learning': (DictionaryLearning, ['default']),
    'mini batch dictionary learning': (MiniBatchDictionaryLearning,
                                       ['default']),
    'fast ica': (FastICA, ['default']),
    'incremental pca': (IncrementalPCA, ['default']),
    'kernel pca': (KernelPCA, ['default']),
    'nmf': (NMF, ['default']),
    'truncated svd': (TruncatedSVD, ['default']),
    'one hot encoder': (OneHotEncoder, ['ohe'])}

# category_encoders below, disable for now, as taking out compatibility
#     self.wrapper_transformer_.cols = list(range(len(inds)))
#     self.wrapper_transformer_.return_df = False
# and because base library isn't up to data - could think about new way
# to re-introduce in the future if needed.
'''
try:
    from category_encoders import (BackwardDifferenceEncoder,
                                   BinaryEncoder, CatBoostEncoder,
                                   HelmertEncoder,
                                   JamesSteinEncoder, LeaveOneOutEncoder,
                                   MEstimateEncoder,
                                   PolynomialEncoder, SumEncoder,
                                   TargetEncoder, WOEEncoder)

    class OneHotEncoderWrapper(OneHotEncoder):

        def fit(self, X, y=None, **kwargs):

            self.return_df = False
            self.cols = [i for i in range(X.shape[1])]

            super().fit(X, y, **kwargs)
            return self

    extra = {
     'backward difference encoder': (BackwardDifferenceEncoder,
                                     ['default']),
     'binary encoder': (BinaryEncoder, ['default']),
     'cat boost encoder': (CatBoostEncoder, ['default']),
     'helmert encoder': (HelmertEncoder, ['default']),
     'james stein encoder': (JamesSteinEncoder, ['default']),
     'leave one out encoder': (LeaveOneOutEncoder, ['default']),
     'm estimate encoder': (MEstimateEncoder, ['default']),
     'polynomial encoder': (PolynomialEncoder, ['default']),
     'sum encoder': (SumEncoder, ['default']),
     'target encoder': (TargetEncoder, ['default']),
     'woe encoder': (WOEEncoder, ['default'])}

    TRANSFORMERS.update(extra)

except ImportError:
    pass
'''


def get_transformer_and_params(transformer_str, extra_params, params,
                               search_type, random_state=None,
                               num_feat_keys=None):

    transformer, extra_transformer_params, transformer_params =\
        get_obj_and_params(transformer_str, TRANSFORMERS, extra_params, params)

    return transformer(**extra_transformer_params), transformer_params
