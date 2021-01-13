from ..helpers.ML_Helpers import (get_obj_and_params,
                                  update_mapping)

from sklearn.decomposition import (PCA, FactorAnalysis,
                                   MiniBatchDictionaryLearning,
                                   DictionaryLearning, FastICA,
                                   IncrementalPCA, KernelPCA,
                                   MiniBatchSparsePCA, NMF, SparsePCA,
                                   TruncatedSVD)

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

        if mapping is None:
            mapping = {}

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

        return super().transform_df(df, base_name=base_name)

    def _proc_new_names(self, feat_names, base_name):

        # Get new names
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

        # Remove old names - using parent method
        feat_names = self._remove_old_names(feat_names)

        # New names come first, then rest of names
        all_names = new_names + feat_names

        return all_names


def _fit_transform_single_transformer(transformer, X, y):

    transformer.fit(X=X, y=y)
    X_trans = transformer.transform(X=X)
    return transformer, X_trans


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
                               random_state=None,
                               num_feat_keys=None):

    transformer, extra_transformer_params, transformer_params =\
        get_obj_and_params(transformer_str, TRANSFORMERS, extra_params, params)

    return transformer(**extra_transformer_params), transformer_params
