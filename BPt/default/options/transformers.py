from ..helpers import get_obj_and_params, all_from_objects

from sklearn.decomposition import (PCA, FactorAnalysis,
                                   MiniBatchDictionaryLearning,
                                   DictionaryLearning, FastICA,
                                   IncrementalPCA, KernelPCA,
                                   MiniBatchSparsePCA, NMF, SparsePCA,
                                   TruncatedSVD)

from sklearn.preprocessing import OneHotEncoder


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
    'one hot encoder': (OneHotEncoder, ['ohe']),
    'dummy coder': (OneHotEncoder, ['dummy code'])}


def get_transformer_and_params(transformer_str, extra_params,
                               params, **kwargs):

    transformer, extra_transformer_params, transformer_params =\
        get_obj_and_params(transformer_str, TRANSFORMERS, extra_params, params)

    return transformer(**extra_transformer_params), transformer_params


all_obj_keys = all_from_objects(TRANSFORMERS)
