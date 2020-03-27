from ..helpers.ML_Helpers import get_obj_and_params, proc_mapping, update_mapping, show_objects
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.decomposition import (PCA, FactorAnalysis,
                                   MiniBatchDictionaryLearning,
                                   DictionaryLearning, FastICA,
                                   IncrementalPCA, KernelPCA,
                                   MiniBatchSparsePCA, NMF, SparsePCA,
                                   TruncatedSVD)

from category_encoders import (OneHotEncoder, BackwardDifferenceEncoder,
                               BinaryEncoder, CatBoostEncoder, HelmertEncoder,
                               JamesSteinEncoder, LeaveOneOutEncoder, MEstimateEncoder,
                               PolynomialEncoder, SumEncoder, TargetEncoder, WOEEncoder)


def ce_conv(parent):
    '''Wrapper function to make classes from category encoders compatible with ABCD_ML transformer'''
    
    class child(parent):
        
        def fit(self, X, y=None, **kwargs):
        
            self.return_df = False
            self.cols = [i for i in range(X.shape[1])]
        
            super().fit(X, y, **kwargs)
            return self
        
    return child

class Transformer_Wrapper(BaseEstimator, TransformerMixin):

    def __init__(self, wrapper_transformer, wrapper_inds, **params):

        self.wrapper_transformer = wrapper_transformer
        self.wrapper_inds = wrapper_inds

        # Set any remaining params to wrapper transformer
        self.wrapper_transformer.set_params(**params)

    def _proc_mapping(self, mapping):

        try:
            self._mapping
            return

        except AttributeError:
            self._mapping = mapping.copy()

        if len(mapping) > 0:
            self.wrapper_inds = proc_mapping(self.wrapper_inds, mapping)

        return

    def fit(self, X, y=None, mapping={}):

        # Need to call fit_transform to figure out change to mapping
        self.fit_transform(X, y, mapping=mapping)
        return self

    def fit_transform(self, X, y=None, mapping={}):

        self._proc_mapping(mapping)

        inds = self.wrapper_inds
        rest_inds = [i for i in range(X.shape[1]) if i not in inds]

        # Fit transform just inds of X
        X_trans = self.wrapper_transformer.fit_transform(X[:, inds])
        self._X_trans_inds = [i for i in range(X_trans.shape[1])]

        new_mapping = {}

        # Many to Many case
        for i in inds:
            new_mapping[i] = self._X_trans_inds

        for cnt in range(len(rest_inds)):
            new_mapping[rest_inds[cnt]] = len(self._X_trans_inds) + cnt

        # Update mapping
        update_mapping(mapping, new_mapping)

        return np.hstack([X_trans, X[:, rest_inds]])

    def transform(self, X):

        # Fit transform just inds of X
        rest_inds = [i for i in range(X.shape[1]) if i not in self.wrapper_inds]
        X_trans = self.wrapper_transformer.transform(X[:, self.wrapper_inds])

        return np.hstack([X_trans, X[:, rest_inds]])

    def transform_df(self, df, base_name='transformer'):

        feat_names = list(df)

        # Transform data as np array
        X = np.array(df).astype(float)
        X_trans = self.transform(X)

        # Get new names
        new_names = self._get_new_df_names(base_name)

        # Remove old names
        df, feat_names = self._remove_old_df_names(df, feat_names)

        # New names come first, then rest of names
        feat_names = new_names + feat_names

        # Replace vals in df with transformed vals and new names
        for i in range(len(feat_names)):
            df[feat_names[i]] = X_trans[:, i]

        return df

    def _get_new_df_names(self, base_name):
        '''Create new feature names for the transformed features'''

        n_trans = len(self._X_trans_inds)
        new_names = [base_name + '_' + str(i) for i in range(n_trans)]

        return new_names

    def _remove_old_df_names(self, df, feat_names):
        '''Create new feature names for the transformed features'''

        to_remove = [feat_names[i] for i in self.wrapper_inds]
        feat_names = [name for name in feat_names if name not in to_remove]
        df = df.drop(to_remove, axis=1)

        return df, feat_names

    def set_params(self, **params):

        if 'wrapper_transformer' in params:
            self.wrapper_transformer = params.pop('wrapper_transformer')
        if 'wrapper_inds' in params:
            self.wrapper_inds = params.pop('wrapper_inds')

        self.wrapper_transformer.set_params(**params)

    def get_params(self, deep=False):

        params = {'wrapper_transformer': self.wrapper_transformer,
                  'wrapper_inds': self.wrapper_inds}

        params.update(self.wrapper_transformer.get_params())

        return params


TRANSFORMERS = {
    'pca': (PCA, ['default', 'pca var search']),
    'sparse pca': (SparsePCA, ['default']),
    'mini batch sparse pca': (MiniBatchSparsePCA, ['default']),
    'factor analysis': (FactorAnalysis, ['default']),
    'dictionary learning': (DictionaryLearning, ['default']),
    'mini batch dictionary learning': (MiniBatchDictionaryLearning, ['default']),
    'fast ica': (FastICA, ['default']),
    'incremental pca': (IncrementalPCA, ['default']),
    'kernel pca': (KernelPCA, ['default']),
    'nmf': (NMF, ['default']),
    'one hot encoder': (ce_conv(OneHotEncoder), ['default']),
    'backward difference encoder': (ce_conv(BackwardDifferenceEncoder), ['default']),
    'binary encoder': (ce_conv(BinaryEncoder), ['default']),
    'cat boost encoder': (ce_conv(CatBoostEncoder), ['default']),
    'helmert encoder': (ce_conv(HelmertEncoder), ['default']),
    'james stein encoder': (ce_conv(JamesSteinEncoder), ['default']),
    'leave one out encoder': (ce_conv(LeaveOneOutEncoder), ['default']),
    'm estimate encoder': (ce_conv(MEstimateEncoder), ['default']),
    'polynomial encoder': (ce_conv(PolynomialEncoder), ['default']),
    'sum encoder': (ce_conv(SumEncoder), ['default']),
    'target encoder': (ce_conv(TargetEncoder), ['default']),
    'woe encoder': (ce_conv(WOEEncoder), ['default']), 
}
                            

def get_transformer_and_params(transformer_str, extra_params, params, search_type,
                               random_state=None, num_feat_keys=None):

    transformer, extra_transformer_params, transformer_params =\
        get_obj_and_params(transformer_str, TRANSFORMERS, extra_params, params,
                           search_type)

    return transformer(**extra_transformer_params), transformer_params


def Show_Transformers(self, transformer=None, show_params_options=False,
                      show_object=False,
                      show_all_possible_params=False):
    '''Print out the avaliable data transformers.

    Parameters
    ----------
    transformer : str or list, optional
        Provide a str or list of strs, where
        each str is the exact transformer str indicator
        in order to show information for only that (or those)
        transformers

    show_params_options : bool, optional
        Flag, if set to True, then will display the ABCD_ML
        param ind options for each transformer.

        (default = False)

    show_object : bool, optional
        Flag, if set to True, then will print the raw transformer
        object.

        (default = False)

    show_all_possible_params: bool, optional
        Flag, if set to True, then will print all
        possible arguments to the classes __init__

        (default = False)
    '''

    show_objects(problem_type=None, obj=transformer,
                 show_params_options=show_params_options,
                 show_object=show_object,
                 show_all_possible_params=show_all_possible_params,
                 AVALIABLE=None, OBJS=TRANSFORMERS)
