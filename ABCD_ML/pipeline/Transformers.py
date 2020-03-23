from ..helpers.ML_Helpers import get_obj_and_params, proc_mapping, update_mapping, show_objects
import numpy as np

from sklearn.decomposition import (PCA, FactorAnalysis,
                                   MiniBatchDictionaryLearning,
                                   DictionaryLearning, FastICA,
                                   IncrementalPCA, KernelPCA,
                                   MiniBatchSparsePCA, NMF, SparsePCA,
                                   TruncatedSVD)

from sklearn.base import BaseEstimator


class Transformer_Wrapper(BaseEstimator):

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
        X_trans_inds = [i for i in range(X_trans.shape[1])]
        self._n_trans = len(X_trans_inds)

        new_mapping = {}

        # Many to Many case
        for i in inds:
            new_mapping[i] = X_trans_inds

        for cnt in range(len(rest_inds)):
            new_mapping[rest_inds[cnt]] = len(X_trans_inds) + cnt

        # Update mapping
        update_mapping(mapping, new_mapping)

        return np.hstack([X_trans, X[:, rest_inds]])

    def transform(self, X):

        # Fit transform just inds of X
        rest_inds = [i for i in range(X.shape[1]) if i not in self.wrapper_inds]
        X_trans = self.wrapper_transformer.transform(X[:, self.wrapper_inds])

        return np.hstack([X_trans, X[:, rest_inds]])

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
}
                            

def get_transformer_and_params(transformer_str, extra_params, params, search_type,
                               inds, random_state=None, n_jobs=1):

    base_trans_obj, extra_trans_params, transformer_params =\
        get_obj_and_params(transformer_str, TRANSFORMERS, extra_params, params,
                           search_type)

    base_trans = base_trans_obj(**extra_trans_params)

    return wrap_transformer(base_trans, transformer_params, inds, search_type,
                            random_state, n_jobs)


def wrap_transformer(base_trans, transformer_params, inds, search_type,
                     random_state, n_jobs):

    if search_type is not None:
        n_jobs = 1

    # Try to set attributes
    try:
        base_trans.n_jobs = n_jobs
    except AttributeError:
        pass

    try:
        base_trans.random_state = random_state
    except AttributeError:
        pass

    transformer = Transformer_Wrapper(base_trans, inds)
    return transformer, transformer_params


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
