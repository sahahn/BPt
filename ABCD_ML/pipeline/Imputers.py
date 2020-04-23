import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from ..helpers.ML_Helpers import get_obj_and_params, show_objects, proc_mapping, replace_model_name


class Regular_Imputer():

    def __init__(self, imputer, inds, copy=True):

        self.imputer = imputer
        self.inds = inds

        self.valid_mask = None
        self.copy = copy

    def _proc_mapping(self, mapping):
        
        try:
            self._mapping
            return
        
        except AttributeError:
            self._mapping = mapping

        if len(mapping) > 0:
            self.inds = proc_mapping(self.inds, mapping)

        return

    def fit(self, X, y=None, mapping=None):

        if mapping is None:
            mapping = {}

        self._proc_mapping(mapping)

        if self.copy:
            X_copy = X.copy()
        else:
            X_copy = X

        impute_X = self.get_impute_X(X_copy)
        self.imputer.fit(impute_X, y)

    def get_impute_X(self, X):

        if self.valid_mask is None:
            self.valid_mask = ~pd.isnull(X).any(axis=0)
            self.valid_mask[self.inds] = False

        valid_extra_X = X[:, self.valid_mask]

        # Unsure if this is needed, need to check
        # valid_extra_X[pd.isnull(valid_extra_X)] = 0

        ind = valid_extra_X.shape[1]
        impute_shape = (np.shape(X)[0], ind + len(self.inds))

        impute_X = np.zeros(impute_shape)
        impute_X[:, :ind] = valid_extra_X

        for i in range(len(self.inds)):
            impute_X[:, [ind+i]] = X[:, [self.inds[i]]]

        return impute_X

    def transform(self, X):

        if self.copy:
            X_copy = X.copy()
        else:
            X_copy = X

        impute_X = self.get_impute_X(X)
        imputed = self.imputer.transform(impute_X)

        ind = np.sum(self.valid_mask)

        for i in range(len(self.inds)):
            X_copy[:, self.inds[i]] = imputed[:, ind+i]

        return X_copy

    def fit_transform(self, X, y=None, mapping=None):

        if mapping == None:
            mapping = {}

        self.fit(X, y, mapping=mapping)
        return self.transform(X)

    def set_params(self, **params):
        self.imputer.set_params(**params)

    def get_params(self, deep=False):

        params = {'imputer': self.imputer,
                  'inds': self.inds,
                  'copy': self.copy}

        return params


class Categorical_Imputer():

    def __init__(self, imputer, encoder_inds=[], ordinal_inds=[], encoders=[],
                 copy=True):

        self.imputer = imputer
        self.encoders = encoders
        self.encoder_inds = encoder_inds
        self.ordinal_inds = ordinal_inds

        self.valid_mask = None
        self.copy = copy

    def get_ohe(self, categories):

        ohe = OneHotEncoder(categories='auto', sparse=False)
        ohe.categories_ = categories
        ohe._legacy_mode = False

        return ohe

    def _proc_mapping(self, mapping):
        
        try:
            self._mapping
            return
        
        except AttributeError:
            self._mapping = mapping

        if len(mapping) > 0:
            self.encoder_inds = proc_mapping(self.encoder_inds, mapping)
            self.ordinal_inds = proc_mapping(self.ordinal_inds, mapping)
        return

    def fit(self, X, y=None, mapping=None):

        if mapping is None:
            mapping = {}

        self._proc_mapping(mapping)

        if self.copy:
            X_copy = X.copy()
        else:
            X_copy = X

        impute_X = self.get_impute_X(X_copy)
        self.imputer.fit(impute_X)

    def get_impute_X(self, X):

        if self.valid_mask is None:
            self.valid_mask = ~pd.isnull(X).any(axis=0)

            all_encoder_inds = []
            for e in self.encoder_inds:
                all_encoder_inds += e

            self.valid_mask[all_encoder_inds] = False
            self.valid_mask[self.ordinal_inds] = False

        valid_extra_X = X[:, self.valid_mask]

        # Unsure if this is needed, need to check
        # valid_extra_X[pd.isnull(valid_extra_X)] = 0

        ind = valid_extra_X.shape[1]
        impute_shape = (np.shape(X)[0],
                        ind + len(self.encoder_inds) + len(self.ordinal_inds))
        impute_X = np.zeros(impute_shape)

        impute_X[:, :ind] = valid_extra_X

        for i in range(len(self.encoder_inds)):

            ordinal = self.to_ordinal(X, self.encoders[i],
                                      self.encoder_inds[i])
            impute_X[:, [ind+i]] = ordinal

        for i in range(len(self.ordinal_inds)):

            ordinal = X[:, [self.ordinal_inds[i]]]
            impute_X[:, [ind+i+len(self.encoder_inds)]] = ordinal

        return impute_X

    def transform(self, X):

        if self.copy:
            X_copy = X.copy()
        else:
            X_copy = X

        impute_X = self.get_impute_X(X)
        imputed = self.imputer.transform(impute_X)

        ind = np.sum(self.valid_mask)

        for i in range(len(self.encoder_inds)):

            imputed_col = imputed[:, [ind+i]]

            encoder = self.get_ohe(self.encoders[i][1])
            encoded = encoder.transform(imputed_col)

            # If originally dummy coded
            if len(self.encoders[i]) > 2:
                encoded = np.delete(encoded, self.encoders[i][2], axis=1)

            X_copy[:, self.encoder_inds[i]] = encoded

        for i in range(len(self.ordinal_inds)):

            X_copy[:, self.ordinal_inds[i]] =\
                imputed[:, ind+i+len(self.encoder_inds)]

        return X_copy

    def to_ordinal(self, X, encoder, inds):
        '''encoder should be the tuple object here'''

        selection = X[:, inds]
        non_nan_mask = ~pd.isnull(selection).any(axis=1)

        non_nan_selection = selection[non_nan_mask]
        if len(encoder) > 2:
            non_nan_selection =\
                self.add_back_dummy_col(non_nan_selection, encoder[2])

        enc = self.get_ohe(encoder[1])
        ordinal = enc.inverse_transform(non_nan_selection)

        to_fill = np.full((len(non_nan_mask), 1), np.nan)
        to_fill[non_nan_mask] = ordinal

        return to_fill

    def add_back_dummy_col(self, non_nan_selection, e):

        nn_shape = non_nan_selection.shape
        reverse_dummy_code = np.zeros((nn_shape[0], nn_shape[1]+1))

        # Add around dropped
        reverse_dummy_code[:, 0:e] = non_nan_selection[:, :e]
        reverse_dummy_code[:, e+1:] = non_nan_selection[:, e:]

        # Recover dropped values
        reverse_dummy_code[:, e] =\
            np.where(np.sum(reverse_dummy_code, axis=1) == 1, 0, 1)

        return reverse_dummy_code

    def fit_transform(self, X, y=None, mapping=None):

        if mapping == None:
            mapping = {}

        self.fit(X, y, mapping=mapping)
        return self.transform(X)

    def set_params(self, **params):
        self.imputer.set_params(**params)

    def get_params(self, deep=False):

        params = {'imputer': self.imputer,
                  'encoder_inds': self.encoder_inds,
                  'ordinal_inds': self.ordinal_inds,
                  'encoders': self.encoders,
                  'copy': self.copy}

        return params


IMPUTERS = {
    'mean': (SimpleImputer, ['mean imp']),
    'median': (SimpleImputer, ['median imp']),
    'most frequent': (SimpleImputer, ['most freq imp']),
    'constant': (SimpleImputer, ['constant imp']),
    'iterative': (IterativeImputer, ['iterative imp']),
}


def get_imputer_and_params(imputer_param, scopes, encoders,
                           base_estimator, base_estimator_params,
                           specs):

    # Unpack params
    imputer_str = imputer_param.obj
    extra_params = imputer_param.extra_params
    params = imputer_param.params
    inds, encoder_inds, ordinal_inds = scopes

    if base_estimator is None:

        base_imputer_obj, extra_imputer_params, imputer_params =\
            get_obj_and_params(imputer_str, IMPUTERS, extra_params, params,
                               specs['search_type'])

        imputer_params = replace_model_name(base_estimator_params)

        # proc imputer params for within obj
        new_imputer_params = {}
        for key in imputer_params:
            new_imputer_params['estimator__' + key] = imputer_params[key]
        imputer_params = new_imputer_params

    else:

        base_imputer_obj, extra_imputer_params, imputer_params =\
            get_obj_and_params('iterative', IMPUTERS, extra_params, 0,
                               search_type=None)

        extra_imputer_params['estimator'] = base_estimator
        if len(inds) == 0:
            extra_imputer_params['initial_strategy'] = 'median'

        # Use the base estimator params !
        imputer_params = replace_model_name(base_estimator_params)

    # This is the actual imputer object
    base_imputer = base_imputer_obj(**extra_imputer_params)

    # Categorical
    if len(inds) == 0:
        encoders_to_pass = []

        for e in encoders:
            if len(e) > 1:
                encoders_to_pass.append(
                    tuple(i.categories_ if hasattr(i, 'categories_') else i
                          for i in e))

        imputer = Categorical_Imputer(base_imputer, encoder_inds,
                                      ordinal_inds, encoders_to_pass)

    # Float or Binary
    else:
        imputer = Regular_Imputer(base_imputer, inds)

    # Try to add random state to base imputer if possible
    try:
        imputer.imputer.random_state = specs['random_state']
    except AttributeError:
        pass

    return imputer, imputer_params


def Show_Imputers(self, imputer=None, show_params_options=False,
                  show_object=False, show_all_possible_params=False):
    '''Print out the avaliable imputers.

    Parameters
    ----------
    imputer : str or list, optional
        Provide a str or list of strs, where
        each str is the exact imputer str indicator
        in order to show information for only that (or those)
        imputers.

    show_params_options : bool, optional
        Flag, if set to True, then will display the ABCD_ML
        param ind options for each imputer.

        (default = False)

    show_object : bool, optional
        Flag, if set to True, then will print the raw data imputer
        object.

        (default = False)

    show_all_possible_params: bool, optional
        Flag, if set to True, then will print all
        possible arguments to the classes __init__

        (default = False)
    '''

    print('Visit: ')
    print('https://scikit-learn.org/stable/modules/impute.html')
    print('For more detailed information on different imputers')
    print('More information through this function is avaliable')
    print('By passing optional extra optional params! Please view',
          'the help function for more info!')
    print('Note: the str indicator actually passed during Evaluate / Test')
    print('is listed as ("str indicator")')
    print('Also for imputers, a model str can be passed instead of',
          'imputer str!')
    print()

    show_objects(problem_type=None, obj=imputer,
                 show_params_options=show_params_options,
                 show_object=show_object,
                 show_all_possible_params=show_all_possible_params,
                 AVALIABLE=None, OBJS=IMPUTERS)
