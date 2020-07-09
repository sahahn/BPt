"""
Feature_Selectors.py
====================================
File with different Feature Selectors
"""
from ..helpers.ML_Helpers import (show_objects, get_possible_init_params,
                                  get_obj_and_params)
from sklearn.feature_selection import SelectPercentile, VarianceThreshold
from ..extensions.Feat_Selectors import RFE_Wrapper, FeatureSelector
import numpy as np
from numpy.random import RandomState
import nevergrad as ng

from sklearn.base import BaseEstimator, clone
from sklearn.feature_selection._base import SelectorMixin
from ..helpers.ML_Helpers import proc_mapping, update_mapping


class FeatureSelectorWrapper(SelectorMixin, BaseEstimator):

    def __init__(self, base_selector, wrapper_inds):
        self.base_selector = base_selector
        self.wrapper_inds = wrapper_inds

    def _proc_mapping(self, mapping):

        try:
            self._mapping
            return
        except AttributeError:
            self._mapping = mapping.copy()

        if len(mapping) > 0:
            self.wrapper_inds_ = proc_mapping(self.wrapper_inds_, mapping)

        return

    def fit(self, X, y=None, mapping=None, **fit_params):

        # Clone base object
        self.base_selector_ = clone(self.base_selector)
        self.wrapper_inds_ = np.copy(self.wrapper_inds)

        # Proc mapping
        if mapping is None:
            mapping = {}

        self._proc_mapping(mapping)

        # Calculate rest of inds
        self.rest_inds_ = [i for i in range(X.shape[1])
                           if i not in self.wrapper_inds_]

        # Attempt fit w/ passing mapping on
        try:
            self.base_selector_.fit(X=X[:, self.wrapper_inds_],
                                    y=y, mapping=mapping, **fit_params)
        except TypeError:
            self.base_selector_.fit(X=X[:, self.wrapper_inds_],
                                    y=y, **fit_params)

        # Grab the just calculated support mask
        support = self.base_selector_.get_support()

        # Generate the new mapping
        new_mapping = {}

        # First half is for updating the index within scope
        cnt = 0
        for i, wrap_ind in enumerate(self.wrapper_inds_):

            # If kept by feat selection
            if support[i]:
                new_mapping[wrap_ind] = cnt
                cnt += 1

            # Else, set to None
            else:
                new_mapping[wrap_ind] = None

        # Next, need to update the mapping for the remaining wrapper inds
        for rem_ind in range(len(self.rest_inds_)):
            new_mapping[self.rest_inds_[rem_ind]] = cnt
            cnt += 1

        # Save for reverse transform
        self._out_mapping = new_mapping.copy()

        # Update the passed mapping
        update_mapping(mapping, new_mapping)

        return self

    def transform(self, X):

        # Transform just wrapper inds
        X_trans = self.base_selector_.transform(X[:, self.wrapper_inds_])
        return_X = np.hstack([X_trans, X[:, self.rest_inds_]])

        return return_X

    def _get_support_mask(self):

        # Create full support as base support + True's for all rest inds
        # i.e., those features originally out of scope
        base_support = self.base_selector_.get_support()
        rest_support = np.ones(len(self.rest_inds_), dtype='bool')
        support = np.concatenate([base_support, rest_support])

        return support

    def inverse_transform(self, X):

        # Get reverse inds
        reverse_inds = proc_mapping(self.wrapper_inds_, self._out_mapping)
        reverse_rest_inds = proc_mapping(self.rest_inds_, self._out_mapping)

        # Reverse
        X_trans =\
            self.base_selector_.inverse_transform(X[:, reverse_inds])

        # Make empty return_X
        all_inds_len = len(self.wrapper_inds_) + len(self.rest_inds_)
        return_X = np.zeros((X.shape[0], all_inds_len), dtype=X.dtype)

        # Fill in return X
        return_X[:, self.wrapper_inds_] = X_trans
        return_X[:, self.rest_inds_] = X[:, reverse_rest_inds]

        return return_X


AVALIABLE = {
        'binary': {
                'univariate selection':
                'univariate selection c',
                'rfe': 'rfe',
                'variance threshold': 'variance threshold',
                'selector': 'selector',
        },
        'regression': {
                'univariate selection':
                'univariate selection r',
                'rfe': 'rfe',
                'variance threshold': 'variance threshold',
                'selector': 'selector',
        },
}

AVALIABLE['categorical'] = AVALIABLE['binary'].copy()

SELECTORS = {
    'univariate selection r': (SelectPercentile,
                               ['base univar fs regression',
                                'univar fs regression dist']),

    'univariate selection c': (SelectPercentile,
                               ['base univar fs classifier',
                                'univar fs classifier dist']),

    'rfe': (RFE_Wrapper, ['base rfe', 'rfe num feats dist']),

    'variance threshold': (VarianceThreshold, ['default']),

    'selector': (FeatureSelector, ['random', 'searchable']),
}


def get_special_selector(feat_selector, feat_selector_params, random_state,
                         num_feat_keys):

    # Init feat selector with mask of random feats
    if random_state is None:
        r_state = RandomState(np.random.randint(1000))
    elif isinstance(random_state, int):
        r_state = RandomState(random_state)
    else:
        r_state = random_state

    init_mask = r_state.random(num_feat_keys)
    feat_selector = feat_selector(mask=init_mask)

    # Figure out param passed
    if 'selector__mask' in feat_selector_params:
        p_name = 'selector__mask'

        # If set to searchable, set to searchable...
        if feat_selector_params[p_name] == 'sets as hyperparameters':

            feat_array = ng.p.Array(init=[.5 for i in range(num_feat_keys)])
            feat_array.set_mutation(sigma=1/6).set_bounds(lower=0, upper=1)
            feat_selector_params[p_name] = feat_array

        elif feat_selector_params[p_name] == 'sets as random features':
            del feat_selector_params[p_name]

    return feat_selector, feat_selector_params


def get_feat_selector_and_params(feat_selector_str, extra_params, params,
                                 search_type, random_state, num_feat_keys):
    '''Returns a scaler based on proced str indicator input,

    Parameters
    ----------
    feat_selector_str : str
        `feat_selector_str` refers to the type of feature selection
        to apply to the saved data during model evaluation.

    extra_params : dict
        Any extra params being passed.
        These can be supplied by creating another dict within extra_params.
        E.g., extra_params[method] = {'method param' : new_value}
        Where method param is a valid argument for that method,
        and method in this case is the str indicator.

    params : int
        The index of the params to use.

    Returns
    ----------
    feature_selector
        A feature selector object with fit and transform methods.

    dict
        The params for this feature selector
    '''

    feat_selector, extra_feat_selector_params, feat_selector_params =\
        get_obj_and_params(feat_selector_str, SELECTORS, extra_params,
                           params, search_type)

    # Special behavior for selector...
    if feat_selector_str == 'selector':

        feat_selector, feat_selector_params =\
            get_special_selector(feat_selector, feat_selector_params,
                                 random_state, num_feat_keys)

        return feat_selector, feat_selector_params

    else:
        # Need to check for estimator, as RFE needs a default param for est.
        # Set as placeholder None if passed
        possible_params = get_possible_init_params(feat_selector)
        if 'estimator' in possible_params:
            extra_feat_selector_params['estimator'] = None

        return (feat_selector(**extra_feat_selector_params),
                feat_selector_params)


def Show_Feat_Selectors(self, problem_type=None, feat_selector_str=None,
                        show_params_options=False, show_object=False,
                        possible_params=False):
    '''Print out the avaliable feature selectors,
    optionally restricted by problem type + other diagnostic args.

    Parameters
    ----------
    problem_type : str or None, optional
        Where `problem_type` is the underlying ML problem

        (default = None)

    feat_selector_str : str or list, optional
        If `feat_selector_str` is passed, will just show the specific
        feat selector, according to the rest of the params passed.
        Note : You must pass the specific feat_selector indicator str
        limited preproc will be done on this input!
        If list, will show all feat selectors within list

        (default = None)

    show_params_options : bool, optional
        Flag, if set to True, then will display the ABCD_ML
        param ind options for each feat selector.

        (default = False)

    show_object : bool, optional
        Flag, if set to True, then will print the
        raw feat_selector object.

        (default = False)

    possible_params: bool, optional
        Flag, if set to True, then will print all
        possible arguments to the classes __init__

        (default = False)
    '''

    print('More information through this function is avaliable')
    print('By passing optional extra optional params! Please view',
          'the help function for more info!')
    print('Note: the str indicator actually passed during Evaluate / Test')
    print('is listed as ("str indicator")')
    print()

    show_objects(problem_type, feat_selector_str,
                 show_params_options, show_object, possible_params,
                 AVALIABLE, SELECTORS)
