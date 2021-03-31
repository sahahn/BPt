"""
Feature_Selectors.py
====================================
File with different Feature Selectors
"""
from ..helpers import (get_possible_init_params,
                       get_obj_and_params, all_from_avaliable)
from sklearn.feature_selection import SelectPercentile, VarianceThreshold
from ...extensions.FeatSelectors import FeatureSelector
from sklearn.feature_selection import RFE
import numpy as np
from numpy.random import RandomState
from ..params.Params import Array


AVALIABLE = {
        'binary': {
                'univariate selection c':
                'univariate selection c',
                'univariate selection':
                'univariate selection c',
                'rfe': 'rfe',
                'variance threshold': 'variance threshold',
                'selector': 'selector',
        },
        'regression': {
                'univariate selection r':
                'univariate selection r',
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
                                'univar fs regression dist',
                                'univar fs r keep more',
                                'univar fs r keep less']),

    'univariate selection c': (SelectPercentile,
                               ['base univar fs classifier',
                                'univar fs classifier dist',
                                'univar fs c keep more',
                                'univar fs c keep less']),

    'rfe': (RFE, ['base rfe', 'rfe num feats dist']),

    'variance threshold': (VarianceThreshold, ['default']),

    'selector': (FeatureSelector, ['random', 'searchable']),
}


def get_special_selector(feat_selector, feat_selector_params,
                         random_state, num_feat_keys):

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

            feat_array = Array(init=[.5 for i in range(num_feat_keys)])
            feat_array.set_mutation(sigma=1/6).set_bounds(lower=0, upper=1)
            feat_selector_params[p_name] = feat_array

        elif feat_selector_params[p_name] == 'sets as random features':
            del feat_selector_params[p_name]

    return feat_selector, feat_selector_params


def get_feat_selector_and_params(feat_selector_str, extra_params, params,
                                 random_state, num_feat_keys):
    '''Returns a scaler based on proc'ed str indicator input,

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

    # Base behavior
    feat_selector, extra_feat_selector_params, feat_selector_params =\
        get_obj_and_params(feat_selector_str, SELECTORS, extra_params, params)

    # Special behavior for selector...
    if feat_selector_str == 'selector':

        # Move to params
        if 'mask' in extra_feat_selector_params:
            feat_selector_params['selector__mask'] =\
                extra_feat_selector_params.pop('mask')

        feat_selector, feat_selector_params =\
            get_special_selector(feat_selector, feat_selector_params,
                                 random_state, num_feat_keys)

        return feat_selector, feat_selector_params

    # Otherwise, need to check for estimator
    # as RFE needs a default param for est.
    # Set as placeholder None if passed
    possible_params = get_possible_init_params(feat_selector)
    if 'estimator' in possible_params:
        extra_feat_selector_params['estimator'] = None

    return (feat_selector(**extra_feat_selector_params),
            feat_selector_params)


all_obj_keys = all_from_avaliable(AVALIABLE)
