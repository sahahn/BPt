"""
Feature_Selectors.py
====================================
File with different Feature Selectors
"""
from ABCD_ML.ML_Helpers import get_obj_and_params
from sklearn.feature_selection import *

AVALIABLE = {
        'binary': {
            'univariate select half':
            'univariate select half classification',
        },
        'regression': {
            'univariate select half':
            'univariate select half regression',
        },
        'categorical': {
            'multilabel': {
            },
            'multiclass': {
                'univariate select half':
                'univariate select half classification',
            }
        }
}

SELECTORS = {
    'univariate select half regression': (SelectPercentile,
                                          ['base univar fs regression']),

    'univariate select half classification': (SelectPercentile,
                                              ['base univar fs classifier']),
}


def get_feat_selector(feat_selector_str, extra_params, param_ind):
    '''From a final feat selector str indicator, return the
    feature selector object.'''

    feat_selector, params = SELECTORS[feat_selector_str]

    # Update with extra params if applicable
    if feat_selector_str in extra_params:
        params.update(extra_params[feat_selector_str])

    return feat_selector(**params)


def get_feat_selector(feat_selector_str, extra_params, param_ind):
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

    param_ind : int
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
                           param_ind)

    return feat_selector(**extra_feat_selector_params), feat_selector_params
