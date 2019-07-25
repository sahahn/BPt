"""
Feature_Selectors.py
====================================
File with different Feature Selectors
"""

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
                                          {score_func: f_regression,
                                           percentile: 50}),

    'univariate select half classification': (SelectPercentile,
                                              {score_func: f_classif,
                                               percentile: 50}),
}


def get_feat_selector(feat_selector_str, extra_params):
    '''From a final feat selector str indicator, return the
    feature selector object.'''

    feat_selector, params = SELECTORS[feat_selector_str]

    # Update with extra params if applicable
    if feat_selector_str in extra_params:
        params.update(extra_params[feat_selector_str])

    return feat_selector(**params)
