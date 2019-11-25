"""
Feature_Selectors.py
====================================
File with different Feature Selectors
"""
from ..helpers.ML_Helpers import (show_objects, get_possible_init_params,
                                  get_obj_and_params)
from sklearn.feature_selection import *


class RFE(RFE):
    def fit(self, X, y):
        '''Override the fit function for slight added functionality,
           specifically allow passing in float % to keep.
        '''

        if isinstance(self.n_features_to_select, float):

            if self.n_features_to_select <= 0:
                self.n_features_to_select = 1

            if self.n_features_to_select < 1:
                divide_by = self.n_features_to_select ** -1
                self.n_features_to_select = X.shape[1] // divide_by

        return self._fit(X, y)


AVALIABLE = {
        'binary': {
                'univariate selection':
                'univariate selection classification',
                'rfe': 'rfe',
                'variance threshold': 'variance threshold',
        },
        'regression': {
                'univariate selection':
                'univariate selection regression',
                'rfe': 'rfe',
                'variance threshold': 'variance threshold',
        },
        'multilabel': {
                'variance threshold': 'variance threshold',
        }
}

AVALIABLE['categorical'] = AVALIABLE['binary'].copy()

SELECTORS = {
    'univariate selection regression': (SelectPercentile,
                                        ['base univar fs regression',
                                         'univar fs regression dist']),

    'univariate selection classification': (SelectPercentile,
                                            ['base univar fs classifier',
                                             'univar fs classifier dist']),

    'rfe': (RFE, ['base rfe', 'rfe num feats dist']),

    'variance threshold': (VarianceThreshold, ['default']),
}


def get_feat_selector_and_params(feat_selector_str, extra_params, params,
                                 search_type):
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

    # Need to check for estimator, as RFE needs a default param for estimator
    # Though, only replaced if not passed in user extra params already.
    possible_params = get_possible_init_params(feat_selector)
    if 'estimator' in possible_params:
            if 'estimator' not in extra_feat_selector_params:
                extra_feat_selector_params['estimator'] = None

    return feat_selector(**extra_feat_selector_params), feat_selector_params


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
