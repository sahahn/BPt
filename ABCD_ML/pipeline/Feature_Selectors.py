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
