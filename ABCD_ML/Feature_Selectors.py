"""
Feature_Selectors.py
====================================
File with different Feature Selectors
"""
from ABCD_ML.ML_Helpers import (get_obj_and_params, get_avaliable_by_type,
                                show_param_options, get_possible_init_params)
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


def Show_Feat_Selectors(self, problem_type=None, feat_selector=None,
                        show_param_ind_options=True,
                        show_feat_selector_object=False,
                        show_all_possible_params=False):
        '''Print out the avaliable feature selectors,
        optionally restricted by problem type + other diagnostic args.

        Parameters
        ----------
        problem_type : {binary, categorical, regression, None}, optional
            Where `problem_type` is the underlying ML problem

            (default = None)

        feat_selector : str or list, optional
            If `feat_selector` is passed, will just show the specific
            feat selector, according to the rest of the params passed.
            Note : You must pass the specific feat_selector indicator str
            limited preproc will be done on this input!
            If list, will show all feat selectors within list

            (default = None)

        show_param_ind_options : bool, optional
            Flag, if set to True, then will display the ABCD_ML
            param ind options for each feat selector.

            (default = True)

        show_feat_selector_object : bool, optional
                Flag, if set to True, then will print the
                raw feat_selector object.

                (default = False)

        show_all_possible_params: bool, optional
                Flag, if set to True, then will print all
                possible arguments to the classes __init__

                (default = False)
        '''

        print('Note: Param distributions with a Rand Distribution')
        print('cannot be used in search_type = "grid"')
        print()

        if feat_selector is not None:
                if isinstance(feat_selector, str):
                        feat_selector = [feat_selector]
                for feat_selector_str in feat_selector:
                        show_feat_selector(feat_selector_str,
                                           show_param_ind_options,
                                           show_feat_selector_object,
                                           show_all_possible_params)
                return

        avaliable_by_type = get_avaliable_by_type(AVALIABLE)

        if problem_type is None:
                for pt in avaliable_by_type:
                        show_type(pt, avaliable_by_type,
                                  show_param_ind_options,
                                  show_feat_selector_object,
                                  show_all_possible_params)
        else:
                show_type(problem_type, avaliable_by_type,
                          show_param_ind_options, show_feat_selector_object,
                          show_all_possible_params)


def show_type(problem_type, avaliable_by_type, show_param_ind_options,
              show_feat_selector_object, show_all_possible_params):

        print('Problem Type:', problem_type)
        print('----------------------------------------')
        print()
        print('Avaliable feat_selectors: ')
        print()

        for feat_selector_str in avaliable_by_type[problem_type]:
                if 'user passed' not in feat_selector_str:
                        show_feat_selector(feat_selector_str,
                                           show_param_ind_options,
                                           show_feat_selector_object,
                                           show_all_possible_params)


def show_feat_selector(feat_selector_str, show_param_ind_options,
                       show_feat_selector_object, show_all_possible_params):

        multilabel, multiclass = False, False

        if 'multilabel ' in feat_selector_str:
                multilabel = True
        elif 'multiclass ' in feat_selector_str:
                multiclass = True

        feat_selector_str = feat_selector_str.replace('multilabel ', '')
        feat_selector_str = feat_selector_str.replace('multiclass ', '')

        print('- - - - - - - - - - - - - - - - - - - - ')
        M = SELECTORS[feat_selector_str]
        print(M[0].__name__, end='')
        print(' ("', feat_selector_str, '")', sep='')
        print('- - - - - - - - - - - - - - - - - - - - ')
        print()

        if multilabel:
                print('(MultiLabel)')
        elif multiclass:
                print('(MultiClass)')

        if show_feat_selector_object:
                print('Feat Selector Object: ', M[0])

        print()
        if show_param_ind_options:
                show_param_options(M[1])

        if show_all_possible_params:
                possible_params = get_possible_init_params(M[0])
                print('All Possible Params:', possible_params)
        print()
