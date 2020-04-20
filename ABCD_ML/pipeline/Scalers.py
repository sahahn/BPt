"""
Scalers.py
====================================
File containing the various data scalers.
"""
from ..helpers.ML_Helpers import (show_param_options, get_possible_init_params,
                                  get_obj_and_params, show_objects)
from sklearn.preprocessing import (MinMaxScaler, RobustScaler, StandardScaler,
                                   PowerTransformer, MaxAbsScaler,
                                   QuantileTransformer, Normalizer)
from ..extensions.Scalers import Winsorizer


# Scalers differs from metrics and models in that the types are not restricted
# by a given problem type. Therefore no AVALIABLE dictionary is neccisary,
# all avaliable scalers are as indexed by SCALERS.
SCALERS = {
    'standard': (StandardScaler, ['base standard']),

    'minmax': (MinMaxScaler, ['base minmax']),

    'maxabs': (MaxAbsScaler, ['default']),

    'robust': (RobustScaler, ['base robust', 'robust gs']),

    'yeo': (PowerTransformer, ['base yeo']),

    'boxcox': (PowerTransformer, ['base boxcox']),

    'winsorize': (Winsorizer, ['base winsorize', 'winsorize gs']),

    'quantile norm': (QuantileTransformer, ['base quant norm']),

    'quantile uniform': (QuantileTransformer, ['base quant uniform']),

    'normalize': (Normalizer, ['default'])
    }


def get_scaler_and_params(scaler_str, extra_params, params, search_type,
                          random_state=None, num_feat_keys=None):
    '''Returns a scaler based on proced str indicator input,

    Parameters
    ----------
    scaler_str : str
        `scaler_str` refers to the type of scaling to apply
        to the saved data during model evaluation.

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
    scaler
        A scaler object with fit and transform methods.

    dict
        The params for this scaler
    '''

    scaler, extra_scaler_params, scaler_params =\
        get_obj_and_params(scaler_str, SCALERS, extra_params, params,
                           search_type)

    return scaler(**extra_scaler_params), scaler_params


def Show_Scalers(self, scaler=None, show_params_options=False,
                 show_object=False,
                 show_all_possible_params=False):
    '''Print out the avaliable data scalers.

    Parameters
    ----------
    scaler : str or list, optional
        Provide a str or list of strs, where
        each str is the exact scaler str indicator
        in order to show information for only that (or those)
        data scalers

    show_params_options : bool, optional
        Flag, if set to True, then will display the ABCD_ML
        param ind options for each data scaler.

        (default = False)

    show_object : bool, optional
        Flag, if set to True, then will print the raw data scaler
        object.

        (default = False)

    show_all_possible_params: bool, optional
        Flag, if set to True, then will print all
        possible arguments to the classes __init__

        (default = False)
    '''

    print('Visit: ')
    print('https://scikit-learn.org/stable/modules/preprocessing.html')
    print('For more detailed information on different scalers',
          '/ preprocessing.')
    print('Note Scalers is used somewhat loosely.')
    print('They describe any transformation on the data that does not')
    print('change the number of columns or data points, and that do')
    print('not require access to the target (y) variable.')
    print('More information through this function is avaliable')
    print('By passing optional extra optional params! Please view',
          'the help function for more info!')
    print('Note: the str indicator actually passed during Evaluate / Test')
    print('is listed as ("str indicator")')
    print()

    show_objects(problem_type=None, obj=scaler,
                 show_params_options=show_params_options,
                 show_object=show_object,
                 show_all_possible_params=show_all_possible_params,
                 AVALIABLE=None, OBJS=SCALERS)
