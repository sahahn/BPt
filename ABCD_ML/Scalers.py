"""
Scalers.py
====================================
File containing the various data scalers.
"""
from ABCD_ML.ML_Helpers import (show_param_options, get_possible_init_params,
                                get_obj_and_params)
from sklearn.preprocessing import (MinMaxScaler, RobustScaler, StandardScaler,
                                   PowerTransformer)


# Scalers differs from metrics and models in that the types are not restricted
# by a given problem type. Therefore no AVALIABLE dictionary is neccisary,
# all avaliable scalers are as indexed by SCALERS.
SCALERS = {
    'standard': (StandardScaler, ['base standard']),

    'minmax': (MinMaxScaler, ['base minmax']),

    'robust': (RobustScaler, ['base robust', 'robust gs']),

    'power': (PowerTransformer, ['base power'])
    }


def get_scaler_objects():

    objs = []
    for scaler_str in SCALERS:

        obj = SCALERS[scaler_str][0]
        obj_params = SCALERS[scaler_str][1]
        objs.append((scaler_str, obj, obj_params))

    return objs


def get_data_scaler_and_params(data_scaler_str, extra_params, param_ind,
                               search_type):
    '''Returns a scaler based on proced str indicator input,

    Parameters
    ----------
    data_scaler_str : str
        `data_scaler_str` refers to the type of scaling to apply
        to the saved data during model evaluation.

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
    scaler
        A scaler object with fit and transform methods.

    dict
        The params for this scaler
    '''

    data_scaler, extra_data_scaler_params, data_scaler_params =\
        get_obj_and_params(data_scaler_str, SCALERS, extra_params, param_ind,
                           search_type)

    return data_scaler(**extra_data_scaler_params), data_scaler_params


def Show_Scalers(self, data_scaler=None, show_param_ind_options=False,
                 show_scaler_object=False,
                 show_all_possible_params=False):
    '''Just calls Show_Data_Scalers'''

    self.Show_Data_Scalers(data_scaler, show_param_ind_options,
                           show_scaler_object, show_all_possible_params)


def Show_Data_Scalers(self, data_scaler=None, show_param_ind_options=False,
                      show_scaler_object=False,
                      show_all_possible_params=False):
    '''Print out the avaliable data scalers.

    Parameters
    ----------
    data_scaler : str or list, optional
        Provide a str or list of strs, where
        each str is the exact data_scaler str indicator
        in order to show information for only that (or those)
        data scalers

    show_param_ind_options : bool, optional
        Flag, if set to True, then will display the ABCD_ML
        param ind options for each data scaler.

        (default = False)

    show_scaler_object : bool, optional
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
    print('Note Scalers / Data_Scalers is used somewhat loosely.')
    print('They describe any transformation on the data that does not')
    print('change the number of columns or data points, and that do')
    print('not require access to the target (y) variable.')
    print('More information through this function is avaliable')
    print('By passing optional extra optional params! Please view',
          'the help function for more info!')
    print('Note: the str indicator actually passed during Evaluate / Test')
    print('is listed as ("str indicator")')
    print()

    if data_scaler is not None:
        if isinstance(data_scaler, str):
                data_scaler = [data_scaler]
        for scaler_str in data_scaler:
                show_scaler(scaler_str, show_param_ind_options,
                            show_scaler_object, show_all_possible_params)
        return

    for scaler in SCALERS:
        show_scaler(scaler, show_param_ind_options,
                    show_scaler_object, show_all_possible_params)


def show_scaler(scaler, show_param_ind_options, show_scaler_object,
                show_all_possible_params):

    print('- - - - - - - - - - - - - - - - - - - - ')
    S = SCALERS[scaler]
    print(S[0].__name__, end='')
    print(' ("', scaler, '")', sep='')
    print('- - - - - - - - - - - - - - - - - - - - ')
    print()

    if show_scaler_object:
        print('Scaler Object: ', S[0])

    print()
    if show_param_ind_options:
        show_param_options(S[1])

    if show_all_possible_params:
            possible_params = get_possible_init_params(S[0])
            print('All Possible Params:', possible_params)
    print()
