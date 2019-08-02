"""
Scalers.py
====================================
File containing the various input data scalers.
"""
from ABCD_ML.ML_Helpers import (show_param_options, get_possible_init_params,
                                get_obj_and_params)
from sklearn.preprocessing import (MinMaxScaler, RobustScaler, StandardScaler,
                                   PowerTransformer)
from sklearn.decomposition import PCA


# Scalers differs from scorers and models in that the types are not restricted
# by a given problem type. Therefore no AVALIABLE dictionary is neccisary,
# all avaliable scalers are as indexed by SCALERS.
SCALERS = {
    'standard': (StandardScaler, ['base standard']),

    'minmax': (MinMaxScaler, ['base minmax']),

    'robust': (RobustScaler, ['base robust', 'robust gs']),

    'power': (PowerTransformer, ['base power']),

    'pca': (PCA, ['base pca', 'pca rs'])
    }


def get_data_scaler_and_params(data_scaler_str, extra_params, param_ind):
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
        get_obj_and_params(data_scaler_str, SCALERS, extra_params, param_ind)

    return data_scaler(**extra_data_scaler_params), data_scaler_params


def Show_Scalers(self, data_scaler=None, show_param_ind_options=True,
                 show_scaler_object=False,
                 show_all_possible_params=False):
    '''Just calls Show_Data_Scalers'''

    self.Show_Data_Scalers(data_scaler, show_param_ind_options,
                           show_scaler_object, show_all_possible_params)


def Show_Data_Scalers(self, data_scaler=None, show_param_ind_options=True,
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

        (default = True)

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
