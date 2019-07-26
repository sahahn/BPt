"""
Scalers.py
====================================
File containing the various input data scalers.
"""
from ABCD_ML.ML_Helpers import get_obj_and_params
from sklearn.preprocessing import (MinMaxScaler, RobustScaler, StandardScaler,
                                   PowerTransformer)
from sklearn.decomposition import PCA


# Scalers differs from scorers and models in that the types are not restricted
# by a given problem type. Therefore no AVALIABLE dictionary is neccisary,
# all avaliable scalers are as indexed by SCALERS.
SCALERS = {
    'standard': (StandardScaler, ['base standard']),

    'minmax': (MinMaxScaler, ['base minmax']),

    'robust': (RobustScaler, ['base robust']),

    'power': (PowerTransformer, ['base power']),

    'pca': (PCA, ['base pca'])
    }


def get_data_scaler(data_scaler_str, extra_params, param_ind):
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


def Show_Scalers(self, show_scaler_help=False, show_default_params=False):
    '''Just calls Show_Data_Scalers'''

    self.Show_Data_Scalers()


def Show_Data_Scalers(self, show_scaler_help=False, show_default_params=False):
    '''Print out the avaliable data scalers.

    Parameters
    ----------
    show_scaler_help : bool, optional
        Flag, if set to True, then will display the full docstring
        for each scaler.

        (default = False)

    show_default_params : bool, optional
        Flag, if set to True, then will display the ABCD_ML
        default parameters for each scaler.

        (default = False)'''

    print('Visit: ')
    print('https://scikit-learn.org/stable/modules/preprocessing.html')
    print('For more detailed information on different scalers',
          '/ preprocessing.')

    for scaler in SCALERS:
        print('str indicator: ', scaler)

        S = SCALERS[scaler]
        print('Scaler object: ', S[0])

        if show_scaler_help:
            print(help(S[0]))
            print()

        if show_default_params:
                print('Default Params: ')

                for p in S[1]:
                    print(p, ':', S[1][p])

        print()
