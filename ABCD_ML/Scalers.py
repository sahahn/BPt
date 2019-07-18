"""
Scalers.py
====================================
File containing the various input data scalers.
"""
from sklearn.preprocessing import (MinMaxScaler, RobustScaler, StandardScaler,
                                   PowerTransformer)

# Scalers differs from scorers and models in that the types are not restricted
# by a given problem type. Therefore no AVALIABLE dictionary is neccisary,
# all avaliable scalers are as indexed by SCALERS.
SCALERS = {
    'standard': (StandardScaler, {'with_mean': True, 'with_std': True}),

    'minmax': (MinMaxScaler, {'feature_range': (0, 1)}),

    'robust': (RobustScaler, {'quantile_range': (5, 95)}),

    'power': (PowerTransformer, {'method': 'yeo-johnson', 'standardize': True})
    }


def get_data_scaler(scaler_str, extra_params):
    '''Returns a scaler based on proced str indicator input,

    Parameters
    ----------
    scaler_str : str
        `scaler_str` refers to the type of scaling to apply
        to the saved data during model evaluation.
        For a full list of supported options call:
        self.show_data_scalers()

    extra_params : dict, optional
        Any extra params being passed.
        These can be supplied by creating another dict within extra_params.
        E.g., extra_params[method] = {'method param' : new_value}
        Where method param is a valid argument for that method,
        and method in this case is the str indicator.
        (default = {})

    Returns
    ----------
    scaler
        A scaler object with fit and transform methods.
    '''

    try:
        scaler, params = SCALERS[scaler_str]
    except KeyError:
        print('Requested scaler with str indicator', scaler_str,
              'does not exist!')

    # Update with extra params if applicable
    if scaler_str in extra_params:
        params.update(extra_params[scaler_str])

    return scaler(**params)
