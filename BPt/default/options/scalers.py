from ..helpers import get_obj_and_params, all_from_objects
from sklearn.preprocessing import (MinMaxScaler, RobustScaler, StandardScaler,
                                   PowerTransformer, MaxAbsScaler,
                                   QuantileTransformer, Normalizer)
from ...extensions.scalers import Winsorizer


# Scalers differs from models in that the types are not restricted
# by a given problem type. Therefore no AVAILABLE dictionary is needed
# all available scalers are as indexed by SCALERS.
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


# @TODO find way to re-use these default ones
def get_scaler_and_params(scaler_str, extra_params, params,
                          **kwargs):
    '''Returns a scaler based on proc'ed str indicator input,

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
        get_obj_and_params(scaler_str, SCALERS, extra_params, params)

    return scaler(**extra_scaler_params), scaler_params


all_obj_keys = all_from_objects(SCALERS)
