"""
ML_Helpers.py
====================================
File with various ML helper functions for ABCD_ML.
These are non-class functions that are used in _ML.py and Scoring.py
"""
import numpy as np
from sklearn.preprocessing import (MinMaxScaler, RobustScaler, StandardScaler,
                                   PowerTransformer)
from ABCD_ML.Models import AVALIABLE


def get_scaler(method, extra_params=None):
    '''Returns a scaler based on the method passed,

    Parameters
    ----------
    method : str
        `method` refers to the type of scaling to apply
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

    method_lower = method.lower()
    params = {}

    if method_lower == 'standard':
        scaler = StandardScaler

    elif method_lower == 'minmax':
        scaler = MinMaxScaler

    elif method_lower == 'robust':
        scaler = RobustScaler
        params = {'quantile_range': (5, 95)}

    elif method_lower == 'power':
        scaler = PowerTransformer
        params = {'method': 'yeo-johnson', 'standardize': True}

    # Check to see if user passed in params,
    # otherwise params will remain default.
    if method in extra_params:
        params.update(extra_params[method])

    scaler = scaler(**params)
    return scaler


def compute_macro_micro(scores, n_repeats, n_splits):
    '''Compute and return scores, as computed froma repeated k-fold.

    Parameters
    ----------
    scores : list or array-like
        Should contain all of the scores
        and have a length of `n_repeats` * `n_splits`

    n_repeats : int
        The number of repeats

    n_splits : int
        The number of splits per repeat

    Returns
    ----------
    float
        The mean macro score

    float
        The standard deviation of the macro score

    float
        The mean micro score

    float
        The standard deviation of the micro score
    '''

    scores = np.array(scores)
    macro_scores = np.mean(np.reshape(scores, (n_repeats, n_splits)), axis=1)

    return (np.mean(macro_scores), np.std(macro_scores),
            np.mean(scores), np.std(scores))


def proc_input(in_vals):
    '''Performs common preproc on a list of str's or
    a single str.'''

    if isinstance(in_vals, list):
        in_vals = [proc_str_input(x) for x in in_vals]
    else:
        in_vals = proc_str_input(in_vals)

    return in_vals


def proc_str_input(in_str):
    '''Perform common preprocs on a str.'''

    in_str = in_str.replace('_', ' ')
    in_str = in_str.lower()

    return in_str


