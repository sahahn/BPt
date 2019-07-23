"""
ML_Helpers.py
====================================
File with various ML helper functions for ABCD_ML.
These are non-class functions that are used in _ML.py and Scoring.py
"""
import numpy as np
import inspect


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
        The standard deviation of the micro score
    '''

    scores = np.array(scores)
    macro_scores = np.mean(np.reshape(scores, (n_repeats, n_splits)), axis=1)

    return (np.mean(macro_scores), np.std(macro_scores),
            np.std(scores))


def proc_input(in_vals):
    '''Performs common preproc on a list of str's or
    a single str.'''

    if isinstance(in_vals, list):
        in_vals = [proc_str_input(x) for x in in_vals]
    else:
        in_vals = proc_str_input(in_vals)

    return in_vals


def proc_str_input(in_str):
    '''Perform common preprocs on a str.
    Speicifcally this function is is used to process user str input,
    as referencing a model_type, metric, or scaler.'''

    if not isinstance(in_str, str):
        return in_str

    in_str = in_str.replace('_', ' ')
    in_str = in_str.lower()
    in_str = in_str.rstrip()

    chunk_replace_dict = {' regressor': '',
                          ' regresure': '',
                          ' classifier': '',
                          ' classifer': ''}

    for chunk in chunk_replace_dict:
        in_str = in_str.replace(chunk, chunk_replace_dict[chunk])

    # This is a dict of of values to replace, if the str ends with that value
    endwith_replace_dict = {' score': '',
                            ' loss': '',
                            ' corrcoef': '',
                            ' ap': ' average precision',
                            ' jac': ' jaccard',
                            ' iou': ' jaccard',
                            ' intersection over union': ' jaccard',
                            }

    for chunk in endwith_replace_dict:
        if in_str.endswith(chunk):
            in_str = in_str.replace(chunk, endwith_replace_dict[chunk])

    startwith_replace_dict = {'rf ': 'random forest ',
                              'lgbm ': 'light gbm ',
                              'svc ': 'svm ',
                              'svr ': 'svm ',
                              }

    for chunk in startwith_replace_dict:
        if in_str.startswith(chunk):
            in_str = in_str.replace(chunk, startwith_replace_dict[chunk])

    # This is a dict where if the input is exactly one
    # of the keys, the value will be replaced.
    replace_dict = {'acc': 'accuracy',
                    'bas': 'balanced accuracy',
                    'ap': 'average precision',
                    'jac': 'jaccard',
                    'iou': 'jaccard',
                    'intersection over union': 'jaccard',
                    'mse': 'mean squared error',
                    'ev': 'explained variance',
                    'mae': 'mean absolute error',
                    'msle': 'mean squared log error',
                    'med ae': 'median absolute error',
                    'rf': 'random forest',
                    'lgbm': 'light gbm',
                    'svc': 'svm',
                    'svr': 'svm',
                    }

    if in_str in replace_dict:
        in_str = replace_dict[in_str]

    return in_str


def get_model_possible_params(model):
    '''Helper function to grab the names of valid arguments to a model

    Parameters
    ----------
    model : sklearn api model reference
        The model object to inspect

    Returns
    ----------
        All valid parameters to the model
    '''
    pos_params = dict(inspect.getmembers(model.__init__.__code__))
    return pos_params['co_varnames']


def get_avaliable_by_type(AVALIABLE):

    avaliable_by_type = {}

    for pt in AVALIABLE:

            avaliable_by_type[pt] = set()

            if pt == 'categorical':
                    for st in AVALIABLE[pt]:
                            for select in AVALIABLE[pt][st]:
                                    avaliable_by_type[pt].add(st + ' ' +
                                                              AVALIABLE[pt]
                                                              [st][select])

            else:
                    for select in AVALIABLE[pt]:
                            avaliable_by_type[pt].add(AVALIABLE[pt][select])

            avaliable_by_type[pt] = list(avaliable_by_type[pt])
            avaliable_by_type[pt].sort()

    return avaliable_by_type
