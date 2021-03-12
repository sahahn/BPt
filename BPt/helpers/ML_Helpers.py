"""
ML_Helpers.py
====================================
File with various ML helper functions for BPt.
These are non-class functions that are used in _ML.py and Scoring.py
"""
import numpy as np
import inspect

from ..default.params.default_params import get_base_params, proc_params
from ..default.params.Params import Params
from copy import deepcopy
from ..main.input_operations import Select
from joblib import hash as joblib_hash
from ..util import is_array_like


def proc_extra_params(extra_params, non_search_params, params=None):

    if extra_params is None or len(extra_params) == 0:
        return non_search_params, params

    for key in extra_params:
        non_search_params[key] = deepcopy(extra_params[key])

        # Override value w/ extra params if also in params
        if params is not None and key in params:
            del params[key]

    return non_search_params, params


def get_obj_and_params(obj_str, OBJS, extra_params, params):

    # First get the object, and process the base params!
    try:
        obj, param_names = OBJS[obj_str]
    except KeyError:
        raise KeyError(repr(obj_str) + ' does not exist!')

    # If params is a str, change it to the relevant index
    if isinstance(params, str):
        try:
            params = param_names.index(params)
        except ValueError:
            print('str', params, 'passed, but not found as an option for',
                  obj_str)
            print('Setting to default base params setting instead!')
            params = 0

    # If passed param ind is a dict, assume that user passed
    if isinstance(params, dict):
        base_params = params.copy()

    # If not a dict passed, grab the param name, then params
    else:

        # Get the actual params
        try:
            param_name = param_names[params]
        except IndexError:
            print('Invalid param ind', params, 'passed for', obj_str)
            print('There are only', len(param_names), 'valid param options.')
            print('Setting to default base params setting instead!')
            param_name = param_names[0]

        base_params = get_base_params(param_name)

    # Process rest of params by search type, and w.r.t to extra params
    non_search_params, params =\
        process_params_by_type(obj_str, base_params, extra_params)

    return obj, non_search_params, params


def process_params_by_type(obj_str, base_params, extra_params):
    '''base params is either a dict or 0'''

    non_search_params, params = {}, {}

    # Start with params as copy of passed base_params
    params = deepcopy(base_params)

    # If params is passed as 0, treat as no search params
    if params == 0:
        params = {}

    # Extract any non-search params from params
    param_keys = list(params)
    for p in param_keys:

        # If not a special search param
        if not isinstance(params[p], Params):
            non_search_params[p] = params.pop(p)

    # Append obj_str to all params still in params
    # as these are the search params
    params = proc_params(params, prepend=obj_str)

    # Process extra params
    non_search_params, params =\
        proc_extra_params(extra_params=extra_params,
                          non_search_params=non_search_params,
                          params=params)

    return non_search_params, params


def get_possible_init_params(model):
    '''Helper function to grab the names of valid arguments to
    classes init

    Parameters
    ----------
    model : object
        The object to inspect

    Returns
    ----------
        All valid parameters to the model
    '''

    try:
        return model._get_param_names()
    except AttributeError:
        pos_params = dict(inspect.getmembers(model.__init__.__code__))
        return pos_params['co_varnames']


def get_possible_params(estimator, method):

    if not hasattr(estimator, method):
        return []

    pos_params = dict(inspect.getmembers(getattr(estimator, method).__code__))
    return pos_params['co_varnames']


def get_possible_fit_params(model):
    '''Helper function to grab the names of valid arguments to
    classes fit method

    Parameters
    ----------
    model : object w/ fit method
        The model object to inspect

    Returns
    ----------
        All valid parameters to the model
    '''
    if not hasattr(model, 'fit'):
        return []

    pos_params = dict(inspect.getmembers(model.fit.__code__))
    return pos_params['co_varnames']


def get_possible_trans_params(model):
    '''Helper function to grab the names of valid arguments to
    classes transform method

    Parameters
    ----------
    model : object w/ fit method
        The model object to inspect

    Returns
    ----------
        All valid parameters to the model
    '''

    if not hasattr(model, 'transform'):
        return []

    pos_params = dict(inspect.getmembers(model.transform.__code__))
    return pos_params['co_varnames']


def get_avaliable_by_type(AVALIABLE):

    avaliable_by_type = {}

    for pt in AVALIABLE:

        avaliable_by_type[pt] = set()
        for select in AVALIABLE[pt]:
            avaliable_by_type[pt].add(AVALIABLE[pt][select])

        avaliable_by_type[pt] = list(avaliable_by_type[pt])
        avaliable_by_type[pt].sort()

    return avaliable_by_type


def get_objects_by_type(problem_type, AVALIABLE=None, OBJS=None):

    avaliable_by_type = get_avaliable_by_type(AVALIABLE)

    objs = []
    for obj_str in avaliable_by_type[problem_type]:

        if 'basic ensemble' not in obj_str:
            obj = OBJS[obj_str][0]
            obj_params = OBJS[obj_str][1]
            objs.append((obj_str, obj, obj_params))

    return objs


def get_objects(OBJS):

    objs = []
    for obj_str in OBJS:

        obj = OBJS[obj_str][0]
        obj_params = OBJS[obj_str][1]
        objs.append((obj_str, obj, obj_params))

    return objs


def proc_problem_type(problem_type, avaliable_by_type):

    if problem_type is not None:
        problem_types = [problem_type]

    else:
        problem_types = list(avaliable_by_type)

    return problem_types


def f_array(in_array):
    return np.array(in_array).astype(float)


def find_ind(X, base_X_mask, X_r, r_ind, mask=True):

    r_dtype = X_r.dtype
    o_dtype = X.dtype

    if r_dtype != o_dtype and mask:
        ind = np.where(np.all(X[:, base_X_mask].astype(r_dtype) == X_r[r_ind],
                       axis=1))

    elif r_dtype != o_dtype:
        ind = np.where(np.all(X.astype(r_dtype) == X_r[r_ind], axis=1))

    elif mask:
        ind = np.where(np.all(X[:, base_X_mask] == X_r[r_ind], axis=1))

    else:
        ind = np.where(np.all(X == X_r[r_ind], axis=1))

    try:
        return ind[0][0]
    except IndexError:
        return None


def replace_with_in_params(params, original, replace):

    new_params = {}

    for key in params:
        new_params[key.replace(original, replace)] = params[key]

    return new_params


def check_replace(objs):

    if isinstance(objs, list):
        return [check_replace(o) for o in objs]

    if isinstance(objs, set):
        new_set = set()
        for o in objs:
            new_set.add(check_replace(o))
        return new_set

    if isinstance(objs, tuple):
        return tuple([check_replace(o) for o in objs])

    if isinstance(objs, dict):
        return {k: check_replace(objs[k]) for k in objs}

    if hasattr(objs, 'get_params'):
        for param in objs.get_params(deep=False):
            new_value = check_replace(getattr(objs, param))
            setattr(objs, param, new_value)

        # Also if has n_jobs replace all with fixed 1
        if hasattr(objs, 'n_jobs'):
            setattr(objs, 'n_jobs', 1)
        if hasattr(objs, 'fix_n_jobs'):
            setattr(objs, 'fix_n_jobs', 1)
        if hasattr(objs, '_n_jobs'):
            try:
                setattr(objs, '_n_jobs', 1)
            except AttributeError:
                pass
        if hasattr(objs, 'n_jobs_'):
            try:
                setattr(objs, 'n_jobs_', 1)
            except AttributeError:
                pass

        # Return objs as changed in place
        return objs

    # If nevergrad / params convert to repr
    if isinstance(objs, Params):
        return repr(objs)

    # Return identity otherwise
    return objs


def hash(objs, steps):
    '''Expects a list'''

    # Make copy with nevergrad / params dists replaced by repr
    hash_steps = check_replace(deepcopy(steps))

    # Hash steps and objs seperate, then combine
    hash_str1 = joblib_hash(objs, hash_name='md5')
    hash_str2 = joblib_hash(hash_steps, hash_name='md5')

    return hash_str1 + hash_str2


def proc_type_dep_str(in_strs, avaliable, problem_type):
    '''Helper function to perform str correction on
    underlying proble type dependent input, e.g., for
    ensemble_types, and to update extra params
    and check to make sure input is valid ect...'''

    as_arr = True
    if not is_array_like(in_strs):
        as_arr = False
        in_strs = [in_strs]

    if not check_avaliable(in_strs, avaliable, problem_type):
        in_strs = proc_input(in_strs)

        if not check_avaliable(in_strs, avaliable, problem_type):
            raise RuntimeError(in_strs, 'are not avaliable for '
                               'this problem type.'
                               'This may be due to the requested object '
                               'being an optional dependency! Check to make '
                               'sure you have the relevant python library '
                               'installed '
                               'and that the passed str contains no typos!')

    avaliable_by_type = get_a_by_type(avaliable, in_strs, problem_type)
    final_strs = [avaliable_by_type[in_str] for in_str in in_strs]

    if as_arr:
        return final_strs
    return final_strs[0]


def check_avaliable(in_strs, avaliable, problem_type):

    avaliable_by_type = get_a_by_type(avaliable, in_strs, problem_type)

    check = np.array([m in avaliable_by_type for
                      m in in_strs]).all()

    return check


def get_a_by_type(avaliable, in_strs, problem_type):

    avaliable_by_type = avaliable[problem_type]

    for s in in_strs:
        if 'Custom ' in s:
            avaliable_by_type[s] = s

    return avaliable_by_type
