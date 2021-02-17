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
from ..main.input_operations import is_special, Select
from joblib import hash as joblib_hash


def compute_micro_macro(scores, n_repeats, n_splits, weights=None):
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
        The standard deviation of the micro score

    float
        The standard deviation of the macro score
    '''

    r_scores = np.reshape(np.array(scores), (n_repeats, n_splits))

    if weights is None:
        macro_scores = np.mean(r_scores, axis=1)
    else:
        r_weights = np.reshape(np.array(weights), (n_repeats, n_splits))
        macro_scores = np.average(r_scores, weights=r_weights, axis=1)

    return (np.mean(macro_scores), np.std(scores), np.std(macro_scores))


def is_array_like(in_val):

    if hasattr(in_val, '__len__') and (not isinstance(in_val, str)) and \
     (not isinstance(in_val, dict)) and (not hasattr(in_val, 'fit')) and \
     (not hasattr(in_val, 'transform')):
        return True
    else:
        return False


def conv_to_list(in_val, amt=1):

    if in_val is None:
        return None

    if not is_array_like(in_val) or is_special(in_val):
        in_val = [in_val for i in range(amt)]

    return in_val


def proc_input(in_vals):
    '''Performs common preproc on a list of str's or
    a single str.'''

    if isinstance(in_vals, list):
        for i in range(len(in_vals)):
            in_vals[i] = proc_str_input(in_vals[i])
    else:
        in_vals = proc_str_input(in_vals)

    return in_vals


def proc_str_input(in_str):

    if not isinstance(in_str, str):
        return in_str

    # Make sure lower-case
    in_str = in_str.lower()

    # Remove regressor or classifier
    chunk_replace_dict = {' regressor': '',
                          ' classifier': ''}
    for chunk in chunk_replace_dict:
        in_str = in_str.replace(chunk, chunk_replace_dict[chunk])

    return in_str


def user_passed_param_check(params, obj_str, search_type):

    if isinstance(params, dict):
        if search_type is None:
            return deepcopy(params), {}
        else:
            return {}, deepcopy(proc_params(params, prepend=obj_str))
    return {}, {}


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
        process_params_by_type(obj, obj_str, base_params, extra_params)

    return obj, non_search_params, params


def process_params_by_type(obj, obj_str, base_params, extra_params):
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
    pos_params = dict(inspect.getmembers(model.fit.__code__))
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


def param_len_check(names, params, _print=print):

    if isinstance(params, dict) and len(names) == 1:
        return params

    try:

        if len(params) > len(names):
            _print('Warning! More params passed than objs')
            _print('Extra params have been truncated.')
            return params[:len(names)]

    # If non list params here
    except TypeError:
        return [0 for i in range(len(names))]

    while len(names) != len(params):
        params.append(0)

    return params


def replace_model_name(base_estimator_params):

    new = {}

    for key in base_estimator_params:
        value = base_estimator_params[key]

        split_key = key.split('__')
        split_key[0] = 'estimator'

        new_key = '__'.join(split_key)
        new[new_key] = value

    return new


def get_avaliable_run_name(name, model_pipeline):

    if hasattr(model_pipeline, 'model'):
        model = model_pipeline.model
    else:
        model = model_pipeline

    if name is None or name == 'default':

        if isinstance(model, Select):
            name = 'select'
        elif isinstance(model, list):
            name = 'special'
        elif hasattr(model, 'obj'):
            if isinstance(model.obj, str):
                name = model.obj
            else:
                name = 'Custom'
        else:
            name = 'Custom'

    return name

def set_n_jobs(obj, n_jobs):

    # Call recursively for list
    if isinstance(obj, list) or isinstance(obj, tuple):
        for o in obj:
            set_n_jobs(o, n_jobs)

    # Check and set for n_jobs
    if hasattr(obj, 'n_jobs'):
        setattr(obj, 'n_jobs', n_jobs)
