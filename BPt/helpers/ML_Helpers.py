"""
ML_Helpers.py
====================================
File with various ML helper functions for BPt.
These are non-class functions that are used in _ML.py and Scoring.py
"""
import numpy as np
import inspect
from .Default_Params import get_base_params, proc_params
from copy import deepcopy
import nevergrad as ng
from ..main.Input_Tools import is_special, Select
from nevergrad.parametrization.core import Constant


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


def proc_extra_params(obj, extra_params, non_search_params, params=None):

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

        try:
            module = params[p].__module__

            # If has a module, but no nevergrad in path, then not a
            # nevergrad param
            if 'nevergrad' not in module:
                non_search_params[p] = params.pop(p)

        # If no module, then not a nevergrad param
        except AttributeError:
            non_search_params[p] = params.pop(p)

    # Append obj_str to all params still in params
    # as these are the search params
    params = proc_params(params, prepend=obj_str)

    # Process extra params
    non_search_params, params =\
        proc_extra_params(obj,
                          extra_params=extra_params,
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


def type_check(ud):
    '''Check if a nevergrad dist'''

    def_dist = [ng.p.Log, ng.p.Scalar, ng.p.Choice, ng.p.TransitionChoice]
    for dd in def_dist:
        if isinstance(ud, dd):
            return True

    types_to_check = [int, float, list, tuple, str, bool, dict, set, Constant]

    for ttc in types_to_check:
        if isinstance(ud, ttc):
            return False

    return True


def proc_mapping(indx, mapping):

    if len(mapping) > 0 and len(indx) > 0:

        # If should proc list...
        if is_array_like(indx[0]):
            return [proc_mapping(i, mapping) for i in indx]

        else:
            new_indx = set()

            for i in indx:
                new = mapping[i]

                if new is None:
                    pass

                # If mapping points to a list of values
                elif isinstance(new, list):
                    for n in new:
                        if n is not None:
                            new_indx.add(n)
                else:
                    new_indx.add(new)

            # Sort, then return
            new_indx = sorted(list(new_indx))
            return new_indx

    else:
        return indx


def update_mapping(mapping, new_mapping):

    # Go through the mapping and update each key with the new mapping
    for key in mapping:

        val = mapping[key]

        if isinstance(val, list):

            new_vals = []
            for v in val:

                if v in new_mapping:

                    new_val = new_mapping[v]
                    if isinstance(new_val, list):
                        new_vals += new_val
                    else:
                        new_vals.append(new_val)

                else:
                    new_vals.append(v)

            as_set = set(new_vals)

            try:
                as_set.remove(None)
            except KeyError:
                pass

            mapping[key] = sorted(list(as_set))

        # Assume int if not list
        else:

            if val in new_mapping:
                mapping[key] = new_mapping[val]


def wrap_pipeline_objs(wrapper, objs, inds, random_state,
                       n_jobs, fix_n_wrapper_jobs, **params):

    # If passed wrapper n_jobs, and != 1, set base obj jobs to 1
    if 'wrapper_n_jobs' in params:
        if params['wrapper_n_jobs'] != 1:
            n_jobs = 1

    # If passed cache locs
    if 'cache_locs' in params:
        cache_locs = params.pop('cache_locs')
    else:
        cache_locs = [None for i in range(len(objs))]

    wrapped_objs = []
    for chunk, ind, cache_loc, fix_n_wrapper_job in zip(objs, inds,
                                                        cache_locs,
                                                        fix_n_wrapper_jobs):

        # Unpack
        name, obj = chunk

        # Pass attributes
        if hasattr(obj, 'n_jobs'):
            setattr(obj, 'n_jobs', n_jobs)

        if hasattr(obj, 'random_state'):
            setattr(obj, 'random_state', random_state)

        wrapped_obj = wrapper(obj, ind,
                              cache_loc=cache_loc,
                              fix_n_wrapper_jobs=fix_n_wrapper_job,
                              **params)
        wrapped_objs.append((name, wrapped_obj))

    return wrapped_objs


def check_for_duplicate_names(objs_and_params):
    '''Checks for duplicate names within an objs_and_params type obj'''

    names = [c[0] for c in objs_and_params]

    # If any repeats
    if len(names) != len(set(names)):
        new_objs_and_params = []

        for obj in objs_and_params:
            name = obj[0]

            if name in names:

                cnt = 0
                used = [c[0] for c in new_objs_and_params]
                while name + str(cnt) in used:
                    cnt += 1

                # Need to change name within params also
                base_obj = obj[1][0]
                base_obj_params = obj[1][1]

                new_obj_params = {}
                for param_name in base_obj_params:

                    p_split = param_name.split('__')
                    new_param_name = p_split[0] + str(cnt)
                    new_param_name += '__' + '__'.join(p_split[1:])

                    new_obj_params[new_param_name] =\
                        base_obj_params[param_name]

                new_objs_and_params.append((name + str(cnt),
                                           (base_obj, new_obj_params)))

            else:
                new_objs_and_params.append(obj)

        return new_objs_and_params
    return objs_and_params


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


def get_reverse_mapping(mapping):

    reverse_mapping = {}
    for m in mapping:
        key = mapping[m]

        if isinstance(key, list):
            for k in key:
                reverse_mapping[k] = m
        else:
            reverse_mapping[key] = m

    return reverse_mapping


def set_n_jobs(obj, n_jobs):

    # Call recursively for list
    if isinstance(obj, list) or isinstance(obj, tuple):
        for o in obj:
            set_n_jobs(o, n_jobs)

    # Check and set for n_jobs
    if hasattr(obj, 'n_jobs'):
        setattr(obj, 'n_jobs', n_jobs)

    # Also check for wrapper_n_jobs
    if hasattr(obj, 'wrapper_n_jobs'):
        setattr(obj, 'wrapper_n_jobs', n_jobs)
