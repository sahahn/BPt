from copy import deepcopy
import numpy as np
import inspect
from ..util import is_array_like


def args_repr(args, kwargs):

    reprs = [repr(arg) for arg in args]
    sorted_keys = sorted(list(kwargs))
    reprs += [key + '=' + repr(kwargs[key]) for key in sorted_keys]
    return ', '.join(reprs)


def get_obj_and_params(obj_str, OBJS, extra_params, params):

    from .params.default_params import get_base_params

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

    from .params.Params import Params
    from .params.default_params import proc_params

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


def proc_extra_params(extra_params, non_search_params, params=None):

    if extra_params is None or len(extra_params) == 0:
        return non_search_params, params

    for key in extra_params:
        non_search_params[key] = deepcopy(extra_params[key])

        # Override value w/ extra params if also in params
        if params is not None and key in params:
            del params[key]

    return non_search_params, params


def proc_type_dep_str(in_strs, avaliable, problem_type):
    '''Helper function to perform str correction on
    underlying problem type dependent input, e.g., for
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


def get_a_by_type(avaliable, in_strs, problem_type):

    avaliable_by_type = avaliable[problem_type]

    for s in in_strs:
        if 'Custom ' in s:
            avaliable_by_type[s] = s

    return avaliable_by_type


def check_avaliable(in_strs, avaliable, problem_type):

    avaliable_by_type = get_a_by_type(avaliable, in_strs, problem_type)

    check = np.array([m in avaliable_by_type for
                      m in in_strs]).all()

    return check


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
                          ' classifier': '',
                          '_regressor': '',
                          '_classifier': ''}

    for chunk in chunk_replace_dict:
        in_str = in_str.replace(chunk, chunk_replace_dict[chunk])

    return in_str


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


def all_from_avaliable(avaliable):

    a = set()
    for pt in avaliable:
        for key in avaliable[pt]:
            a.add(key)

    return a


def all_from_objects(objects):

    a = set()
    for key in objects:
        a.add(key)

    return a


def coarse_any_obj_check(in_str):
    from .options.all_keys import get_all_keys

    all_keys = get_all_keys()

    # Get proc str
    proc_str = proc_str_input(in_str)

    if proc_str not in all_keys:
        raise RuntimeError('Passed obj=' + str(in_str) + ' does not '
                           'correspond '
                           'to any avaliable default options from BPt!')
