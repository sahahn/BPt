"""
ML_Helpers.py
====================================
File with various ML helper functions for ABCD_ML.
These are non-class functions that are used in _ML.py and Scoring.py
"""
import numpy as np
import inspect
from .Default_Params import get_base_params, proc_params, show
import subprocess
from copy import deepcopy


def compute_macro_micro(scores, n_repeats, n_splits, weights=None):
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

    r_scores = np.reshape(np.array(scores), (n_repeats, n_splits))

    if weights is None:
        macro_scores = np.mean(r_scores, axis=1)
    else:
        r_weights = np.reshape(np.array(weights), (n_repeats, n_splits))
        macro_scores = np.average(r_scores, weights=r_weights, axis=1)

    return (np.mean(macro_scores), np.std(macro_scores),
            np.std(scores))


def is_array_like(in_val):

    if hasattr(in_val, '__len__') and (not isinstance(in_val, str)) and \
     (not isinstance(in_val, dict)) and (not hasattr(in_val, 'fit')):
        return True
    else:
        return False


def conv_to_list(in_val, amt=1):

    if in_val is None:
        return None

    if not is_array_like(in_val):
        in_val = [in_val for i in range(amt)]

    return in_val


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
    as referencing a model, metric, or scaler.'''

    if not isinstance(in_str, str):
        return in_str

    in_str = in_str.replace('_', ' ')
    in_str = in_str.lower()
    in_str = in_str.rstrip()

    chunk_replace_dict = {' regressor': '',
                          ' regresure': '',
                          ' classifier': '',
                          ' classifer': '',
                          ' classification': ''}

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
                            ' logistic': '',
                            }

    for chunk in endwith_replace_dict:
        if in_str.endswith(chunk):
            in_str = in_str.replace(chunk, endwith_replace_dict[chunk])

    startwith_replace_dict = {'rf ': 'random forest ',
                              'lgbm ': 'light gbm ',
                              'lightgbm ': 'light gbm ',
                              'svc ': 'svm ',
                              'svr ': 'svm ',
                              'neg ': '',
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


def user_passed_param_check(params, obj_str, search_type):

    if isinstance(params, dict):
        if search_type is None:
            return deepcopy(params), {}
        else:
            return {}, deepcopy(proc_params(params, prepend=obj_str))
    return {}, {}


def get_obj_and_params(obj_str, OBJS, extra_params, params, search_type):

    try:
        obj, param_names = OBJS[obj_str]
    except KeyError:
        print('Requested:', obj_str, 'does not exist!')

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

    # Special case if search type None, convert param grid to
    # be one set of params
    if search_type is None:

        params = base_params.copy()

        if obj_str in extra_params:
            params.update(extra_params[obj_str])

        return obj, params, {}

    # Otherwise, prepend obj_str to all keys in base params
    params = proc_params(base_params, prepend=obj_str)

    # Update with extra params if applicable
    extra_obj_params = {}
    if obj_str in extra_params:
        extra_obj_params = extra_params[obj_str]

        # If any user passed args, and also in param grid, remove.
        for ex_param in extra_obj_params:
            key = obj_str + '__' + ex_param

            if key in params:
                del params[key]

    return obj, extra_obj_params, params


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


def show_objects(problem_type=None, obj=None,
                 show_params_options=True, show_object=False,
                 show_all_possible_params=False, AVALIABLE=None, OBJS=None):

        if obj is not None:
            objs = conv_to_list(obj)

            for obj in objs:
                show_obj(obj, show_params_options, show_object,
                         show_all_possible_params, OBJS)
            return

        if AVALIABLE is not None:

            avaliable_by_type = get_avaliable_by_type(AVALIABLE)
            problem_types = proc_problem_type(problem_type, avaliable_by_type)

            for pt in problem_types:
                    show_type(pt, avaliable_by_type,
                              show_params_options,
                              show_object,
                              show_all_possible_params, OBJS)

        else:

            for obj in OBJS:
                show_obj(obj, show_params_options, show_object,
                         show_all_possible_params, OBJS)


def show_type(problem_type, avaliable_by_type, show_params_options,
              show_object, show_all_possible_params, OBJS):

        print('Avaliable for Problem Type:', problem_type)
        print('----------------------------------------')
        print()
        print()

        for obj_str in avaliable_by_type[problem_type]:

            if 'basic ensemble' in obj_str:

                print('- - - - - - - - - - - - - - - - - - - - ')
                print('("basic ensemble")')
                print('- - - - - - - - - - - - - - - - - - - - ')
                print()

            elif 'user passed' not in obj_str:
                show_obj(obj_str, show_params_options, show_object,
                         show_all_possible_params, OBJS)


def show_obj(obj_str, show_params_options, show_object,
             show_all_possible_params, OBJS):

        print('- - - - - - - - - - - - - - - - - - - - ')
        OBJ = OBJS[obj_str]
        print(OBJ[0].__name__, end='')
        print(' ("', obj_str, '")', sep='')
        print('- - - - - - - - - - - - - - - - - - - - ')
        print()

        if show_object:
            print('Object: ', OBJ[0])

        print()
        if show_params_options:
            show_param_options(OBJ[1])

        if show_all_possible_params:
            possible_params = get_possible_init_params(OBJ[0])
            print('All Possible Params:', possible_params)
        print()


def show_param_options(param_options):

    print('Param Indices')
    print('-------------')

    for ind in range(len(param_options)):

        print()
        print(ind, ":", sep='')
        print()
        print('"', param_options[ind], '"', sep='')
        show(param_options[ind])
        print()
    print('-------------')


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

    types_to_check = [int, float, list, tuple, str, bool, dict, set]

    for ttc in types_to_check:
        if isinstance(ud, ttc):
            return False

    return True


def mem_check():

    process = subprocess.Popen('free', stdout=subprocess.PIPE)
    output, error = process.communicate()
    output = output.decode("utf-8")

    output = output.split('\n')

    cats = ['total', 'used', 'free', 'shared', 'buff/cache', 'available']

    mem = [o for o in output[1].split(' ') if len(o) > 0][1:]
    swap = [o for o in output[2].split(' ') if len(o) > 0][1:]

    print('mem')
    for m, c in zip(mem, cats):
        print(c + ':', m)
    print()
    print('swap')
    for m, c in zip(swap, cats):
        print(c + ':', m)


def proc_mapping(indx, mapping):
    
    if len(mapping) > 0 and len(indx) > 0:

        # If should proc list... 
        if is_array_like(indx[0]):
            return [proc_mapping(i, mapping) for i in indx]

        else:
            new_indx = set()
            
            for i in indx:
                new = mapping[i]

                # If mapping points to a list of values
                if isinstance(new, list):
                    for n in new:
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
                    
            mapping[key] = sorted(list(set(new_vals)))

        # Assume int if not list
        else:

            if val in new_mapping:
                mapping[key] = new_mapping[val]


def wrap_pipeline_objs(wrapper, objs, inds, random_state,
                       n_jobs, **params):

    # If passed wrapper n_jobs, and != 1, set base obj jobs to 1
    if 'wrapper_n_jobs' in params:
        if params['wrapper_n_jobs'] != 1:
            n_jobs = 1

    wrapped_objs = []
    for chunk, ind in zip(objs, inds):

        name, obj = chunk

        # Try to set attributes
        try:
            obj.n_jobs = n_jobs
        except AttributeError:
            pass

        try:
            obj.random_state = random_state
        except AttributeError:
            pass

        wrapped_obj = wrapper(obj, ind, **params)
        wrapped_objs.append((name, wrapped_obj))

    return wrapped_objs


def update_extra_params(extra_params, orig_strs, conv_strs):
    '''Helper method to update extra params in the case
    where model_types or scaler str indicators change,
    and they were refered to in extra params as the original name.

    Parameters
    ----------
    orig_strs : list
        List of original str indicators.

    conv_strs : list
        List of final-proccesed str indicators, indices should
        correspond to the order of orig_strs
    '''

    for i in range(len(orig_strs)):
        if orig_strs[i] in extra_params:
            extra_params[conv_strs[i]] =\
                extra_params[orig_strs[i]]

    return extra_params


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


def proc_type_dep_str(in_strs, avaliable, extra_params, problem_type):
    '''Helper function to perform str correction on
    underlying proble type dependent input, e.g., for
    metric or ensemble_types, and to update extra params
    and check to make sure input is valid ect...'''

    conv_strs = proc_input(in_strs)

    if not check_avaliable(conv_strs, avaliable, problem_type):
        raise RuntimeError(conv_strs, 'are not avaliable for this problem type')

    avaliable_by_type = get_a_by_type(avaliable, conv_strs, problem_type)
    final_strs = [avaliable_by_type[conv_str] for conv_str in conv_strs]

    extra_params = update_extra_params(extra_params, in_strs, final_strs)
    
    return final_strs, extra_params

def check_avaliable(in_strs, avaliable, problem_type):

    avaliable_by_type = get_a_by_type(avaliable, in_strs, problem_type)

    check = np.array([m in avaliable_by_type for
                      m in in_strs]).all()

    return check


def get_a_by_type(avaliable, in_strs, problem_type):

    avaliable_by_type = avaliable[problem_type]

    for s in in_strs:
        if 'user passed' in s:
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

