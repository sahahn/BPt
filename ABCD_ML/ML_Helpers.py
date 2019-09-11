"""
ML_Helpers.py
====================================
File with various ML helper functions for ABCD_ML.
These are non-class functions that are used in _ML.py and Scoring.py
"""
import numpy as np
import inspect
from ABCD_ML.Default_Params import get_base_params, proc_params, show


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


def is_array_like(in_val):

    if hasattr(in_val, '__len__') and (not isinstance(in_val, str)) and \
     (not isinstance(in_val, dict)) and (not hasattr(in_val, 'fit')):
        return True
    else:
        return False


def conv_to_list(in_val):

    if in_val is None:
        return None

    if not is_array_like(in_val):
        in_val = [in_val]

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
    as referencing a model_type, metric, or scaler.'''

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


def user_passed_param_check(params, obj_str):

    if isinstance(params, dict):
        return proc_params(params, prepend=obj_str)
    return {}


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

    # If passed param ind is a dict, assume that a grid of params passed
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
        params = proc_params(base_params)

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

            if pt == 'categorical':
                    for st in AVALIABLE[pt]:
                            avaliable_by_type[pt + ' ' + st] = set()

                            key = pt + ' ' + st

                            for select in AVALIABLE[pt][st]:
                                    avaliable_by_type[key].add(
                                        AVALIABLE[pt][st][select])

                            avaliable_by_type[key] =\
                                list(avaliable_by_type[key])
                            avaliable_by_type[key].sort()

            else:
                    avaliable_by_type[pt] = set()
                    for select in AVALIABLE[pt]:
                            avaliable_by_type[pt].add(AVALIABLE[pt][select])

                    avaliable_by_type[pt] = list(avaliable_by_type[pt])
                    avaliable_by_type[pt].sort()

    return avaliable_by_type


def get_objects_by_type(problem_type, AVALIABLE=None, OBJS=None):
    '''problem_type must be binary, regression or categorical multilabel or
    categorical multiclass'''

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
        if problem_type == 'categorical':
            problem_types = ['categorical multilabel',
                             'categorical multiclass']

        else:
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
        O = OBJS[obj_str]
        print(O[0].__name__, end='')
        print(' ("', obj_str, '")', sep='')
        print('- - - - - - - - - - - - - - - - - - - - ')
        print()

        if show_object:
                print('Object: ', O[0])

        print()
        if show_params_options:
                show_param_options(O[1])

        if show_all_possible_params:
                possible_params = get_possible_init_params(O[0])
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


