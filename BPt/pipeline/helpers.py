import tempfile
import random
import numpy as np
import os
import inspect
from copy import deepcopy
from joblib import hash as joblib_hash
from ..util import is_array_like
from ..dataset.data_file import DataFile
from ..default.params.Params import Params


def to_memmap(X):

    f = os.path.join(tempfile.gettempdir(), str(random.random()))
    x = np.memmap(f, dtype=X.dtype, shape=X.shape, mode='w+')
    x[:] = X

    return f, X.dtype, X.shape


def from_memmap(X):

    X_file, X_type, X_shape = X
    X = np.memmap(X_file, dtype=X_type, shape=X_shape, mode='c')
    return X


def is_ng(p):

    try:
        return 'nevergrad' in p.__module__
    except AttributeError:
        return False


def get_grid_params(params):

    # Set grid params
    grid_params = {}
    for p in params:

        if hasattr(params[p], 'to_grid'):
            grid_params[p] = params[p].to_grid()
        elif is_ng(params[p]):
            raise RuntimeError('Passing nevergrad parameters directly is not '
                               'supported. Pass as a BPt Params equivalent.')
        else:
            grid_params[p] = params[p]

    return grid_params


def _get_fis_list(estimators, prop):

    fis = []

    # Go through each trained estimator
    for est in estimators:
        if hasattr(est, prop):
            fi = getattr(est, prop)

            # If any is None, return None
            if fi is None:
                return None

            # Try to squeeze values
            try:
                fi = fi.squeeze()
            except AttributeError:
                pass
            
            # Add to list
            fis.append(fi)

        # If any don't, return None
        else:
            return None

    return fis


def get_concat_fis_len(estimators, prop):

    fis = _get_fis_list(estimators, prop)

    if fis is None:
        return None

    # Return length of each
    return [len(c) for c in fis]

def get_concat_fis(estimators, prop):

    fis = _get_fis_list(estimators, prop)

    if fis is None:
        return None

    # Return concat
    return np.concatenate(fis)


def get_mean_fis(estimators, prop):

    fis = _get_fis_list(estimators, prop)

    # Make sure all same len
    try:
        if len(set([len(x) for x in fis])) != 1:
            return None
    except TypeError:
        return None

    # Return as mean over axis 0
    return np.mean(np.array(fis), axis=0)

def check_for_nested_loader(objs):
    '''Go through in nested manner and see if any
    objects are instance of BPtLoader.'''

    from .BPtLoader import BPtLoader

    def _check_for_nested_loader(objs):
        
        if isinstance(objs, BPtLoader):
            return True

        elif isinstance(objs, (list, set, tuple, frozenset)):
            return any([_check_for_nested_loader(o) for o in objs])

        elif isinstance(objs, dict):
            return any([_check_for_nested_loader(objs[k] for k in objs)])

        elif hasattr(objs, 'get_params'):
            return any([_check_for_nested_loader(getattr(objs, param))
                        for param in objs.get_params(deep=False)])

        return False

    return _check_for_nested_loader(objs)

def get_nested_final_estimator(estimator):

    # Init loop
    final_estimator = estimator

    # Check for nested in loops
    while hasattr(final_estimator, '_final_estimator') and \
        getattr(final_estimator, '_final_estimator') is not None:
        final_estimator = getattr(final_estimator, '_final_estimator')

    return final_estimator

def proc_mapping(indx, mapping):

    # Special case, if passed index of Ellipsis
    # return as is.
    if indx is Ellipsis:
        return indx

    # If non-empty mapping and non-empty index
    if len(mapping) > 0 and len(indx) > 0:

        # If should proc list...
        if is_array_like(indx[0]):
            return [proc_mapping(i, mapping) for i in indx]

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

    # Base case return indx
    return indx

# @ TODO change lists in mapping interally to sets???


def update_mapping(mapping, new_mapping):

    # In case where new_mapping is empty,
    # return the original mapping as is
    if len(new_mapping) == 0:
        return

    # Go through the mapping and update each key with the new mapping
    for key in mapping:
        val = mapping[key]

        # In case of list
        if isinstance(val, list):

            # Generate a new list of values
            new_vals = []
            for v in val:

                # If the value is valid / in the new mapping
                if v in new_mapping:
                    new_val = new_mapping[v]

                    # Case where the new values is also a list
                    if isinstance(new_val, list):
                        new_vals += new_val
                    else:
                        new_vals.append(new_val)

                else:
                    new_vals.append(v)

            # Treat as set to remove duplicates
            as_set = set(new_vals)

            # Don't keep None's if any in list
            try:
                as_set.remove(None)
            except KeyError:
                pass

            # Exists case where None was the only element in the
            # list, in that case just set to None
            if len(as_set) == 0:
                mapping[key] = None

            # Otherwise set as list in sorted order
            else:
                mapping[key] = sorted(list(as_set))

        # Assume int if not list
        else:

            if val in new_mapping:
                mapping[key] = new_mapping[val]


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


def set_n_jobs(obj, n_jobs):

    # Call recursively for list
    if isinstance(obj, list) or isinstance(obj, tuple):
        for o in obj:
            set_n_jobs(o, n_jobs)

    # Check and set for n_jobs
    if hasattr(obj, 'n_jobs'):
        setattr(obj, 'n_jobs', n_jobs)


def get_possible_params(estimator, method):

    if not hasattr(estimator, method):
        return []

    pos_params = dict(inspect.getmembers(getattr(estimator, method).__code__))
    return pos_params['co_varnames']

def file_mapping_to_str(file_mapping):

    # Remove NaN first
    if np.nan in file_mapping:
        del file_mapping[np.nan]

    # Instead of full dict, return str representation for faster hash
    str_rep = ''.join([str(k) + file_mapping[k].quick_hash_repr() for k in file_mapping])

    return str_rep


def check_replace(objs):

    jobs_aliases = ['n_jobs', 'fix_n_jobs', '_fix_n_jobs',
                    'fix_n_jobs_', '_n_jobs', 'n_jobs_']

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

        if len(objs) == 0:
            return objs

        # Check first element, if data file
        # assume this is dict of data files
        # and return right away
        if isinstance(objs[list(objs)[0]], DataFile):

            # Return fast hash str repr of file mapping
            return file_mapping_to_str(objs)

        # Check n_jobs in dict
        for k in objs:
            for j_a in jobs_aliases:
                if k == j_a:
                    objs[k] = 1

        return {k: check_replace(objs[k]) for k in objs}

    if hasattr(objs, 'get_params'):
        for param in objs.get_params(deep=False):
            new_value = check_replace(getattr(objs, param))
            setattr(objs, param, new_value)

        # Also if has n_jobs replace all with fixed 1
        # trying in a bunch of different ways
        for j_a in jobs_aliases:
            if hasattr(objs, j_a):
                try:
                    setattr(objs, j_a, 1)
                except AttributeError:
                    pass

        # Return objs as changed in place
        return objs

    # If nevergrad / params convert to repr
    if isinstance(objs, Params):
        return repr(objs)

    # Return identity otherwise
    return objs


def pipe_hash(objs, steps):
    '''Expects a list'''

    # Make copy with nevergrad / params dists replaced by repr
    hash_steps = check_replace(deepcopy(steps))

    # Hash steps and objs separate, then combine
    hash_str1 = joblib_hash(objs, hash_name='md5')
    hash_str2 = joblib_hash(hash_steps, hash_name='md5')

    return hash_str1 + hash_str2


def list_loader_hash(X_col, file_mapping, y, estimator):

    # Convert X_col to data files,  then str, then hash
    as_data_files_str = [file_mapping[int(key)].quick_hash_repr() for key in X_col]
    hash_str1 = joblib_hash(as_data_files_str, hash_name='md5')

    # Hash y
    hash_str2 = joblib_hash(y, hash_name='md5')

    # Hash the estimator - w/ extra special check
    hash_estimator_copy = check_replace(deepcopy(estimator))
    hash_str3 = joblib_hash(hash_estimator_copy, hash_name='md5')

    return hash_str1 + hash_str2 + hash_str3


def replace_with_in_params(params, original, replace):

    new_params = {}

    for key in params:
        new_params[key.replace(original, replace)] = params[key]

    return new_params


def check_om(out_mapping):

    # Worried about possibility
    # make sure out_mapping at these stages
    # is 1:1
    flag = False

    for key in out_mapping:
        if key != out_mapping[key]:
            flag = True

    # Issue warning if comes up
    if flag:
        import warnings
        warnings.warn('Maybe an issue with updating inds correctly, carefully validate results.')
