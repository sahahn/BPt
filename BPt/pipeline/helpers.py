import tempfile
import random
import numpy as np
import os
import inspect
from copy import deepcopy
from joblib import hash as joblib_hash
from ..util import is_array_like
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


def get_mean_fis(estimators, prop):

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

            fis.append(fi)

        # If any don't, return None
        else:
            return None

    # Make sure all same len
    if len(set([len(x) for x in fis])) != 1:
        return None

    # Return as mean
    return np.mean(np.array(fis), axis=0)


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


def pipe_hash(objs, steps):
    '''Expects a list'''

    # Make copy with nevergrad / params dists replaced by repr
    hash_steps = check_replace(deepcopy(steps))

    # Hash steps and objs separate, then combine
    hash_str1 = joblib_hash(objs, hash_name='md5')
    hash_str2 = joblib_hash(hash_steps, hash_name='md5')

    return hash_str1 + hash_str2


def replace_with_in_params(params, original, replace):

    new_params = {}

    for key in params:
        new_params[key.replace(original, replace)] = params[key]

    return new_params
