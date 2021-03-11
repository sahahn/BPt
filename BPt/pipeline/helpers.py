import tempfile
import random
import numpy as np
import os


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

            fis.append(fi)

        # If any don't, return None
        else:
            return None

    # Make sure all same len
    if len(set([len(x) for x in fis])) != 1:
        return None

    # Return as mean
    return np.mean(np.array(fis), axis=0)


def is_array_like(in_val):

    if hasattr(in_val, '__len__') and (not isinstance(in_val, str)) and \
     (not isinstance(in_val, dict)) and (not hasattr(in_val, 'fit')) and \
     (not hasattr(in_val, 'transform')):
        return True

    return False


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
