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
                               'supported. Pass as a BPt Params equivilent.')
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