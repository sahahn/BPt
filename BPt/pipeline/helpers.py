import tempfile
import random
import numpy as np
import os


def f_array(in_array, tp='float32'):
    return np.array(in_array).astype(tp)


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


def extract_values(value):

    if is_ng(value):

        # If a choice obj
        if hasattr(value, 'choices'):

            # Unpack choices
            choices = []
            for c in range(len(value.choices)):

                # Check for nested
                choice_value = extract_values(value.choices[c])
                if isinstance(choice_value, list):
                    choices += choice_value
                else:
                    choices.append(choice_value)

            return choices

        # if scalar type
        elif hasattr(value, 'integer'):

            # If cast to integer
            if value.integer:

                lower = value.bounds[0]
                if len(lower) == 1:
                    lower = int(lower[0])
                else:
                    lower = None

                upper = value.bounds[1]
                if len(upper) == 1:
                    upper = int(upper[0])
                else:
                    upper = None

                if lower is not None and upper is not None:
                    return list(range(lower, upper+1))

        elif hasattr(value, 'value'):
            return value.value

        # All other cases
        raise RuntimeError('Could not convert nevergrad',
                           value, 'to grid search parameter!')

    else:
        return value


def get_grid_params(params):

    # Set grid params
    grid_params = {}
    for p in params:
        grid_params[p] = extract_values(params[p])

    return grid_params
