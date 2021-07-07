from joblib import Parallel, delayed, effective_n_jobs, hash as joblib_hash
from copy import deepcopy
import numpy as np
import os


class DataFile():

    def __init__(self, loc, load_func):

        self.loc = os.path.abspath(loc)
        self.load_func = load_func

    def _load(self):
        return self.load_func(self.loc)

    def load(self):
        return self._load()

    def reduce(self, reduce_func):

        # Init cache if not already
        if not hasattr(self, '_cached_reduce'):
            self._cached_reduce = {}

        # Hash the reduce func w/ joblib hash
        func_hash = joblib_hash(reduce_func)

        # If has already been hashed before
        if func_hash in self._cached_reduce:
            return self._cached_reduce[func_hash]

        # Apply the function
        reduced = reduce_func(self.load())

        # Cache result before returning
        self._cached_reduce[func_hash] = reduced

        return reduced

    def __lt__(self, other):
        return self.loc < other.loc

    def __eq__(self, other):
        return self.loc == other.loc

    def __hash__(self):
        return hash(self.loc)

    def __deepcopy__(self, memo):
        return DataFile(deepcopy(self.loc, memo), self.load_func)

    def __repr__(self):
        return 'DataFile(loc=' + repr(self.loc) + ')'

    def __str__(self):
        return self.__repr__()


def mp_single_load(files, reduce_func):

    # Create proxy to fill with values
    proxy = np.zeros(shape=(len(files)))

    # Get reduced from each file
    for f in range(len(files)):
        proxy[f] = files[f].reduce(reduce_func)

    return proxy


def load_data_file_proxy(values, reduce_func, file_mapping, n_jobs=1):

    # Replace n_jobs w/ effective n_jobs
    n_jobs = effective_n_jobs(n_jobs)

    # Can at most be number of files
    n_jobs = min([n_jobs, len(values)])

    # Create proxy to fill in
    proxy = values.copy()

    # Generate splits based on n_jobs
    splits = np.array_split(np.array(values), n_jobs)

    # Nested func for multi-proc, to vectorize
    def change_to_map(x):
        return file_mapping[x]
    v_func = np.vectorize(change_to_map)

    # Apply v_func to each split
    file_splits = [v_func(split) for split in splits]

    # Load w/ joblib Parallel
    output = Parallel(n_jobs=n_jobs,
                      backend="threading")(delayed(mp_single_load)(
                       files=files, reduce_func=reduce_func)
                      for files in file_splits)

    # Fill proxy with the concatenated output
    proxy[:] = np.concatenate(output)

    return proxy

