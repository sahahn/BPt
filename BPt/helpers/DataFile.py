from joblib import Parallel, delayed
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

    def __lt__(self, other):
        return self.loc < other.loc

    def __eq__(self, other):
        return self.loc == other.loc

    def __hash__(self):
        return hash(self.loc)

    def __deepcopy__(self, memo):
        return DataFile(deepcopy(self.loc, memo), self.load_func)


def mp_load(files, reduce_funcs):

    proxy = np.zeros((len(files), len(reduce_funcs)))
    for f in range(len(files)):

        data = files[f].load()
        for r in range(len(reduce_funcs)):
            proxy[f, r] = reduce_funcs[r](data)

    return proxy


def mp_single_load(files, reduce_func):

    # Create proxy to fill with values
    proxy = np.zeros(shape=(len(files)))

    for f in range(len(files)):

        # Load file
        data = files[f].load()

        # Reduce and add to proxy
        proxy[f] = reduce_func(data)

    return proxy


def load_data_file_proxy(values, reduce_func, file_mapping, n_jobs=1):

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
    output = Parallel(n_jobs=n_jobs)(delayed(mp_single_load)(
                      files=files, reduce_func=reduce_func)
                      for files in file_splits)

    # Fill proxy with the concatenated output
    proxy[:] = np.concatenate(output)

    return proxy


def load_data_file_proxies(data, reduce_funcs,
                           data_file_keys, file_mapping,
                           n_jobs=1):

    data_file_proxies = [data[data_file_keys].copy()
                         for _ in range(len(reduce_funcs))]
    data_files = data[data_file_keys]

    # Single core version
    if n_jobs == 1:

        for col in data_files:
            for subject in data_files.index:

                file_key = data_files.at[subject, col]
                data = file_mapping[file_key].load()

                for r in range(len(reduce_funcs)):
                    data_file_proxies[r].at[subject, col] =\
                        reduce_funcs[r](data)

    # Multi-core version
    else:

        col_names = list(data_files)
        data_files = np.array(data_files)
        for col in range(data_files.shape[1]):
            splits = np.array_split(data_files[:, col], n_jobs)

            def change_to_map(x):
                return file_mapping[x]
            v_func = np.vectorize(change_to_map)
            file_splits = [v_func(split) for split in splits]

            output = Parallel(n_jobs=n_jobs)(delayed(mp_load)(
                files=files, reduce_funcs=reduce_funcs)
                for files in file_splits)
            output = np.vstack(output)

            for i in range(len(reduce_funcs)):
                data_file_proxies[i][col_names[col]] = output[:, i]

    return data_file_proxies
