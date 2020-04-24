from joblib import wrap_non_picklable_objects


class Data_File():

    def __init__(self, loc, load_func):

        self.loc = loc
        self.load_func = wrap_non_picklable_objects(load_func)

    def _load(self):
        return self.load_func(self.loc)

    def load(self):
        return self._load()


def load_data_file_proxies(data, reduce_funcs,
                           data_file_keys, file_mapping):

    data_file_proxies = [data[data_file_keys].copy()
                         for _ in range(len(reduce_funcs))]

    data_files = data[data_file_keys]

    for col in data_files:
        for subject in data_files.index:

            file_key = data_files.loc[subject, col]
            data = file_mapping[file_key].load()

            for r in range(len(reduce_funcs)):
                data_file_proxies[r].loc[subject, col] = reduce_funcs[r](data)

    return data_file_proxies
