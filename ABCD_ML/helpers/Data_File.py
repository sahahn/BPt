from copy import deepcopy
from joblib import wrap_non_picklable_objects

class Data_File():

    def __init__(self, loc, load_func):

        self.loc = loc
        self.load_func = wrap_non_picklable_objects(load_func)

    def _load(self):
        return self.load_func(self.loc)

    def load(self):
        return self._load()

