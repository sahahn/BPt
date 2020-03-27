from copy import deepcopy

class Data_File():

    def __init__(self, loc, load_func, in_memory=False):

        self.loc = loc
        self.load_func = load_func
        self.in_memory = in_memory

        if self.in_memory:
            self.data = self._load()

    def _load(self):

        return self.load_func(self.loc)

    def load(self, copy=True):

        if self.in_memory:
            if copy:
                return deepcopy(self.data)
            
            return self.data
        
        return self._load()

