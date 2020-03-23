

class Data_File():

    def __init__(self, loc, load_func):

        self.loc = loc
        self.load_func = load_func

    def load(self):

        return self.load_func(self.loc)

