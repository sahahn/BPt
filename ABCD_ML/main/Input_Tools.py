

class Select(list):
    input_type = 'select'

    def __repr__(self):
        return 'Select(' + super().__repr__() + ')'
    def __str__(self):
        return self.__repr__()

def is_select(obj):

    try:
        if obj.input_type == 'select':
            return True
        return False

    except AttributeError:
        return False

class Duplicate(list):
    '''The Duplicate object is an ABCD_ML specific Input wrapper.
    It is designed to be cast on a list of valid scope parameters, e.g., 
    
    ::

        scope=Duplicate(['float', 'cat'])

    Such that the corresponding pipeline piece will be duplicated for every
    entry within Duplicate. In this case, two copies of the base object will be
    made, where both have the same remaining non-scope params (i.e., obj, params, extra_params),
    but one will have a scope of 'float' and the other 'cat'. 
    
    Consider the following exentended example, where loaders is being specified within Model_Pipeline:

    ::
        
        loaders = Loader(obj='identity', scope=Duplicate(['float', 'cat']))

    Is transformed in post processing / equivalent to

    ::

        loaders = [Loader(obj='identity', scope='float'),
                   Loader(obj='identity', scope='cat')]

    '''

    input_type = 'duplicate'

    def __repr__(self):
        return 'Duplicate(' + super().__repr__() + ')'
    def __str__(self):
        return self.__repr__()

def is_duplicate(obj):

    try:
        if obj.input_type == 'duplicate':
            return True
        return False

    except AttributeError:
        return False

class Pipe(list):
    input_type = 'pipe'

    def __repr__(self):
        return 'Pipe(' + super().__repr__() + ')'
    def __str__(self):
        return self.__repr__()

def is_pipe(obj):

    try:
        if obj.input_type == 'pipe':
            return True
        return False

    except AttributeError:
        return False

class Value_Subset():
    input_type = 'value_subset'

    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __repr__(self):
        return 'Value_Subset(name=' + str(self.name) + ', value=' + str(self.value) + ')'
    def __str__(self):
        return self.__repr__()

def is_value_subset(obj):

    try:
        if obj.input_type == 'value_subset':
            return True
        return False

    except AttributeError:
        return False

def is_special(obj):
    return hasattr(obj, 'input_type')

