from ..helpers.VARS import SCOPES

class Select(list):
    input_type = 'select'

    def __repr__(self):
        return 'Select(' + super().__repr__() + ')'
    def __str__(self):
        return 'Select(' + super().__repr__() + ')'

def is_select(obj):

    try:
        if obj.input_type == 'select':
            return True
        return False

    except AttributeError:
        return False


class Duplicate(list):
    input_type = 'duplicate'

    def __repr__(self):
        return 'Duplicate(' + super().__repr__() + ')'
    def __str__(self):
        return 'Duplicate(' + super().__repr__() + ')'

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
        return 'Pipe(' + super().__repr__() + ')'

def is_pipe(obj):

    try:
        if obj.input_type == 'pipe':
            return True
        return False

    except AttributeError:
        return False

class Scope():

    input_type = 'scope'

    def __init__(self, value):
        self.value = value
    def __repr__(self):
        return 'Scope(' + self.value.__repr__() + ')'
    def __str__(self):
        return 'Scope(' + self.value.__repr__() + ')'

def is_scope(obj):

    try:
        if obj.input_type == 'scope':
            return True
        return False

    except AttributeError:
        return False


def cast_input_to_scopes(scopes):

    # If input already a scope, don't wrap
    if is_scope(scopes):
        return scopes

    # If input is native list - or special input list like,
    # cast each member of the list to be a scope
    elif isinstance(scopes, list):
        for i in range(len(scopes)):
            scopes[i] = cast_input_to_scopes(scopes[i])

        return scopes

    # Otherwise, just wrap in Scope class
    else:
        return Scope(scopes)


def is_special(obj):

    try:
        obj.input_type
        return True
    except AttributeError:
        return False
