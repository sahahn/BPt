class Select(list):
    is_select = True

    def __repr__(self):
        return 'Select(' + super().__repr__() + ')'
    def __str__(self):
        return 'Select(' + super().__repr__() + ')'


def is_select(obj):

    try:
        if obj.is_select:
            return True
        return False

    except AttributeError:
        return False
