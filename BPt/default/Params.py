import nevergrad as ng


def is_array_like(in_val):

    if hasattr(in_val, '__len__') and (not isinstance(in_val, str)) and \
     (not isinstance(in_val, dict)) and (not hasattr(in_val, 'fit')) and \
     (not hasattr(in_val, 'transform')):
        return True
    else:
        return False


def args_repr(args, kwargs):

    reprs = [repr(arg) for arg in args]
    sorted_keys = sorted(list(kwargs))
    reprs += [key + '=' + repr(kwargs[key]) for key in sorted_keys]
    return ', '.join(reprs)


class Params():

    def _class_name(self):
        return type(self).__name__

    def _extra_repr(self):

        if not hasattr(self, 'extra'):
            return ''

        base = ''
        for info in self.extra:
            base += '.' + info['name'] + '('
            base += args_repr(info['args'], info['kwargs'])
            base += ')'

        return base

    def __repr__(self):

        base = self._class_name() + '('
        base += args_repr(self.args_, self.kwargs_) + ')'
        return base + self._extra_repr()

    def __init__(self, *args, **kwargs):

        self.args_ = list(args)
        self.kwargs_ = kwargs

        super().__init__(*self.args_, **self.kwargs_)

    def set_integer_casting(self):

        if not hasattr(self, 'extra'):
            self.extra = []

        info = {}
        info['name'] = 'set_integer_casting'
        info['args'] = []
        info['kwargs'] = {}
        self.extra.append(info)

        super().set_integer_casting()

        return self

    def set_mutation(self, *args, **kwargs):

        if not hasattr(self, 'extra'):
            self.extra = []

        info = {}
        info['name'] = 'set_mutation'
        info['args'] = list(args)
        info['kwargs'] = kwargs
        self.extra.append(info)

        super().set_mutation(*args, **kwargs)

        return self

    def set_bounds(self, *args, **kwargs):

        if not hasattr(self, 'extra'):
            self.extra = []

        info = {}
        info['name'] = 'set_bounds'
        info['args'] = list(args)
        info['kwargs'] = kwargs
        self.extra.append(info)

        super().set_bounds(*args, **kwargs)

        return self


def choice_to_grid(self):

    if 'choices' in self.kwargs_:
        choices = self.kwargs_['choices']
    else:
        choices = self.args_[0]

    as_grid = []
    for choice in choices:

        # Check for nested
        if hasattr(choice, 'to_grid'):
            values = choice.to_grid()
            if isinstance(values, list):
                as_grid += values
            else:
                as_grid.append(values)

        # If not a function with to_grid, add as is
        else:
            as_grid.append(choice)

    return as_grid


def undefined_to_grid(self):
    raise RuntimeError(self._class_name() + ' does not support GridSearch.')


class TransitionChoice(Params, ng.p.TransitionChoice):
    to_grid = choice_to_grid


class Choice(Params, ng.p.Choice):
    to_grid = choice_to_grid


class Array(Params, ng.p.Array):
    to_grid = undefined_to_grid


class Scalar(Params, ng.p.Scalar):

    def to_grid(self):

        # If not interger
        if not self.integer:
            raise RuntimeError('Scalar cannot convert to grid unless cast '
                               'to integer.')

        lower = self.bounds[0]
        if len(lower) == 1:
            lower = int(lower[0])
        else:
            lower = None

        upper = self.bounds[1]
        if len(upper) == 1:
            upper = int(upper[0])
        else:
            upper = None

        if lower is not None and upper is not None:
            return list(range(lower, upper+1))

        raise RuntimeError('Scalar cannot convert to grid unless cast '
                           'both lower and upper are set')


class Log(Params, ng.p.Log):
    to_grid = undefined_to_grid


class Tuple(Params, ng.p.Tuple):
    to_grid = undefined_to_grid


class Instrumentation(Params, ng.p.Instrumentation):
    to_grid = undefined_to_grid


class Dict(Params, ng.p.Dict):
    to_grid = undefined_to_grid


# @TODO Add more support for different to_grid
