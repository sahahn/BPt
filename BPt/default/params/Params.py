import nevergrad as ng
from ..helpers import args_repr


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

        # Default consistent choice value
        if isinstance(self, Choice):
            self.indices.value = [0]

    @property
    def descriptors(self):
        '''Stores the descriptor information for this parameter.'''
        return super().descriptors

    @descriptors.setter
    def descriptors(self, descriptors):
        super().descriptors = descriptors


class ValParams(Params):

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

    def set_recombination(self, *args, **kwargs):
        '''Sets a recombination mutation.

        For example:

        ::

            array.set_recombination(ng.p.mutation.Crossover(axis=(0, 1)))

        '''

        if not hasattr(self, 'extra'):
            self.extra = []

        info = {}
        info['name'] = 'set_recombination'
        info['args'] = list(args)
        info['kwargs'] = kwargs
        self.extra.append(info)

        super.set_recombination(*args, **kwargs)


# Override doc string
ValParams.set_integer_casting.__doc__ =\
    ng.p.Scalar.set_integer_casting.__doc__
ValParams.set_mutation.__doc__ =\
    ng.p.Scalar.set_mutation.__doc__
ValParams.set_bounds.__doc__ =\
    ng.p.Scalar.set_bounds.__doc__


def choice_to_grid(self):
    '''This method will attempt to convert from the
    current BPt / nevergrad style parameter to a sklearn-grid
    search compatible one.'''

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
    '''This parameter does not support converting to
    sklearn-grid compatible params.'''
    raise RuntimeError(self._class_name() + ' does not support GridSearch.')


class TransitionChoice(Params, ng.p.TransitionChoice):
    '''BPt parameter wrapper around :class:`nevergrad.p.TransitionChoice`.'''
    to_grid = choice_to_grid


class Choice(Params, ng.p.Choice):
    '''BPt parameter wrapper around :class:`nevergrad.p.Choice`.

    Examples
    ---------
    Constructing a Choice parameter:

    .. ipython:: python

        import BPt as bp
        class_weight_choice = bp.p.Choice(['balanced', None])
        class_weight_choice

    This parameter can then be set when setting :ref:`api.pipeline_pieces`.
    In the example below we will set it to be the params when
    constructing a logistic ridge model.

    .. ipython:: python

        params = {'class_weight': class_weight_choice}
        model = bp.Model('ridge', params=params)
        model

    '''
    to_grid = choice_to_grid


class Array(ValParams, ng.p.Array):
    '''BPt parameter wrapper around :class:`nevergrad.p.Array`.'''
    to_grid = undefined_to_grid


class Scalar(ValParams, ng.p.Scalar):
    '''BPt parameter wrapper around :class:`nevergrad.p.Scalar`.'''

    def to_grid(self):
        '''This method will attempt to convert from the
           current BPt / nevergrad style parameter to a sklearn-grid
           search compatible one.

           Since the base value is a scaler, a lower
           and upper bound must be set, and also this
           Scaler param must have been cast to an integer first.
           '''

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


class Log(ValParams, ng.p.Log):
    '''BPt parameter wrapper around :class:`nevergrad.p.Log`.'''
    to_grid = undefined_to_grid


class Tuple(Params, ng.p.Tuple):
    '''BPt parameter wrapper around :class:`nevergrad.p.Tuple`.'''
    to_grid = undefined_to_grid


class Instrumentation(Params, ng.p.Instrumentation):
    '''BPt parameter wrapper around :class:`nevergrad.p.Instrumentation`.'''
    to_grid = undefined_to_grid


class Dict(Params, ng.p.Dict):
    '''BPt parameter wrapper around :class:`nevergrad.p.Dict`.'''

    def to_grid(self):
        '''This method will attempt to convert from the
           current BPt / nevergrad style parameter to a sklearn-grid
           search compatible one.

           In this case it will just assume the dictionary is
           a value, and will return a python dictionary, with
           to_grid called recursively if their are any other Params
           stored in the values of the dictionary.
        '''

        new_dict = {}

        for key in self._content:

            # Will this work with grid search style params?
            if isinstance(self._content[key], Params):
                new_val = self._content[key].to_grid()
            elif isinstance(self._content[key], ng.p.Constant):
                new_val = self._content[key].value
            else:
                new_val = self._content[key]

            new_dict[key] = new_val

        return new_dict


# @TODO Add more support for different to_grid?
