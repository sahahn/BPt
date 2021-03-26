from .BPtEvaluator import BPtEvaluator
from .input_operations import BPtInputMixIn
from copy import deepcopy
import pandas as pd


def clean_str(in_str):

    in_str = str(in_str)
    in_str = in_str.replace('"', '')
    in_str = in_str.replace("'", '')

    return in_str


def str_to_option(option):

    key, name = option.split('=')
    o = Option(value=None, name=name, key=key)
    return o


def str_to_options(option):

    # If single str option
    if option.count('=') == 1:
        return [str_to_option(option)]

    elif option.count('=') == 0:
        raise KeyError(repr(option) + 'is not a valid key.')

    # Otherwise multiple
    return [str_to_option(opt_str)
            for opt_str in option.split(', ')]


def add_scores(cols, evaluator, attr_name):

    if not hasattr(evaluator, attr_name):
        return

    attr = getattr(evaluator, attr_name)
    for key in attr:
        val = attr[key]

        try:
            cols[attr_name + '_' + key].append(val)
        except KeyError:
            cols[attr_name + '_' + key] = [val]


class Option(BPtInputMixIn):
    '''This is a special BPt input class designed
    to be used with :class:`Compare`. It is used as
    a formal way to represent comparison options.

    Parameters
    ----------
    value : option value
        The explicit value in which the parameter
        it is passed to will take.

    name : str or 'repr', optional
        The name of this value, where this name
        is used to index the :class:`CompareDict`
        of results.

        If left as default value of 'repr' then
        a str representation of the value will
        be automatically created.

        ::

            default = 'repr'

    key : str or None, optional
        This parameter should typically not be
        set directly. It is instead inferred by the
        context in which the option is eventually used.

        ::

            default = None

    '''

    def __init__(self, value, name='repr', key=None):

        self.value = value

        # The name is either by default
        # the representation or a custom name.
        if name == 'repr':

            if hasattr(self.value, 'obj') and \
             isinstance(getattr(self.value, 'obj'), str):
                self.name = clean_str(getattr(self.value, 'obj'))
            else:
                self.name = clean_str(repr(value))

        else:
            self.name = clean_str(name)

        if not isinstance(self.name, str):
            raise RuntimeError('name must be a str.')

        self.key = key

    def __hash__(self):
        return hash(self.name) + hash(self.key)

    def __repr__(self):
        return self.key + '=' + self.name

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        return (self.name, self.key) == (other.name, other.key)

    def __ne__(self, other):
        return not(self == other)

    # Sort by key then name if keys are equal
    def __lt__(self, other):

        if self.key == other.key:
            return self.name < other.name
        return self.key < other.key

    def __gt__(self, other):

        if self.key == other.key:
            return self.name > other.name
        return self.key > other.key


class Compare(BPtInputMixIn):
    '''This is a special BPt input class which
    can be used as a helper to more easily run comparison
    analysis between a few choice of parameters.

    Parameters
    ------------
    options : list of values or :class:`Option`
        This parameter should be a list of options
        in which to try each one of. You may pass
        these options either directly as values, for example ::

            options = ['option1', 'option2']

        Or as instances of :class:`Option`. This
        second strategy is reccomended when the underlying
        options are objects or something more complex then strings,
        for example between two :class:`Pipeline` ::

            pipe1 = bp.Pipeline([bp.Model('elastic')])
            pipe2 = bp.Pipeline([bp.Model('ridge')])

            options = [bp.Option(pipe1, name='pipe1'),
                       bp.Option(pipe2, name='pipe2')]


    Notes
    ---------
    | Usage of this object is designed to passed as input
        to :func:`evaluate`. Only parameters within :func:`evaluate`
        parameters `pipeline` and `problem_spec` (or their associated extra
        params) can be passed Compare. That said, some options, while valid
        may still make downstream intreptation more difficult, e.g., passing
        problem_type with two different Compare values will work, but
        will yield results with different metrics.

    | When to use Compare? It may be tempting to use Compare to evaluate
        different configurations of hyper-parameters, but in most cases
        this type of fine-grained usage is discouraged. On a conceptual level,
        the usage of Compare should be used to compare the actual underlying
        topic of interest! For example, if it
        of interest to the underlying research topic,
        then Compare can be used between two different :class:`Pipeline`.
        If instead this is not the key point of interest, but you still wish to
        try two different, say, :class:`Model`, then you would be better off
        nesting this choice as a hyper-parameter to optimize
        (in this case see: :class:`Select`).
    '''

    def __init__(self, options):

        self.options = []
        for option in options:

            # If not Option, cast to Option
            if not isinstance(option, Option):
                self.options.append(Option(option))

            # Otherwise, add as is
            else:
                self.options.append(option)

    def _check_args(self):
        '''If called here, then Compare is made
        up of pipeline pieces.'''

        for option in self.options:
            option.value._check_args()


class Options():

    def __init__(self, *args, **kwargs):

        if len(args) == 1 and len(kwargs) == 0:
            self._single_arg(args[0])

        elif len(args) == 0 and len(kwargs) > 0:
            self._just_kwargs(kwargs)

    def _single_arg(self, first_arg):

        if isinstance(first_arg, list):

            options = []
            for option in first_arg:

                # skip if None
                if option is None:
                    continue

                # If option, add
                elif isinstance(option, Option):
                    options.append(option)

                # If options, add all existing options
                elif isinstance(option, Options):
                    options += option.options

                # Otherwise treat as query str and
                # cast to option.
                elif isinstance(option, str):
                    options += str_to_options(option)

                else:
                    raise RuntimeError('Unknown option:' + repr(option))

            self.options = options

        else:

            if isinstance(first_arg, str):

                try:
                    self.options = str_to_options(first_arg)

                except KeyError:
                    # If Failed, treat as Option with just value
                    self.options = [Option(value=None,
                                           name=first_arg,
                                           key='')]

            elif isinstance(first_arg, Option):
                self.options = [first_arg]
            else:
                self.options = first_arg

    def _just_kwargs(self, kwargs):

        self.options = [Option(value=None, name=repr(kwargs[key]), key=key)
                        for key in kwargs]

    def __hash__(self):
        return sum([hash(o) for o in self.options])

    def __repr__(self):
        return 'Options(' + ', '.join([repr(o) for
                                       o in sorted(self.options)]) + ')'

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        return sorted(self.options) == sorted(other.options)

    def __ne__(self, other):
        return not(self == other)

    def is_subset(self, other):

        in_other = [option for option in self.options
                    if option in other.options]

        return len(in_other) == len(self.options)

    def is_just_name_subset(self, other):

        if len(self.options) > 1:
            return False

        name = self.options[0].name
        names = [option.name for option in other.options]

        if name in names:
            return True

        return False


class CompareDict(dict):
    '''This object is simmilar to a python dictionary,
    but is used to stored results from analyses run using
    :class:`Compare`.
    '''

    def _cast_to_options(self, k):

        # options can be None
        if k is None:
            return None

        if not isinstance(k, Options):
            k = Options(k)

        return k

    def __setitem__(self, k, v):

        k = self._cast_to_options(k)
        return super().__setitem__(k, v)

    def __getitem__(self, k):

        # Get input key as cast to Options
        k = self._cast_to_options(k)

        # First try to treat as single key
        try:
            return super().__getitem__(k)

        # If that fails, then check to see
        # if this index works as a subset.
        except KeyError:

            subset_keys = [key for key in self.keys() if k.is_subset(key)]

            # If fails here, try single subset
            if len(subset_keys) == 0:

                subset_keys = [key for key in self.keys()
                               if k.is_just_name_subset(key)]

                if len(subset_keys) == 0:
                    raise KeyError(repr(k) + ' is not a valid key.')

            # If just one, return that one
            if len(subset_keys) == 1:
                return self.__getitem__(subset_keys[0])

            # If any subsets, return a subset of this
            # dict with only those subsets
            return CompareDict({key: self[key] for key in subset_keys})

    def __repr__(self):
        return 'CompareDict(' + super().__repr__() + ')'

    def __str__(self):
        return self.__repr__()

    def summary(self):
        '''Return a pandas DataFrame with summary
        information as broken down by all of the different
        original :class:`Compare` options.'''

        # Get example value to base summary on
        ex = self.__getitem__(list(self.keys())[0])

        # if evaluation results
        if isinstance(ex, BPtEvaluator):
            return self._evaluator_summary()

        # @TODO add more options

    def _evaluator_summary(self):

        keys = list(self.keys())
        repr_key = keys[0]
        option_keys = [o.key for o in repr_key.options]
        cols = {key: [] for key in option_keys}

        for key in list(self.keys()):

            for o in key.options:
                cols[o.key].append(o.name)

            # Add values
            evaluator = self[key]
            add_scores(cols, evaluator, attr_name='mean_scores')
            add_scores(cols, evaluator, attr_name='std_scores')
            add_scores(cols, evaluator, attr_name='mean_timing')

        summary = pd.DataFrame.from_dict(cols)
        summary = summary.set_index(option_keys)

        return summary


def _make_compare_copies(objs, key, compare):

    # If not already compare dict, cast to compare dict
    # under empty string to represent base
    if not isinstance(objs, CompareDict):
        objs = CompareDict({None: objs})

    # Store all new objects
    new_objs = CompareDict()

    # For each compare value
    for option in compare.options:

        # Set key in option
        option.key = key

        # For each existing object
        for obj_key in objs:

            # Get copy
            new_obj = deepcopy(objs[obj_key])

            # Assign value in place
            new_obj.set_params(**{key: option.value})

            # New key as existing key + new option
            new_obj_key = Options([obj_key, option])

            # Add to new objs under new key
            if new_obj_key in new_objs:
                raise RuntimeError('key collision!')

            new_objs[new_obj_key] = new_obj

    return new_objs


def _compare_check(obj):

    # Check for any params set to Compare
    params = obj.get_params(deep=True)

    for key in params:
        if isinstance(params[key], Compare):

            # Update obj to be list of problem
            # specs init'ed with different options
            obj = _make_compare_copies(obj, key, params[key])

    return obj


def _merge_compare(pipe, ps):

    # If both are not dicts
    if not isinstance(pipe, CompareDict) and not isinstance(ps, CompareDict):
        return pipe, ps

    # New will stored new dict
    new = CompareDict()

    # If both are dicts
    if isinstance(pipe, CompareDict) and isinstance(ps, CompareDict):

        # Save each combined combination
        for pipe_key in pipe:
            for ps_key in ps:
                new[[pipe_key, ps_key]] =\
                    (deepcopy(pipe[pipe_key]), deepcopy(ps[ps_key]))

        return new

    # If just ps
    elif isinstance(ps, CompareDict):
        for ps_key in ps:
            new[ps_key] = (deepcopy(pipe), deepcopy(ps[ps_key]))

        return new

    # If just pipe
    for pipe_key in pipe:
        new[pipe_key] = (deepcopy(pipe[pipe_key]), deepcopy(ps))

    return new
