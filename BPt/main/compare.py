from numpy.lib.shape_base import split
from .BPtEvaluator import BPtEvaluator
from .input_operations import BPtInputMixIn
from copy import deepcopy, copy
import pandas as pd
from .stats_helpers import compute_corrected_ttest
import numpy as np
from itertools import combinations
from math import factorial
from .helpers import clean_str
import pickle as pkl
import os
from ..util import get_unique_str_markers


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

        if self.key is None:
            return self.name

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

    Examples
    ---------
    Compare is used with :func:`evaluate`, for example:

    .. ipython:: python

        from BPt.datasets import load_cali
        data = bp.datasets.load_cali()
        data.shape

        pipe_options = bp.Compare([bp.Option(bp.Model('elastic'),
                                             name='elastic'),
                                   bp.Option(bp.Model('ridge'),
                                             name='ridge')])

        bp.evaluate(pipe_options, data, progress_bar=False).summary()

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

    def _uniquify(self):

        for option in self.options:

            # Recursive check first
            if hasattr(option.value, '_uniquify'):
                option.value._uniquify()

            # Then replace with copy
            option.value = copy(option.value)


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
            
            # Cast w/ clean str if numeric
            if isinstance(k, (int, float)):
               k = clean_str(k)

            # To options
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

    def summary(self, **kwargs):
        '''Return a pandas DataFrame with summary
        information as broken down by all of the different
        original :class:`Compare` options.'''

        # Get example value to base summary on
        ex = self.__getitem__(list(self.keys())[0])

        # @TODO make sure all of same time

        # if Evaluation results
        if isinstance(ex, BPtEvaluator):
            if self._check_multiple_problem_types():
                return self._split_evaluator_summary(by='problem_type', **kwargs)
            return self._evaluator_summary(**kwargs)

        # @TODO add more options

        raise RuntimeError('Base options not comparable.')

    def _check_multiple_problem_types(self):

        problem_types = []
        for key in list(self.keys()):
            problem_types.append(self[key].ps.problem_type)

        if len(set(problem_types)) > 1:
            return True
        return False

    def _split_evaluator_summary(self, by='problem_type', **kwargs):

        # @TODO add more by options + auto detect

        subsets = {}
        for key in list(self.keys()):

            if by == 'problem_type':
                split = self[key].ps.problem_type

            # Add to subset by split value
            try:
                subsets[split][key] = self[key]
            except KeyError:
                subsets[split] = {key: self[key]}

        # Convert to individual summary dfs
        summary_dfs = {split: CompareDict(subsets[split]).summary(**kwargs)
                       for split in subsets}

        # Return separate summary per split in special 
        # MultipleSummary display object
        return MultipleSummary(summary_dfs)

    def _evaluator_summary(self, **kwargs):

        # TODO what about if the metrics
        # computed vary ... or one was
        # run w/o timing or something, handle these
        # cases

        # Un-pack kwargs
        show_timing = False
        if 'show_timing' in kwargs:
            show_timing = kwargs['show_timing']
        show_std = True
        if 'show_std' in kwargs:
            show_std = kwargs['show_std']

        # Setup
        keys = list(self.keys())
        repr_key = keys[0]
        option_keys = [o.key for o in repr_key.options]
        cols = {key: [] for key in option_keys}

        # Get flags for extra to show
        flags = self._check_evaluator_difs()

        # Go through each evaluator in self
        for key in list(self.keys()):

            for o in key.options:
                cols[o.key].append(o.name)

            # Add values
            evaluator = self[key]
            add_scores(cols, evaluator, attr_name='mean_scores')

            if show_std:
                add_scores(cols, evaluator, attr_name='std_scores')

            if show_timing:
                add_scores(cols, evaluator, attr_name='mean_timing')

            # Add any extra in flags
            for attr in flags:
                try:
                    cols[attr].append(getattr(evaluator, attr))
                except KeyError:
                    cols[attr] = [getattr(evaluator, attr)]

        summary = pd.DataFrame.from_dict(cols)
        summary = summary.set_index(option_keys)

        return summary

    def _check_evaluator_difs(self):

        # Init
        attrs = ['n_subjects', 'n_folds']
        attr_values = {attr: [] for attr in attrs}

        # Go through all
        for key in list(self.keys()):
            
            # Add each value
            for attr in attrs:
                attr_values[attr].append(getattr(self[key], attr))

        # If any not all save value, return flags
        flags = []
        for attr in attr_values:
            if len(set(attr_values[attr])) > 1:
                flags.append(attr)
        
        return flags


    def pairwise_t_stats(self, metric='first'):
        '''This method performs pair-wise t-test
        comparisons between all different options,
        assuming this object holds instances of :class:`BPtEvaluator`.
        The method used to generate t-test comparisons here is based off the
        example code from:
        https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_stats.html

        .. note::
            In the case that the sizes of the training and validation sets
            at each fold vary dramatically, it is unclear if this
            statistics are still valid.
            In that case, the mean train size and mean validation sizes
            are employed when computing statistics.

        Parameters
        ----------
        metric : str, optional
            This method compares the metrics
            produced for only one valid
            metric / scorer. Notably
            all :class:`BPtEvaluator` must have
            been evaluated with respect to this
            scorer. By default the reserved key, 'first'
            indicates that just whatever scorer is first
            should be used to produce the pairwise
            t statistics.

            ::

                default = 'first'

        Returns
        --------
        stats_df : pandas DataFrame
            A DataFrame comparing all pairwise
            combinations of the original :class:`Compare` options.
            't_stat' and 'p_val' columns will be generated for
            each comparison representing the corrected t_stat
            for non-independence of folds and the corresponding
            Bonferroni correctted p values (for multiple comparisons
            from comparing all pairwise combinations). See the
            referenced scikit-learn example for more information.
        '''

        # @TODO Clean up code and add error handling.

        ex_data_point = self.__getitem__(list(self.keys())[0])
        metrics = list(ex_data_point.mean_scores)

        # If metric is first
        if metric == 'first':
            metric = metrics[0]

        # Make sure valid metric
        if metric not in metrics:
            raise RuntimeError(f'{metric} not in avaliable {metrics}')

        n_comparisons = (
            factorial(len(self))
            / (factorial(2) * factorial(len(self) - 2))
        )

        pairwise_t_test = []
        for model_i, model_k in combinations(self, 2):

            # Grab scores
            model_i_scores = np.array(self[model_i].scores[metric])
            model_k_scores = np.array(self[model_k].scores[metric])

            # Skip if equal
            if np.array_equal(model_i_scores, model_k_scores):
                continue

            # Compute differences
            differences = model_i_scores - model_k_scores

            n = len(model_i_scores)
            df = n - 1

            # Use the mean train / test size
            n_train = np.mean([len(ti) for ti in self[model_i].train_subjects])
            n_test = np.mean([len(ti) for ti in self[model_i].val_subjects])

            # Do t-test
            t_stat, p_val =\
                compute_corrected_ttest(differences, df, n_train, n_test)

            # Implement Bonferroni correction
            p_val *= n_comparisons

            # Bonferroni can output p-values higher than 1
            p_val = 1 if p_val > 1 else p_val

            # Gen key entry names
            mi_option_names = [o.name for o in model_i.options]
            mk_option_names = [o.name for o in model_k.options]
            opt_names = mi_option_names + mk_option_names

            # Append
            pairwise_t_test.append(opt_names + [t_stat, p_val])

        ex = list(self)[0]
        key_names = [str(o.key) for o in ex.options]
        col_names = [key + ' (1)' for key in key_names] +\
            [key + ' (2)' for key in key_names]

        pairwise_comp_df = pd.DataFrame(
            pairwise_t_test, columns=col_names + ['t_stat', 'p_val'])

        return pairwise_comp_df

    def pairwise_bayesian():
        '''This method has not yet been implemented'''

        '''
        pairwise_bayesian = []

        for model_i, model_k in combinations(range(len(model_scores)), 2):
            model_i_scores = model_scores.iloc[model_i].values
            model_k_scores = model_scores.iloc[model_k].values
            differences = model_i_scores - model_k_scores
            t_post = t(
                df, loc=np.mean(differences),
                scale=corrected_std(differences, n_train, n_test)
            )
            worse_prob = t_post.cdf(rope_interval[0])
            better_prob = 1 - t_post.cdf(rope_interval[1])
            rope_prob = t_post.cdf(rope_interval[1]) - t_post.cdf(rope_interval[0])

            pairwise_bayesian.append([worse_prob, better_prob, rope_prob])

        pairwise_bayesian_df = (pd.DataFrame(
            pairwise_bayesian,
            columns=['worse_prob', 'better_prob', 'rope_prob']
        ).round(3))

        pairwise_comp_df = pairwise_comp_df.join(pairwise_bayesian_df)
        pairwise_comp_df
        '''

        return

class MultipleSummary():

    def __init__(self, summary_dfs):
        self.summary_dfs = summary_dfs

    def _repr_html_(self):

        template = """<div style="float: left; padding: 10px;">
        <h3>{0}</h3>{1}</div>"""

        html = ''
        for name in self.summary_dfs:
            html += template.format(name, self.summary_dfs[name]._repr_html_())
            html += '\n'

        return html

    def __repr__(self):

        return repr(self.summary_dfs)


def compare_dict_from_existing(results):
    '''Load in a :class:`CompareDict` from some combination
    of already run instances of :class:`BPtEvaluator`.

    Parameters
    -----------
    results : dict, list or str
        This parameter can be passed in a number
        of different ways.

        1. As a dictionary, where key-values
           correspond to either an instance of
           :class:`BPtEvaluator` or the str location
           of a pickle saved :class:`BPtEvaluator`.

        2. As a list, of either :class:`BPtEvaluator`
           instances, str file locations where they are
           saved, or a mix.

        3. As a str, corresponding to the location of
           a directory, where every file in the directory
           is a pickle saved instance of :class:`BPtEvaluator`.

        Note: The key values in the compare dictionary
        will be in the case that a dictionary is passed, the
        same keys as in the passed dictionary. In the case
        a list is passed, just a simple str index, or lastly
        in the case that either a directory of saved files
        or list of all files is passed, the key names will be
        auto-determined based on the passed file names.
    '''

    compare_dict = CompareDict()

    def _add_element(key, element):

        # If result, add directly
        if isinstance(element, BPtEvaluator):
            compare_dict[key] = element

        # If str, load as pickle
        elif isinstance(element, str):
            with open(element, 'rb') as f:
                compare_dict[key] = pkl.load(f)

        # Error if anything else
        else:
            raise RuntimeError(f'Unable to add {element} to CompareDict.')

    def _list_of_files_case(files):

        # Get unique names to add each file under
        file_unique_names = get_unique_str_markers(files)

        # Add all
        for key, file in zip(file_unique_names, files):
            _add_element(key, file)

    # Dict case
    if isinstance(results, dict):
        for key in results:
            _add_element(key, results[key])

    # List case
    elif isinstance(results, list):

        # Check for case where passing custom
        # list of files
        if all([isinstance(e, str) for e in results]):
            _list_of_files_case(results)

        # Otherwise add each as is
        else:
            for i, element in enumerate(results):
                _add_element(str(i), element)

    # If str, treat as directory of results case
    elif isinstance(results, str):

        # If doesn't exist
        if not os.path.exists(results):
            raise RuntimeError("Passed input as str, but loc does not exist.")

        # Get list of files
        files = [os.path.join(results, file_name)
                 for file_name in os.listdir(results)]

        # Go to list of files case
        _list_of_files_case(files)

    return compare_dict


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
