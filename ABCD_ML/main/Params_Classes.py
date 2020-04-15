from copy import deepcopy
from sklearn.base import BaseEstimator
import pandas as pd
from ..helpers.ML_Helpers import conv_to_list, proc_input

from ..helpers.VARS import ORDERED_NAMES
from ..main.Input_Tools import is_duplicate, is_pipe, is_select, is_special
from pprint import PrettyPrinter
import inspect

def proc_all(base_obj):

    if base_obj is None:
        return

    proc_name(base_obj, 'obj')
    proc_name(base_obj, 'base_model')
    proc_name(base_obj, 'models')
    proc_name(base_obj, 'metric')

def proc_name(base_obj, name):

    if hasattr(base_obj, name):
        obj = getattr(base_obj, name)

        if isinstance(obj, list):
            for o in obj:
                proc_all(o)

        elif hasattr(obj, 'obj'):
            proc_all(obj)

        elif isinstance(obj, str):
            setattr(base_obj, name, proc_input(obj))

class Params(BaseEstimator):

    def set_values(self, args):

        for key in args:
            if key != 'self' and hasattr(self, key):

                try:
                    if isinstance(args[key], str):
                        if args[key] != 'default':
                            setattr(self, key, args[key])
                    else:
                        setattr(self, key, args[key])

                # If value error, set to key
                except ValueError:
                    setattr(self, key, args[key])

    def copy(self):
        return deepcopy(self)

class Check():

    def check_args(self):

        if hasattr(self, 'obj'):
            self._check_obj()

        if hasattr(self, 'params'):
            self._check_params()

        if hasattr(self, 'obj') and hasattr(self, 'params'):
            self._check_obj_and_params()

        if hasattr(self, 'scope'):
            self._check_scope()

        if hasattr(self, 'extra_params'):
            self._check_extra_params()

        if hasattr(self, 'base_model'):
            self._check_base_model()

        try:
            self.check_extra_args()
        except AttributeError:
            pass

    def _check_obj(self):

        obj = getattr(self, 'obj')

        if obj is None:
            raise IOError('passed obj cannot be None, to ignore the object itself',
                          'set it within Model_Pipeline to None, not obj here.')

        if isinstance(obj, list) and not is_pipe(obj):
            raise IOError('You may only pass a list of objs with special input',
                          'Pipe()')

    def _check_params(self):

        params = getattr(self, 'params')

        if params is None:
            setattr(self, 'params', 0)

    def _check_obj_and_params(self):

        obj = getattr(self, 'obj')
        params = getattr(self, 'params')

        if is_pipe(obj):
            
            if not isinstance(params, list):

                if params == 0:
                    new_params = [0 for i in range(len(obj))]
                    setattr(self, 'params', new_params)
                else:
                    raise IOError('obj is passed as Pipe, but params was not. Make sure',
                                'params is either a list or Pipe with the same length as',
                                'self.obj')

            elif len(params) != len(obj):

                if params == [0]:
                    new_params = [0 for i in range(len(obj))]
                    setattr(self, 'params', new_params)
                else:
                    raise IOError('obj is passed as Pipe, Make sure',
                                'params is either a list or Pipe with the same length as',
                                'self.obj')

    def _check_scope(self):

        pass
        # Nothing to check as of right now
        # scope = getattr(self, 'scope')
        
    def _check_extra_params(self):

        extra_params = getattr(self, 'extra_params')
        if extra_params is None:
            setattr(self, 'extra_params', {})

        elif not isinstance(extra_params, dict):
            raise IOError('extra params must be a dict!')

    def _check_base_model(self):

        base_model = getattr(self, 'base_model')

        # None is okay
        if base_model is None:
            return

        if not hasattr(base_model, '_is_model'):
            raise IOError('base_model must be either None or a valid Model / Ensemble',
                          'set of wrapeper params!')

class Piece(Params, Check):
    pass

class Loader(Piece):

    def __init__(self, obj, params=0, scope='data files', extra_params=None):

        self.obj = obj
        self.params = params
        self.scope = scope
        self.extra_params = extra_params

        self.check_args()

class Imputer(Piece):

    def __init__(self, obj, params=0, scope='float',
                 base_model=None, extra_params=None):

        self.obj = obj
        self.params = params
        self.scope = scope
        self.base_model = base_model
        self.extra_params = extra_params

        self.check_args()

class Scaler(Piece):

    def __init__(self, obj, params=0, scope='float', extra_params=None):

        self.obj = obj
        self.params = params
        self.scope = scope
        self.extra_params = extra_params

        self.check_args()

class Transformer(Piece):

    def __init__(self, obj, params=0, scope='float', extra_params=None):

        self.obj = obj
        self.params = params
        self.scope = scope
        self.extra_params = extra_params

        self.check_args()

class Sampler(Piece):

    def __init__(self, obj, params=0, sample_on='targets', extra_params=None):

        self.obj = obj
        self.params = params

        # Don't cast sample on to scope for now...
        self.sample_on = sample_on
        self.extra_params = extra_params

        self.check_args()

    def add_strat_u_name(self, func):
        self.sample_on = func(self.sample_on)

class Feat_Selector(Piece):

    def __init__(self, obj, params=0, base_model=None, extra_params=None):

        self.obj = obj
        self.params = params
        self.base_model = base_model
        self.extra_params = extra_params

        self.check_args()

class Model(Piece):

    def __init__(self, obj, params=0, extra_params=None):

        self.obj = obj
        self.params = params
        self.extra_params = extra_params
        self._is_model = True

        self.check_args()

class Ensemble(Piece):

    def __init__(self, obj, models, params=0, needs_split=False,
                 des_split=.2, single_estimator=False, cv=3,
                 extra_params=None):

        self.obj = obj
        self.models = models
        self.params = params
        self.needs_split = needs_split
        self.des_split = des_split
        self.single_estimator = single_estimator
        self.cv = cv
        self.extra_params = extra_params
        self._is_model = True

        self.check_args()

    def check_extra_args(self):

        if isinstance(self.models, list):
            for model in self.models:
                if not hasattr(model, '_is_model'):
                    raise IOError('All models must be valid Model/Ensemble param wrapped!')

        else:
            if not hasattr(self.models, '_is_model'):
                raise IOError('Passed model in models must be a valid Model/Ensemble, i.e., param wrapped.')

class Param_Search(Params):

    def __init__(self, search_type='RandomSearch',
                 splits=3, n_iter=10, metric='default',
                 weight_metric=False):
        
        self.search_type = search_type
        self.splits = splits
        self.n_iter = n_iter
        self.metric = metric
        self.weight_metric = weight_metric

        self._splits_vals = None
        self._n_jobs = 1

    def set_split_vals(self, vals):
        self._splits_vals = vals

    def check_args(self):

        if isinstance(self.metric, list):
            raise IOError('metric within Param Search cannot be list-like')
        if isinstance(self.weight_metric, list):
            raise IOError('weight_metric within Param Search cannot be list-like')

class Feat_Importance(Params):

    def __init__(self, obj, metric='default'):
        # Likely want other params here!

        self.obj = obj
        self.metric = metric

class Drop_Strat():

    def __init__(self):

        self.obj = None
        self.params = 0

    def check_args(self):
        return

    def proc_input(self):
        return

class Model_Pipeline(Params):
    '''feat_importances : Feat_Importance or list of, optional
            Provide here either a single, or list of Feat_Importance param objects
            in which to specify what importance values, and with what settings should be computed.
            See the Feat_Importance param obj for more info.
    '''

    def __init__(self, loaders=None, imputers='default',
                 scalers='default', transformers=None,
                 samplers=None, feat_selectors=None,
                 model='default', param_search=None,
                 feat_importances='default',
                 cache=None):

        self.loaders = loaders
        
        if imputers == 'default':
            imputers = [Imputer('mean', scope='float'),
                        Imputer('median', scope='cat')]
        self.imputers = imputers

        if scalers == 'default':
            scalers = Scaler('standard')
        self.scalers = scalers

        self.transformers = transformers
        self.samplers = samplers

        # Special place holder case for drop strat, for compat.
        self._drop_strat = Drop_Strat()

        self.feat_selectors = feat_selectors

        if model == 'default':
            model = Model('linear')
        self.model = model

        self.param_search = param_search

        if feat_importances == 'default':
            feat_importances = Feat_Importance('base')
        self.feat_importances = feat_importances

        self.cache = cache

        self._n_jobs = 1

        # Perform all preproc on input which can be run
        # more then once, these are essentially checks on the input
        self._proc_checks()

    def _proc_all_pieces(self, func):
        '''Helper to proc all pieces with a function
        that accepts a list of pieces'''

        for param_name in ORDERED_NAMES:

            if param_name != '_drop_strat':
                params = getattr(self, param_name)

                if params is None:
                    new_params = params

                elif isinstance(params, list):
                    new_params = func(params)

                # If not list-
                else:
                    new_params = func([params])
                    if isinstance(new_params, list) and len(new_params) == 1:
                        new_params = new_params[0]

                setattr(self, param_name, new_params)

    def _proc_duplicates(self, params):

        if isinstance(params, list):
            new_params = deepcopy(params)
            del new_params[:]

            if len(params) == 1:
                return self._proc_duplicates(params[0])

            for p in params:
                new_params.append(self._proc_duplicates(p))

            return new_params

        elif hasattr(params, 'scope'):
            scopes = getattr(params, 'scope')
            new_params = []

            if is_duplicate(scopes):
                for scope in scopes:
                    new = deepcopy(params)
                    new.scope = scope
                    new_params.append(new)

                return new_params

        return params

    def _proc_input(self, params):

        for p in params:
            if isinstance(p, list):
                self._proc_input(p)
            else:
                proc_all(p)

        return params

    def _check_args(self, params):

        for p in params:
            if isinstance(p, list):
                self._check_args(p)
            else:
                p.check_args()

        return params

    def check_imputer(self, data):

        # If no NaN, no imputer
        if not pd.isnull(data).any().any():
            self.imputer = None

    def check_samplers(self, func):
        if self.samplers is not None:
            if isinstance(self.samplers, list):
                for s in self.samplers:
                    s.add_strat_u_name(func)
            else:
                self.samplers.add_strat_u_name(func)

    def preproc(self, n_jobs):

        # Base proc checks
        self._proc_checks()

        # Store n_jobs
        self.set_n_jobs(n_jobs)
        
    def _proc_checks(self):

        # Check for duplicate scopes
        self._proc_all_pieces(self._proc_duplicates)

        # Proc input
        self._proc_all_pieces(self._proc_input)
        proc_all(self.param_search)
        proc_all(self.feat_importances)

        # Double check input args in case something changed
        self._proc_all_pieces(self._check_args)

        # Proc param search if not None
        if self.param_search is not None:
            self.param_search.check_args()

    def set_n_jobs(self, n_jobs):

        # If no param search, each objs base n_jobs is
        # the passed n_jobs
        if self.param_search is None:
            self._n_jobs = n_jobs

        # Otherwise, base jobs are 1, and the search_n_jobs
        # are set to passed n_jobs
        else:
            self._n_jobs = 1
            self.param_search._n_jobs = n_jobs

    def get_ordered_pipeline_params(self):

        # Conv all to list & return in order as a deep copy
        return deepcopy([conv_to_list(getattr(self, piece_name))
                         for piece_name in ORDERED_NAMES])

    def params_print(self, params, indent, _print=print, end='\n'):

        if not isinstance(params, list):
            _print(' ' * indent, params, sep='', end=end)
            return

        elif len(params) == 1:
            self.params_print(params[0], indent, _print=print)
            return

        elif not is_special(params):
            
            _print('[', sep='', end='')
            self._p_stack.append(']')
            self.params_print(params[0], indent, _print=_print, end='')
            _print(',', sep='')
            indent += 1

        elif is_select(params):

            _print('Select([', sep='', end='')
            self._p_stack.append('])')
            self.params_print(params[0], indent, _print=_print, end='')
            _print(',', sep='')
            indent += 9

        for param in params[1:-1]:
            self.params_print(param, indent, _print=_print, end='')
            _print(',', sep='')
    
        self.params_print(params[-1], indent, _print=_print, end='')
        _print(self._p_stack.pop(), end=end)

        return

    def print_all(self, _print=print):

        self._p_stack = []
        
        _print('Model_Pipeline')
        _print('--------------')

        pipeline_params = self.get_ordered_pipeline_params()
        for name, params in zip(ORDERED_NAMES, pipeline_params):

            if name == '_drop_strat':
                pass

            elif params is None:
                pass

            else:
                _print(name + '=\\')
                self.params_print(params, 0, _print=_print)
                _print()

        _print('param_search=\\')
        _print(self.param_search)
        _print()
        _print('feat_importances=\\')
        _print(self.feat_importances)
        _print()

        if self.cache is not None:
            _print('cache =', self.cache)
        _print()

class Problem_Spec(Params):

    def __init__(self, problem_type='regression',
                 target=0, metric='default', weight_metric=False,
                 scope='all', subjects='all',
                 n_jobs='default',
                 random_state='default'):
        '''Problem Spec is defined as an object of params encapsulating the set of
        parameters shared by modelling class functions
        :func:`Evaluate <ABCD_ML.ABCD_ML.Evaluate>` and :func:`Test <ABCD_ML.ABCD_ML.Test>`

        Parameters
        ----------
        problem_type : str, optional

            - 'regression'
                For ML on float/continuous target data.

            - 'binary'
                For ML on binary target data.

            - 'categorical'
                For ML on categorical target data, as multiclass.

            (default = 'regression')

        target : int or str, optional
            The loaded target in which to use during modelling.
            This can be the int index (as assigned by order loaded in,
            e.g., first target loaded is 0, then the next is 1),
            or the name of the target column.
            If only one target is loaded, just leave as default of 0.

            (default = 0)

        metric : str or list, optional
            Indicator str for which metric(s) to use when calculating
            average validation score in Evaluate, or Test set score in Test.

            A list of str's can be passed as well, in this case, scores for
            all of the requested metrics will be calculated and returned.

            Note: If using a Param_Search, the Param_Search object has a
            metric parameter as well. This metric describes the metric optimized
            in a parameter search. In the case that this metric is left as 'default',
            the the first metric passed here will be used in Param_Search!

            For a full list of supported metrics please view the docs at :ref:`Metrics`

            If left as 'default', assign a reasonable metric based on the
            passed problem type.

            - 'regression'  : 'r2'
            - 'binary'      : 'macro roc auc'
            - 'categorical' : 'matthews'

            (default = 'default')

        weight_metric : bool, list of, optional
            If True, then the metric of interest will be weighted within
            each repeated fold by the number of subjects in that validation set.
            This parameter only typically makes sense for custom split behavior where 
            validation folds may end up with differing sizes.
            When default CV schemes are employed, there is likely no point in
            applying this weighting, as the validation folds will have simmilar sizes.

            If you are passing mutiple metrics, then you can also pass a 
            list of values for weight_metric, with each value set as boolean True or False,
            specifying if the corresponding metric by index should be weighted or not.

            (default = False)

        scope : key str or Scope obj, optional
            This parameter allows the user to optionally
            run an expiriment with just a subset of the loaded features
            / columns.

            See :ref:`Scopes` for a more detailed explained / guide on how scopes
            are defined and used within ABCD_ML.

            In general, scopes can be passed as either a preset key (seen below),
            or more complex composition created with the Scope wrapper. See the options
            below

            - 'float'
                To apply to all non-categorical columns, in both
                loaded data and covars.

            - 'data'
                To apply to all loaded data columns only.

            - 'data files'
                To apply to just columns which were originally loaded as data files.

            - 'float covars' or 'fc'
                To apply to all non-categorical, float covars columns only.

            - 'all'
                To apply to everything, regardless of float/cat or data/covar.

            - 'cat' or 'categorical'
                To apply to just loaded categorical data.

            - 'covars'
                To apply to all loaded covar columns only.

            - array-like of strs (not list)
                Can pass specific col names in as array-like
                to select only those cols.

            - array-like of wildcards / mixed
                See :ref:`Scopes` for how to use these more complex input
                options

            - Duplicate 
                Can pass scopes wrapped in Duplicate, to replicate the base
                object for each seperate scope.

        subjects : str, array-like or Value_Subset, optional
            This parameter allows the user to optionally run Evaluate or Test
            with just a subset of the loaded subjects. It is notably distinct
            from the `train_subjects`, and `test_subjects` parameters directly
            avaliable to Evaluate and Test, as those parameters typically refer
            to train/test splits. Specifically, any value specified for this 
            subjects parameter will be applied AFTER selecting the relevant train
            or test subset.

            One use case for this parameter might be specifying subjects of just one
            sex, where you would still want the same training set for example, but just want
            to test sex specific models.

            If set to 'all' (as is by default), all avaliable subjects will be
            used.

            `subjects` can accept either a specific array of subjects,
            or even a loc of a text file (formatted one subject per line) in
            which to read from.

            A special wrapper, Value_Subset, can also be used to specify more specific,
            specifically value specific, subsets of subjects to use. This wrapper can be used
            as follows, pass Value_Subset(name, value), where name is the name of a loader Strat
            column / feature, and value is the subset of values from that column to select subjects by.
            E.g., in the example above where you want to specify just a single sex, you could pass
            Value_Subset('sex', 0), to pass along only subjects with value == 0.
            You may also pass a list-like set of multiple columns to name, in this case, the overlap
            across all passed names will be computed, E.G., Value_Subset(['sex', 'race'], 0), would select
            only subjects with a value of 0 in the computed unique overlap across 'sex' and 'race'.

            (default = 'all')

        

        '''

        self.problem_type = problem_type
        self.target = target

        if metric == 'default':
            default_metrics = {'regression': 'r2',
                               'binary': 'macro roc auc',
                               'categorical': 'matthews'}
            metric = default_metrics[self.problem_type]

        self.metric = metric
        self.weight_metric = weight_metric
        self.scope = scope
        self.subjects = subjects

        self.n_jobs = n_jobs
        self.random_state = random_state

        self._final_subjects = None

        self._proc_checks()

    def _proc_checks(self):
        
        proc_all(self)
        

    def set_final_subjects(self, final_subjects):
        self._final_subjects = final_subjects

    def get_model_spec(self):

        return {'problem_type': self.problem_type,
                'random_state': self.random_state}

    def print_all(self, _print=print):

        pp = PrettyPrinter(indent=2)

        _print('Problem_Spec')
        _print('------------')
        _print('problem_type =', self.problem_type)
        _print('target =', self.target)
        _print('metric =', self.metric)
        _print('weight_metric =', self.weight_metric)
        _print('scope =', self.scope)
        _print('subjects =', self.subjects)
        _print('len(subjects) =', len(self._final_subjects),
               '(before overlap w/ train/test subjects)')
        _print('n_jobs', self.n_jobs)
        _print('random_state', self.random_state)
        _print()
