from copy import deepcopy
from sklearn.base import BaseEstimator
import pandas as pd
from ..helpers.ML_Helpers import conv_to_list, proc_input

from ..helpers.VARS import ORDERED_NAMES
from ..main.Input_Tools import is_duplicate, is_pipe, is_select, cast_input_to_scopes

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

        scope = getattr(self, 'scope')

        if isinstance(scope, list) and not is_duplicate(scope):
            raise IOError('Passed scope may not be list-like, unless it',
                          'is cast as a Duplicate')

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


class Loader(Params, Check):

    def __init__(self, obj, params=0, scope='data files', extra_params=None):

        self.obj = obj
        self.params = params
        self.scope = cast_input_to_scopes(scope)
        self.extra_params = extra_params

        self.check_args()

class Imputer(Params, Check):

    def __init__(self, obj, params=0, scope='float',
                 base_model=None, extra_params=None):

        self.obj = obj
        self.params = params
        self.scope = cast_input_to_scopes(scope)
        self.base_model = base_model
        self.extra_params = extra_params

        self.check_args()

class Scaler(Params, Check):

    def __init__(self, obj, params=0, scope='float', extra_params=None):

        self.obj = obj
        self.params = params
        self.scope = cast_input_to_scopes(scope)
        self.extra_params = extra_params

        self.check_args()

class Transformer(Params, Check):

    def __init__(self, obj, params=0, scope='float', extra_params=None):

        self.obj = obj
        self.params = params
        self.scope = cast_input_to_scopes(scope)
        self.extra_params = extra_params

        self.check_args()

class Sampler(Params, Check):

    def __init__(self, obj, params=0, sample_on='targets', extra_params=None):

        self.obj = obj
        self.params = params

        # Don't cast sample on to scope for now...
        self.sample_on = sample_on
        self.extra_params = extra_params

        self.check_args()

    def add_strat_u_name(self, func):
        self.sample_on = func(self.sample_on)

class Feat_Selector(Params, Check):

    def __init__(self, obj, params=0, base_model=None, extra_params=None):

        self.obj = obj
        self.params = params
        self.base_model = base_model
        self.extra_params = extra_params

        self.check_args()

class Model(Params, Check):

    def __init__(self, obj, params=0, extra_params=None):

        self.obj = obj
        self.params = params
        self.extra_params = extra_params
        self._is_model = True

        self.check_args()

class Ensemble(Params, Check):

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

class Model_Pipeline(Params):

    def __init__(self, loaders=None, imputers='default',
                 scalers='default', transformers=None,
                 samplers=None, feat_selectors=None,
                 model='default', param_search=None,
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
        self.cache = cache

        self._n_jobs = 1

        # Perform all preproc on input which can be run
        # more then once, these are essentially checks on the input
        self._proc_checks()

    def _proc_all_pieces(self, func):
        '''Helper to proc all pieces with a function
        that accepts a list of pieces'''

        for param_name in ORDERED_NAMES:
            params = getattr(self, param_name)

            if params is None:
                new_params = params

            elif isinstance(params, list):
                new_params = func(params)

            # If not list-
            else:
                new_params = func([params])
                if len(new_params) == 1:
                    new_params = new_params[0]

            setattr(self, param_name, new_params)

    def _proc_duplicates(self, params):

        # Want to preserve type
        new_params = deepcopy(params)
        del new_params[:]

        for param in params:
            new_params += self._proc_duplicate(param)

        return new_params

    def _proc_duplicate(self, param):

        if hasattr(param, 'scope'):
            scopes = getattr(param, 'scope')
            new_params = []

            if is_duplicate(scopes):
                for scope in scopes:
                    new = deepcopy(param)
                    new.scope = scope
                    new_params.append(new)

                return new_params
        return [param]

    def _proc_input(self, params):

        for p in range(len(params)):
            try:
                params[p].obj = proc_input(params[p].obj)
            except AttributeError:
                pass

        return params

    def _check_args(self, params):

        for p in params:
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

        # Double check input args in case something changed
        self._proc_all_pieces(self._check_args)

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

    def print_all(self, _print=print):

        _print('Model_Pipeline')
        _print('--------------')
        _print('loaders:'),
        _print('  ', self.loaders)
        _print('imputers:')
        _print('  ', self.imputers)
        _print('scalers:')
        _print('  ', self.scalers)
        _print('transformers:')
        _print('  ', self.transformers)
        _print('samplers:')
        _print('  ', self.samplers)
        _print('feat_selectors:')
        _print('  ', self.feat_selectors)
        _print('models:')
        _print('  ', self.model)
        _print('param_search:')
        _print('  ', self.param_search)
        _print('cache =', self.cache)
        _print()

class Problem_Spec(Params):

    def __init__(self, problem_type='regression',
                 target=0, metric='default', weight_metric=False,
                 scope='all', subjects='all',
                 feat_importance='default', n_jobs='default',
                 random_state='default'):

        self.problem_type = problem_type
        self.target = target
        self.metric = metric
        self.weight_metric = weight_metric
        self.scope = scope
        self.subjects = subjects

        if feat_importance == 'default':
            feat_importance = Feat_Importance('base')
        self.feat_importance = feat_importance

        self.n_jobs = n_jobs
        self.random_state = random_state

        self._final_subjects = None

    def set_final_subjects(self, final_subjects):
        self._final_subjects = final_subjects

    def get_model_spec(self):

        return {'problem_type': self.problem_type,
                'random_state': self.random_state}

    def print_all(self, _print=print):

        _print('Problem_Spec')
        _print('------------')
        _print('problem_type =', self.problem_type)
        _print('target =', self.target)
        _print('metric =', self.metric)
        _print('weight_metric =', self.weight_metric)
        _print('scope =', self.scope)
        _print('subjects =', self.subjects)
        _print('feat_importance:')
        _print('  ', self.feat_importance)
        _print('n_jobs', self.n_jobs)
        _print('random_state', self.random_state)
        _print()
