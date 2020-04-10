from copy import deepcopy
from sklearn.base import BaseEstimator
import pandas as pd
from .ML_Helpers import conv_to_list
from .Input_Tools import cast_input_to_scopes

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

class ML_Params(Params):

    def __init__(self, random_state=None):
        '''Init with default parameters'''

        self.problem_type = 'regression'
        self.target = 0
        self.model = 'linear'
        self.model_params = 0
        self.metric = 'r2'
        self.weight_metric = False
        self.loader = None
        self.loader_scope = 'data files'
        self.loader_params = 0
        self.imputer = ['mean', 'median']
        self.imputer_scope = ['float', 'cat']
        self.imputer_params = 0
        self.scaler = 'standard'
        self.scaler_scope = 'float'
        self.scaler_params = 0
        self.transformer = None
        self.transformer_scope = 'float'
        self.transformer_params = 0
        self.sampler = None
        self.sample_on = 'targets'
        self.sampler_params = 0
        self.feat_selector = None
        self.feat_selector_params = None
        self.ensemble = None
        self.ensemble_split = .2
        self.ensemble_params = 0
        self.splits = 3
        self.n_repeats = 2
        self.search_type = None
        self.search_splits = 3
        self.search_n_iter = 10
        self.scope = 'all'
        self.subjects = 'all'
        self.feat_importances = 'base'
        self.feat_importances_params = 0
        self.n_jobs = 1
        self.random_state = random_state
        self.compute_train_score = False
        self.cache = None
        self.extra_params = {}

        self._final_subjects = None
        self._split_vals = None
        self._search_split_vals = None
        self._base_n_jobs = None

    def setup(self):

        self._cast_scopes()
        self._conv_to_lists()
        self._set_base_n_jobs()

    def _cast_scopes(self):
        
        to_conv = ['loader_scope', 'imputer_scope', 'scaler_scope',
                   'transformer_scope', 'sample_on', 'scope']

        for c in to_conv:
            as_scope = cast_input_to_scopes(getattr(self, c))
            setattr(self, c, as_scope)

    def _conv_to_lists(self):

        to_conv = ["model", "model_params", "metric",
                   "weight_metric", "loader", "loader_scope",
                   "loader_params", "imputer", "imputer_scope",
                   "imputer_params", "scaler", "scaler_scope",
                   "scaler_params", "transformer", "transformer_scope",
                   "transformer_params", "sampler", "sample_on",
                   "sampler_params", "feat_selector", "feat_selector_params",
                   "ensemble", "ensemble_params", "feat_importances", 
                   "feat_importances_params"]

        for c in to_conv:
            as_list = conv_to_list(getattr(self, c))
            setattr(self, c, as_list)

    def _set_base_n_jobs(self):

        # Essentially, if going to wrap in a search object
        # will use n_jobs for param search, so set all
        # internal base pipeline objs to n_jobs = 1
        if self.search_type is None:
            self._base_n_jobs = self.n_jobs
        else:
            self._base_n_jobs = 1

    def _get_param_names(self):
        return [i for i in self.__dict__.keys() if i[:1] != '_']

    def check_imputer(self, data):

        # If no NaN, no imputer
        if not pd.isnull(data).any().any():
            self.imputer = None

    def set_final_subjects(self, final_subjects):
        self._final_subjects = final_subjects

    def set_split_vals(self, split_vals, search_split_vals):
        self._split_vals = split_vals
        self._search_split_vals = search_split_vals

    def _get_scope_str(self, obj):

        if len(obj) > 50:
            return 'Custom Scope of len = ' + str(len(obj))
        else:
            return obj

    def print_model_params(self, test=False, _print=print):

        if test:
            _print('----Running Test----')
        else:
            _print('----Running Evaluate----')

        _print()

        _print('Experiment Params:')
        _print('problem_type =', self.problem_type)
        _print('target =', self.target)
        _print('metric =', self.metric)

        if not test:
            _print('splits =', self.splits)
            _print('n_repeats =', self.n_repeats)
        
        if self.weight_metric:
            _print('weight_metric =', self.weight_metric)

        _print()
        _print('Model Pipeline Params:')

        _print('model =', self.model)

        if self.loader is not None:
            _print('loader =', self.loader)
            _print('loader_scope =', self._get_scope_str(self.loader_scope))

        if self.imputer is not None:
            _print('imputer =', self.imputer)
            _print('imputer_scope =', self._get_scope_str(self.imputer_scope))

        if self.scaler is not None:
            _print('scaler =', self.scaler)
            _print('scaler_scope =', self._get_scope_str(self.scaler_scope))

        if self.transformer is not None:
            _print('transformer =', self.transformer)
            _print('transformer_scope =', self._get_scope_str(self.transformer_scope))

        if self.sampler is not None:
            _print('sampler =', self.sampler)
            _print('sample_on =', self.sample_on)

        if self.feat_selector is not None:
            _print('feat_selector', self.feat_selector)
           
        ensmb_flag = self.ensemble is not None
        if isinstance(self.model, list) or ensmb_flag:

            _print('ensemble =', self.ensemble)
            _print('ensemble_split =', self.ensemble_split)

        _print()
        _print('Search Params:')
        _print('search_type =', self.search_type)

        if self.search_type is not None:
            _print('search_splits =', self.search_splits)
            _print('search_n_iter =', self.search_n_iter)

            if self.model_params != 0:
                _print('model_params =', self.model_params)

            if self.loader_params != 0:
                _print('loader_params =', self.loader_params)

            if self.imputer_params != 0:
                _print('imputer_params =', self.imputer_params)

            if self.scaler_params != 0:
                _print('scaler_params =', self.scaler_params)

            if self.transformer_params != 0:
                _print('transformer_params =', self.transformer_params)

            if self.sampler_params != 0:
                _print('sampler_params =', self.sampler_params)

            if self.feat_selector_params != 0:
                 _print('feat_selector_params =',
                        self.feat_selector_params)

            if self.ensemble_params != 0:
                _print('ensemble_params =', self.ensemble_params)

        _print()
        _print('Remaining Params:')

        _print('n_jobs =', self.n_jobs)
        _print('scope =', self._get_scope_str(self.scope))

        if len(self.subjects) > 20:
            _print('subjects = Custom passed keys with len =',
                    len(self.subjects))
        else:
            _print('subjects =', self.subjects)

        if self.compute_train_score:
            _print('compute_train_score =', self.compute_train_score)

        _print('random_state =', self.random_state)

        if self.feat_importances is not None:

            _print('feat_importances =',
                    self.feat_importances)
            _print('feat_importances_params =',
                    self.feat_importances_params)

        if self.cache is not None:
            _print('cache =', self.cache)
        if len(self.extra_params) > 0:
            _print('extra_params =', self.extra_params)

        _print()
        
