from sklearn.preprocessing import FunctionTransformer

from ..helpers.ML_Helpers import (proc_input,
                                  user_passed_param_check, update_extra_params,
                                  check_for_duplicate_names, wrap_pipeline_objs,
                                  proc_type_dep_str, param_len_check)

from .extensions.Col_Selector import ColTransformer, InPlaceColTransformer
from sklearn.ensemble import VotingClassifier, VotingRegressor

from sklearn.pipeline import Pipeline
from copy import deepcopy

import numpy as np


class Pieces():

    def __init__(self, obj_strs, param_strs, scopes=None, user_passed_objs=None,
                 problem_type='regression', search_type=None,
                 random_state=None, Data_Scopes=None, n_jobs=1,
                 extra_params=None, _print=print):
        '''Accept all params for sub-classes, but if not relevant just set to None
        and ignore.'''

        self.obj_strs = obj_strs
        self.param_strs = param_strs
        self.scopes = scopes

        self.user_passed_objs = user_passed_objs
        self.problem_type = problem_type
        self.search_type = search_type
        self.random_state = random_state
        self.Data_Scopes = Data_Scopes
        self.n_jobs = n_jobs

        self.AVAILABLE = None

        if extra_params is None:
            self.extra_params = {}
        else:
            self.extra_params = extra_params

        self._print = _print

        # Process according to child class
        self.process()

    def process(self):

        if self.obj_strs is None:
            self.objs = []
            self.obj_params = {}

        else:
            self.objs, self.obj_params = self._process()

    # Overide in children class
    def _process(self):
        return [], {}

    def _get_objs_and_params(self, get_func, names, params):
        '''Helper function to grab scaler / feat_selectors and
        their relevant parameter grids'''

        # Make the object + params based on passed settings
        objs_and_params = []
        for name, param in zip(names, params):

            if 'user passed' in name:
                objs_and_params.append(self._get_user_passed_obj_params(name, param))

            else:
                objs_and_params.append((name, get_func(name, self.extra_params,
                                                       param, self.search_type,
                                                       self.random_state,
                                                       self.Data_Scopes.num_feat_keys)
                                       ))

        # Perform extra proc, to split into objs and merged param dict
        objs, params = self._proc_objs_and_params(objs_and_params)

        return objs, params

    def _checks(self, names, params, scopes=None):

        conv_names = self._proc_in_str(names)

        # Check + fill length of passed params + names
        params = param_len_check(conv_names, params, _print=self._print)

        # Check search type
        params = self._check_params_by_search(params)

        # Try proc scopes
        if scopes is not None:
            conv_names, params, scopes = self._scope_check(conv_names, params, scopes)

            return conv_names, params, scopes

        return conv_names, params

    def _proc_in_str(self, in_strs):

        conv_strs = proc_input(in_strs)
        self.extra_params = update_extra_params(self.extra_params, in_strs, conv_strs)
        return conv_strs

    def _check_params_by_search(self, params):

        if self.search_type is None:

            if (np.array(params) != 0).any():
                self._print('Note, Search type is set to None! Therefore no',
                            'hyper-param search will be conducted, even',
                            'though params were passed.',
                            'Params will be still be set if passing default values.')
                self._print()

        return params

    def _scope_check(self, names, params, scopes):

        # First just case where one obj passed
        if len(names) == 1 and len(scopes) > 1:
            scopes = [scopes]

        # Now the lengths should be equal or else bad input
        if len(names) != len(scopes):

            raise RuntimeError('Warning: non equal number of',
                               names, 'passed as', scopes)

        new_names, new_scopes, new_params = [], [], []
        for name, scope, param in zip(names, scopes, params):

            if isinstance(scope, tuple):
                for s in scope:

                    new_names.append(name)
                    new_scopes.append(s)
                    new_params.append(param)

            else:

                new_names.append(name)
                new_scopes.append(scope)
                new_params.append(param)

        return new_names, new_params, new_scopes

    def _get_user_passed_obj_params(self, name, param):

        user_obj = self.user_passed_objs[name]

        # proc as necc. user passed params
        extra_user_obj_params, user_obj_params =\
            user_passed_param_check(param, name, self.search_type)

        # If passing a user object, kind of stupid to pass default, non search params
        # via a dict..., but hey...
        try:
            user_obj.set_params(**extra_user_obj_params)
        except AttributeError:
            pass

        # Return in obj_param format
        return (name, (user_obj, user_obj_params))

    def _proc_objs_and_params(self, objs_and_params):

        # If two of same object passed, change name
        objs_and_params = check_for_duplicate_names(objs_and_params)

        # Construct the obj as list of (name, obj) tuples, checking each base
        # objects params
        objs = [(c[0], self._check_params(c[1][0])) for c in objs_and_params]

        # Grab the params, and merge them into one dict of all params
        params = {k: v for params in objs_and_params
                  for k, v in params[1][1].items()}

        return objs, params

    def _check_params(self, obj):

        try:
            obj.random_state = self.random_state
        except AttributeError:
            pass

        try:
            obj.n_jobs = self.n_jobs
        except AttributeError:
            pass
        
        return obj


class Type_Pieces(Pieces):

    def _base_type_process(self, func):

        # Perform checks first
        obj_strs, param_strs = self._checks(self.obj_strs, self.param_strs)

        # Then call get objs and params
        objs, obj_params =\
            self._get_objs_and_params(func, obj_strs, param_strs)

        return objs, obj_params

    def _proc_in_str(self, in_strs):
        '''Helper function to perform str correction on
        underlying proble type dependent input, e.g., for
        metric or ensemble_types, and to update extra params
        and check to make sure input is valid ect...'''

        final_strs, self.extra_params =\
            proc_type_dep_str(in_strs, self.AVAILABLE, self.extra_params, self.problem_type)

        return final_strs


class Scope_Pieces(Pieces):

    def _wrap_pipeline_objs(self, wrapper, objs, scopes, **params):

        inds = [self.Data_Scopes.get_inds_from_scope(scope) for scope in scopes]

        objs = wrap_pipeline_objs(wrapper, objs, inds,
                                  random_state=self.random_state,
                                  n_jobs=self.n_jobs, **params)

        return objs

    def _make_col_version(self, objs, params, scopes):

        # Make objects first
        col_objs = []

        for i in range(len(objs)):
            name, obj = objs[i][0], objs[i][1]
            inds = self.Data_Scopes.get_inds_from_scope(scopes[i])

            col_obj =\
                ('col_' + name, InPlaceColTransformer([(name, obj, inds)],
                 remainder='passthrough', sparse_threshold=0))

            col_objs.append(col_obj)

        # Change params to reflect objs
        col_params = {}

        for key in params:
            name = key.split('__')[0]
            new_name = 'col_' + name + '__' + key

            col_params[new_name] = params[key]

        return col_objs, col_params
    

class Models(Type_Pieces):

    name = 'models'

    def _process(self):

        from .Models import get_base_model_and_params, AVALIABLE
        self.AVAILABLE = AVALIABLE.copy()

        return self._base_type_process(get_base_model_and_params)

    def apply_ensembles(self, ensembles, ensemble_split):

        from .Ensembles import Ensemble_Wrapper

        wrapper = Ensemble_Wrapper(self.obj_params,
                                   ensembles.obj_params,
                                   ensembles._get_base_ensembler,
                                   self.n_jobs)

        # Replace models w/ ensembled models
        self.objs = wrapper.wrap_ensemble(self.objs, ensembles.objs,
                                          ensemble_split, self.random_state)

        # Get updated params
        self.obj_params, ensembles.obj_param = wrapper.get_updated_params()

        # Lastly, ensembles objs should now be set to just empty list
        ensembles.objs = []


class Loaders(Scope_Pieces):

    name = 'loaders'

    def _get_split_by_tuple(self, passed_obj_strs, passed_param_strs, passed_scopes):

        non_tuple_strs, tuple_strs = [], []
        non_tuple_params, tuple_params = [], []
        non_tuple_scopes, tuple_scopes = [], []

        for i in range(len(passed_obj_strs)):

            if isinstance(passed_obj_strs[i], tuple):
                tuple_strs.append(passed_obj_strs[i])
                tuple_scopes.append(passed_scopes[i])

                # If not within index set to 0
                try:
                    param_str = passed_param_strs[i]
                except IndexError:
                    param_str = 0

                # Passed params not tuple, just set to tuple version of same value
                if isinstance(param_str, tuple):
                    tuple_params.append(param_str)
                else:
                    as_tuple = tuple([param_str for j in range(len(passed_obj_strs[i]))])
                    tuple_params.append(as_tuple)

            else:
                non_tuple_strs.append(passed_obj_strs[i])
                non_tuple_scopes.append(passed_scopes[i])

                try:
                    non_tuple_params.append(passed_param_strs[i])
                except IndexError:
                    non_tuple_params.append(0)

        tuples = (tuple_strs, tuple_params, tuple_scopes)
        non_tuples = (non_tuple_strs, non_tuple_params, non_tuple_scopes)

        return tuples, non_tuples

    def _process_tuples(self, tuples, get_func):
        
        # Process all of the tuples
        cnt = 0
        ordered_tuple_pipelines, ordered_tuple_scopes = [], []
        passed_loader_params = {}

        for x in range(len(tuples[0])):
            strs, params, scopes = tuples[0][x], tuples[1][x], tuples[2][x]

            # Conv to list
            loader_strs = list(strs)
            loader_params = list(params)

            # Set scopes
            original_len = len(loader_strs)
            loader_scopes = [scopes for i in range(original_len)]

            checked = self._checks(loader_strs, loader_params, loader_scopes)

            # If pass scope will return obj, param, scope
            loaders, loader_params =\
                self._get_objs_and_params(get_func, checked[0], checked[1])

            loader_scopes = checked[2]
            
            ordered_tuple_pipeline = []
            ordered_tuple_scope = []

            # Handle duplicates if any
            n_duplicates = int(len(loaders) / original_len)
            for i in range(n_duplicates):

                # Create a pipeline of the combined objects
                name = 'loader_combo' + str(cnt)
                rel_loaders = [loaders[l] for l in range(i, len(loaders), n_duplicates)]
                ordered_tuple_pipeline.append((name, Pipeline(steps = rel_loaders)))
                
                # Keep track of scope order too
                ordered_tuple_scope.append(loader_scopes[i])
                
                # Proc params,
                keys = [step[0] for step in rel_loaders]
                rel_keys = [key for key in loader_params if key.split('__')[0] in keys]
                pipeline_params = {name + '__' + key: loader_params[key] for key in rel_keys}

                # No order, so just add
                passed_loader_params.update(pipeline_params)
                cnt += 1

            ordered_tuple_pipelines.append(ordered_tuple_pipeline)
            ordered_tuple_scopes.append(ordered_tuple_scope)

        return passed_loader_params, (ordered_tuple_pipelines, ordered_tuple_scopes)

    def _process_non_tuples(self, non_tuples, get_func, passed_loader_params):

        checked = self._checks(non_tuples[0], non_tuples[1], non_tuples[2])

        loaders, loader_params =\
            self._get_objs_and_params(get_func, checked[0], checked[1])
        loader_scopes = checked[2]

        passed_loader_params.update(loader_params)
        return passed_loader_params, (loaders, loader_scopes)

    def _merge(self, ordered_tuples, ordered_non_tuples,
               passed_obj_strs, original_scopes, passed_loader_params):

        # Init and set final loaders, params, scopes
        passed_loaders = []     
        passed_loader_scopes = []

        cnt1, cnt2 = 0, 0

        # Merge back together in the right order
        for i in range(len(passed_obj_strs)):

            if isinstance(passed_obj_strs[i], tuple):
                passed_loaders += ordered_tuples[0][cnt1]
                passed_loader_scopes += ordered_tuples[1][cnt1]
                
                cnt1 += 1
        
            else:
                
                if isinstance(original_scopes[i], tuple):
                    for j in range(len(original_scopes[i])):
                        passed_loaders.append(ordered_non_tuples[0][cnt2])
                        passed_loader_scopes.append(ordered_non_tuples[1][cnt2])
                        cnt2 += 1
                
                else:
                    passed_loaders.append(ordered_non_tuples[0][cnt2])
                    passed_loader_scopes.append(ordered_non_tuples[1][cnt2])
                    cnt2 += 1

        return passed_loaders, passed_loader_scopes, passed_loader_params

    def _process_base(self):

        from .Loaders import get_loader_and_params

        # Passed as just () case
        if isinstance(self.obj_strs, tuple):
            passed_obj_strs = [self.obj_strs]
            passed_param_strs = [self.param_strs]
        else:
            passed_obj_strs = self.obj_strs
            passed_param_strs = self.param_strs

        if isinstance(self.scopes, tuple):
            passed_scopes = [self.scopes]
        else:
            passed_scopes = self.scopes

        # Save a copy of original scopes
        original_scopes = deepcopy(passed_scopes)

        # Split by tuple, non tuple
        tuples, non_tuples =\
            self._get_split_by_tuple(passed_obj_strs, passed_param_strs, passed_scopes)

        # Proc tuples
        passed_loader_params, ordered_tuples =\
            self._process_tuples(tuples, get_loader_and_params)

        # Process the non-tuples
        passed_loader_params, ordered_non_tuples =\
            self._process_non_tuples(non_tuples, get_loader_and_params, passed_loader_params)

        
        # Merge everything together & return
        return self._merge(ordered_tuples, ordered_non_tuples,
                           passed_obj_strs, original_scopes, passed_loader_params)

    def _process(self):

        from .Loaders import Loader_Wrapper

        # Process according to passed tuples or not
        passed_loaders, passed_loader_scopes, passed_loader_params =\
            self._process_base()
        
        # The base objects have been created, but they need to be wrapped in the loader wrapper
        params = {'file_mapping': self.Data_Scopes.file_mapping,
                  'wrapper_n_jobs': self.n_jobs}

        passed_loaders =\
            self._wrap_pipeline_objs(Loader_Wrapper,
                                     passed_loaders,
                                     passed_loader_scopes,
                                     **params)

        return passed_loaders, passed_loader_params


class Imputers(Scope_Pieces):

    name = 'imputers'

    def _process(self):

        # Perform initial checks
        obj_strs, param_strs, scopes =\
            self._checks(self.obj_strs, self.param_strs, self.scopes)

        # Make the imputers and params combo
        imputers_and_params =\
            [self._get_imputer(imputer_str, imputer_param, scope)
                for imputer_str, imputer_param, scope in zip(
                    obj_strs, param_strs, scopes)]

        # Remove None's if any
        while None in imputers_and_params:
            imputers_and_params.remove(None)

        # Perform proc objs
        imputers, imputer_params =\
            self._proc_objs_and_params(imputers_and_params)

        # Make column transformers for skipping strat cols
        skip_strat_scopes = ['all' for i in range(len(imputers))]

        col_imputers, col_imputer_params =\
            self._make_col_version(imputers, imputer_params,
                                    skip_strat_scopes)

        return col_imputers, col_imputer_params

    def _get_imputer(self, imputer_str, imputer_param, scope):

        from .Imputers import get_imputer_and_params

        # First grab the correct params based on scope
        if scope == 'cat' or scope == 'categorical':
            scope = 'categorical'

            cat_inds, ordinal_inds =\
                self.Data_Scopes.get_cat_ordinal_inds()

            valid_cat_inds = []
            for c in cat_inds:
                if len(c) > 0:
                    valid_cat_inds.append(c)
            cat_inds = valid_cat_inds

            inds = []

            # If scope doesn't match any actual data, skip
            if len(ordinal_inds) == 0 and len(cat_inds) == 0:
                return None

        else:

            if scope == 'float':
                scope = 'float'
                keys = 'float'
            else:
                scope = 'custom'
                keys = scope

            inds = self.Data_Scopes.get_inds_from_scope(keys)
            cat_inds = []
            ordinal_inds = []

            # If scope doesn't match any actual data, skip
            if len(inds) == 0:
                return None

        # Determine base estimator based off str + scope
        base_estimator, base_estimator_params =\
            self._proc_imputer_base_estimator(imputer_str,
                                              imputer_param,
                                              scope)

        cat_encoders = self.Data_Scopes.cat_encoders
        for c_encoder in cat_encoders:
            if c_encoder is None:
                raise RuntimeError('Impution on multilabel-type covars is not',
                                   'currently supported!')

        imputer, imputer_params =\
            get_imputer_and_params(imputer_str, self.extra_params,
                                   imputer_param, self.search_type, inds,
                                   cat_inds, ordinal_inds, cat_encoders,
                                   base_estimator, base_estimator_params)

        name = 'imputer_' + imputer_str + '_' + scope

        # Update imputer params with name of imputer obj
        new_imputer_params = {}
        for key in imputer_params:
            new_imputer_params[name + '__' + key] = imputer_params[key]

        return name, (imputer, new_imputer_params)

    def _proc_imputer_base_estimator(self, imputer_str, imputer_param, scope):

        # For now, custom passed params assume and only work with
        # float/regression type.
        if scope == 'float' or scope == 'custom':
            problem_type = 'regression'

        # Assumes for model_types binary and multiclass have the same options
        else:
            problem_type = 'binary'

        # Check if the passed imputer str is a valid Model
        try:

            base_model_obj = Models(obj_strs=[imputer_str],
                                    param_strs=[imputer_param],
                                    problem_type=problem_type,
                                    search_type=None,
                                    random_state=self.random_state,
                                    n_jobs=self.n_jobs,
                                    extra_params=self.extra_params)

            # By passing search type None, should ensure that returned params
            # are not preprended with anything. This function needs more work...
            return base_model_obj.objs[0][1], base_model_obj.obj_params

        except RuntimeError:
            return None, {}


class Scalers(Scope_Pieces):

    name = 'scalers'

    def _process(self):

        from .Scalers import get_scaler_and_params

        # Perform checks first
        obj_strs, param_strs, scopes =\
            self._checks(self.obj_strs, self.param_strs, self.scopes)

        # Then call get objs and params
        objs, obj_params =\
            self._get_objs_and_params(get_scaler_and_params,
                                      obj_strs, param_strs)

        # Wrap in col_transformer for scope
        col_scalers, col_scaler_params =\
                self._make_col_version(objs, obj_params,
                                       scopes)

        return col_scalers, col_scaler_params


class Transformers(Scope_Pieces):

    name = 'transformers'

    def _process(self):

        from .Transformers import get_transformer_and_params, Transformer_Wrapper

        # Perform checks first
        obj_strs, param_strs, scopes =\
            self._checks(self.obj_strs, self.param_strs, self.scopes)

        # Then call get objs and params
        objs, obj_params =\
            self._get_objs_and_params(get_transformer_and_params,
                                      obj_strs, param_strs)

        transformers =\
            self._wrap_pipeline_objs(Transformer_Wrapper,
                                     objs,
                                     scopes)

        return transformers, obj_params


class Samplers(Scope_Pieces):

    name = 'samplers'

    def _process(self):

        # Perform checks first
        obj_strs, param_strs, sample_on =\
            self._checks(self.obj_strs, self.param_strs, self.scopes)

        recover_strats = self._get_recover_strats(len(obj_strs), sample_on)

        # Get the scalers and params
        scalers_and_params =\
            [self._get_sampler(sampler_str, sampler_param, on,
                               recover_strat) for
             sampler_str, sampler_param, on, recover_strat in
             zip(obj_strs, param_strs, sample_on, recover_strats)]

        samplers, sampler_params =\
            self._proc_objs_and_params(scalers_and_params)

        # Change Random State + n_jobs
        samplers = self._check_and_replace_samplers(samplers,
                                                    'random_state',
                                                    self.random_state)

        samplers = self._check_and_replace_samplers(samplers,
                                                    'n_jobs',
                                                    self.n_jobs)

        return samplers, sampler_params

    def _get_recover_strats(self, num_samplers, sample_on):

        # Creates binary mask of, True if any strat inds used
        uses_strat = [len(self._proc_sample_on(sample_on[i])[1]) > 0 for
                      i in range(num_samplers)]

        # If never use strat, set all to False
        if True not in uses_strat:
            return [False for i in range(num_samplers)]

        # If strat is used, then just need Trues up to but not including
        # the last true.
        last_true = len(uses_strat) - 1 - uses_strat[::-1].index(True)

        trues = [True for i in range(last_true)]
        falses = [False for i in range(num_samplers - last_true)]

        return trues + falses

    def _proc_sample_on(self, on):

        # On should be either a list, or a single str
        if isinstance(on, str):
            on = [on]
        else:
            on = list(on)

        sample_strat_keys = [o for o in on if o in self.Data_Scopes.strat_keys]

        # Set sample_target
        sample_target = False
        if len(sample_strat_keys) != len(on):
            sample_target = True

        return sample_target, sample_strat_keys
    
    def _get_sampler(self, sampler_str, sampler_param, on, recover_strat):

        from .Samplers import get_sampler_and_params

        # Grab sample_target and sample_strat_keys, from on
        sample_target, sample_strat_keys = self._proc_sample_on(on)

        # Set strat inds and sample_strat
        strat_inds = self.Data_Scopes.get_strat_inds()
        sample_strat = self.Data_Scopes.get_train_inds_from_keys(sample_strat_keys)

        # Set categorical flag
        categorical = True
        if self.problem_type == 'regression':
            categorical = False

        cat_inds, ordinal_inds =\
            self.Data_Scopes.get_cat_ordinal_inds()
        covars_inds = cat_inds + [[o] for o in ordinal_inds]

        sampler, sampler_params =\
            get_sampler_and_params(sampler_str, self.extra_params,
                                   sampler_param, self.search_type,
                                   strat_inds=strat_inds,
                                   sample_target=sample_target,
                                   sample_strat=sample_strat,
                                   categorical=categorical,
                                   recover_strat=recover_strat,
                                   covars_inds=covars_inds)

        return sampler_str, (sampler, sampler_params)

    def _check_and_replace_samplers(self, samplers, param_name, replace_value):

        for i in range(len(samplers)):

            try:
                getattr(samplers[i][1].sampler, param_name)
                setattr(samplers[i][1].sampler, param_name, replace_value)

            except AttributeError:
                pass

        return samplers


class Feat_Selectors(Type_Pieces):

    name = 'feat_selectors'

    def _process(self):

        from .Feature_Selectors import get_feat_selector_and_params, AVALIABLE
        self.AVAILABLE = AVALIABLE

        # Base
        objs, obj_params = self._base_type_process(get_feat_selector_and_params)
 
        # If any base estimators, replace with a model
        objs = self._replace_base_rfe_estimator(objs)

        return objs, obj_params

    def _replace_base_rfe_estimator(self, feat_selectors):
        '''Check feat selectors for a RFE model'''

        for i in range(len(feat_selectors)):

            try:
                base_model_str = feat_selectors[i][1].estimator

                # Default behavior is use ridge
                if base_model_str is None:
                    base_model_str = 'ridge'

                base_model_obj = Models(obj_strs=[base_model_str],
                                        param_strs=[0],
                                        problem_type=self.problem_type,
                                        search_type=None,
                                        random_state=self.random_state,
                                        n_jobs=self.n_jobs,
                                        extra_params=self.extra_params)

                feat_selectors[i][1].estimator = base_model_obj.objs[0][1]
            
            except AttributeError:
                pass

        return feat_selectors


class Ensembles(Type_Pieces):

    name = 'ensembles'

    def _process(self):

        from .Ensembles import get_ensemble_and_params, AVALIABLE
        self.AVAILABLE = AVALIABLE
        
        return self._base_type_process(get_ensemble_and_params)

    def _get_base_ensembler(self, models):

        # If wrapping in ensemble, set n_jobs for ensemble
        # and each indv model, make sure 1
        for model in models:
            try:
                model[1].n_jobs = 1
            except AttributeError:
                pass

            # Ensemble of des ensembles case
            if hasattr(model[1], 'estimators'):
                for estimator in model[1].estimators:
                    try:
                        estimator.n_jobs = 1
                    except AttributeError:
                        pass

        if self.problem_type == 'regression':
            return VotingRegressor(models, n_jobs=self.n_jobs)

        return VotingClassifier(models, voting='soft', n_jobs=self.n_jobs)


class Drop_Strat(Pieces):

    name = 'drop_strat'

    def _process(self):

        non_strat_inds = self.Data_Scopes.get_inds_from_scope('all')
        identity = FunctionTransformer(validate=False)

        # Make base col_transformer, just for dropping strat cols
        col_transformer =\
            ColTransformer(transformers=[('keep_all_but_strat_inds',
                                          identity, non_strat_inds)],
                           remainder='drop', sparse_threshold=0)

        # Put in list, to easily add to pipeline
        drop_strat = [('drop_strat', col_transformer)]
        return drop_strat, {}

        