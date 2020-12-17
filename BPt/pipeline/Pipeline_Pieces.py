from ..main.Input_Tools import is_pipe, is_select

from ..helpers.ML_Helpers import (check_for_duplicate_names,
                                  wrap_pipeline_objs,
                                  proc_type_dep_str, param_len_check,
                                  conv_to_list,
                                  process_params_by_type, replace_model_name)

from .ScopeObjs import ScopeTransformer, ScopeModel
from sklearn.ensemble import VotingClassifier, VotingRegressor

from sklearn.pipeline import Pipeline
from copy import deepcopy

from .Selector import selector_wrapper
from sklearn.compose import TransformedTargetRegressor
from .BPtSearchCV import wrap_param_search

import numpy as np


def process_input_types(obj_strs, param_strs, scopes):

    # I think want to fill in missing param_strs w/ 0's here
    param_strs = param_len_check(obj_strs, param_strs)
    return obj_strs, param_strs, scopes


def get_scope_name(scope):

    scope_name = ' ' + str(scope)
    if scope_name == ' all':
        scope_name = ''
    elif len(scope_name) > 10:
        scope_name = scope_name[:10] + '...'

    return scope_name


class Pieces():

    def __init__(self, user_passed_objs, Data_Scopes, spec):
        # problem_type, random_state, n_jobs, search_type are stored in spec

        # Class values
        self.user_passed_objs = user_passed_objs
        self.Data_Scopes = Data_Scopes
        self.spec = spec.copy()

        # This value with be replaced in child classes
        self.AVAILABLE = None

    def process(self, params):
        '''params is a list of Param piece classes, or potentially a list
        like custom object,
        e.g., Select. Each actual param object might vary a bit on
        what it includes'''

        if params is None or (isinstance(params, list) and len(params) == 0):
            return [], {}

        # In case of recursive calls, make sure params are as list
        params = conv_to_list(params)

        # Check for select
        select_mask = np.array([is_select(param) for param in params])
        if select_mask.any():
            return self._process_with_select(params, select_mask)

        # Otherwise, process as normal
        return self._process(params)

    def _process_with_select(self, params, select_mask):

        # Check & Process for Select()
        # All params have been conv_to_list, s.t. if Select is present
        # Then it is as [Select([...])] or [Select([...]), ...]

        # Init objs as empty list of correct size w/ just place holders
        objs = [None for i in range(len(params))]

        # Process everything but the select groups first
        # Putting the recursive call to process here... even though I
        # think call to
        # just _process could work too. The logic is, I don't think it will
        # hurt,
        # and if more options beyond Select are added to process later... this
        # will hopefully cover those cases.
        non_select_objs, obj_params =\
            self.process([i for idx, i in enumerate(params) if
                         not select_mask[idx]])

        # Update right spot in objs
        for obj, ind in zip(non_select_objs, np.where(~select_mask)[0]):
            objs[ind] = obj

        # Split up select params
        select_params = [i for idx, i in enumerate(params) if select_mask[idx]]

        # Next, process each group of select params seperately
        cnt = 0
        for s_params, ind in zip(select_params, np.where(select_mask)[0]):

            # Recursive call to process... s.t., can handle nested Select...
            # oh god
            s_objs, s_obj_params = self.process(list(s_params))

            # Wrap in selector object
            name = self.name + '_selector' + str(cnt)
            s_obj, s_obj_params = selector_wrapper(s_objs, s_obj_params, name)

            # Can update final params
            objs[ind] = s_obj
            obj_params.update(s_obj_params)
            cnt += 1

        return objs, obj_params

    # Overide in children class
    def _process(self, params):
        return [], {}

    def _get_objs_and_params(self, get_func, params):
        '''Helper function to grab scaler / feat_selectors and
        their relevant parameter grids'''

        # Save copy of the original spec
        spec = self.spec.copy()

        # Make the object + params based on passed settings
        objs_and_params = []
        for param in params:

            name, param_str = param.obj, param.params
            extra_params = param.extra_params

            # If this param has an associated param search of non None,
            # set class to that search type.
            # Otherwise, set spec to copy of the original class
            if hasattr(param, 'param_search') and param.param_search is not None:
                self.spec['search_type'] = param.param_search.search_type
            else:
                self.spec = spec.copy()

            if 'Custom ' in name:
                objs_and_params.append(
                    self._get_user_passed_obj_params(name, param_str,
                                                     extra_params))

            else:

                # Set the original number of feat keys based on the
                # original scope
                num_feat_keys =\
                    len(self.Data_Scopes.get_inds_from_scope(param.scope))

                objs_and_params.append(
                    (name, get_func(name, extra_params,
                                    param_str, self.spec['search_type'],
                                    self.spec['random_state'],
                                    num_feat_keys)
                     ))

        # Perform extra proc, to split into objs and merged param dict
        objs, params = self._proc_objs_and_params(objs_and_params)

        return objs, params

    def _get_user_passed_obj_params(self, name, param, extra_params):

        # Get copy of user_obj
        user_obj = deepcopy(self.user_passed_objs[name])

        # Proc as necc. user passed params
        extra_user_obj_params, user_obj_params =\
            process_params_by_type(obj=user_obj,
                                   obj_str=name,
                                   base_params=deepcopy(param),
                                   extra_params=extra_params)

        # If passing a user object, kind of stupid to pass default, non search
        # params, via a dict..., but hey give it a try
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

        if hasattr(obj, 'random_state'):
            setattr(obj, 'random_state', self.spec['random_state'])

        if hasattr(obj, 'n_jobs'):
            setattr(obj, 'n_jobs', self.spec['n_jobs'])

        if hasattr(obj, 'needs_cat_inds'):
            if getattr(obj, 'needs_cat_inds'):

                # Get cat inds and set
                cat_inds = self.Data_Scopes.get_inds_from_scope('categorical')
                obj.set_params(cat_inds=cat_inds)

        return obj

    def replace_base_estimator(self, objs, obj_params, params):

        for i in range(len(objs)):

            if (hasattr(objs[i][1], 'estimator') and
               params[i].base_model is not None):

                # Proc base model type
                model_spec = self.spec.copy()
                base_model_type = params[i].base_model_type
                if base_model_type is not None:
                    if base_model_type == 'default':

                        # Default regression
                        model_spec['problem_type'] = 'regression'

                        # Check for categorical
                        scope = params[i].scope
                        if scope == 'cat' or scope == 'categorical':
                            model_spec['problem_type'] = 'categorical'

                    else:
                        model_spec['problem_type'] = base_model_type

                # Grab the base estimator
                base_model_obj = Models(self.user_passed_objs,
                                        self.Data_Scopes,
                                        model_spec)
                base_objs, base_params =\
                    base_model_obj.process(params[i].base_model)

                # Replace the estimator
                objs[i][1].estimator = base_objs[0][1]

                # Replace the model name in the params with estimator
                base_params = replace_model_name(base_params)

                # Add the name of the object pre-pended
                base_params = {objs[i][0] + '__' + p_name: base_params[p_name]
                               for p_name in base_params}

                obj_params.update(base_params)

        return objs, obj_params

    def _wrap_pipeline_objs(self, wrapper, objs, scopes,
                            fix_n_wrapper_jobs, **params):

        inds = [self.Data_Scopes.get_inds_from_scope(scope)
                for scope in scopes]

        objs = wrap_pipeline_objs(wrapper, objs, inds,
                                  random_state=self.spec['random_state'],
                                  n_jobs=self.spec['n_jobs'],
                                  fix_n_wrapper_jobs=fix_n_wrapper_jobs,
                                  **params)

        return objs

    def _make_col_version(self, objs, params, scopes):

        # Make objs + params
        col_objs, col_params = [], {}

        for i in range(len(objs)):
            name, obj = objs[i][0], objs[i][1]
            inds = self.Data_Scopes.get_inds_from_scope(scopes[i])

            # Get scope name
            scope_name = get_scope_name(scopes[i])

            # Create the col obj as a ScopeTransformer
            col_obj = (name + scope_name,
                       ScopeTransformer(estimator=obj, inds=inds))
            col_objs.append(col_obj)

            # Change any associated params
            for key in params:
                split_key = key.split('__')

                # If this is an associated param w/ this obj
                if split_key[0] == name + '__':

                    # Replace name with estimator
                    name = split_key[0] + scope_name
                    split_key[0] = 'estimator'

                    # Create new name
                    new_name = '__'.join([name] + split_key)

                    # Save under new name
                    col_params[new_name] = params[key]

        return col_objs, col_params


class Type_Pieces(Pieces):

    def _base_type_process(self, params, func):

        # Check / replace by type
        params = self._check_type(params)

        # Then call get objs and params
        objs, obj_params =\
            self._get_objs_and_params(func, params)

        return objs, obj_params

    def _check_type(self, params):

        for p in range(len(params)):
            params[p].obj =\
                proc_type_dep_str(params[p].obj, self.AVAILABLE,
                                  self.spec['problem_type'])
        return params


class Models(Type_Pieces):

    name = 'models'

    def _process(self, params):

        from .Models import get_base_model_and_params, AVALIABLE
        self.AVAILABLE = AVALIABLE.copy()

        # If any ensembles
        ensemble_mask = np.array([hasattr(params[i], 'models')
                                  for i in range(len(params))])

        # Seperate non-ensemble objs and obj_params
        non_ensemble_params = [i for idx, i in enumerate(params)
                               if not ensemble_mask[idx]]

        # Run initial base type process on the non ensemble models
        non_ensemble_objs, non_ensemble_obj_params =\
            self._base_type_process(non_ensemble_params,
                                    get_base_model_and_params)

        # Perform wrap checks on non_ensembles
        non_ensemble_objs, non_ensemble_obj_params =\
            self.check_wraps(non_ensemble_objs,
                             non_ensemble_obj_params,
                             non_ensemble_params)

        # If just models, i.e., no ensembles to process,
        # can return here just the non_ensemble_objs + params
        if not ensemble_mask.any():
            return non_ensemble_objs, non_ensemble_obj_params

        # Assume ensemble case from here
        # Process the ensemble objs and params
        ensembled_objs, ensembled_obj_params =\
            self._process_ensembles(params, ensemble_mask)

        # In mixed case, merge with non_ensemble
        ensembled_objs += non_ensemble_objs
        ensembled_obj_params.update(non_ensemble_obj_params)
        return ensembled_objs, ensembled_obj_params

    def _process_ensembles(self, params, ensemble_mask):

        from .Ensembles import Ensemble_Wrapper

        # Seperate the ensemble params from all passed params
        ensemble_params = [i for idx, i in enumerate(params)
                           if ensemble_mask[idx]]

        # Get base ensemble objs. and obj_params
        ensembles =\
            Ensembles(self.user_passed_objs, self.Data_Scopes, self.spec)
        ensemble_objs, ensemble_obj_params = ensembles.process(ensemble_params)

        # For each ensemble, go through and process
        ensembled_objs, ensembled_obj_params = [], {}
        for i in range(len(ensemble_params)):

            # Recursively process the base ensembles models
            model_objs, model_obj_params =\
                self.process(ensemble_params[i].models)

            # Check for a base model - process if found
            if ensemble_params[i].base_model is not None:
                final_estimator, final_estimator_params =\
                    self.process(ensemble_params[i].base_model)
            else:
                final_estimator, final_estimator_params = None, {}

            # If any passed ensemble params, then need to select just the
            # params associated with this ensemble
            this_ensemble_name = ensemble_objs[i][0]
            this_ensemble_obj_params =\
                {key: ensemble_obj_params[key] for key
                 in ensemble_obj_params
                 if this_ensemble_name == key.split('__')[0]}

            # Create wrapper object for building the ensemble
            wrapper = Ensemble_Wrapper(model_obj_params,
                                       this_ensemble_obj_params,
                                       ensembles._get_base_ensembler,
                                       n_jobs=self.spec['n_jobs'],
                                       random_state=self.spec['random_state'])

            # Use the wrapper to get the ensembled_obj
            ensembled_objs +=\
                wrapper.wrap_ensemble(
                    models=model_objs,
                    ensemble=ensemble_objs[i],
                    ensemble_params=ensemble_params[i],
                    final_estimator=final_estimator,
                    final_estimator_params=final_estimator_params
                    )

            # The wrapper keeps track of the changes to params,
            # Add the final updated params to the collective
            # ensembled_obj_params
            ensembled_obj_params.update(wrapper.get_updated_params())

        # Perform wrap checks on the ensembles
        ensembled_objs, ensembled_obj_params =\
            self.check_wraps(ensembled_objs,
                             ensembled_obj_params,
                             ensemble_params)

        return ensembled_objs, ensembled_obj_params

    def check_wraps(self, objs, obj_params, params):
        '''Check for first target scaler wrap, then nested param search,
        then scope model wrap.'''

        # Check ensembled_objs for target_scaler
        for i in range(len(params)):
            target_scaler = params[i].target_scaler

            objs[i], obj_params =\
                self.wrap_target_scaler(target_scaler, objs[i], obj_params)

        # Check for nested param search
        for i in range(len(params)):

            param_search = params[i].param_search

            objs[i], obj_params =\
                wrap_param_search(param_search, objs[i], obj_params)

        # For now am wrapping all models in Scope wrap
        # Could / maybe should change this in the future.
        for i in range(len(params)):
            scope = params[i].scope
            objs[i], obj_params =\
                self.wrap_model_scope(scope, objs[i], obj_params)

        return objs, obj_params

    def wrap_target_scaler(self, target_scaler, model_obj, model_params):

        if self.spec['problem_type'] != 'regression':
            return model_obj, model_params
        if target_scaler is None:
            return model_obj, model_params

        # Process and get the base scaler_obj + params
        base_scaler_obj = Target_Scalers(self.user_passed_objs,
                                         self.Data_Scopes,
                                         self.spec)
        scaler_objs, scaler_params =\
            base_scaler_obj.process(target_scaler)
        scaler_obj = scaler_objs[0]

        # Unwrap into name + base
        model_name, base_model = model_obj[0], model_obj[1]
        scaler_name, base_scaler = scaler_obj[0], scaler_obj[1]

        # Now, wrap the model + scaler in the transformed target regressor
        base_wrapper_model =\
            TransformedTargetRegressor(regressor=base_model,
                                       transformer=base_scaler)
        wrapped_name = 'scale_target_' + model_name
        wrapper_model_obj = (wrapped_name, base_wrapper_model)

        # Need to update model params with new nested model name
        model_param_names = list(model_params)
        for param_name in model_param_names:
            if param_name.startswith(model_name + '__'):

                new_base = wrapped_name + '__regressor__'
                new_param_name =\
                    param_name.replace(model_name + '__', new_base, 1)

                model_params[new_param_name] = model_params.pop(param_name)

        # Need to also update / add any scaler params
        for param_name in scaler_params:
            if param_name.startswith(scaler_name + '__'):

                new_base = wrapped_name + '__transformer__'
                new_param_name =\
                    param_name.replace(scaler_name + '__', new_base, 1)

                model_params[new_param_name] = scaler_params[param_name]

        return wrapper_model_obj, model_params

    def wrap_model_scope(self, scope, model, model_params):

        # Check if scope is 'all' or functionally 'all'
        # Only wrap if not functionally all
        inds = self.Data_Scopes.get_inds_from_scope(scope)
        all_inds = self.Data_Scopes.get_inds_from_scope('all')

        if len(inds) == len(all_inds):
            return model, model_params

        # Get scope name
        scope_name = get_scope_name(scope)
        name = model[0]

        # Get scope model - under same name as model
        scope_model = (name + scope_name,
                       ScopeModel(estimator=model[1], inds=inds))

        # Change any associated params with this model obj
        param_keys = list(model_params)
        for key in param_keys:
            split_key = key.split('__')

            # If this is an associated param w/ this obj
            if split_key[0] == name + '__':

                # Replace name with estimator
                name = split_key[0] + scope_name
                split_key[0] = 'estimator'

                # Create new name
                new_name = '__'.join([name] + split_key)

                # Replace under new name
                model_params[new_name] = model_params.pop(key)

        return scope_model, model_params


class Loaders(Pieces):

    name = 'loaders'

    def _process_base(self, params):

        from .Loaders import get_loader_and_params

        # Check for pipe first
        pipe_mask = np.array([is_pipe(param.obj) for param in params])

        # Init objs as empty list of correct size w/ just place holders
        objs = [None for i in range(len(params))]

        # Process everything but the select groups first
        non_pipe_params = [i for idx, i in enumerate(params)
                           if not pipe_mask[idx]]

        # obj_params has no order, so init with non pipe params
        non_pipe_objs, obj_params =\
            self._get_objs_and_params(get_loader_and_params, non_pipe_params)

        # Update right spot in objs
        for obj, ind in zip(non_pipe_objs, np.where(~pipe_mask)[0]):
            objs[ind] = obj

        # Split up pipe params
        pipe_params = [i for idx, i in enumerate(params) if pipe_mask[idx]]

        # Next, process each group of pipe params seperately
        cnt = 0
        for p_params, ind in zip(pipe_params, np.where(pipe_mask)[0]):

            # Need to move objs and params to seperate objs to pass along
            p_sep_params = []
            for o, p in zip(p_params.obj, p_params.params):
                base = deepcopy(p_params)
                base.obj = o
                base.params = p
                p_sep_params.append(base)

            p_objs, p_obj_params =\
                self._get_objs_and_params(get_loader_and_params, p_sep_params)

            # Create loader pipeline
            name = 'loader_pipe' + str(cnt)
            p_obj = (name, Pipeline(steps=p_objs))

            # Add pipeline name to params
            with_name =\
                {name + '__' + key: p_obj_params[key] for key in p_obj_params}

            # Can update final params
            objs[ind] = p_obj
            obj_params.update(with_name)
            cnt += 1

        return objs, obj_params

    def _process(self, params):

        from .Loaders import Loader_Wrapper

        # Extract scopes + cache loc
        passed_loader_scopes = [p.scope for p in params]
        passed_cache_locs = [p.cache_loc for p in params]
        passed_fix_n_wrapper_jobs = [p.fix_n_wrapper_jobs for p in params]

        # Process according to passed tuples or not
        passed_loaders, passed_loader_params =\
            self._process_base(params)

        # The base objects have been created, but they need to be wrapped
        # in the loader wrapper.
        pass_params = {'file_mapping': self.Data_Scopes.file_mapping,
                       'wrapper_n_jobs': self.spec['n_jobs'],
                       'cache_locs': passed_cache_locs}

        passed_loaders =\
            self._wrap_pipeline_objs(Loader_Wrapper,
                                     passed_loaders,
                                     passed_loader_scopes,
                                     passed_fix_n_wrapper_jobs,
                                     **pass_params)

        return passed_loaders, passed_loader_params


class Imputers(Pieces):

    name = 'imputers'

    def _process(self, params):

        from .Imputers import get_imputer_and_params

        # Then call get objs and params
        objs, obj_params =\
            self._get_objs_and_params(get_imputer_and_params,
                                      params)

        # Replace any base estimators
        objs, obj_params =\
            self.replace_base_estimator(objs, obj_params, params)

        # Wrap in col_transformer for scope
        col_imputers, col_imputer_params =\
            self._make_col_version(objs, obj_params,
                                   [p.scope for p in params])

        return col_imputers, col_imputer_params


class Target_Scalers(Pieces):

    name = 'target scalers'

    def _process(self, params):

        from .Scalers import get_scaler_and_params

        # Then call get objs and params
        objs, obj_params =\
            self._get_objs_and_params(get_scaler_and_params,
                                      params)

        return objs, obj_params


class Scalers(Pieces):

    name = 'scalers'

    def _process(self, params):

        from .Scalers import get_scaler_and_params

        # Then call get objs and params
        objs, obj_params =\
            self._get_objs_and_params(get_scaler_and_params,
                                      params)

        # Wrap in col_transformer for scope
        col_scalers, col_scaler_params =\
            self._make_col_version(objs, obj_params,
                                   [p.scope for p in params])

        return col_scalers, col_scaler_params


class Transformers(Pieces):

    name = 'transformers'

    def _process(self, params):

        from .Transformers import (get_transformer_and_params,
                                   Transformer_Wrapper)

        # Extract scopes + cache loc
        passed_scopes = [p.scope for p in params]
        passed_cache_locs = [p.cache_loc for p in params]
        passed_fix_n_wrapper_jobs = [p.fix_n_wrapper_jobs for p in params]
        pass_params = {'cache_locs': passed_cache_locs}

        # Then call get objs and params
        objs, obj_params =\
            self._get_objs_and_params(get_transformer_and_params,
                                      params)

        transformers =\
            self._wrap_pipeline_objs(Transformer_Wrapper,
                                     objs,
                                     passed_scopes,
                                     passed_fix_n_wrapper_jobs,
                                     **pass_params)

        return transformers, obj_params


class Feat_Selectors(Type_Pieces):

    name = 'feat_selectors'

    def _process(self, params):

        from .Feature_Selectors import get_feat_selector_and_params, AVALIABLE
        self.AVAILABLE = AVALIABLE

        # Standard base type process
        objs, obj_params =\
            self._base_type_process(params, get_feat_selector_and_params)

        # If any base estimators, replace with a model
        objs, obj_params =\
            self.replace_base_estimator(objs, obj_params, params)

        # Wrap in feat select wrapper
        feat_objs, feat_params =\
            self._wrap_feat_select(objs, obj_params,
                                   [p.scope for p in params])

        return feat_objs, feat_params

    def _wrap_feat_select(self, objs, params, scopes):
        from .Feature_Selectors import FeatureSelectorWrapper

        # Make wrapped objs
        feat_objs = []
        for i in range(len(objs)):

            # Unpack name + obj
            name, obj = objs[i][0], objs[i][1]

            # Get wrapper inds from scope
            wrapper_inds = self.Data_Scopes.get_inds_from_scope(scopes[i])

            # Wrap in feat selector wrapper, which handles scope +
            # keeping track of updating mapping
            feat_obj = ('feat_' + name,
                        FeatureSelectorWrapper(base_selector=obj,
                                               wrapper_inds=wrapper_inds))
            feat_objs.append(feat_obj)

        # Update params
        feat_params = {}
        for key in params:
            name = key.split('__')[0]
            new_key = key.replace(name, 'base_selector', 1)
            new_name = 'feat_' + name + '__' + new_key
            feat_params[new_name] = params[key]

        return feat_objs, feat_params


class Ensembles(Type_Pieces):

    name = 'ensembles'

    def _process(self, params):

        from .Ensembles import get_ensemble_and_params, AVALIABLE
        self.AVAILABLE = AVALIABLE

        return self._base_type_process(params,
                                       get_ensemble_and_params)

    def _get_base_ensembler(self, models):

        # @TODO Might want to reflect choice of ensemble / model n_jobs here?

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

        if self.spec['problem_type'] == 'regression':
            return VotingRegressor(models, n_jobs=self.spec['n_jobs'])

        return VotingClassifier(models, voting='soft',
                                n_jobs=self.spec['n_jobs'])

