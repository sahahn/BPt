from sklearn.preprocessing import FunctionTransformer
from ..main.Input_Tools import is_pipe, is_select

from ..helpers.ML_Helpers import (check_for_duplicate_names,
                                  wrap_pipeline_objs,
                                  proc_type_dep_str, param_len_check,
                                  conv_to_list,
                                  process_params_by_type)

from ..extensions.Col_Selector import ColDropStrat, InPlaceColTransformer
from ..extensions.Scope_Model import Scope_Model
from sklearn.ensemble import VotingClassifier, VotingRegressor

from sklearn.pipeline import Pipeline
from copy import deepcopy

from .Selector import selector_wrapper

import numpy as np


def process_input_types(obj_strs, param_strs, scopes):

    # I think want to fill in missing param_strs w/ 0's here
    param_strs = param_len_check(obj_strs, param_strs)
    return obj_strs, param_strs, scopes


class Pieces():

    def __init__(self, user_passed_objs, Data_Scopes, spec, _print=print):
        # problem_type, random_state, n_jobs, search_type are stored in spec

        # Class values
        self.user_passed_objs = user_passed_objs
        self.Data_Scopes = Data_Scopes
        self.spec = spec
        self._print = _print

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

        # Make the object + params based on passed settings
        objs_and_params = []
        for param in params:

            name, param_str = param.obj, param.params
            extra_params = param.extra_params

            if 'user passed' in name:
                objs_and_params.append(
                    self._get_user_passed_obj_params(name, param_str,
                                                     extra_params))

            else:
                objs_and_params.append(
                    (name, get_func(name, extra_params,
                                    param_str, self.spec['search_type'],
                                    self.spec['random_state'],
                                    self.Data_Scopes.num_feat_keys)
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
                                   extra_params=extra_params,
                                   search_type=self.spec['search_type'])

        # If passing a user object, kind of stupid to pass default, non search
        # params
        # via a dict..., but hey give it a try
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
            obj.random_state = self.spec['random_state']
        except AttributeError:
            pass

        try:
            obj.n_jobs = self.spec['n_jobs']
        except AttributeError:
            pass

        return obj

    def check_model_scope(self, scope, model, params):

        if scope != 'all':
            inds = self.Data_Scopes.get_inds_from_scope(scope)
            name = model[0]
            scoped_model =\
                (name, Scope_Model(model[1], inds))
            params[name + '__needs_mapping'] = True

            return scoped_model, params

        return model, params

    def replace_base_estimator(self, objs, obj_params, params):

        from ..helpers.ML_Helpers import replace_model_name

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
                                        model_spec, self._print)
                base_objs, base_params =\
                    base_model_obj.process(params[i].base_model)

                # Replace the estimator
                objs[i][1].estimator = base_objs[0][1]

                # Replace the model name in the params with estimator
                base_params = replace_model_name(base_params)
                obj_params.update(base_params)

        return objs, obj_params


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


class Scope_Pieces(Pieces):

    def _wrap_pipeline_objs(self, wrapper, objs, scopes, **params):

        inds = [self.Data_Scopes.get_inds_from_scope(scope)
                for scope in scopes]

        objs = wrap_pipeline_objs(wrapper, objs, inds,
                                  random_state=self.spec['random_state'],
                                  n_jobs=self.spec['n_jobs'], **params)

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

    def _process(self, params):

        from .Models import get_base_model_and_params, AVALIABLE
        self.AVAILABLE = AVALIABLE.copy()

        # If any ensembles
        ensemble_mask = np.array([hasattr(params[i], 'models')
                                  for i in range(len(params))])

        # Seperate non-ensemble objs and obj_params
        non_ensemble_params = [i for idx, i in enumerate(params)
                               if not ensemble_mask[idx]]
        non_ensemble_objs, non_ensemble_obj_params =\
            self._base_type_process(non_ensemble_params,
                                    get_base_model_and_params)

        # If any non-ensemble models have scope != 'all'
        for i in range(len(non_ensemble_params)):
            scope = non_ensemble_params[i].scope

            non_ensemble_objs[i], non_ensemble_obj_params =\
                self.check_model_scope(scope, non_ensemble_objs[i],
                                       non_ensemble_obj_params)

        # If not any ensembles, return the non_ensembles
        if not ensemble_mask.any():
            return non_ensemble_objs, non_ensemble_obj_params

        # Assume ensemble case from here
        from .Ensembles import Ensemble_Wrapper

        # Seperate the ensemble params
        ensemble_params = [i for idx, i in enumerate(params)
                           if ensemble_mask[idx]]

        # Get base ensemble objs. by proc all at once
        ensembles = Ensembles(self.user_passed_objs, self.Data_Scopes,
                              self.spec, _print=self._print)
        ensemble_objs, ensemble_obj_params = ensembles.process(ensemble_params)

        # Get the base models + model_params for each ensemble
        model_objs_and_params = [self.process(e.models)
                                 for e in ensemble_params]

        # For each ensemble, wrap the corr. base models
        ensembled_objs, ensembled_obj_params = [], {}
        for i in range(len(ensemble_params)):

            model_objs, model_obj_params = model_objs_and_params[i]

            # If any passed ensemble params, then need to select just the
            # params associated with this ensemble
            this_ensemble_name = ensemble_objs[i][0]
            this_ensemble_params =\
                {key: ensemble_obj_params[key] for key
                 in ensemble_obj_params
                 if this_ensemble_name == key.split('__')[0]}

            # Create wrapper object for building the ensemble
            wrapper = Ensemble_Wrapper(model_obj_params,
                                       this_ensemble_params,
                                       ensembles._get_base_ensembler,
                                       self.spec['n_jobs'])

            # Get the now ensembled_objs
            ensembled_objs +=\
                wrapper.wrap_ensemble(model_objs, ensemble_objs[i],
                                      ensemble_params[i].des_split,
                                      self.spec['random_state'],
                                      ensemble_params[i].single_estimator,
                                      ensemble_params[i].is_des)

            # And add the params
            ensembled_obj_params.update(wrapper.get_updated_params())

        # Check ensembled_objs for scope != 'all'
        for i in range(len(ensemble_params)):
            scope = ensemble_params[i].scope

            ensembled_objs[i], ensembled_obj_params =\
                self.check_model_scope(scope, ensembled_objs[i],
                                       ensembled_obj_params)

        # In mixed case, merge with non_ensemble
        ensembled_objs += non_ensemble_objs
        ensembled_obj_params.update(non_ensemble_obj_params)
        return ensembled_objs, ensembled_obj_params


class Loaders(Scope_Pieces):

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

            # Can update final params
            objs[ind] = p_obj
            obj_params.update(p_obj_params)
            cnt += 1

        return objs, obj_params

    def _process(self, params):

        from .Loaders import Loader_Wrapper

        # Extract scopes
        passed_loader_scopes = [p.scope for p in params]

        # Process according to passed tuples or not
        passed_loaders, passed_loader_params =\
            self._process_base(params)

        # The base objects have been created, but they need to be wrapped
        # in the loader wrapper.
        params = {'file_mapping': self.Data_Scopes.file_mapping,
                  'wrapper_n_jobs': self.spec['n_jobs']}

        passed_loaders =\
            self._wrap_pipeline_objs(Loader_Wrapper,
                                     passed_loaders,
                                     passed_loader_scopes,
                                     **params)

        return passed_loaders, passed_loader_params


class Imputers(Scope_Pieces):

    name = 'imputers'

    def _process(self, params):

        from .Imputers import get_imputer_and_params

        # Then call get objs and params
        objs, obj_params =\
            self._get_objs_and_params(get_imputer_and_params,
                                      params)

        print(params)

        print(objs, obj_params)

        # Replace any base estimators
        objs, obj_params =\
            self.replace_base_estimator(objs, obj_params, params)

        print(objs, obj_params)

        # Wrap in col_transformer for scope
        col_imputers, col_imputer_params =\
            self._make_col_version(objs, obj_params,
                                   [p.scope for p in params])

        return col_imputers, col_imputer_params


class Scalers(Scope_Pieces):

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


class Transformers(Scope_Pieces):

    name = 'transformers'

    def _process(self, params):

        from .Transformers import (get_transformer_and_params,
                                   Transformer_Wrapper)

        # Then call get objs and params
        objs, obj_params =\
            self._get_objs_and_params(get_transformer_and_params,
                                      params)

        transformers =\
            self._wrap_pipeline_objs(Transformer_Wrapper,
                                     objs,
                                     [p.scope for p in params])

        return transformers, obj_params


class Samplers(Scope_Pieces):

    name = 'samplers'

    def _process(self, params):

        sample_on = [p.sample_on for p in params]
        recover_strats = self._get_recover_strats(sample_on)

        # Get the scalers and params
        scalers_and_params =\
            [self._get_sampler(param, on, recover_strat)
             for param, on, recover_strat in zip(params, sample_on,
                                                 recover_strats)]

        samplers, sampler_params =\
            self._proc_objs_and_params(scalers_and_params)

        # Change Random State + n_jobs
        samplers = self._check_and_replace_samplers(samplers,
                                                    'random_state',
                                                    self.spec['random_state'])

        samplers = self._check_and_replace_samplers(samplers,
                                                    'n_jobs',
                                                    self.spec['n_jobs'])

        return samplers, sampler_params

    def _get_recover_strats(self, sample_on):

        # Creates binary mask of, True if any strat inds used
        uses_strat = [len(self._proc_sample_on(s)[1]) > 0 for s in sample_on]

        # If never use strat, set all to False
        if True not in uses_strat:
            return [False for i in range(len(sample_on))]

        # If strat is used, then just need Trues up to but not including
        # the last true.
        last_true = len(uses_strat) - 1 - uses_strat[::-1].index(True)

        trues = [True for i in range(last_true)]
        falses = [False for i in range(len(sample_on) - last_true)]

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

    def _get_sampler(self, param, on, recover_strat):

        from .Samplers import get_sampler_and_params

        # Grab sample_target and sample_strat_keys, from on
        sample_target, sample_strat_keys = self._proc_sample_on(on)

        # Set strat inds and sample_strat
        strat_inds = self.Data_Scopes.get_strat_inds()
        sample_strat =\
            self.Data_Scopes.get_train_inds_from_keys(sample_strat_keys)

        # Set categorical flag
        categorical = True
        if self.spec['problem_type'] == 'regression':
            categorical = False

        cat_inds, ordinal_inds =\
            self.Data_Scopes.get_cat_ordinal_inds()
        covars_inds = cat_inds + [[o] for o in ordinal_inds]

        # Get the sampler
        sampler, sampler_params =\
            get_sampler_and_params(param, self.spec['search_type'],
                                   strat_inds=strat_inds,
                                   sample_target=sample_target,
                                   sample_strat=sample_strat,
                                   categorical=categorical,
                                   recover_strat=recover_strat,
                                   covars_inds=covars_inds)

        return param.obj, (sampler, sampler_params)

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

    def _process(self, params):

        from .Feature_Selectors import get_feat_selector_and_params, AVALIABLE
        self.AVAILABLE = AVALIABLE

        # Standard base type process
        objs, obj_params =\
            self._base_type_process(params, get_feat_selector_and_params)

        # If any base estimators, replace with a model
        objs, obj_params =\
            self.replace_base_estimator(objs, obj_params, params)

        return objs, obj_params


class Ensembles(Type_Pieces):

    name = 'ensembles'

    def _process(self, params):

        from .Ensembles import get_ensemble_and_params, AVALIABLE
        self.AVAILABLE = AVALIABLE

        return self._base_type_process(params,
                                       get_ensemble_and_params)

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

        if self.spec['problem_type'] == 'regression':
            return VotingRegressor(models, n_jobs=self.spec['n_jobs'])

        return VotingClassifier(models, voting='soft',
                                n_jobs=self.spec['n_jobs'])


class Drop_Strat(Pieces):

    name = 'drop_strat'

    def _process(self, params):

        non_strat_inds = self.Data_Scopes.get_inds_from_scope('all')
        identity = FunctionTransformer(validate=False)

        # Make base col_transformer, just for dropping strat cols
        col_transformer =\
            ColDropStrat(transformers=[('keep_all_but_strat_inds',
                                        identity, non_strat_inds)],
                         remainder='drop', sparse_threshold=0)

        # Put in list, to easily add to pipeline
        drop_strat = [('drop_strat', col_transformer)]
        return drop_strat, {}
