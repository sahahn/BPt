import pandas as pd

import numpy as np
import time

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from .extensions.Col_Selector import InPlaceColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from collections import Counter

from .Models import MODELS
from ..helpers.ML_Helpers import (conv_to_list, proc_input,
                                  get_possible_init_params,
                                  get_obj_and_params,
                                  user_passed_param_check,
                                  f_array, replace_with_in_params,
                                  type_check, mem_check)

from .Models import AVALIABLE as AVALIABLE_MODELS
from .Feature_Selectors import AVALIABLE as AVALIABLE_SELECTORS
from .Metrics import AVALIABLE as AVALIABLE_METRICS
from .Ensembles import AVALIABLE as AVALIABLE_ENSEMBLES
from .Feat_Importances import AVALIABLE as AVALIABLE_IMPORTANCES

from .Samplers import get_sampler_and_params
from .Feature_Selectors import get_feat_selector_and_params
from .Metrics import get_metric
from .Scalers import get_scaler_and_params
from .Imputers import get_imputer_and_params
from .Ensembles import (get_ensemble_and_params, Basic_Ensemble,
                        DES_Ensemble)
from .Feat_Importances import get_feat_importances_and_params

from .Nevergrad import NevergradSearchCV
import os

MEM_MODE = False


class Model_Pipeline():
    '''Helper class for handling all of the different parameters involved in
    model training, scaling, handling different datatypes ect...
    '''

    def __init__(self, ML_params, CV, search_split_vals,
                 all_data_keys, targets_key, covar_scopes,
                 cat_encoders, progress_bar, _print=print):
        ''' Init function for Model

        Parameters
        ----------
        ML_params : dict
            Dictionary of different ML params, the following must be included,
            (See the docstring for ABCD_ML.Set_Default_ML_Params for a more
            detailed description of all parameters contained within ML_params)

            - model_types : str or list of str,
                Each string refers to a type of model to train.
                If a list of strings is passed then an ensemble model
                will be created over all individual models.
            - metrics : str or list
                Metric / scorer str indicator, or list of. If list, the
                metric in the first index will be used during model selection.
            - imputers : str, list or None
                filler.
            - imputer_scopes : str, list or None
                filler.
            - scalers : str, list or None
                str indicator (or list of) for what type of data scaling
                to use if any. If list, data will scaled in list order.
            - scaler_scopes : str, list or None
                The scope of each scaler passed, determining which columns
                that scaler should act on.
            - samplers : str, list or None
                Re-sampler objects
            - sample_on : str or list or None
                params for samplers.
            - feat_selectors : str, list or None
                str indicator (or list of) for what type of feat selector(s)
                to use if any. If list, selectors will be applied in that
                order.
            - n_splits : int
                The number of folds to use during the Evaluate repeated
                k-fold.
            - n_repeats : int
                The number of repeats to do during the Evaluate repeated
                k-fold.
            - search_splits : int, str or list
                The number of internal folds to use during modeling training
                / parameter selection
            - ensemble_type : str or list of str
                Each string refers to a type of ensemble to train,
                or 'basic ensemble' for base behavior.
            - ensemble_split : float, int or None
                If a an ensemble_type that requires fitting is passed,
                i.e., not "basic ensemble", then this param is
                the porportion of the train_data within each fold to
                use towards fitting the ensemble.
            - search_type : {str, None}
                The type of parameter search to conduct if any.
            - model_type_params : int, str or list of
                The index or str name of the param index for `model_type`
            - imputer_params : int, str or list of
                The index or str name of the param index for `imputer`
            - scaler_params : int, str or list of
                The index or str name of the param index for `scaler`
            - feat_selector_params : int, str or list of
                The index or str name of the param index for `feat_selector`
            - ensemble_type_params : int, str or list of
                The index or str name of the param index for `ensemble_type`
            - n_jobs : int
                The number of jobs to use during model training.
            - search_n_iter : int
                The number of random searches to conduct in random search
                models.
            - feats_to_use : {'all', 'data', 'covars'}
                The subset of data to use, either all avaliable, just
                the data, or just the covars.
            - compute_train_score : bool
                If True, computes the training score of the model pipeline,
                if false only computes the testing score.
            - random_state : int or None
                The random state to use for CV splits / within modeling.
            - cache : str or None
                Cache dr
            - extra_params : dict
                The dictionary of any extra params to be passed to models or
                data scalers.

        CV : ABCD_ML CV
            The class defined ABCD_ML CV object for defining
            custom validation splits.

        search_split_vals : pandas Series or None
            Use if search_splits not int.

        all_data_keys : dict
            Contains data_keys, covars_keys, strat_keys and cat_keys,
            as the column names within all_data.

        targets_key : str or list
            The str or list corresponding to the column keys for the targets
            within the data passed to Evaluate_Model or Test Model.

        target_encoder : sklearn encoder or list of
            The encoder or list of encoders, used in the case of targets
            needing to be transformed in some way.

        covar_scopes : dict
            Contains the different cols / scopes for each type
            of covar.

        cat_encoders : list of tuples
            The encoders for conv'ing categorical.

        progress_bar : None or tqdm obj
            Either None, to not use progress bar or a tqdm object, to
            display progress.

        _print : func, optional
            The print function to use, by default the python print,
            but designed to be passed ABCD_ML._ML_print

        '''

        # Set class parameters
        self.CV = CV
        self.search_split_vals = search_split_vals
        self.data_keys = all_data_keys['data_keys']
        self.covars_keys = all_data_keys['covars_keys']
        self.strat_keys = all_data_keys['strat_keys']
        self.cat_keys = all_data_keys['cat_keys']
        self.targets_key = targets_key
        self.covar_scopes = covar_scopes
        self.cat_encoders = cat_encoders
        self.progress_bar = progress_bar
        self._print = _print

        # Un-pack ML_params
        self.model_strs = conv_to_list(ML_params['model'])
        self.metric_strs = conv_to_list(ML_params['metric'])
        self.imputer_strs = conv_to_list(ML_params['imputer'])
        self.imputer_scopes = conv_to_list(ML_params['imputer_scope'])
        self.scaler_strs = conv_to_list(ML_params['scaler'])
        self.scaler_scopes = conv_to_list(ML_params['scaler_scope'])
        self.sampler_strs = conv_to_list(ML_params['sampler'])
        self.sample_on = conv_to_list(ML_params['sample_on'])
        self.feat_selector_strs = conv_to_list(ML_params['feat_selector'])
        self.feat_importances_strs =\
            conv_to_list(ML_params['feat_importances'])
        self.splits = ML_params['splits']
        self.n_repeats = ML_params['n_repeats']
        self.search_splits = ML_params['search_splits']
        self.ensemble_strs = conv_to_list(ML_params['ensemble'])
        self.ensemble_split = ML_params['ensemble_split']
        self.n_jobs = ML_params['n_jobs']
        self.search_n_iter = ML_params['search_n_iter']
        self.compute_train_score = ML_params['compute_train_score']
        self.random_state = ML_params['random_state']
        self.cache = ML_params['cache']
        self.extra_params = ML_params['extra_params'].copy()

        # Un-pack param search ML_params
        self.search_type = ML_params['search_type']

        self.model_params =\
            conv_to_list(ML_params['model_params'])
        self.imputer_params =\
            conv_to_list(ML_params['imputer_params'])
        self.scaler_params =\
            conv_to_list(ML_params['scaler_params'])
        self.sampler_params =\
            conv_to_list(ML_params['sampler_params'])
        self.feat_selector_params =\
            conv_to_list(ML_params['feat_selector_params'])
        self.ensemble_params =\
            conv_to_list(ML_params['ensemble_params'])
        self.feat_importances_params =\
            conv_to_list(ML_params['feat_importances_params'])

        # Default params just sets (sub)problem type for now
        self._set_default_params()

        self.subjects_to_use = ML_params['subjects_to_use']

        # Set all_keys based on feats_to_use
        self._set_all_keys(ML_params['feats_to_use'])

        # Process inputs
        self._check_all_for_user_passed()
        self._process_model()
        self._process_feat_selectors()
        self._process_imputers()
        self._process_scalers()
        self._process_samplers()
        self._process_ensemble_types()
        self._process_metrics()
        self._process_feat_importances()

        # Make the model pipeline, save to self.base_model_pipeline
        self._make_drop_strat()
        self._set_base_model_pipeline()

    def _set_default_params(self):
        self.problem_type = ''
        self.n_splits = None
        self.user_passed_objs = {}

        self.flags = {'linear': False,
                      'tree': False}

    def _set_all_keys(self, feats_to_use):
        '''Strat keys then target keys should always be last,
           as to no effect earlier indexing.'''

        # Always add self.strat_keys + targets_key at end
        if isinstance(self.targets_key, str):
            t_key = [self.targets_key]
        else:
            t_key = self.targets_key

        keys_at_end = self.strat_keys + t_key

        if isinstance(feats_to_use, str):

            if feats_to_use == 'data' or feats_to_use == 'd':
                self.all_keys = self.data_keys + keys_at_end
            elif feats_to_use == 'covars' or feats_to_use == 'c':
                self.all_keys = self.covars_keys + keys_at_end
            elif feats_to_use == 'all' or feats_to_use == 'a':
                self.all_keys =\
                    self.data_keys + self.covars_keys + keys_at_end
            else:
                self._print('Invalid param:', feats_to_use, 'passed for feats',
                            'to use! Make sure you pass "data", "covars",',
                            '"all"',
                            'or a custom array!')

        # Assume user passed list, if not str
        else:
            self.all_keys = list(feats_to_use) + keys_at_end

        self._print('Using:', len(self.all_keys) - len(keys_at_end), 'feats',
                    level='size')

    def _get_subjects_overlap(self, subjects):
        '''Computer overlapping subjects with subjects
        to use.'''

        overlap = self.subjects_to_use.intersection(set(subjects))
        return np.array(list(overlap))

    def _get_train_inds_from_keys(self, keys):
        '''Assume target is always last within self.all_keys ...
        maybe a bad assumption.'''

        inds = [self.all_keys.index(k) for k in keys if k in self.all_keys]
        return inds

    def _check_all_for_user_passed(self):
        '''If not str passed passed for various str indicators,
        assume that a user passed object was passed instead'''

        cnt = 0

        self.model_strs, cnt =\
            self._check_for_user_passed(self.model_strs, cnt)

        self.feat_selector_strs, cnt =\
            self._check_for_user_passed(self.feat_selector_strs, cnt)

        self.imputer_strs, cnt =\
            self._check_for_user_passed(self.imputer_strs, cnt)

        self.scaler_strs, cnt =\
            self._check_for_user_passed(self.scaler_strs, cnt)

        self.sampler_strs, cnt =\
            self._check_for_user_passed(self.sampler_strs, cnt)

        self.ensemble_strs, cnt =\
            self._check_for_user_passed(self.ensemble_strs, cnt)

    def _check_for_user_passed(self, objs, cnt):

        if objs is not None:
            for o in range(len(objs)):
                if not isinstance(objs[o], str):

                    save_name = 'user passed' + str(cnt)
                    cnt += 1

                    self.user_passed_objs[save_name] = objs[o]
                    objs[o] = save_name

        return objs, cnt

    def _check_params_by_search(self, params):

        if self.search_type is None:

            if (np.array(params) != 0).any():
                self._print('Search type is set to None! Therefore no',
                            'hyper-param search will be conducted, even',
                            'though params were passed.')
                self._print()

                params = [0 for i in range(len(params))]

        return params

    def _process_model(self):
        '''Class function to convert input model to final
        str indicator, based on problem type and common input correction.
        '''

        self.model_strs = self._proc_type_dep_str(self.model_strs,
                                                  AVALIABLE_MODELS)

        self.models, self.model_params =\
            self._get_objs_and_params(self._get_base_model,
                                      self.model_strs, self.model_params)

    def _process_feat_selectors(self):
        '''Class function to convert input feat selectors to a final
        set of feat_selector objects along with parameters,
        based on problem type and common input correction.
        '''

        if self.feat_selector_strs is not None:

            self.feat_selector_strs =\
                self._proc_type_dep_str(self.feat_selector_strs,
                                        AVALIABLE_SELECTORS)

            # Get the feat_selectors tuple, and merged params grid / distr dict
            self.feat_selectors, self.feat_selector_params =\
                self._get_objs_and_params(get_feat_selector_and_params,
                                          self.feat_selector_strs,
                                          self.feat_selector_params)

            # If any base estimators, replace with a model
            self._replace_base_rfe_estimator()

        else:
            self.feat_selectors = []
            self.feat_selector_params = {}

    def _process_metrics(self):
        '''Process self.metric_strs and set to proced version.'''

        self.metric_strs = self._proc_type_dep_str(self.metric_strs,
                                                   AVALIABLE_METRICS)

        self.metrics = [get_metric(metric_str)
                        for metric_str in self.metric_strs]

        # Define the metric to be used in model selection
        self.metric = self.metrics[0]

    def _process_imputers(self):

        if self.imputer_strs is not None:

            conv_imputer_strs = proc_input(self.imputer_strs)
            self._update_extra_params(self.imputer_strs, conv_imputer_strs)

            self.imputer_params =\
                self._param_len_check(conv_imputer_strs, self.imputer_params)

            self.imputer_params =\
                self._check_params_by_search(self.imputer_params)

            self.imputer_scopes = self._scope_len_check(conv_imputer_strs,
                                                        self.imputer_scopes,
                                                        "imputers")

            imputers_and_params =\
                [self._get_imputer(imputer_str, imputer_param, scope)
                 for imputer_str, imputer_param, scope in zip(
                     conv_imputer_strs,
                     self.imputer_params,
                     self.imputer_scopes)]

            while None in imputers_and_params:
                imputers_and_params.remove(None)

            imputers, imputer_params =\
                self._proc_objs_and_params(imputers_and_params)

            # Make column transformers for skipping strat cols
            skip_strat_scopes = ['n' for i in range(len(imputers))]

            self.col_imputers, self.col_imputer_params =\
                self._make_col_version(imputers, imputer_params,
                                       skip_strat_scopes)

        else:
            self.col_imputers = []
            self.col_imputer_params = {}

    def _get_imputer(self, imputer_str, imputer_param, scope):

        # First grab the correct params based on scope
        if scope == 'c' or scope == 'categorical':
            scope = 'categorical'

            cat_inds, ordinal_inds =\
                self._get_cat_ordinal_inds()

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

            if scope == 'f' or scope == 'float':
                scope = 'float'
                keys = 'a'
            else:
                scope = 'custom'
                keys = scope

            inds = self._get_inds_from_scope(keys)
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

        for c_encoder in self.cat_encoders:
            if c_encoder is None:
                raise RuntimeError('Impution on multilabel-type covars is not',
                                   'currently supported!')

        imputer, imputer_params =\
            get_imputer_and_params(imputer_str, self.extra_params,
                                   imputer_param, self.search_type, inds,
                                   cat_inds, ordinal_inds, self.cat_encoders,
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

        avaliable_by_type =\
            self._get_avaliable_by_type(AVALIABLE_MODELS, [imputer_str],
                                        problem_type)

        # If a not a valid model type, then no base estimator
        if imputer_str in avaliable_by_type:
            base_model_str = avaliable_by_type[imputer_str]
            base_model = self._get_base_model(base_model_str,
                                              self.extra_params, imputer_param,
                                              self.search_type)

            base_estimator = base_model[0]
            base_estimator_params = base_model[1]
        else:
            base_estimator = None
            base_estimator_params = {}

        return base_estimator, base_estimator_params

    def _process_scalers(self):
        '''Processes self.scaler to be a list of
        (name, scaler) tuples, and then creates col_scalers
        from that.'''

        if self.scaler_strs is not None:

            # Get converted scaler str and update extra params
            conv_scaler_strs = proc_input(self.scaler_strs)
            self._update_extra_params(self.scaler_strs, conv_scaler_strs)

            # Get the scalers tuple, and scaler_params grid / distr
            scalers, scaler_params =\
                self._get_objs_and_params(get_scaler_and_params,
                                          conv_scaler_strs,
                                          self.scaler_params)

            self.scaler_scopes = self._scope_len_check(scalers,
                                                       self.scaler_scopes,
                                                       "scalers")

            self.col_scalers, self.col_scaler_params =\
                self._make_col_version(scalers, scaler_params,
                                       self.scaler_scopes)

        else:
            self.col_scalers = []
            self.col_scaler_params = {}

    def _make_col_version(self, objs, params, scopes):

        # Make objects first
        col_objs = []

        for i in range(len(objs)):
            name, obj = objs[i][0], objs[i][1]
            inds = self._get_inds_from_scope(scopes[i])

            col_obj =\
                ('col_' + name, InPlaceColumnTransformer([(name, obj, inds)],
                 remainder='passthrough', sparse_threshold=0))

            col_objs.append(col_obj)

        # Change params to reflect objs
        col_params = {}

        for key in params:
            name = key.split('__')[0]
            new_name = 'col_' + name + '__' + key

            col_params[new_name] = params[key]

        return col_objs, col_params

    def _get_inds_from_scope(self, scope):
        '''Return inds from scope, with a check to remove strat
        and target keys always.'''

        if isinstance(scope, str):

            # Lower-case first char only
            scope = scope.lower()[0]

            # All non categorical
            if scope == 'a':
                keys = [k for k in self.all_keys if k not in self.cat_keys]

            # Just data_keys
            elif scope == 'd':
                keys = self.data_keys.copy()

            # Just non-categorical covars
            elif scope == 'c':
                keys = [k for k in self.all_keys if
                        k not in self.cat_keys and
                        k not in self.data_keys]

            # Everything (but strat and targets!)
            elif scope == 'n':
                keys = self.all_keys.copy()

            # Wrong str case
            else:
                self._print('Warning! Passed scaler scope of:', scope,
                            'is invalid!')
                self._print('Setting scope to data only by default.')
                keys = self.data_keys.copy()

        # If not str then assume list / array like containing col names / keys
        else:
            keys = scope

        # Need to remove all strat keys + target_keys, if there regardless
        if isinstance(self.targets_key, list):
            t_keys = self.targets_key
        else:
            t_keys = [self.targets_key]

        for key in self.strat_keys + t_keys:
            try:
                keys.remove(key)
            except ValueError:
                pass

        inds = self._get_train_inds_from_keys(keys)
        return inds

    def _process_samplers(self):
        '''Class function to convert input sampler strs to
        a resampling object.
        '''

        if self.sampler_strs is not None:

            # Initial proc
            conv_sampler_strs = proc_input(self.sampler_strs)
            self._update_extra_params(self.sampler_strs, conv_sampler_strs)

            # Performing proc on input lengths
            self.sampler_params =\
                self._param_len_check(conv_sampler_strs, self.sampler_params)

            self.sampler_params =\
                self._check_params_by_search(self.sampler_params)

            self.sample_on = self._scope_len_check(conv_sampler_strs,
                                                   self.sample_on,
                                                   'samplers')

            recover_strats = self._get_recover_strats(len(conv_sampler_strs))

            # Get the scalers and params
            scalers_and_params =\
                [self._get_sampler(sampler_str, sampler_param, on,
                                   recover_strat) for
                 sampler_str, sampler_param, on, recover_strat in
                 zip(conv_sampler_strs, self.sampler_params, self.sample_on,
                     recover_strats)]

            self.samplers, self.sampler_params =\
                self._proc_objs_and_params(scalers_and_params)

            # Change some sampler params if applicable
            self._check_and_replace_samplers('random_state',
                                             self.random_state)

            if self.search_type is None:
                self._check_and_replace_samplers('n_jobs',
                                                 self.n_jobs)

        else:
            self.samplers = []
            self.sampler_params = {}

    def _get_recover_strats(self, num_samplers):

        recover_strats = []

        # Creates binary mask of, True if any strat inds used
        uses_strat = [len(self._proc_sample_on(self.sample_on[i])[1]) > 0 for
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

        sample_strat_keys = [o for o in on if o in self.strat_keys]

        # Set sample_target
        sample_target = False
        if len(sample_strat_keys) != len(on):
            sample_target = True

        return sample_target, sample_strat_keys

    def _get_sampler(self, sampler_str, sampler_param, on, recover_strat):

        # Grab sample_target and sample_strat_keys, from on
        sample_target, sample_strat_keys = self._proc_sample_on(on)

        # Set strat inds and sample_strat
        strat_inds = self._get_train_inds_from_keys(self.strat_keys)
        sample_strat = self._get_train_inds_from_keys(sample_strat_keys)

        # Set categorical flag
        categorical = True
        if self.problem_type == 'regression':
            categorical = False

        cat_inds, ordinal_inds =\
            self._get_cat_ordinal_inds()
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

    def _process_ensemble_types(self):
        '''Processes ensemble types'''

        if self.ensemble_strs is not None:

            self.ensemble_strs = self._proc_type_dep_str(self.ensemble_strs,
                                                         AVALIABLE_ENSEMBLES)

            # If basic ensemble is in any of the ensemble_strs,
            # ensure it is the only one.
            if np.array(['basic ensemble' in ensemble for
                        ensemble in self.ensemble_strs]).any():

                if len(self.ensemble_strs) > 1:

                    self._print('Warning! "basic ensemble" ensemble type',
                                'passed within a list of ensemble types.')
                    self._print('In order to use multiple ensembles',
                                'they cannot include "basic ensemble".')
                    self._print('Setting to just "basic ensemble" ensemble',
                                ' type!')

                    self.ensemble_strs = ['basic ensemble']

            self.ensembles, self.ensemble_params =\
                self._get_objs_and_params(get_ensemble_and_params,
                                          self.ensemble_strs,
                                          self.ensemble_params)

        else:
            self.ensembles = [('basic ensemble', None)]
            self.ensemble_params = {}

    def _process_feat_importances(self):
        '''Process the feat importances.
        Note: cannot pass user define objects here.'''

        if self.feat_importances_strs is not None:

            # Proc input strs and params
            names = self._proc_type_dep_str(self.feat_importances_strs,
                                            AVALIABLE_IMPORTANCES)
            params = self._param_len_check(names, self.feat_importances_params)

            # Create the feat importance objects
            self.feat_importances =\
                [get_feat_importances_and_params(name, self.extra_params,
                                                 param, self.problem_type,
                                                 self.n_jobs)
                 for name, param in zip(names, params)]

        else:
            self.feat_importances = []

    def _make_drop_strat(self):
        '''This creates a columntransformer in order to drop
        the strat cols from X!'''

        non_strat_inds = self._get_inds_from_scope('n')
        identity = FunctionTransformer(validate=False)

        # Make base col_transformer, just for dropping strat cols
        col_transformer =\
            ColumnTransformer(transformers=[('keep_all_but_strat_inds',
                                             identity, non_strat_inds)],
                              remainder='drop', sparse_threshold=0)

        # Put in list, to easily add to pipeline
        self.drop_strat = [('drop_strat', col_transformer)]

    def _proc_type_dep_str(self, in_strs, avaliable):
        '''Helper function to perform str correction on
        underlying proble type dependent input, e.g., for
        metric or ensemble_types, and to update extra params
        and check to make sure input is valid ect...'''

        conv_strs = proc_input(in_strs)

        assert self._check_avaliable(conv_strs, avaliable),\
            "Error " + conv_strs + ' are not avaliable for this problem type'

        avaliable_by_type = self._get_avaliable_by_type(avaliable, conv_strs)
        final_strs = [avaliable_by_type[conv_str] for conv_str in conv_strs]

        self._update_extra_params(in_strs, final_strs)
        return final_strs

    def _check_avaliable(self, in_strs, avaliable):

        avaliable_by_type = self._get_avaliable_by_type(avaliable, in_strs)

        check = np.array([m in avaliable_by_type for
                          m in in_strs]).all()

        return check

    def _get_avaliable_by_type(self, avaliable, in_strs, problem_type='class'):

        if problem_type == 'class':
            problem_type = self.problem_type

        avaliable_by_type = avaliable[problem_type]

        for s in in_strs:
            if 'user passed' in s:
                avaliable_by_type[s] = s

        return avaliable_by_type

    def _update_extra_params(self, orig_strs, conv_strs):
        '''Helper method to update class extra params in the case
        where model_types or scaler str indicators change,
        and they were refered to in extra params as the original name.

        Parameters
        ----------
        orig_strs : list
            List of original str indicators.

        conv_strs : list
            List of final-proccesed str indicators, indices should
            correspond to the order of orig_strs
        '''

        for i in range(len(orig_strs)):
            if orig_strs[i] in self.extra_params:
                self.extra_params[conv_strs[i]] =\
                    self.extra_params[orig_strs[i]]

    def _get_objs_and_params(self, get_func, names, params):
        '''Helper function to grab scaler / feat_selectors and
        their relevant parameter grids'''

        # Check + fill length of passed params + names
        params = self._param_len_check(names, params)

        # Check search type
        params = self._check_params_by_search(params)

        # Make the object + params based on passed settings
        objs_and_params = []
        for name, param in zip(names, params):
            if 'user passed' in name:

                user_obj = self.user_passed_objs[name]
                user_obj_params = user_passed_param_check(param, name)

                objs_and_params.append((name, (user_obj, user_obj_params)))

            else:

                objs_and_params.append((name, get_func(name, self.extra_params,
                                                       param, self.search_type)
                                        ))

        objs, params = self._proc_objs_and_params(objs_and_params)
        return objs, params

    def _param_len_check(self, names, params):

        if isinstance(params, dict) and len(names) == 1:
            return params

        if len(params) > len(names):
            self._print('Warning! More params passed than objs')
            self._print('Extra params have been truncated.')
            params = params[:len(names)]

        while len(names) != len(params):
            params.append(0)

        return params

    def _scope_len_check(self, objs, scopes, error_name):

        scope_name = 'scopes'
        if error_name == 'imputers':
            scope_name = 'sample_on'

        # If more scopes then objs passed, assume passed inds
        if len(scopes) > len(objs):
            scopes = [scopes]

        if len(objs) != len(scopes):

            if error_name == 'imputers':

                self._print('Warning: non equal number of', error_name,
                            'passed as scopes! imputers without scope will be',
                            'ignored.')

            else:
                self._print('Warning: non equal number of', error_name,
                            'passed as', scope_name, '! Filling remaining',
                            scope_name, 'with', scopes[0])

                while len(objs) != len(scopes):
                    scopes.append(scopes[0])

        return scopes

    def _proc_objs_and_params(self, objs_and_params):

        # If two of same object passed, change name
        objs_and_params = self.check_for_duplicate_names(objs_and_params)

        # Construct the obj as list of (name, obj) tuples
        objs = [(c[0], c[1][0]) for c in objs_and_params]

        # Grab the params, and merge them into one dict of all params
        params = {k: v for params in objs_and_params
                  for k, v in params[1][1].items()}

        return objs, params

    def check_for_duplicate_names(self, objs_and_params):

        names = [c[0] for c in objs_and_params]

        # If any repeats
        if len(names) != len(set(names)):
            new_objs_and_params = []

            for obj in objs_and_params:
                name = obj[0]

                if name in names:

                    cnt = 1
                    used = [c[0] for c in new_objs_and_params]
                    while name + str(cnt) in used:
                        cnt += 1

                    # Need to change name within params also
                    base_obj = obj[1][0]
                    base_obj_params = obj[1][1]

                    new_obj_params = {}
                    for param_name in base_obj_params:

                        p_split = param_name.split('__')
                        new_param_name = p_split[0] + str(cnt)
                        new_param_name += '__' + '__'.join(p_split[1:])

                        new_obj_params[new_param_name] =\
                            base_obj_params[param_name]

                    new_objs_and_params.append((name + str(cnt),
                                               (base_obj, new_obj_params)))

                else:
                    new_objs_and_params.append(obj)

            return new_objs_and_params
        return objs_and_params

    def _get_cat_ordinal_inds(self):

        cat_keys = self.covar_scopes['categorical']
        ordinal_keys = self.covar_scopes['ordinal categorical']

        cat_inds = [self._get_inds_from_scope(k) for k in cat_keys]
        ordinal_inds = self._get_inds_from_scope(ordinal_keys)

        return cat_inds, ordinal_inds

    def _replace_base_rfe_estimator(self):
        '''Check feat selectors for a RFE model'''

        for i in range(len(self.feat_selectors)):

            try:
                base_model_str = self.feat_selectors[i][1].estimator

                # Default behavior is use linear
                if base_model_str is None:
                    base_model_str = 'linear'

                base_model_str =\
                    self._proc_type_dep_str([base_model_str],
                                            AVALIABLE_MODELS)[0]

                self.feat_selectors[i][1].estimator =\
                    self._get_base_model(base_model_str, self.extra_params, 0,
                                         None)[0]

            except AttributeError:
                pass

    def _check_and_replace_samplers(self, param_name, replace_value):

        for i in range(len(self.samplers)):

            try:
                getattr(self.samplers[i][1].sampler, param_name)
                setattr(self.samplers[i][1].sampler, param_name, replace_value)

            except AttributeError:
                pass

    def Evaluate(self, data, train_subjects, splits_vals=None):
        '''Method to perform a full evaluation
        on a provided model type and training subjects, according to
        class set parameters.

        Parameters
        ----------
        data : pandas DataFrame
            ABCD_ML formatted, with both training and testing data.

        train_subjects : array-like
            An array or pandas Index of the train subjects should be passed.

        splits_vals : pandas Series or None, optional
            Subjects with unique vals, for doing leave one out split

            (default = None)

        Returns
        ----------
        array-like of array-like
            numpy array of numpy arrays,
            where each internal array contains the raw scores as computed for
            all passed in metrics, computed for each fold within
            each repeat.
            e.g., array will have a length of `n_repeats` * n_splits
            (num folds) and each internal array will have the same length
            as the number of metrics.
        '''

        # Set train_subjects according to self.subjects_to_use
        train_subjects = self._get_subjects_overlap(train_subjects)

        # Init raw_preds_df
        self._init_raw_preds_df(train_subjects)

        # Setup the desired eval splits
        subject_splits =\
            self._get_eval_splits(train_subjects, splits_vals)

        all_train_scores, all_scores = [], []
        fold_ind = 0

        if self.progress_bar is not None:
            repeats_bar = self.progress_bar(total=self.n_repeats,
                                            desc='Repeats')

            folds_bar = self.progress_bar(total=self.n_splits,
                                          desc='Folds')

        # For each split with the repeated K-fold
        for train_subjects, test_subjects in subject_splits:

            # Fold name verbosity
            repeat = str((fold_ind // self.n_splits) + 1)
            fold = str((fold_ind % self.n_splits) + 1)
            self._print(level='name')
            self._print('Repeat: ', repeat, '/', self.n_repeats, ' Fold: ',
                        fold, '/', self.n_splits, sep='', level='name')

            if self.progress_bar is not None:
                repeats_bar.n = int(repeat) - 1
                repeats_bar.refresh()

                folds_bar.n = int(fold) - 1
                folds_bar.refresh()

            if MEM_MODE:
                print('Pre-Test')
                mem_check()

            # Run actual code for this evaluate fold
            start_time = time.time()
            train_scores, scores = self.Test(data, train_subjects,
                                             test_subjects, fold_ind)

            if MEM_MODE:
                print('Post-Test')
                mem_check()

            # Time by fold verbosity
            elapsed_time = time.time() - start_time
            time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
            self._print('Time Elapsed:', time_str, level='time')

            # Score by fold verbosity
            if self.compute_train_score:
                for i in range(len(self.metric_strs)):
                    self._print('train ', self.metric_strs[i], ': ',
                                train_scores[i], sep='', level='score')

            for i in range(len(self.metric_strs)):
                self._print('val ', self.metric_strs[i], ': ',
                            scores[i], sep='', level='score')

            all_train_scores.append(train_scores)
            all_scores.append(scores)
            fold_ind += 1

        if self.progress_bar is not None:
            repeats_bar.n = self.n_repeats
            repeats_bar.refresh()
            repeats_bar.close()

            folds_bar.n = self.n_splits
            folds_bar.refresh()
            folds_bar.close()

        # If any local feat importances
        for feat_imp in self.feat_importances:
            feat_imp.set_final_local()

        # self.micro_scores = self._compute_micro_scores()

        # Return all scores
        return (np.array(all_train_scores), np.array(all_scores),
                self.raw_preds_df, self.feat_importances)

    def Test(self, data, train_subjects, test_subjects, fold_ind='test'):
        '''Method to test given input data, training a model on train_subjects
        and testing the model on test_subjects.

        Parameters
        ----------
        data : pandas DataFrame
            ABCD_ML formatted, with both training and testing data.

        train_subjects : array-like
            An array or pandas Index of train subjects should be passed.

        test_subjects : array-like
            An array or pandas Index of test subjects should be passed.

        Returns
        ----------
        array-like
            A numpy array of scores as determined by the passed
            metric/scorer(s) on the provided testing set.
        '''

        # Ensure train and test subjects are just the requested overlap
        train_subjects = self._get_subjects_overlap(train_subjects)
        test_subjects = self._get_subjects_overlap(test_subjects)

        # Ensure data being used is just the selected col / feats
        data = data[self.all_keys]

        # Init raw_preds_df
        if fold_ind == 'test':

            if self.compute_train_score:
                self._init_raw_preds_df(np.concatenate([train_subjects,
                                                        test_subjects]))
            else:
                self._init_raw_preds_df(test_subjects)

        # Assume the train_subjects and test_subjects passed here are final.
        train_data = data.loc[train_subjects]
        test_data = data.loc[test_subjects]

        self._print('Train subjects:', train_data.shape[0], level='size')
        self._print('Val/Test subjects:', test_data.shape[0], level='size')

        # Wrap in search CV / set to self.Model, final pipeline
        self._set_model_pipeline(train_data)

        # Train the model(s)
        self._train_model(train_data)

        # Proc the different feat importances
        self._proc_feat_importance(train_data, test_data, fold_ind)

        # Get the scores
        if self.compute_train_score:
            train_scores = self._get_scores(train_data, 'train_', fold_ind)
        else:
            train_scores = 0

        scores = self._get_scores(test_data, '', fold_ind)

        if fold_ind == 'test':

            return (train_scores, scores, self.raw_preds_df,
                    self.feat_importances)

        return train_scores, scores

    def _get_eval_splits(self, train_subjects, splits_vals):

        if splits_vals is None:

            self.n_splits = self.splits

            subject_splits =\
                self.CV.repeated_k_fold(train_subjects, self.n_repeats,
                                        self.n_splits, self.random_state,
                                        return_index=False)

        else:

            # Set num splits
            self.n_splits =\
                self.CV.get_num_groups(train_subjects, splits_vals)

            # Generate the leave-out CV
            subject_splits =\
                self.CV.repeated_leave_one_group_out(train_subjects,
                                                     self.n_repeats,
                                                     splits_vals,
                                                     return_index=False)

        return subject_splits

    def _update_model_ensemble_params(self, to_add, model=True, ensemble=True):

        if model:
            new_model_params = {}
            for key in self.model_params:
                new_model_params[to_add + '__' + key] =\
                    self.model_params[key]
            self.model_params = new_model_params

        if ensemble:

            new_ensemble_params = {}
            for key in self.ensemble_params:
                new_ensemble_params[to_add + '__' + key] =\
                    self.ensemble_params[key]
            self.ensemble_params = new_ensemble_params

    def _basic_ensemble(self, models, name, ensemble=False):

        if len(models) == 1:
            return models

        else:
            basic_ensemble = Basic_Ensemble(models)
            self._update_model_ensemble_params(name, ensemble=ensemble)

            return [(name, basic_ensemble)]

    def _basic_ensemble_pipe(self, models, name):
        '''Models as [(name, obj), ect...]'''

        basic_ensemble = self._basic_ensemble(models, name, ensemble=True)
        return self._make_model_pipeline(basic_ensemble)

    def _set_base_model_pipeline(self):

        # Check if basic ensemble
        if self.ensembles[0][1] is None:

            self.base_model_pipeline =\
                self._basic_ensemble_pipe(self.models, self.ensembles[0][0])

        # Otherwise special ensembles
        else:

            new_ensembles = []

            for ensemble in self.ensembles:
                ensemble_obj = ensemble[1][0]
                ensemble_extra_params = ensemble[1][1]
                ensemble_name = ensemble[0]

                try:
                    single_estimator =\
                        ensemble_extra_params.pop('single_estimator')
                except KeyError:

                    self._print('Assuming that this ensemble does not build',
                                'on a single estimator/model',
                                'if this is not the case, pass',
                                'single_estimator: True, in ensemble params')
                    single_estimator = False

                # If needs a single estimator, but multiple models passed,
                # wrap in ensemble!
                if single_estimator:

                    se_ensemb_name = 'ensemble for single est'
                    models = self._basic_ensemble(self.models,
                                                  se_ensemb_name)
                else:
                    models = self.models

                try:
                    needs_split = ensemble_extra_params.pop('needs_split')
                except KeyError:
                    self._print('Assuming this ensemble needs a split!')
                    self._print('Pass needs_split: False, in ensemble params',
                                'if this is not the case.')
                    needs_split = True

                if needs_split:

                    # Init with default params
                    ensemble = ensemble_obj()

                    new_ensembles.append(
                        (ensemble_name, DES_Ensemble(models,
                                                     ensemble,
                                                     ensemble_name,
                                                     self.ensemble_split,
                                                     ensemble_extra_params,
                                                     self.random_state)))

                    self._update_model_ensemble_params(ensemble_name)

                else:

                    new_ensembles.append(
                        (ensemble_name,
                         ensemble_obj(base_estimator=models[0][1],
                                      **ensemble_extra_params)))

                    # Have to change model name to base_estimator
                    self.model_params =\
                        replace_with_in_params(self.model_params, models[0][0],
                                               'base_estimator')

                    self._update_model_ensemble_params(ensemble_name,
                                                       ensemble=False)

            e_of_e = 'ensemble of ensembles'
            self.base_model_pipeline =\
                self._basic_ensemble_pipe(new_ensembles, e_of_e)

    def _make_model_pipeline(self, models):
        '''Make the model pipeline object

        Returns
        ----------
        imblearn Pipeline
            Pipeline object with all relevant column specific data
            scalers, and then the passed in model.
        '''

        steps = self.col_scalers + self.col_imputers + self.samplers \
            + self.drop_strat + self.feat_selectors + models

        if self.cache is not None:
            os.makedirs(self.cache, exist_ok=True)

        model_pipeline = Pipeline(steps, memory=self.cache)

        return model_pipeline

    def _set_model_pipeline(self, train_data):

        # Set Model_Pipeline as Model
        self.Model =\
            self._get_search_model(self.base_model_pipeline,
                                   self._get_search_cv(train_data.index))

    def _get_base_fitted_pipeline(self):

        if self.search_type is None:
            base_pipeline = self.Model
        else:
            base_pipeline = self.Model.best_estimator_

        return base_pipeline

    def _get_base_fitted_model(self):

        base_pipeline = self._get_base_fitted_pipeline()
        last_name = base_pipeline.steps[-1][0]
        base_model = base_pipeline[last_name]

        return base_model

    def _set_model_flags(self):

        base_model = self._get_base_fitted_model()

        try:
            base_model.coef_
            self.flags['linear'] = True
        except AttributeError:
            pass

        try:
            base_model.feature_importances_
            self.flags['tree'] = True
        except AttributeError:
            pass

    def _proc_feat_importance(self, train_data, test_data, fold_ind):

        # Ensure model flags are set / there are feat importances to proc
        if len(self.feat_importances) > 0:
            self._set_model_flags()
        else:
            return

        # Process each feat importance
        for feat_imp in self.feat_importances:

            split = feat_imp.split

            # Init global feature df
            if fold_ind == 0 or fold_ind == 'test':

                X, y = self._get_X_y(train_data, X_as_df=True)
                feat_imp.init_global(X, y)

            # Local init - Test
            if fold_ind == 'test':

                if split == 'test':
                    X, y = self._get_X_y(test_data, X_as_df=True)

                elif split == 'train':
                    X, y = self._get_X_y(train_data, X_as_df=True)

                elif split == 'all':
                    X, y =\
                        self._get_X_y(pd.concat([train_data, test_data]),
                                      X_as_df=True)

                feat_imp.init_local(X, y, test=True, n_splits=None)

            # Local init - Evaluate
            elif fold_ind % self.n_splits == 0:

                X, y = self._get_X_y(pd.concat([train_data, test_data]),
                                     X_as_df=True)

                feat_imp.init_local(X, y, n_splits=self.n_splits)

            self._print('Calculate', feat_imp.name, 'feat importances',
                        level='name')

            # Get base fitted model
            base_model = self._get_base_fitted_model()

            # Optionally proc train, though train is always train
            if feat_imp.get_data_needed_flags(self.flags):
                X_train = self._proc_X_train(train_data)
            else:
                X_train = None

            # Test depends on scope
            if split == 'test':
                test = test_data
            elif split == 'train':
                test = train_data
            elif split == 'all':
                test = pd.concat([train_data, test_data])

            # Always proc test.
            X_test, y_test = self._proc_X_test(test)

            try:
                fold = fold_ind % self.n_splits
            except TypeError:
                fold = 'test'

            # Process the feature importance, provide all needed
            feat_imp.proc_importances(base_model, X_test, y_test=y_test,
                                      X_train=X_train, scorer=self.metric,
                                      fold=fold)

            # For local, need an intermediate average, move df to dfs
            if isinstance(fold_ind, int):
                if fold_ind % self.n_splits == self.n_splits-1:
                    feat_imp.proc_local()

    def _get_ensemble_split(self, train_data):
        '''Split the train subjects further only if an ensemble split
        is defined and ensemble type has been changed from basic ensemble!'''

        ensemble_data = None

        if (self.ensemble_split is not None and
           self.ensemble_strs[0] != 'basic ensemble'):

            train_subjects, ensemble_subjects =\
                self.CV.train_test_split(train_data.index,
                                         test_size=self.ensemble_split,
                                         random_state=self.random_state)

            # Set ensemble data
            ensemble_data = train_data.loc[ensemble_subjects]

            # Set train_data to new smaller set
            train_data = train_data.loc[train_subjects]

            self._print('Performed extra ensemble train data split',
                        level='name')
            self._print('New Train size:', train_data.shape[0], level='size')
            self._print('Ensemble Val size:', ensemble_data.shape[0],
                        level='size')

        return train_data, ensemble_data

    def _get_X_y(self, data, X_as_df=False, copy=False):
        '''Helper method to get X,y data from ABCD ML formatted df.

        Parameters
        ----------
        data : pandas DataFrame
            ABCD ML formatted.

        X_as_df : bool, optional
            If True, return X as a pd DataFrame,
            otherwise, return as a numpy array

            (default = False)

        copy : bool, optional
            If True, return a copy of X

            (default = False)

        Returns
        ----------
        array-like
            X data for ML
        array-like
            y target for ML
        '''

        if copy:
            X = data.drop(self.targets_key, axis=1).copy()
            y = data[self.targets_key].copy()
        else:
            X = data.drop(self.targets_key, axis=1)
            y = data[self.targets_key]

        if not X_as_df:
            X = np.array(X).astype(float)

        y = np.array(y).astype(float)

        return X, y

    def _get_search_cv(self, train_data_index):

        if self.search_split_vals is None:
            search_cv = self.CV.k_fold(train_data_index, self.search_splits,
                                       random_state=self.random_state,
                                       return_index=True)

        else:
            search_cv = self.CV.leave_one_group_out(train_data_index,
                                                    self.search_split_vals,
                                                    return_index=True)

        return search_cv

    def _train_model(self, train_data):
        '''Helper method to train a models given
        a str indicator and training data.

        Parameters
        ----------
        train_data : pandas DataFrame
            ABCD_ML formatted, training data.

        Returns
        ----------
        sklearn api compatible model object
            The trained model.
        '''

        # Data, score split
        X, y = self._get_X_y(train_data)

        # Fit the model
        self.Model.fit(X, y)

        # If a search object, show the best params
        self._show_best_params()

    def _show_best_params(self):

        try:
            name = self.Model.name
        except AttributeError:
            return

        if name == 'nevergrad':

            all_params = [self.col_scaler_params, self.col_imputer_params,
                          self.sampler_params, self.feat_selector_params,
                          self.model_params, self.ensemble_params]
        names = ['Scalers', 'Imputers', 'Samplers', 'Feat Selectors',
                 'Models', 'Ensembles']

        self._print('Params Selected by Best Pipeline:', level='params')

        for params, name in zip(all_params, names):

            if len(params) > 0:

                to_show = []

                base = list(params)[0].split('__')[0]
                all_ps = self.Model.best_estimator_[base].get_params()

                for p in params:

                    ud = params[p]

                    if type_check(ud):
                        nm = p.replace(base + '__', '')
                        to_show.append(nm + ': ' + str(all_ps[nm]))

                if len(to_show) > 0:
                    self._print(name, level='params')

                    for show in to_show:
                        self._print(show, level='params')

                    self._print('', level='params')

    def _get_search_model(self, model, search_cv):
        '''Passed model as a created pipeline object + search_cv info,
        if search_type isn't None, creates the search wrapper.'''

        if self.search_type is None:
            return model

        # Set the search params
        search_params = {}
        search_params['optimizer_name'] = self.search_type
        search_params['estimator'] = model
        search_params['cv'] = search_cv
        search_params['scoring'] = self.metric
        search_params['n_jobs'] = self.n_jobs
        search_params['n_iter'] = self.search_n_iter
        search_params['random_state'] = self.random_state

        # Merge the different params / grids of params
        all_params = {}
        all_params.update(self.col_scaler_params)
        all_params.update(self.col_imputer_params)
        all_params.update(self.sampler_params)
        all_params.update(self.feat_selector_params)
        all_params.update(self.model_params)
        all_params.update(self.ensemble_params)

        # Create the search object
        search_params['param_distributions'] = all_params
        search_model = NevergradSearchCV(**search_params)

        return search_model

    def _get_base_model(self, model_type, extra_params, model_type_params,
                        search_type):

        model, extra_model_params, model_type_params =\
            get_obj_and_params(model_type, MODELS, self.extra_params,
                               model_type_params, search_type)

        # Set class param values from possible model init params
        possible_params = get_possible_init_params(model)

        if 'n_jobs' in possible_params:
            if self.search_type is None:
                extra_model_params['n_jobs'] = self.n_jobs
            else:
                extra_model_params['n_jobs'] = 1

        if 'random_state' in possible_params:
            extra_model_params['random_state'] = self.random_state

        # Init model, w/ any user passed params + class params
        model = model(**extra_model_params)

        return model, model_type_params

    def _get_scores(self, test_data, eval_type, fold_ind):
        '''Helper method to get the scores of
        the trained model saved in the class on input test data.
        For all metrics/scorers.

        Parameters
        ----------
        test_data : pandas DataFrame
            ABCD ML formatted test data.

        eval_type : {'train_', ''}

        fold_ind : int or 'test'

        Returns
        ----------
        float
            The score of the trained model on the given test data.
        '''

        # Data, score split
        X_test, y_test = self._get_X_y(test_data)

        # Add raw preds to raw_preds_df
        self._add_raw_preds(X_test, y_test, test_data.index, eval_type,
                            fold_ind)

        # Get the scores
        scores = [metric(self.Model, X_test, y_test)
                  for metric in self.metrics]

        return np.array(scores)

    def _add_raw_preds(self, X_test, y_test, subjects, eval_type, fold_ind):

        if fold_ind == 'test':
            fold = 'test'
            repeat = ''
        else:
            fold = str((fold_ind % self.n_splits) + 1)
            repeat = str((fold_ind // self.n_splits) + 1)

        self.classes = np.unique(y_test)

        try:
            raw_prob_preds = self.Model.predict_proba(X_test)
            pred_col = eval_type + repeat + '_prob'

            if len(np.shape(raw_prob_preds)) == 3:

                for i in range(len(raw_prob_preds)):
                    p_col = pred_col + '_class_' + str(self.classes[i])
                    class_preds = [val[1] for val in raw_prob_preds[i]]
                    self.raw_preds_df.loc[subjects, p_col] = class_preds

            elif len(np.shape(raw_prob_preds)) == 2:

                for i in range(np.shape(raw_prob_preds)[1]):
                    p_col = pred_col + '_class_' + str(self.classes[i])
                    class_preds = raw_prob_preds[:, i]
                    self.raw_preds_df.loc[subjects, p_col] = class_preds

            else:
                self.raw_preds_df.loc[subjects, pred_col] = raw_prob_preds

        except AttributeError:
            pass

        raw_preds = self.Model.predict(X_test)
        pred_col = eval_type + repeat

        if len(np.shape(raw_preds)) == 2:
            for i in range(np.shape(raw_preds)[1]):
                p_col = pred_col + '_class_' + str(self.classes[i])
                class_preds = raw_preds[:, i]
                self.raw_preds_df.loc[subjects, p_col] = class_preds

        else:
            self.raw_preds_df.loc[subjects, pred_col] = raw_preds

        self.raw_preds_df.loc[subjects, pred_col + '_fold'] = fold

        # Make copy of true values
        if len(np.shape(y_test)) > 1:
            for i in range(len(self.targets_key)):
                self.raw_preds_df.loc[subjects, self.targets_key[i]] =\
                    y_test[:, i]

        elif isinstance(self.targets_key, list):
            t_base_key = '_'.join(self.targets_key[0].split('_')[:-1])
            self.raw_preds_df.loc[subjects, 'multiclass_' + t_base_key] =\
                y_test

        else:
            self.raw_preds_df.loc[subjects, self.targets_key] = y_test

    def _compute_micro_scores(self):

        micro_scores = []

        # For each metric
        for metric in self.metrics:
            score_func = metric._score_func
            sign = metric._sign

            if 'needs_proba=True' in metric._factory_args():
                prob = '_prob'
            else:
                prob = ''

            eval_type = ''

            by_repeat = []
            for repeat in range(1, self.n_repeats+1):

                p_col = eval_type + str(repeat) + prob

                try:
                    pred_col = self.raw_preds_df[p_col]
                    valid_subjects = pred_col[~pred_col.isnull()].index

                except KeyError:

                    if len(self.classes) == 2:
                        pred_col = self.raw_preds_df[p_col + '_class_' +
                                                     str(self.classes[1])]
                        valid_subjects = pred_col[~pred_col.isnull()].index
                    else:
                        pred_col = self.raw_preds_df[[p_col + '_class_' +
                                                      str(i) for
                                                      i in self.classes]]
                        valid_subjects =\
                            pred_col[~pred_col.isnull().any(axis=1)].index

                # Only for the target is one col case
                truth = self.raw_preds_df.loc[valid_subjects, self.targets_key]
                predicted = pred_col.loc[valid_subjects]

                by_repeat.append(sign * score_func(truth, predicted))
            micro_scores.append(by_repeat)
        return micro_scores

    def _init_raw_preds_df(self, subjects):

        self.raw_preds_df = pd.DataFrame(index=subjects)

    def _get_objs_from_pipeline(self, names_objs):

        pipeline = self._get_base_fitted_pipeline()

        names = [n[0] for n in names_objs]
        objs = [pipeline[n] for n in names]

        return objs

    def _proc_X_test(self, test_data):

        scalers = self._get_objs_from_pipeline(self.col_scalers)
        imputers = self._get_objs_from_pipeline(self.col_imputers)
        drop_strat = self._get_objs_from_pipeline(self.drop_strat)
        feat_selectors = self._get_objs_from_pipeline(self.feat_selectors)

        # Grab the test data, X as df + copy
        X_test, y_test = self._get_X_y(test_data, X_as_df=True, copy=True)

        feat_names = list(X_test)

        # Apply pipeline operations in place
        for scaler in scalers:
            X_test[feat_names] = scaler.transform(f_array(X_test))
        for imputer in imputers:
            X_test[feat_names] = imputer.transform(f_array(X_test))

        # Make sure to keep track of col changes w/ drop + feat_selector
        for drop in drop_strat:

            valid_inds = np.array(drop.transformers[0][2])
            feat_names = np.array(feat_names)[valid_inds]
            X_test = X_test[feat_names]

        for feat_selector in feat_selectors:

            feat_mask = feat_selector.get_support()
            feat_names = np.array(feat_names)[feat_mask]

            X_test[feat_names] = feat_selector.transform(X_test)
            X_test = X_test[feat_names]

        return X_test, y_test

    def _proc_X_train(self, train_data):

        scalers = self._get_objs_from_pipeline(self.col_scalers)
        imputers = self._get_objs_from_pipeline(self.col_imputers)
        samplers = self._get_objs_from_pipeline(self.samplers)
        drop_strat = self._get_objs_from_pipeline(self.drop_strat)
        feat_selectors = self._get_objs_from_pipeline(self.feat_selectors)

        X_train, y_train = self._get_X_y(train_data)

        for scaler in scalers:
            X_train = scaler.transform(X_train)
        for imputer in imputers:
            X_train = imputer.transform(np.array(X_train))
        for sampler in samplers:
            X_train, y_train = sampler.fit_resample(X_train, y_train)
        for drop in drop_strat:
            X_train = drop.transform(X_train)
        for feat_selector in feat_selectors:
            X_train = feat_selector.transform(X_train)

        return X_train


class Regression_Model_Pipeline(Model_Pipeline):
    '''Child class of Model for regression problem types.'''

    def _set_default_params(self):
        '''Set default params'''

        super()._set_default_params()
        self.problem_type = 'regression'


class Binary_Model_Pipeline(Model_Pipeline):
    '''Child class of Model for binary problem types.'''

    def _set_default_params(self):
        '''Set default params'''

        super()._set_default_params()
        self.problem_type = 'binary'


class Categorical_Model_Pipeline(Model_Pipeline):
    '''Child class of Model for categorical problem types.'''

    def _set_default_params(self):
        '''Set default params'''

        super()._set_default_params()
        self.problem_type = 'categorical'


class Multilabel_Model_Pipeline(Categorical_Model_Pipeline):
    '''Child class of Model for multilabel problem types.'''

    def _set_default_params(self):
        '''Set default params'''

        super()._set_default_params()
        self.problem_type = 'multilabel'

    def _process_samplers(self):

        if self.sampler_strs is not None:
            raise RuntimeError('Samplers with multilabel data is not',
                               'currently supported!')

        else:
            self.samplers = []
            self.sampler_params = {}
