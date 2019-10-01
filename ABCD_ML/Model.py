import numpy as np
import time

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

import shap
import pandas as pd
from collections import Counter

from ABCD_ML.Models import MODELS
from ABCD_ML.ML_Helpers import (conv_to_list, proc_input,
                                get_possible_init_params,
                                get_obj_and_params,
                                user_passed_param_check,
                                f_array, replace_with_in_params)

from ABCD_ML.Models import AVALIABLE as AVALIABLE_MODELS
from ABCD_ML.Feature_Selectors import AVALIABLE as AVALIABLE_SELECTORS
from ABCD_ML.Metrics import AVALIABLE as AVALIABLE_METRICS
from ABCD_ML.Ensembles import AVALIABLE as AVALIABLE_ENSEMBLES

from ABCD_ML.Samplers import get_sampler_and_params
from ABCD_ML.Feature_Selectors import get_feat_selector_and_params
from ABCD_ML.Metrics import get_metric
from ABCD_ML.Scalers import get_scaler_and_params
from ABCD_ML.Imputers import get_imputer_and_params
from ABCD_ML.Ensembles import (get_ensemble_and_params, Basic_Ensemble,
                               DES_Ensemble)


class Model():
    '''Helper class for handling all of the different parameters involved in model
    training, scaling, handling different datatypes ect...
    '''

    def __init__(self, ML_params, CV, search_split_vals,
                 all_data_keys, targets_key, targets_encoder, covar_scopes,
                 cat_encoders, progress_bar, param_search_verbose,
                 _print=print):
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
                The number of folds to use during the Evaluate_Model repeated
                k-fold.
            - n_repeats : int
                The number of repeats to do during the Evaluate_Model repeated
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
            - search_type : {'random', 'grid', None}
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
            - class_weight : str or None
                For categorical / binary problem_types, for setting different
                class weights.
            - n_jobs : int
                The number of jobs to use during model training.
            - search_n_iter : int
                The number of random searches to conduct in random search
                model types.
            - feats_to_use : {'all', 'data', 'covars'}
                The subset of data to use, either all avaliable, just
                the data, or just the covars.
            - compute_train_score : bool
                If True, computes the training score of the model pipeline,
                if false only computes the testing score.
            - random_state : int or None
                The random state to use for CV splits / within modeling.
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

        calc_base_feature_importances : bool
            If set to True, will store the base feature importances
            when running Evaluate or Test. Note, base feature importances
            are only avaliable for tree-based or linear models, specifically
            those with either coefs_ or feature_importance_ attributbes.

        calc_shap_feature_importances : bool
            If set to True, will calculate SHapley Additive exPlanations
            for the model when running Evaluate or Test. Note: any case
            where the model isnt tree or linear based, e.g. an ensemble of
            different methods, or non-linear svm, these values are estimated
            by a kernel function which is very compute intensive.

        progress_bar : None or tqdm obj
            Either None, to not use progress bar or a tqdm object, to
            display progress.

        param_search_verbose : int
            Argument passed to sklearn param search objects, random or grid.

        _print : func, optional
            The print function to use, by default the python print,
            but designed to be passed ABCD_ML._ML_print

        Notes
        ----------
        The Model class processes model_type, metric and scaler,
        as model_types, metrics and scalers (the plural...).
        This design decision was made to support both single str indicator
        input for any of these options, or a list of str indicators.
        '''

        # Set class parameters
        self.CV = CV
        self.search_split_vals = search_split_vals
        self.data_keys = all_data_keys['data_keys']
        self.covars_keys = all_data_keys['covars_keys']
        self.strat_keys = all_data_keys['strat_keys']
        self.cat_keys = all_data_keys['cat_keys']
        self.targets_key = targets_key
        self.targets_encoder = targets_encoder
        self.covar_scopes = covar_scopes
        self.cat_encoders = cat_encoders
        self.progress_bar = progress_bar
        self.param_search_verbose = param_search_verbose
        self._print = _print

        # Un-pack ML_params
        self.model_types = conv_to_list(ML_params['model_type'])
        self.metrics = conv_to_list(ML_params['metric'])
        self.imputers = conv_to_list(ML_params['imputer'])
        self.imputer_scopes = conv_to_list(ML_params['imputer_scope'])
        self.scalers = conv_to_list(ML_params['scaler'])
        self.scaler_scopes = conv_to_list(ML_params['scaler_scope'])
        self.samplers = conv_to_list(ML_params['sampler'])
        self.sample_on = conv_to_list(ML_params['sample_on'])
        self.feat_selectors = conv_to_list(ML_params['feat_selector'])
        self.splits = ML_params['splits']
        self.n_repeats = ML_params['n_repeats']
        self.search_splits = ML_params['search_splits']
        self.ensemble_types = conv_to_list(ML_params['ensemble_type'])
        self.ensemble_split = ML_params['ensemble_split']
        self.class_weight = ML_params['class_weight']
        self.n_jobs = ML_params['n_jobs']
        self.search_n_iter = ML_params['search_n_iter']
        self.compute_train_score = ML_params['compute_train_score']
        self.random_state = ML_params['random_state']
        self.calc_base_feature_importances =\
            ML_params['calc_base_feature_importances']
        self.calc_shap_feature_importances =\
            ML_params['calc_shap_feature_importances']
        self.extra_params = ML_params['extra_params'].copy()

        # Un-pack param search ML_params
        self.search_type = ML_params['search_type']

        self.model_type_params =\
            conv_to_list(ML_params['model_type_params'])
        self.imputer_params =\
            conv_to_list(ML_params['imputer_params'])
        self.scaler_params =\
            conv_to_list(ML_params['scaler_params'])
        self.sampler_params =\
            conv_to_list(ML_params['sampler_params'])
        self.feat_selector_params =\
            conv_to_list(ML_params['feat_selector_params'])
        self.ensemble_type_params =\
            conv_to_list(ML_params['ensemble_type_params'])

        # Default params just sets (sub)problem type for now
        self._set_default_params()

        self.subjects_to_use = ML_params['subjects_to_use']

        # Set all_keys based on feats_to_use
        self._set_all_keys(ML_params['feats_to_use'])

        # Process inputs
        self._check_all_for_user_passed()
        self._process_model_types()
        self._process_feat_selectors()
        self._process_metrics()
        self._process_imputers()
        self._process_scalers()
        self._process_samplers()
        self._process_ensemble_types()

        # Make the model pipeline, save to self.model_pipeline
        self._make_drop_strat()
        self._set_model_pipeline()

    def _set_default_params(self):
        self.problem_type = ''
        self.n_splits = None
        self.user_passed_objs = {}
        self.shap_dfs = []

        # Flags for feat importance things
        self.ensemble_flag = False
        self.linear_flag = False
        self.tree_flag = False

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

        self.model_types, cnt =\
            self._check_for_user_passed(self.model_types, cnt)

        self.feat_selectors, cnt =\
            self._check_for_user_passed(self.feat_selectors, cnt)

        self.imputers, cnt =\
            self._check_for_user_passed(self.imputers, cnt)

        self.scalers, cnt =\
            self._check_for_user_passed(self.scalers, cnt)

        self.samplers, cnt =\
            self._check_for_user_passed(self.samplers, cnt)

        self.ensemble_types, cnt =\
            self._check_for_user_passed(self.ensemble_types, cnt)

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

    def _process_model_types(self):
        '''Class function to convert input model types to final
        str indicator, based on problem type and common input correction.
        '''

        self.model_types = self._proc_type_dep_str(self.model_types,
                                                   AVALIABLE_MODELS)

        self.models, self.model_params =\
            self._get_objs_and_params(self._get_base_model,
                                      self.model_types, self.model_type_params)

    def _process_feat_selectors(self):
        '''Class function to convert input feat selectors to a final
        set of feat_selector objects along with parameters,
        based on problem type and common input correction.
        '''

        if self.feat_selectors is not None:

            feat_selector_strs =\
                self._proc_type_dep_str(self.feat_selectors,
                                        AVALIABLE_SELECTORS)

            # Get the feat_selectors tuple, and merged params grid / distr dict
            self.feat_selectors, self.feat_selector_params =\
                self._get_objs_and_params(get_feat_selector_and_params,
                                          feat_selector_strs,
                                          self.feat_selector_params)

            # If any base estimators, replace with a model
            self._replace_base_rfe_estimator()

        else:
            self.feat_selectors = []
            self.feat_selector_params = {}

    def _process_metrics(self):
        '''Process self.metrics and set to proced version.'''

        self.metric_strs = self._proc_type_dep_str(self.metrics,
                                                   AVALIABLE_METRICS)

        self.metrics = [get_metric(metric_str)
                        for metric_str in self.metric_strs]

        # Define the metric to be used in model selection
        self.metric = self.metrics[0]

    def _process_imputers(self):

        if self.imputers is not None:

            conv_imputer_strs = proc_input(self.imputers)
            self._update_extra_params(self.imputers, conv_imputer_strs)

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

            inds = []

            # If scope doesn't match any actual data, skip
            if len(cat_inds) == 0 and len(ordinal_inds) == 0:
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
            base_model = self._get_base_model(base_model_str, imputer_param,
                                              self.search_type)
            base_estimator = base_model[0]
            base_estimator_params = base_model[2]
        else:
            base_estimator = None
            base_estimator_params = {}

        return base_estimator, base_estimator_params

    def _process_scalers(self):
        '''Processes self.scaler to be a list of
        (name, scaler) tuples, and then creates col_scalers
        from that.'''

        if self.scalers is not None:

            # Get converted scaler str and update extra params
            conv_scaler_strs = proc_input(self.scalers)
            self._update_extra_params(self.scalers, conv_scaler_strs)

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
                ('col_' + name, ColumnTransformer([(name, obj, inds)],
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

        if self.samplers is not None:

            # Initial proc
            conv_sampler_strs = proc_input(self.samplers)
            self._update_extra_params(self.samplers, conv_sampler_strs)

            # Performing proc on input lengths
            self.sampler_params =\
                self._param_len_check(conv_sampler_strs, self.sampler_params)

            self.sampler_params =\
                self._check_params_by_search(self.sampler_params)

            self.sample_on = self._scope_len_check(conv_sampler_strs,
                                                   self.sample_on,
                                                   'imputers')

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

        # Set targets_encoder if multilabel
        targets_encoder = None
        if self.targets_encoder is not None:

            # If multi-label, target_encoder saved in index 1
            if len(self.targets_encoder) > 1:
                targets_encoder = self.targets_encoder[1]

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
                                   targets_encoder=targets_encoder,
                                   recover_strat=recover_strat,
                                   covars_inds=covars_inds)

        return sampler_str, (sampler, sampler_params)

    def _process_ensemble_types(self):
        '''Processes ensemble types'''

        if self.ensemble_types is not None:

            self.ensemble_types = self._proc_type_dep_str(self.ensemble_types,
                                                          AVALIABLE_ENSEMBLES)

            # If basic ensemble is in any of the ensemble_strs,
            # ensure it is the only one.
            if np.array(['basic ensemble' in ensemble for
                        ensemble in self.ensemble_types]).any():

                if len(self.ensemble_types) > 1:

                    self._print('Warning! "basic ensemble" ensemble type',
                                'passed within a list of ensemble types.')
                    self._print('In order to use multiple ensembles',
                                'they cannot include "basic ensemble".')
                    self._print('Setting to just "basic ensemble" ensemble',
                                ' type!')

                    self.ensemble_types = ['basic ensemble']

            self.ensembles, self.ensemble_params =\
                self._get_objs_and_params(get_ensemble_and_params,
                                          self.ensemble_types,
                                          self.ensemble_type_params)

        else:
            self.ensembles = [('basic ensemble', None)]
            self.ensemble_params = {}

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

        return cat_keys, ordinal_inds

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
                    self._get_base_model(base_model_str, 0, None)[0]

            except AttributeError:
                pass

    def _check_and_replace_samplers(self, param_name, replace_value):

        for i in range(len(self.samplers)):

            try:
                getattr(self.samplers[i][1].sampler, param_name)
                setattr(self.samplers[i][1].sampler, param_name, replace_value)

            except AttributeError:
                pass

    def Evaluate_Model(self, data, train_subjects, splits_vals=None):
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

            # Run actual code for this evaluate fold
            start_time = time.time()
            train_scores, scores = self.Test_Model(data, train_subjects,
                                                   test_subjects, fold_ind)

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

        # Average feature importances across folds / repeats
        self._average_feature_importances()

        # Return all scores
        return (np.array(all_train_scores), np.array(all_scores),
                self.raw_preds_df)

    def Test_Model(self, data, train_subjects, test_subjects, fold_ind='test'):
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

        # Check for any NaN
        self._nan_check(data)

        # Init raw_preds_df
        if fold_ind == 'test':
            self._init_raw_preds_df(test_subjects)

        # Assume the train_subjects and test_subjects passed here are final.
        train_data = data.loc[train_subjects]
        test_data = data.loc[test_subjects]

        self._print('Train subjects:', train_data.shape[0], level='size')
        self._print('Val/Test subjects:', test_data.shape[0], level='size')

        # Wrap in search CV / set to self.Model, final pipeline
        self._set_model(train_data)

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
            return train_scores, scores, self.raw_preds_df

        return train_scores, scores

    def _get_eval_splits(self, train_subjects, splits_vals):

        if splits_vals is None:

            self.n_splits = self.splits

            subject_splits =\
                self.CV.repeated_k_fold(train_subjects, self.n_repeats,
                                        self.n_splits, self.random_state,
                                        return_index=False)

        else:

            self.n_splits =\
                len(np.unique(splits_vals.loc[train_subjects]))

            subject_splits =\
                self.CV.repeated_leave_one_group_out(train_subjects,
                                                     self.n_repeats,
                                                     splits_vals,
                                                     return_index=False)

        return subject_splits

    def _nan_check(self, data):
        '''If no nans, can set imputers to None, regardless
        of passed params.'''

        nans = pd.isnull(data).any().any()

        if nans is False:
            self.col_imputers = []
            self.col_imputer_params = {}

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

    def _basic_ensemble(self, models, name):

        if len(models) == 1:
            return models

        else:
            basic_ensemble = Basic_Ensemble(models)
            self._update_model_ensemble_params(name, ensemble=False)

            return [(name, basic_ensemble)]

    def _basic_ensemble_pipe(self, models, name):
        '''Models as [(name, obj), ect...]'''

        basic_ensemble = self._basic_ensemble(models, name)
        return self._make_model_pipeline(basic_ensemble)

    def _set_model_pipeline(self):

        # Check if basic ensemble
        if self.ensembles[0][1] is None:

            self.model_pipeline =\
                self._basic_ensemble_pipe(self.models, self.ensembles[0][0])

        # Otherwise special ensembles
        else:

            new_model_params = {}
            new_ensemble_params = {}
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

                else:

                    new_ensembles.append(
                        (ensemble_name,
                         ensemble_obj(base_estimator=models[0][1],
                                      **ensemble_extra_params)))

                    # Have to change model name to base_estimator
                    self.model_params =\
                        replace_with_in_params(self.model_params, models[0][0],
                                               'base_estimator')

                self._update_model_ensemble_params(ensemble_name)

            self.model_pipeline =\
                self._basic_ensemble_pipe(new_ensembles,
                                          'ensemble of ensembles')
            self._update_model_ensemble_params(ensemble_name, model=False)

    def _make_model_pipeline(self, models):
        '''Make the model pipeline object

        Returns
        ----------
        sklearn Pipeline
            Pipeline object with all relevant column specific data
            scalers, and then the passed in model.
        '''

        steps = self.col_scalers + self.col_imputers + self.samplers \
            + self.drop_strat + self.feat_selectors + models

        model_pipeline = Pipeline(steps)

        return model_pipeline

    def _set_model(self, train_data):

        # Get search CV
        search_cv = self._get_search_cv(train_data.index)

        # Wrap in search object
        self.Model = self._get_search_model(self.model_pipeline, search_cv)

    def _proc_feat_importance(self, train_data, test_data, fold_ind):

        if self.calc_base_feature_importances:
            if fold_ind == 0 or fold_ind == 'test':
                self._init_feature_importances(train_data)

        if self.calc_shap_feature_importances:

            # If fold ind test, init with test_data
            if fold_ind == 'test':
                self._init_shap_df(test_data)

            # Otherwise, using Evaluate, init with all data only every n_splits
            elif fold_ind % self.n_splits == 0:
                self._init_shap_df(pd.concat([train_data, test_data]))

        # Compute feature importance
        self._get_feature_importance(train_data, test_data)

        # Need a check to add self.shap_df to self.shap_dfs
        # Only during Evalaute, not for test
        if isinstance(fold_ind, int) and self.calc_shap_feature_importances:
            if fold_ind % self.n_splits == self.n_splits-1:
                self.shap_dfs.append(self.shap_df)

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

        # Convert/decode y/score if needed
        y = self._conv_targets(y)

        return X, y

    def _conv_targets(self, y):
        '''Returns y, overriden by Categorical_Model

        Parameters
        ----------
        y : array-like
            ML target

        Returns
        ----------
        array-like
            input y as is
        '''
        return y

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

    def _get_search_model(self, model, search_cv):
        '''Passed model as a created pipeline object + search_cv info,
        if search_type isn't None, creates the search wrapper.'''

        if self.search_type is None:
            return model

        # Set the search params
        search_params = {}
        search_params['iid'] = False
        search_params['verbose'] = self.param_search_verbose
        search_params['estimator'] = model
        search_params['pre_dispatch'] = 'n_jobs - 1'
        search_params['cv'] = search_cv
        search_params['scoring'] = self.metric
        search_params['n_jobs'] = self.n_jobs

        if self.search_type == 'random':
            search_params['n_iter'] = self.search_n_iter

        # Merge the different params / grids of params
        # into one dict.
        all_params = {}
        all_params.update(self.col_scaler_params)
        all_params.update(self.col_imputer_params)
        all_params.update(self.sampler_params)
        all_params.update(self.feat_selector_params)
        all_params.update(self.model_params)
        all_params.update(self.ensemble_params)

        # Create the search model
        if self.search_type == 'random':
            search_params['random_state'] = self.random_state
            search_params['param_distributions'] = all_params
            search_model = RandomizedSearchCV(**search_params)

        else:
            search_params['param_grid'] = all_params
            search_model = GridSearchCV(**search_params)

        return search_model

    def _get_base_model(self, model_type, extra_params, model_type_params,
                        search_type):

        model, extra_model_params, model_type_params =\
            get_obj_and_params(model_type, MODELS, self.extra_params,
                               model_type_params, self.search_type)

        # Set class param values from possible model init params
        possible_params = get_possible_init_params(model)

        if 'class_weight' in possible_params:
            extra_model_params['class_weight'] = self.class_weight

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

    def _fit_ensemble(self, models, ensemble_data, i):

        # Grab data
        ensemble_X, ensemble_y = self._get_X_y(ensemble_data)

        # Grab ensemble + params from ind (i)
        ensemble_model = self.ensembles[i][0]
        ensemble_params = self.ensembles[i][1]

        # Init the ensemble model & fit
        ensemble_model = ensemble_model(models, **ensemble_params)
        ensemble_model.fit(ensemble_X, ensemble_y)

        return ensemble_model

    def _get_one_or_basic_ensemble(self, models):

        if len(models) == 1:
            model = models[0]
        else:
            model = Ensemble_Model(models)

        return model

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

        fold = str((fold_ind % self.n_splits) + 1)
        repeat = str((fold_ind // self.n_splits) + 1)

        try:
            raw_prob_preds = self.Model.predict_proba(X_test)
            pred_col = eval_type + repeat + '_prob'

            if len(np.shape(raw_prob_preds)) == 3:

                for i in range(len(raw_prob_preds)):
                    p_col = pred_col + '_class_' + str(i)
                    class_preds = [val[1] for val in raw_prob_preds[i]]
                    self.raw_preds_df.loc[subjects, p_col] = class_preds

            elif len(np.shape(raw_prob_preds)) == 2:

                for i in range(np.shape(raw_prob_preds)[1]):
                    p_col = pred_col + '_class_' + str(i)
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
                p_col = pred_col + '_class_' + str(i)
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

    def _init_feature_importances(self, data):

        X, y = self._get_X_y(data, X_as_df=True)
        feat_names = list(X)

        self.feature_importances = pd.DataFrame(columns=feat_names)

    def _init_raw_preds_df(self, subjects):

        self.raw_preds_df = pd.DataFrame(index=subjects)

    def _init_shap_df(self, data):

        self.shap_df, y = self._get_X_y(data, X_as_df=True, copy=True)

        for col in self.shap_df.columns:
            self.shap_df[col].values[:] = 0

    def _get_feature_importance(self, train_data, test_data):

        base_model = self._check_feat_importance_type()

        if self.calc_base_feature_importances:
            self._print('Calculate base feature importances', level='name')
            self._get_base_feature_importance(base_model, test_data)

        if self.calc_shap_feature_importances:
            self._print('Calculate shap feature importances', level='name')
            self._get_shap_feature_importance(base_model, train_data,
                                              test_data)

    def _get_base_feature_importance(self, base_model, test_data):

        if self.tree_flag or self.linear_flag:

            X_test = self._proc_X_test(test_data)

            if self.linear_flag:
                feat_importance = np.squeeze(base_model.coef_)
            elif self.tree_flag:
                feat_importance = base_model.feature_importances_

            # For both
            self._add_to_feature_importances(list(X_test), feat_importance)

    def _get_shap_feature_importance(self, base_model, train_data, test_data):

        if self.tree_flag or self.linear_flag:

            X_test = self._proc_X_test(test_data)

            if self.linear_flag:
                X_train = self._proc_X_train(train_data)
                explainer =\
                    shap.LinearExplainer(base_model, X_train,
                                         feature_dependence="independent")

            elif self.tree_flag:
                explainer = shap.TreeExplainer(base_model)

            shap_values = self._get_shap_values(explainer, X_test)

        else:
            shap_values, X_test =\
                self._get_kernel_shap_values(train_data, test_data)

        # Set to df
        self._add_new_shap_values(X_test, shap_values)

    def _add_new_shap_values(self, X_test, shap_values):

        shap_df = X_test.copy()
        shap_df[list(X_test)] = shap_values

        self.shap_df.update(shap_df)

    def _check_feat_importance_type(self):

        base_model = None

        # Try grabbing a base model, if it doesnt exist,
        # it means there is some type of ensemble being used
        try:
            base_model = self.Model[self.model_types[0]]
        except:
            self.ensemble_flag = True

        if self.ensemble_flag:

            try:
                base_model = self.Model.best_estimator_[self.model_types[0]]
                self.ensemble_flag = False
            except:
                self.ensemble_flag = True

        if not self.ensemble_flag:

            try:
                base_model.coef_
                self.linear_flag = True
            except AttributeError:
                pass

            try:
                base_model.feature_importances_
                self.tree_flag = True
            except AttributeError:
                pass

        return base_model

    def _get_objs_from_pipeline(self, names_objs, model, as_steps=False):
        '''Assumes that the self.Model is a pipeline object only,
        no ensemble, or within a search object
        '''

        names = [n[0] for n in names_objs]

        try:
            objs = [model[n] for n in names]
        except TypeError:
            objs = [model.best_estimator_[n] for n in names]

        if as_steps:
            objs = [(names[i], objs[i]) for i in range(len(names))]

        return objs

    def _proc_X_test(self, test_data):

        scalers = self._get_objs_from_pipeline(self.col_scalers, self.Model)
        imputers = self._get_objs_from_pipeline(self.col_imputers, self.Model)
        drop_strat = self._get_objs_from_pipeline(self.drop_strat, self.Model)
        feat_selectors = self._get_objs_from_pipeline(self.feat_selectors,
                                                      self.Model)

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
            X_test[feat_names] = drop.transform(X_test)

        for feat_selector in feat_selectors:

            feat_mask = feat_selector.get_support()
            feat_names = np.array(feat_names)[feat_mask]

            X_test[feat_names] = feat_selector.transform(X_test)
            X_test = X_test[feat_names]

        return X_test

    def _proc_X_train(self, train_data):

        scalers = self._get_objs_from_pipeline(self.col_scalers, self.Model)
        imputers = self._get_objs_from_pipeline(self.col_imputers, self.Model)
        samplers = self._get_objs_from_pipeline(self.samplers, self.Model)
        drop_strat = self._get_objs_from_pipeline(self.drop_strat, self.Model)
        feat_selectors = self._get_objs_from_pipeline(self.feat_selectors,
                                                      self.Model)

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

    def _add_to_feature_importances(self, feat_names, feat_importance):

        feat_importance_dict = {name: importance for name, importance in
                                zip(feat_names, feat_importance)}

        self.feature_importances =\
            self.feature_importances.append(feat_importance_dict,
                                            ignore_index=True)
        self.feature_importances = self.feature_importances.fillna(0)

    def _get_shap_values(self, explainer, X_test):

        shap_values = explainer.shap_values(X_test)
        return self._proc_shap_values(shap_values)

    def _proc_shap_values(self, shap_values):
        return shap_values

    def _get_kernel_shap_values(self, train_data, test_data):

        X_train, y_train = self._get_X_y(train_data)
        X_test, y_test = self._get_X_y(test_data, X_as_df=True)

        # Generate summary of X_train, w/ k = 10 default
        X_train_summary = shap.kmeans(X_train, 10)

        explainer = self._get_kernel_explainer(self.Model, X_train_summary)

        shap_values = explainer.shap_values(np.array(X_test), l1_reg='aic',
                                            n_samples='auto')

        return self._proc_shap_values(shap_values), X_test

    def _get_kernel_explainer(self, model, X_train_summary):
        '''Base behavior for binary / multi-class'''

        explainer = shap.KernelExplainer(model.predict_proba,
                                         X_train_summary, link='logit')

        return explainer

    def _average_feature_importances(self):

        if self.calc_shap_feature_importances:

            # Set to copy of 1st one to start, as base to fill
            self.shap_df = self.shap_dfs[0].copy()

            # Only need to average if more than one repeat
            if len(self.shap_dfs) > 1:

                shap_df_arrays = [np.array(df) for df in self.shap_dfs]
                mean_shap_array = np.mean(shap_df_arrays, axis=0)

                self.shap_df[list(self.shap_df)] = mean_shap_array

            # Reset self.shap_dfs to clear memory
            self.shap_dfs = []


class Regression_Model(Model):
    '''Child class of Model for regression problem types.'''

    def _set_default_params(self):
        '''Set default params'''

        super()._set_default_params()
        self.problem_type = 'regression'

    def _get_kernel_explainer(self, model, X_train_summary):

        explainer = shap.KernelExplainer(model.predict, X_train_summary,
                                         link='identity')

        return explainer


class Binary_Model(Model):
    '''Child class of Model for binary problem types.'''

    def _set_default_params(self):
        '''Set default params'''

        super()._set_default_params()
        self.problem_type = 'binary'

    def _proc_shap_values(self, shap_values):
        return shap_values[1]


class Categorical_Model(Model):
    '''Child class of Model for categorical problem types.'''

    def _set_default_params(self):
        '''Set default params'''

        super()._set_default_params()
        self.problem_type = 'categorical'
        self.sub_problem_type = 'multilabel'

    def _check_avaliable(self, in_strs, avaliable):

        check = super()._check_avaliable(in_strs, avaliable)

        if not check and self.sub_problem_type == 'multilabel':
            self._print('Not all input supports multilabel,')
            self._print('Switching to multiclass for compatibility!')

            self.sub_problem_type = 'multiclass'
            check = super()._check_avaliable(in_strs, avaliable)

        return check

    def _get_avaliable_by_type(self, avaliable, in_strs, problem_type='class'):

        if problem_type == 'class':
            avaliable_by_type =\
                avaliable[self.problem_type][self.sub_problem_type]
        else:
            if problem_type in avaliable[self.problem_type]:
                avaliable_by_type =\
                    avaliable[self.problem_type][problem_type]
            elif problem_type in avaliable:
                avaliable_by_type = avaliable[problem_type]
            else:
                avaliable_by_type = {}

        for s in in_strs:
            if 'user passed' in s:
                avaliable_by_type[s] = s

        return avaliable_by_type

    def _conv_targets(self, y):
        '''Overrides parent method, if the sub problem type
        is multi-class, then the target will need to be inverse transform.

        Parameters
        ----------
        y : array-like
            ML target

        Returns
        ----------
        array-like
            inverse encoded y if multiclass, otherwise input y
        '''

        # If multiclass, convert to correct score format
        if self.sub_problem_type == 'multiclass':
            y = self.targets_encoder[1].inverse_transform(y).squeeze()

        else:
            y = y.astype(int)

        return y

    def _init_shap_df(self, data):

        self.shap_df = []

        for i in range(len(self.targets_key)):

            shap_df, y = self._get_X_y(data, X_as_df=True, copy=True)

            for col in shap_df.columns:
                shap_df[col].values[:] = 0

            self.shap_df.append(shap_df)

    def _add_new_shap_values(self, X_test, shap_values):

        for i in range(len(shap_values)):

            shap_df = X_test.copy()
            shap_df[list(X_test)] = shap_values[i]

            self.shap_df[i].update(shap_df)

    def _average_feature_importances(self):

        if self.calc_shap_feature_importances:

            # Set to copy of 1st one to start, as base to fill
            self.shap_df = self.shap_dfs[0].copy()

            # Only need to average if more than one repeat
            if len(self.shap_dfs) > 1:

                # For each class within the target variable
                for i in range(len(self.shap_df)):

                    shap_df_arrays = [np.array(df[i]) for df in self.shap_dfs]
                    mean_shap_array = np.mean(shap_df_arrays, axis=0)

                    self.shap_df[i][list(self.shap_df[i])] = mean_shap_array

            # Reset self.shap_dfs to clear memory
            self.shap_dfs = []
