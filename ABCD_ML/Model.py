import numpy as np

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from ABCD_ML.Models import MODELS
from ABCD_ML.Ensemble_Model import Ensemble_Model
from ABCD_ML.ML_Helpers import (conv_to_list, proc_input,
                                get_possible_init_params,
                                get_possible_fit_params,
                                get_obj_and_params)

from ABCD_ML.Models import AVALIABLE as AVALIABLE_MODELS
from ABCD_ML.Samplers import AVALIABLE as AVALIABLE_SAMPLERS
from ABCD_ML.Feature_Selectors import AVALIABLE as AVALIABLE_SELECTORS
from ABCD_ML.Scorers import AVALIABLE as AVALIABLE_SCORERS
from ABCD_ML.Ensembles import AVALIABLE as AVALIABLE_ENSEMBLES

from ABCD_ML.Samplers import get_sampler_and_params
from ABCD_ML.Feature_Selectors import get_feat_selector_and_params
from ABCD_ML.Scorers import get_scorer
from ABCD_ML.Scalers import get_data_scaler_and_params
from ABCD_ML.Ensembles import get_ensemble_and_params


class Model():
    '''Helper class for handling all of the different parameters involved in model
    training, scaling, handling different datatypes ect...
    '''

    def __init__(self, model_types, ML_params, model_type_param_ind, CV,
                 data_inds, cat_inds, targets_key, targets_encoder,
                 ensemble_type, ensemble_split, _print=print):
        ''' Init function for Model

        Parameters
        ----------
        model_types : str or list of str,
            Each string refers to a type of model to train.
            If a list of strings is passed then an ensemble model
            will be created over all individual models.
            For a full list of supported options call:
            ABCD_ML.Show_Model_Types(), with optional problem type parameter.

        ML_params : dict
            Dictionary of different ML params, the following must be included,
            (See the docstring for ABCD_ML.Set_Default_ML_Params for a more
            detailed description of all parameters contained within ML_params)

            - metrics : str or list,
                Metric / scorer str indicator, or list of. If list, the
                metric in the first index will be used during model selection.
            - data_scalers : str, list or None
                str indicator (or list of) for what type of data scaling
                to use if any. If list, data will scaled in list order.
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
            - int_cv : int
                The number of internal folds to use during modeling training
                / parameter selection
            - search_type : {'random', 'grid', None}
                The type of parameter search to conduct if any.
            - data_scaler_param_ind : int, str or list of
                The index or str name of the param index for `data_scaler`
            - feat_selector_param_ind : int, str or list of
                The index or str name of the param index for `feat_selector`
            - class_weight : str or None
                For categorical / binary problem_types, for setting different
                class weights.
            - n_jobs : int
                The number of jobs to use during model training.
            - n_iter : int
                The number of random searches to conduct in random search
                model types.
            - random_state : int or None
                The random state to use for CV splits / within modeling.
            - extra_params : dict
                The dictionary of any extra params to be passed to models or
                data scalers.

        model_type_param_ind : int, str or list of
            The index or str name of the param index for `model_type`

        CV : ABCD_ML CV
            The class defined ABCD_ML CV object for defining
            custom validation splits.

        data_inds : list
            List of column inds within data, as passed to Evaluate_Model or
            Test_Model, that correspond to the columns which should be scaled
            with the chosen data scaler(s) (if any).

        cat_inds : list
            List of column inds within data, as passed to Evaluate_Model or
            Test_Model, that correspond to the columns within covars that
            are categorical (binary or categorical).

        targets_key : str or list
            The str or list corresponding to the column keys for the targets
            within the data passed to Evaluate_Model or Test Model.

        target_encoder : sklearn encoder or list of
            The encoder or list of encoders, used in the case of targets
            needing to be transformed in some way.

        ensemble_type : str or list of str,
            Each string refers to a type of ensemble to train,
            or 'basic ensemble' for base behavior.

        ensemble_split : float, int or None
            If a an ensemble_type that requires fitting is passed,
            i.e., not "basic ensemble", then this param is
            the porportion of the train_data within each fold to
            use towards fitting the ensemble.

        _print : func, optional
            The print function to use, by default the python print,
            but designed to be passed ABCD_ML._print

        Notes
        ----------
        The Model class processes model_type, metric/scorer and data_scaler,
        as model_types, metrics/scorers and data_scalers (the plural...).
        This design decision was made to support both single str indicator
        input for any of these options, or a list of str indicators.
        '''

        # Set class parameters
        self.model_types = conv_to_list(model_types)
        self.CV = CV
        self.data_inds = data_inds
        self.cat_inds = cat_inds
        self.targets_key = targets_key
        self.targets_encoder = targets_encoder
        self._print = _print

        # Un-pack ML_params
        self.metrics = conv_to_list(ML_params['metric'])
        self.data_scalers = conv_to_list(ML_params['data_scaler'])
        self.samplers = conv_to_list(ML_params['sampler'])
        self.feat_selectors = conv_to_list(ML_params['feat_selector'])
        self.n_splits = ML_params['n_splits']
        self.n_repeats = ML_params['n_repeats']
        self.int_cv = ML_params['int_cv']
        self.class_weight = ML_params['class_weight']
        self.n_jobs = ML_params['n_jobs']
        self.n_iter = ML_params['n_iter']
        self.random_state = ML_params['random_state']
        self.extra_params = ML_params['extra_params']

        # Un-pack param search ML_params
        self.search_type = ML_params['search_type']

        self.model_type_param_inds =\
            conv_to_list(ML_params['model_type_param_ind'])
        self.data_scaler_param_inds =\
            conv_to_list(ML_params['data_scaler_param_ind'])
        self.sampler_param_inds =\
            conv_to_list(ML_params['sampler_param_ind'])
        self.feat_selector_param_inds =\
            conv_to_list(ML_params['feat_selector_param_ind'])

        self.ensemble_types = conv_to_list(ensemble_type)
        self.ensemble_split = ensemble_split

        # Default params just sets (sub)problem type for now
        self._set_default_params()

        # Process inputs
        self._process_model_types()
        self._process_feat_selectors()
        self._process_scorers()
        self._process_data_scalers()
        self._process_samplers()
        self._process_ensemble_types()

    def _set_default_params(self):
        self.user_passed_models = []
        self.upmi = 0

    def _process_model_types(self):
        '''Class function to convert input model types to final
        str indicator, based on problem type and common input correction.
        '''

        self._check_user_passed_models()

        self.model_types = self._proc_type_dep_str(self.model_types,
                                                   AVALIABLE_MODELS)

        if self.search_type is None:
            self.model_type_param_inds =\
                [0 for i in range(len(self.model_types))]

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
                                          self.feat_selector_param_inds)

            # If any base estimators, replace with a model
            self._replace_base_estimator()

        else:
            self.feat_selectors = []
            self.feat_selector_params = {}

    def _process_scorers(self):
        '''Process self.metrics and set self.scorers and self.scorer,
        as well as save the str processed final scorer_strs for verbose output.
        '''

        self.scorer_strs = self._proc_type_dep_str(self.metrics,
                                                   AVALIABLE_SCORERS)

        self.scorers = [get_scorer(scorer_str)
                        for scorer_str in self.scorer_strs]

        # Define the scorer to be used in model selection
        self.scorer = self.scorers[0]

    def _process_data_scalers(self):
        '''Processes self.data_scaler to be a list of
        (name, scaler) tuples, and then creates col_data_scalers
        from that.'''

        if self.data_scalers is not None:

            # Get converted scaler str and update extra params
            conv_data_scaler_strs = proc_input(self.data_scalers)
            self._update_extra_params(self.data_scalers, conv_data_scaler_strs)

            # Get the data_scalers tuple, and data_scaler_params grid / distr
            data_scalers, data_scaler_params =\
                self._get_objs_and_params(get_data_scaler_and_params,
                                          conv_data_scaler_strs,
                                          self.data_scaler_param_inds)

            # Create a list of tuples (just like self.data_scalers), but
            # with column versions of the scalers.
            self.col_data_scalers =\
                [('col_' + name, ColumnTransformer([(name, scaler,
                                                    self.data_inds)],
                 remainder='passthrough', sparse_threshold=0))
                 for name, scaler in data_scalers]

            # Create col_data_scaler_params from data_scaler_params
            self.col_data_scaler_params = {}

            for key in data_scaler_params:
                name = key.split('__')[0]
                new_name = 'col_' + name + '__' + key

                self.col_data_scaler_params[new_name] =\
                    data_scaler_params[key]

        else:
            self.col_data_scalers = []
            self.col_data_scaler_params = {}

    def _process_samplers(self):
        '''Class function to convert input sampler strs to
        a resampling object.
        '''

        if self.samplers is not None:

            sampler_strs =\
                self._proc_type_dep_str(self.samplers, AVALIABLE_SAMPLERS)

            self.samplers, self.sampler_params =\
                self._get_objs_and_params(get_sampler_and_params,
                                          sampler_strs,
                                          self.sampler_param_inds)

            # Replace random state
            self.samplers =\
                self._check_and_replace(self.samplers, 'random_state',
                                        self.random_state)

            # Replace categorical feats
            self.samplers =\
                self._check_and_replace(self.samplers, 'categorical_features',
                                        self.cat_inds)

            # N jobs if search type is None
            if self.search_type is None:
                self.samplers =\
                    self._check_and_replace(self.samplers,
                                            'n_jobs',
                                            self.n_jobs)

        else:
            self.samplers = []
            self.sampler_params = {}

    def _process_ensemble_types(self):
        '''Processes ensemble types to be a list of
        (ensemble, ensemble params) tuples.
        '''

        self.ensemble_strs = self._proc_type_dep_str(self.ensemble_types,
                                                     AVALIABLE_ENSEMBLES)

        # If basic ensemble is in any of the ensemble_strs,
        # ensure it is the only one.
        if np.array(['basic ensemble' in ensemble_str for
                     ensemble_str in self.ensemble_strs]).any():

            if len(self.ensemble_strs) > 1:

                self._print('Warning! "basic ensemble" ensemble type passed',
                            'within a list of ensemble types.')
                self._print('In order to use multiple ensembles',
                            'they cannot include "basic ensemble".')
                self._print('Setting to just "basic ensemble" ensemble type!')

                self.ensemble_strs = ['basic ensemble']

        # Grab the ensembles to use as a list of tuples with,
        # (ensemble, ensemble params).
        self.ensembles = [get_ensemble_and_params(ensemble_str,
                                                  self.extra_params)
                          for ensemble_str in self.ensemble_strs]

    def _check_user_passed_models(self):
        '''If not str passed as model type, assume it
        to be a user passed model.'''

        for m in range(len(self.model_types)):
            if not isinstance(self.model_types[m], str):

                self.user_passed_models.append(self.model_types[m])
                self.model_types[m] = 'user passed'

    def _proc_type_dep_str(self, in_strs, avaliable):
        '''Helper function to perform str correction on
        underlying proble type dependent input, e.g., for
        scorer or ensemble_types, and to update extra params
        and check to make sure input is valid ect...'''

        conv_strs = proc_input(in_strs)

        assert self._check_avaliable(conv_strs, avaliable),\
            "Error " + conv_strs + ' are not avaliable for this problem type'

        avaliable_by_type = self._get_avaliable_by_type(avaliable)
        final_strs = [avaliable_by_type[conv_str] for conv_str in conv_strs]

        self._update_extra_params(in_strs, final_strs)
        return final_strs

    def _check_avaliable(self, in_strs, avaliable):

        avaliable_by_type = self._get_avaliable_by_type(avaliable)

        check = np.array([m in avaliable_by_type for
                          m in in_strs]).all()

        return check

    def _get_avaliable_by_type(self, avaliable):
        return avaliable[self.problem_type]

    def _update_extra_params(self, orig_strs, conv_strs):
        '''Helper method to update class extra params in the case
        where model_types or data_scaler str indicators change,
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

    def _get_objs_and_params(self, get_func, names, param_inds):
        '''Helper function to grab data_scaler / feat_selectors and
        their relevant parameter grids'''

        # Grab necc. info w/ given get_func
        objs_and_params = [(name, get_func(name, self.extra_params, ind,
                            self.search_type))
                           for name, ind in zip(names, param_inds)]

        # Construct the obj as list of (name, obj) tuples
        objs = [(c[0], c[1][0]) for c in objs_and_params]

        # Grab the params, and merge them into one dict of all params
        params = {k: v for params in objs_and_params
                  for k, v in params[1][1].items()}

        return objs, params

    def _replace_base_estimator(self):
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

    def _check_and_replace(self, objs, param_name, replace_value):

        for i in range(len(objs)):

            try:
                getattr(objs[i][1], param_name)
                setattr(objs[i][1], param_name, replace_value)

            except AttributeError:
                pass

        return objs

    def Evaluate_Model(self, data, train_subjects):
        '''Method to perform a full repeated k-fold evaluation
        on a provided model type and training subjects, according to
        class set parameters.

        Parameters
        ----------
        data : pandas DataFrame
            ABCD_ML formatted, with both training and testing data.

        train_subjects : array-like
            An array or pandas Index of the train subjects should be passed.

        Returns
        ----------
        array-like of array-like
            numpy array of numpy arrays,
            where each internal array contains the raw scores as computed for
            all passed in metrics, computed for each fold within
            each repeat.
            e.g., array will have a length of `n_repeats` * `n_splits`,
            and each internal array will have the same length as the number of
            metrics.
        '''

        # Setup the desired splits, using the passed in train subjects
        subject_splits = self.CV.repeated_k_fold(train_subjects,
                                                 self.n_repeats, self.n_splits,
                                                 self.random_state,
                                                 return_index=False)

        all_scores = []

        # For each split with the repeated K-fold
        for train_subjects, test_subjects in subject_splits:

            scores = self.Test_Model(data, train_subjects, test_subjects)
            all_scores.append(scores)

        # Return all scores
        return np.array(all_scores)

    def Test_Model(self, data, train_subjects, test_subjects):
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

        # Assume the train_subjects and test_subjects passed here are final.
        train_data = data.loc[train_subjects]
        test_data = data.loc[test_subjects]

        # Train the model(s)
        self._train_models(train_data)

        # Get the score on the test set
        scores = self._get_scores(test_data)
        return scores

    def _train_models(self, train_data):
        '''Given training data, train the model(s), from the
        class model_types.

        Parameters
        ----------
        train_data : pandas DataFrame
            ABCD_ML formatted, training data.

        Returns
        ----------
        sklearn api compatible model object
            The trained single model, or some ensemble of models.
            Ensemble objects with predict funcs.
        '''

        # Split the train data further, if nec. for ensemble split
        train_data, ensemble_data = self._get_ensemble_split(train_data)

        # User passed model index should be reset to 0 here
        self.upmi = 0

        # Train all of the models
        models = []
        mt_and_mt_params = zip(self.model_types, self.model_type_param_inds)
        for model_type, model_type_param_ind in mt_and_mt_params:

            models.append(self._train_model(train_data, model_type,
                                            model_type_param_ind))

        # If special ensemble passed, fit each one seperate
        if ensemble_data:
            models = [self._fit_ensemble(models, ensemble_data, i)
                      for i in range(len(self.ensembles))]

        # If multiple base models, or ensembles of models,
        # either make a basic ensemble or if len == 1, use just that 1st model
        self.model = self._get_one_or_basic_ensemble(models)

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

            # Set ensemble data to X_y
            ensemble_data = train_data.loc[ensemble_subjects]

            # Set train_data to new smaller set
            train_data = train_data.loc[train_subjects]

        return train_data, ensemble_data

    def _get_X_y(self, data):
        '''Helper method to get X,y data from ABCD ML formatted df.

        Parameters
        ----------
        data : pandas DataFrame
            ABCD ML formatted.

        Returns
        ----------
        array-like
            X data for ML
        array-like
            y target for ML
        '''

        X = np.array(data.drop(self.targets_key, axis=1))
        y = np.array(data[self.targets_key])

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

    def _train_model(self, train_data, model_type, model_type_param_ind):
        '''Helper method to train a single model type given
        a str indicator and training data.

        Parameters
        ----------
        train_data : pandas DataFrame
            ABCD_ML formatted, training data.

        model_type : str
            The final processed str indicator for which model_type to load from
            MODELS constant.

        model_type_param_ind : int
            The index of the param grid / search space for the given model
            type.

        Returns
        ----------
        sklearn api compatible model object
            The trained model.
        '''

        # Create the internal base k-fold indices to pass to model
        base_int_cv = self.CV.k_fold(train_data.index, self.int_cv,
                                     random_state=self.random_state,
                                     return_index=True)

        # Create the model
        model = self._get_model(model_type, model_type_param_ind, base_int_cv)

        # Data, score split
        X, y = self._get_X_y(train_data)

        # Fit the model
        model.fit(X, y, **self.fit_params)

        return model

    def _get_model(self, model_type, model_type_param_ind, base_int_cv):

        # Grab the base model, model if changed, and model params grid/distr
        model, model_type, model_type_params =\
            self._get_base_model(model_type, model_type_param_ind,
                                 self.search_type)

        # Create the model pipeline object
        model = self._make_model_pipeline(model, model_type)

        if self.search_type is None:
            return model

        # Set the search params
        search_params = {}
        search_params['iid'] = False
        search_params['estimator'] = model
        search_params['pre_dispatch'] = 'n_jobs - 1'
        search_params['cv'] = base_int_cv
        search_params['scoring'] = self.scorer
        search_params['n_jobs'] = self.n_jobs

        if self.search_type == 'random':
            search_params['n_iter'] = self.n_iter

        # Merge the different params / grids of params
        # into one dict.
        all_params = {}
        all_params.update(self.col_data_scaler_params)
        all_params.update(self.sampler_params)
        all_params.update(self.feat_selector_params)
        all_params.update(model_type_params)

        # Create the search model
        if self.search_type == 'random':
            search_params['random_state'] = self.random_state
            search_params['param_distributions'] = all_params
            search_model = RandomizedSearchCV(**search_params)

        else:
            search_params['param_grid'] = all_params
            search_model = GridSearchCV(**search_params)

        return search_model

    def _get_base_model(self, model_type, model_type_param_ind, search_type):

        # Check for user passed model
        if model_type == 'user passed':

            user_model = self.user_passed_models[self.upmi]
            user_model_type = 'user passed' + str(self.upmi)
            self.upmi += 1

            return user_model, user_model_type, {}

        model, extra_model_params, model_type_params =\
            get_obj_and_params(model_type, MODELS, self.extra_params,
                               model_type_param_ind, search_type)

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

        # Set class param values from possible model fit params
        possible_fit_params = get_possible_fit_params(model)

        # This dict should reset here to empty every time
        self.fit_params = {}

        if 'categorical_feature' in possible_fit_params:
            self.fit_params[model_type + '__' + 'categorical_feature'] =\
                self.cat_inds

        # Init model, w/ any user passed params + class params
        model = model(**extra_model_params)

        return model, model_type, model_type_params

    def _make_model_pipeline(self, model, model_type):
        '''Provided a model & model type (model str indicator),
        return a sklearn pipeline with proceeding self.col_data_scalers,
        and then self.samplers, then self.feat_selectors
        (which should all just be empty list if None) and then the model,
        w/ model_type as its unique name.

        Parameters
        ----------
        model : sklearn api model
            The base model, w/ parameters already provided

        model_type : str
            The final str indicator for this model, also
            the name that the model will be saved under within
            the Pipeline object.

        Returns
        ----------
        sklearn Pipeline
            Pipeline object with all relevant column specific data
            scalers, and then the passed in model.
        '''

        steps = self.col_data_scalers + self.samplers + self.feat_selectors \
            + [(model_type, model)]

        model_pipeline = Pipeline(steps)

        return model_pipeline

    def _fit_ensemble(self, models, ensemble_data, i):

        # Grab ensemble + params from ind (i)
        ensemble_model = self.ensembles[i][0]
        ensemble_params = self.ensembles[i][1]

        # Init the ensemble model
        ensemble_model = ensemble_model(models, **ensemble_params)

        # Fit the ensemble model
        ensemble_X, ensemble_y = self._get_X_y(ensemble_data)
        ensemble_model.fit(ensemble_X, ensemble_y)

        return ensemble_model

    def _get_one_or_basic_ensemble(self, models):

        if len(models) == 1:
            model = models[0]
        else:
            model = Ensemble_Model(models)

        return model

    def _get_scores(self, test_data):
        '''Helper method to get the scores of
        the trained model saved in the class on input test data.
        For all metrics/scorers.

        Parameters
        ----------
        test_data : pandas DataFrame
            ABCD ML formatted test data.

        Returns
        ----------
        float
            The score of the trained model on the given test data.
        '''

        # Data, score split
        X_test, y_test = self._get_X_y(test_data)

        # Get the scores
        scores = [scorer(self.model, X_test, y_test)
                  for scorer in self.scorers]

        return np.array(scores)


class Regression_Model(Model):
    '''Child class of Model for regression problem types.'''

    def _set_default_params(self):
        '''Set default params'''

        super()._set_default_params()
        self.problem_type = 'regression'


class Binary_Model(Model):
    '''Child class of Model for binary problem types.'''

    def _set_default_params(self):
        '''Set default params'''

        super()._set_default_params()
        self.problem_type = 'binary'


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
            self._print('Checking compatability with multiclass!')

            self.sub_problem_type = 'multiclass'
            check = super()._check_avaliable(in_strs, avaliable)

        return check

    def _get_avaliable_by_type(self, avaliable):
        return avaliable[self.problem_type][self.sub_problem_type]

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

        return y
