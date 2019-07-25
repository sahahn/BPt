import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from ABCD_ML.Ensemble_Model import Ensemble_Model
from ABCD_ML.ML_Helpers import (conv_to_list, proc_input,
                                get_model_possible_params)

from ABCD_ML.Models import MODELS

from ABCD_ML.Models import AVALIABLE as AVALIABLE_MODELS
from ABCD_ML.Feature_Selectors import AVALIABLE as AVALIABLE_SELECTORS
from ABCD_ML.Scorers import AVALIABLE as AVALIABLE_SCORERS

from ABCD_ML.Feature_Selectors import get_feat_selector
from ABCD_ML.Scorers import get_scorer
from ABCD_ML.Scalers import get_data_scaler


class Model():
    '''Helper class for handling all of the different parameters involved in model
    training, scaling, handling different datatypes ect...
    '''

    def __init__(self, model_types, ML_params, CV, data_keys, targets_key,
                 targets_encoder, verbose=True):
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

        CV : ABCD_ML CV
            The class defined ABCD_ML CV object for defining
            custom validation splits.

        data_keys : list
            List of column keys within data passed to Evaluate_Model or
            Test_Model, that correspond to the columns which should be scaled
            with the chosen data scaler (if any).

        targets_key : str or list
            The str or list corresponding to the column keys for the targets
            within the data passed to Evaluate_Model or Test Model.

        target_encoder : sklearn encoder or list of
            The encoder or list of encoders, used in the case of targets
            needing to be transformed in some way.

        verbose : bool
            If set to true will display extra diagnostic / print output.

        Notes
        ----------
        The Model class processes model_type, metric/scorer and data_scaler,
        as model_types, metrics/scorers and data_scalers (the plural...).
        This design decision was made to support both single str indicator
        input for any of these options, or a list of str indicators.
        '''

        # Set class parameters
        self.model_types = model_types
        self.CV = CV
        self.data_keys = data_keys
        self.targets_key = targets_key
        self.targets_encoder = targets_encoder
        self.verbose = verbose

        # Un-pack ML_params
        self.metrics = ML_params['metric']
        self.data_scalers = ML_params['data_scaler']
        self.feat_selectors = ML_params['feat_selector']
        self.n_splits = ML_params['n_splits']
        self.n_repeats = ML_params['n_repeats']
        self.int_cv = ML_params['int_cv']
        self.class_weight = ML_params['class_weight']
        self.n_jobs = ML_params['n_jobs']
        self.n_iter = ML_params['n_iter']
        self.random_state = ML_params['random_state']
        self.extra_params = ML_params['extra_params']

        # Default params just sets (sub)problem type for now
        self._set_default_params()
        self.col_data_scalers = []
        self.user_passed_models, self.upmi = [], 0

        # Process model_types, scorers and scalers from str indicator input
        self._process_model_types()
        self._process_feat_selectors()
        self._process_scorers()
        self._process_data_scalers()

    def _print(self, *args):
        '''Overriding the print function to allow for
        customizable verbosity within class methods

        Parameters
        ----------
        args
            Anything that would be passed to default python print
        '''

        if self.verbose:
            print(*args)

    def _set_default_params(self):
        '''Overriden by child classes'''
        pass

    def _process_model_types(self):
        '''Class function to convert input model types to final
        str indicator, based on problem type and common input correction.
        Also handles updating extra params, if applicable.'''

        self.model_types = conv_to_list(self.model_types)
        self._check_user_passed_models()

        self.model_types = self._proc_type_dep_str(self.model_types,
                                                   AVALIABLE_MODELS)

    def _check_user_passed_models(self):
        '''If not str passed as model type, assume it
        to be a user passed model.'''

        for m in range(len(self.model_types)):
            if not isinstance(self.model_types[m], str):

                self.user_passed_models.append(self.model_types[m])
                self.model_types[m] = 'user passed'

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

    def _process_feat_selectors(self):

        if self.feat_selectors is not None:

            feat_selector_strs =\
                self._proc_type_dep_str(self.feat_selectors,
                                        AVALIABLE_SELECTORS)

            # Set feat selectors to list of (name, scaler) tuples.
            self.feat_selectors = [(fs_str,
                                    get_feat_selector(fs_str,
                                                      self.extra_params))
                                   for fs_str in feat_selector_strs]

        else:
            self.feat_selectors = []

    def _proc_type_dep_str(self, in_str, avaliable):

        in_strs = conv_to_list(in_str)
        conv_strs = proc_input(in_strs)

        assert self._check_avaliable(conv_strs, avaliable),\
            "Error " + conv_strs + ' are not avaliable for this problem type'

        avaliable_by_type = self._get_avaliable_by_type(avaliable)
        final_strs = [avaliable_by_type[conv_str] for conv_str in conv_strs]

        self._update_extra_params(in_strs, final_strs)
        return final_strs

    def _get_avaliable_by_type(self, avaliable):
        return avaliable[self.problem_type]

    def _check_avaliable(self, in_strs, avaliable):

        avaliable_by_type = self._get_avaliable_by_type(avaliable)

        check = np.array([m in avaliable_by_type for
                          m in in_strs]).all()

        return check

    def _process_data_scalers(self):
        '''Processes self.data_scaler to be a list of
        (name, scaler) tuples.'''

        if self.data_scalers is not None:

            # If not a list of data scalers, convert to list
            self.data_scalers = conv_to_list(self.data_scalers)

            # Get converted scaler str and update extra params
            conv_scaler_strs = proc_input(self.data_scalers)
            self._update_extra_params(self.data_scalers, conv_scaler_strs)

            # Set data scalers to list of (name, scaler) tuples.
            self.data_scalers = [(name,
                                  get_data_scaler(name, self.extra_params))
                                 for name in conv_scaler_strs]

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

        # Set column specific data scalers
        if self.data_scalers is not None:
            self._set_col_data_scalers(train_data)

        # Train the model(s)
        self._train_models(train_data)

        # Get the score on the test set
        scores = self._get_scores(test_data)
        return scores

    def _set_col_data_scalers(self, data):
        '''Convert the data scaler to column specific data scalers,
        based on the numerical index of the data keys within data.
        Save that to self.col_data_scalers.

        Parameters
        ----------
        data : pandas DataFrame
            ABCD ML formatted, the data to base the column indices from.
        '''

        # Grab the numerical indices for the data-only keys
        data_inds = self._get_data_inds(data)

        # Create a list of tuples (just like self.data_scalers), but
        # with column versions of the scalers.
        self.col_data_scalers = [('col_' + name,
                                 ColumnTransformer([(name, scaler, data_inds)],
                                                   remainder='passthrough',
                                                   sparse_threshold=0)
                                  )
                                 for name, scaler in self.data_scalers]

    def _get_data_inds(self, data):
        '''Grabs the numerical column indices for the data keys
        within data minus targets.

        Parameters
        ----------
        data : pandas DataFrame
            ABCD_ML formatted, assumed to still contain targets

        Returns
        ----------
        list
            The numerical indices for the data keys within data minus
            targets.
        '''

        data_without_targets = data.drop(self.targets_key, axis=1)
        data_inds = [data_without_targets.columns.get_loc(k)
                     for k in self.data_keys]

        return data_inds

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
            The trained single model, or Ensemble_Model of models.
        '''

        # Reset user passed model index to 0
        self.upmi = 0

        models = []
        for model_type in self.model_types:
            models.append(self._train_model(train_data, model_type))

        # Set self.model to be either an ensemble or single model
        if len(models) == 1:
            self.model = models[0]
        else:
            self.model = Ensemble_Model(models)

    def _train_model(self, train_data, model_type):
        '''Helper method to train a single model type given
        a str indicator and training data.

        Parameters
        ----------
        train_data : pandas DataFrame
            ABCD_ML formatted, training data.

        model_type : str
            The final processed str indicator for which model_type to load from
            MODELS constant.

        Returns
        ----------
        sklearn api compatible model object
            The trained model.
        '''

        # Create the internal base k-fold
        base_int_cv = self.CV.k_fold(train_data.index, self.int_cv,
                                     random_state=self.random_state,
                                     return_index=True)

        # Create the model
        model = self._get_model(model_type, base_int_cv)

        # Data, score split
        X, y = self._get_X_y(train_data)

        # Fit the model
        model.fit(X, y)

        return model

    def _get_model(self, model_type, base_int_cv, base_model_flag=False):
        '''Get a model object from a given model type, called recursively to
        build any model that has a base model (e.g. Grid Search)

        Parameters
        ----------
        model_type : str
            The final processed str indicator for which model_type to load from
            MODELS constant.

        base_int_cv : CV output list of tuples
            The internal cv index output to be passed to a classifier

        base_model_flag : bool, optional
            If this flag is set to True, it means this method
            was called recursively.
            (default = False)

        Returns
        ----------
        sklearn api compatible model object
            The requested model object initialized with parameters,
            and ready to be fit.
        '''

        # Check for user passed model
        if model_type == 'user passed':

            user_model = self.user_passed_models[self.upmi]
            user_model_type = 'user passed' + str(self.upmi)
            self.upmi += 1

            model = self._make_model_pipeline(user_model, user_model_type)
            return model

        estimator, base_model_type = None, None

        # Check for if a base model should be created
        if ' gs' in model_type or ' rs' in model_type:
            base_model_type = MODELS[model_type][1]['estimator']
        if ' cal' in model_type:
            base_model_type = MODELS[model_type][1]['base_estimator']

        # Set estimator,
        if base_model_type is not None:
            estimator = self._get_model(base_model_type, base_int_cv, True)

        # Grab the right model and params
        model = MODELS[model_type][0]
        possible_params = get_model_possible_params(model)

        model_params = MODELS[model_type][1].copy()
        model_params = self._replace_params(model_params, possible_params,
                                            base_int_cv, estimator,
                                            base_model_flag)

        # Check to see if there are any user passed model params to update
        extra_model_params = {}
        if model_type in self.extra_params:
            extra_model_params = self.extra_params[model_type]
        model_params.update(extra_model_params)

        # Create model
        model = model(**model_params)

        # In the case that estimator is None, that means this model is
        # is the base model. In that case, we set the model to be a Pipeline
        # object with the proceeding data scalers, as this should be applied
        # on the base estimator.
        if estimator is None:
            model = self._make_model_pipeline(model, model_type)

        return model

    def _replace_params(self, params, possible_params, base_int_cv,
                        estimator, base_model_flag):
        '''Helper method to replace default values with provided params,
        with actual values saved within the class.

        Parameters
        ----------
        params : dict
            Dictionary with parameter values to be replaced.

        base_int_cv : CV output list of tuples
            The internal cv index output to be passed to a classifier

        estimator : model or None
            Either a model object passed to be set for the estimator param
            or None if not applicable.
            Specifically, if estimator is None, then this is the final model,
            if has a specific value then it means these are the params for
            a grid search or calibrated overarching-esque object.

        base_model_flag : bool
            If set to True, then that means we are
            replacing the parameters for a base model within
            another estimator. In that case, we set n_jobs if
            avaliable to 1.

        Returns
        ----------
        dict
            The input dictionary with applicable values transformed.
        '''

        if 'cv' in possible_params:
            params['cv'] = base_int_cv

        if 'scoring' in possible_params:
            params['scoring'] = self.scorer

        if 'class_weight' in possible_params:
            params['class_weight'] = self.class_weight

        if 'n_jobs' in possible_params:
            if base_model_flag:
                params['n_jobs'] = 1
            else:
                params['n_jobs'] = self.n_jobs

        if 'n_iter' in possible_params:
            params['n_iter'] = self.n_iter

        if 'random_state' in possible_params:
            params['random_state'] = self.random_state

        if 'estimator' in params:
            if type(params['estimator']) == str and estimator is not None:
                params['estimator'] = estimator

        if 'base_estimator' in params:
            if type(params['base_estimator']) == str and estimator is not None:
                params['base_estimator'] = estimator

        return params

    def _make_model_pipeline(self, model, model_type):
        '''Provided a model & model type (model str indicator),
        return a sklearn pipeline with proceeding self.col_data_scalers,
        and then the model.

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

        steps = self.col_data_scalers + self.feat_selectors \
            + [(model_type, model)]

        model_pipeline = Pipeline(steps)

        return model_pipeline

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


class Regression_Model(Model):
    '''Child class of Model for regression problem types.'''

    def _set_default_params(self):
        '''Overrides parent method'''
        self.problem_type = 'regression'


class Binary_Model(Model):
    '''Child class of Model for binary problem types.'''

    def _set_default_params(self):
        '''Overrides parent method'''
        self.problem_type = 'binary'


class Categorical_Model(Model):
    '''Child class of Model for categorical problem types.'''

    def _set_default_params(self):
        '''Overrides parent method'''
        self.problem_type = 'categorical'
        self.sub_problem_type = 'multilabel'

    def _get_conv_model_types(self):
        '''Overrides parent method, categorical models have
        a special case with sub_problem_type, this method handles that.

        Returns
        ----------
        list
            List of final str indicator model_types, indices should
            correspond to the order os self.model_types, where the str
            in the same index should represent the converted version.
        '''

        conv_model_types = proc_input(self.model_types)

        # Check first to see if all model names are in multilabel
        if np.array([m in AVALIABLE_MODELS['categorical']['multilabel']
                    for m in conv_model_types]).all():

            conv_model_types =\
                [AVALIABLE_MODELS['categorical']['multilabel'][m]
                 for m in conv_model_types]

        # Then check for multiclass, if multilabel not avaliable
        elif np.array([m in AVALIABLE_MODELS['categorical']['multiclass']
                      for m in conv_model_types]).all():

            conv_model_types =\
                [AVALIABLE_MODELS['categorical']['multiclass'][m]
                 for m in conv_model_types]

            # Set the cat conv flag to be true
            self.sub_problem_type = 'multiclass'
            self._print('Not all model types passed have multilabel support!',
                        'Using multiclass instead.')

        else:
            assert 0 == 1, "Selected model type(s) not avaliable."

        return conv_model_types

    def _get_avaliable_scorers(self):
        '''Overrides parent method, adding subproblem_type
        Get the avaliable scorers by problem type.

        Returns
        ----------
        dict
            Dictionary of avaliable scorers, with value as final str
            indicator.
        '''

        return AVALIABLE_SCORERS[self.problem_type][self.sub_problem_type]

    def _get_avaliable_by_type(self, avaliable):
        return avaliable[self.problem_type][self.sub_problem_type]

    def _check_avaliable(self, in_strs, avaliable):

        check = super()._check_avaliable(in_strs, avaliable)

        if not check and self.sub_problem_type == 'multilabel':
            self._print('Not all input supports multilabel,')
            self._print('Checking compatability with multiclass!')

            self.sub_problem_type = 'multiclass'
            check = super()._check_avaliable(in_strs, avaliable)

        return check

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
