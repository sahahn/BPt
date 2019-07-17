import numpy as np
from ABCD_ML.ML_Helpers import (get_scaler, proc_input,
                                get_model_possible_params)
from ABCD_ML.Models import AVALIABLE, MODELS
from ABCD_ML.Scoring import AVALIABLE as AVALIABLE_SCORERS
from ABCD_ML.Scoring import get_scorer
from ABCD_ML.Ensemble_Model import Ensemble_Model


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
            ABCD_ML.show_model_types(), with optional problem type parameter.

        ML_params : dict
            Dictionary of different ML params, the following must be included,
            (See the docstring for ABCD_ML.set_default_ML_params for a more
            detailed description of all parameters contained within ML_params)

            - metric : str
                Metric / scorer str indicator
            - data_scaler : str or None
                str indicator for what type of data scaling to use if any.
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
        '''

        # Set class parameters
        self.model_types = model_types
        self.CV = CV
        self.data_keys = data_keys
        self.targets_key = targets_key
        self.targets_encoder = targets_encoder
        self.verbose = verbose

        # Un-pack ML_params
        self.metric = ML_params['metric']
        self.data_scaler = ML_params['data_scaler']
        self.n_splits = ML_params['n_splits']
        self.n_repeats = ML_params['n_repeats']
        self.int_cv = ML_params['int_cv']
        self.class_weight = ML_params['class_weight']
        self.n_jobs = ML_params['n_jobs']
        self.n_iter = ML_params['n_iter']
        self.random_state = ML_params['random_state']
        self.extra_params = ML_params['extra_params']

        # Set problem type info and proc. model_type
        self._set_default_params()
        self._process_model_type()
        self._scorer_from_string()

        # Get the data scaler and scorer
        self.data_scaler = get_scaler(self.data_scaler, self.extra_params)
   
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

    def _process_model_type(self):
        '''Class function to convert input model types to final
        str indicator, based on problem type and common input correction.
        Also handles updating extra params, if applicable.'''

        # If not a list of model types, convert to list
        if not isinstance(self.model_types, list):
            self.model_types = [self.model_types]

        # Get the converted version of the model type passed in
        conv_model_types = self._get_conv_model_types()

        # If any extra params passed for the model, change to conv'ed name
        self._update_extra_params(conv_model_types)

        # Set model type to the conv'ed version
        self.model_types = conv_model_types

    def _get_conv_model_types(self):
        '''Method to grab the converted version of the model_names saved
        in self.model_types.

        Returns
        ----------
        list
            List of final str indicator model_types, indices should
            correspond to the order os self.model_types, where the str
            in the same index should represent the converted version.
        '''

        # Base behavior for binary / regression
        conv_model_types = proc_input(self.model_types)

        assert np.array([m in AVALIABLE[self.problem_type]
                        for m in conv_model_types]).all(), \
            "Selected model type(s) not avaliable."

        conv_model_types = [AVALIABLE[self.problem_type][m]
                            for m in conv_model_types]
        return conv_model_types

    def _update_extra_params(self, conv_model_types):
        '''Helper method to update class extra params in the case
        that get_conv_model_types changed any input str indicators.

        Parameters
        ----------
        conv_model_types : list
            List of final str indicator model_types, indices should
            correspond to the order os self.model_types, where the str
            in the same index should represent the converted version.
        '''

        for m in range(len(conv_model_types)):
            if self.model_types[m] in self.extra_params:

                self.extra_params[conv_model_types[m]] = \
                    self.extra_params[self.model_types[m]]

    def _get_avaliable_scorers(self):
        '''Get the avaliable scorers by problem type.

        Returns
        ----------
        dict
            Dictionary of avaliable scorers, with value as final str
            indicator.
        '''

        return AVALIABLE_SCORERS[self.problem_type]

    def _scorer_from_string(self):
        '''Process self.metric and set self.scorer'''

        conv_metric = proc_input(self.metric)
        avaliable_scorers = self._get_avaliable_scorers()

        assert conv_metric in avaliable_scorers, \
            "Selected metric is not avaliable with this (sub)problem type."
        scorer_str = avaliable_scorers[conv_metric]

        self.scorer = get_scorer(scorer_str)

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
        list of floats
            The raw score as computed for each fold within each repeat,
            e.g., list will have a length of `n_repeats` * `n_splits`
        '''

        # Setup the desired splits, using the passed in train subjects
        subject_splits = self.CV.repeated_k_fold(train_subjects,
                                                 self.n_repeats, self.n_splits,
                                                 self.random_state,
                                                 return_index=False)

        scores = []

        # For each split with the repeated K-fold
        for train_subjects, test_subjects in subject_splits:

            score = self.Test_Model(data, train_subjects, test_subjects)
            scores.append(score)

        # Return the list of scores
        return scores

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
        float
            The score as determined by the passed metric/scorer on the
            provided testing set.
        '''

        # Assume the train_subjects and test_subjects passed here are final.
        train_data = data.loc[train_subjects]
        test_data = data.loc[test_subjects]

        # Scale the data
        train_data, test_data = self._scale_data(train_data, test_data)

        # Train the model(s)
        self._train_models(train_data)

        # Get the score on the test set
        score = self._get_score(test_data)
        return score

    def _scale_data(self, train_data, test_data):
        '''Wrapper function to take in train/test data,
        and if not None, fit + transform a data scaler on the train data,
        and then transform the test data.

        Parameters
        ----------
        train_data : pandas DataFrame
            ABCD_ML formatted df, with the subset of training data only

        test_data : pandas DataFrame
            ABCD_ML formatted df, with the subset of testing data only

        Returns
        ----------
        pandas DataFrame
            ABCD_ML formatted, the scaled training data

        pandas DataFrame
            ABCD_ML formatted, the scaled testing data
        '''

        if self.data_scaler is not None:

            train_data[self.data_keys] = \
                self.data_scaler.fit_transform(train_data[self.data_keys])

            test_data[self.data_keys] = \
                self.data_scaler.transform(test_data[self.data_keys])

        return train_data, test_data

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

    def _get_model(self, model_type, base_int_cv, base_model=False):
        '''Get a model object from a given model type, called recursively to
        build any model that has a base model (e.g. Grid Search)

        Parameters
        ----------
        model_type : str
            The final processed str indicator for which model_type to load from
            MODELS constant.

        base_int_cv : CV output list of tuples
            The internal cv index output to be passed to a classifier

        base_model : bool, optional
            Flag to determine if the passed params are for the final model,
            or if they are for a base_model (the model object within a grid or
            random search). Set to False if final model, True if base model.

        Returns
        ----------
        sklearn api compatible model object
            The requested model object initialized with parameters,
            and ready to be fit.
        '''

        estimator = None

        # If gs or rs in name (grid or random search)
        # recursively build the base_model / estimator.
        if ' gs' in model_type or ' rs' in model_type:

            base_model_type = MODELS[model_type][1]['estimator']
            estimator = self._get_model(base_model_type, base_int_cv,
                                        base_model=True)

        # Grab the right model and params
        model = MODELS[model_type][0]
        possible_params = get_model_possible_params(model)

        model_params = MODELS[model_type][1].copy()
        model_params = self._replace_params(model_params, possible_params,
                                            base_int_cv, estimator=estimator,
                                            base_model=base_model)

        # Check to see if there are any user passed model params to update
        extra_model_params = {}
        if model_type in self.extra_params:
            extra_model_params = self.extra_params[model_type]
        model_params.update(extra_model_params)

        # Create model
        model = model(**model_params)
        return model

    def _replace_params(self, params, possible_params, base_int_cv,
                        estimator=None, base_model=False):
        '''Helper method to replace default values with provided params,
        with actual values saved within the class.

        Parameters
        ----------
        params : dict
            Dictionary with parameter values to be replaced.

        base_int_cv : CV output list of tuples
            The internal cv index output to be passed to a classifier

        estimator : model or None, optional
            Either a model object passed to be set for the estimator param
            or None if not applicable.
            (default = None)

        base_model : bool, optional
            Flag to determine if the passed params are for the final model,
            or if they are for a base_model (the model object within a grid or
            random search). Set to False if final model, True if base model.

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
            if base_model:
                params['n_jobs'] = self.n_jobs
            else:
                params['n_jobs'] = 1

        if 'n_iter' in possible_params:
            params['n_iter'] = self.n_iter

        if 'random_state' in possible_params:
            params['random_state'] = self.random_state

        if 'estimator' in params:
            if type(params['estimator']) == str and estimator is not None:
                params['estimator'] = estimator

        return params

    def _get_score(self, test_data):
        '''Helper method to get the score of
        the trained model saved in the class on input test data.

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

        # Get the score
        score = self.scorer(self.model, X_test, y_test)
        return score

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
        if np.array([m in AVALIABLE['categorical']['multilabel']
                    for m in conv_model_types]).all():

            conv_model_types = [AVALIABLE['categorical']['multilabel'][m]
                                for m in conv_model_types]

        # Then check for multiclass, if multilabel not avaliable
        elif np.array([m in AVALIABLE['categorical']['multiclass']
                      for m in conv_model_types]).all():

            conv_model_types = [AVALIABLE['categorical']['multiclass'][m]
                                for m in conv_model_types]

            # Set the cat conv flag to be true
            self.sub_problem_type = 'multiclass'
            self._print('Not all model types passed have multilabel support, \
                using multiclass instead.')

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
