import numpy as np
from ABCD_ML.ML_Helpers import get_scaler
from ABCD_ML.Scoring import scorer_from_string
from ABCD_ML.Models import AVALIABLE
from ABCD_ML.Train_Models import get_model
from ABCD_ML.Ensemble_Model import Ensemble_Model


class Model():

    def __init__(self, CV, model_types, data_scaler, data_keys,
                 score_keys, score_encoder, int_cv, metric,
                 class_weight='balanced', random_state=None, n_jobs=1,
                 extra_params={}, verbose=True):

        self.CV = CV
        self.model_types = model_types
        self.data_scaler = data_scaler
        self.data_keys = data_keys
        self.score_key = score_key
        self.score_encoder = score_encoder
        self.int_cv = int_cv
        self.metric = metric
        self.class_weight = class_weight
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.extra_params = extra_params
        self.verbose = verbose

        self.set_default_params()
        self.check_default_metric()
        self.process_model_type()
        self.data_scaler = get_scaler(self.data_scaler, self.extra_params)
        self.scorer = scorer_from_string(self.metric)

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

    def set_default_params(self):
        pass

    def check_default_metric(self):

        if self.metric == 'default':
            self.metric = self.default_metric
            self._print('Default metric passed, setting metric ' +
                        self.default_metric + '.')

    def process_model_type(self):

        # If not a list of model types, convert to list
        if not isinstance(self.model_types, list):
            self.model_types = [self.model_types]

        # Get the converted version of the model type passed in
        conv_model_types = self.get_conv_model_types()

        # If any extra params passed for the model, change to conv'ed name
        self.update_extra_params(conv_model_types)

        # Set model type to the conv'ed version
        self.model_types = conv_model_types

    def get_conv_model_types(self):

        # Base behavior for binary / regression

        conv_model_types = [m.replace('_', ' ').lower()
                            for m in self.model_types]

        assert np.array([m in AVALIABLE[self.problem_type]
                        for m in conv_model_types]).all(), \
            "Selected model type(s) not avaliable."

        conv_model_types = [AVALIABLE[problem_type][m]
                            for m in conv_model_types]
        return conv_model_types

    def update_extra_params(self, conv_model_types):

        for m in range(len(conv_model_types)):
            if self.model_types[m] in self.extra_params:

                self.extra_params[conv_model_types[m]] = \
                    self.extra_params[self.model_types[m]]

    def evaluate_model(self, data, train_subjects, n_splits, n_repeats):

        # Setup the desired splits, using the passed in train subjects
        subject_splits = self.CV.repeated_k_fold(train_subjects,
                                                 n_repeats, n_splits,
                                                 self.random_state, False)

        scores = []

        # For each split with the repeated K-fold
        for train_subjects, test_subjects in subject_splits:

            score = self.test_model(data, train_subjects, test_subjects)
            scores.append(score)

        # Return the list of scores
        return scores

    def test_model(self, data, train_subjects, test_subjects):

        # Assume the train_subjects and test_subjects passed here are final.
        train_data = data.loc[train_subjects]
        test_data = data.loc[test_subjects]

        # Scale the data
        train_data, test_data = self.scale_data(train_data, test_data)

        # Train the model(s)
        self.train_models(train_data)

        # Get the score on the test set
        score = self.get_score(test_data)
        return score

    def scale_data(self, train_data, test_data):
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

            test_data[data_keys] = \
                self.data_scaler.transform(test_data[self.data_keys])

        return train_data, test_data

    def train_models(self, train_data):

        models = []
        for model_type in self.model_types:
            models.append(self.train_model(train_data, model_type))

        # Set self.model to be either an ensemble or single model
        if len(models) == 1:
            self.model = models[0]
        else:
            self.model = Ensemble_Model(models)

    def train_model(self, train_data, model_type):

        # Create the internal base k-fold and scorer
        base_int_cv = self.CV.k_fold(train_data.index, self.int_cv,
                                     random_state=self.random_state,
                                     return_index=True)

        # Create the model
        model = get_model(model_type, base_int_cv, self.scorer,
                          self.class_weight, self.n_jobs, self.extra_params)

        # Data, score split
        X, y = self.get_X_y(train_data)

        # Fit the model
        model.fit(X, y)

        return model

    def get_score(self, test_data):

        # Data, score split
        X_test, y_test = self.get_X_y(test_data)

        # Get the score
        score = self.scorer(self.model, X_test, y_test)
        return score

    def get_X_y(self, data):

        X = np.array(data.drop(self.score_key, axis=1))
        y = np.array(data[self.score_key])

        # Convert/decode y/score if needed
        y = self.conv_score(y)

        return X, y

    def conv_score(self, y):
        return y


class Regression_Model(Model):

    def set_default_params(self):
        self.problem_type = 'regression'
        self.default_metric = 'r2 score'


class Binary_Model(Model):

    def set_default_params(self):
        self.problem_type = 'binary'
        self.default_metric = 'roc auc'


class Categorical_Model(Model):

    def set_default_params(self):
        self.problem_type = 'categorical'
        self.sub_problem_type = 'multilabel'
        self.default_metric = 'weighted roc auc'

    def get_conv_model_types(self):

        conv_model_types = [m.replace('_', ' ').lower()
                            for m in self.model_types]

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

    def conv_score(self, y):

        # If multiclass, convert to correct score format
        if self.sub_problem_type == 'multiclass':
            y = self.score_encoder[1].inverse_transform(y).squeeze()

        return y











        

    