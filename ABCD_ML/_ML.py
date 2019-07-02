'''
ABCD_ML Project

Main wrappers for Machine Learning functions
'''

from ABCD_ML.Ensemble_Model import Ensemble_Model
from ABCD_ML.Train_Models import train_regression_model, train_binary_model
from ABCD_ML.ML_Helpers import get_scaler, metric_from_string, compute_macro_micro

import numpy as np


#import warnings
#warnings.filterwarnings("ignore", category=DeprecationWarning)



def get_regression_score(model, X_test, y_test, metric='r2', target_transform=None):
    '''Computes a regression score, provided a model and testing data with labels and metric'''
    
    preds = model.predict(X_test)

    if target_transform == 'log':
        preds = np.exp(preds)

    metric_func = metric_from_string(metric)
    score = metric_func(y_test, preds)

    return score

def get_binary_score(model, X_test, y_test, metric='roc'):
    '''Computes a binary score, provided a model and testing data with labels and metric'''

    preds = model.predict_proba(X_test)
    preds = [p[1] for p in preds]
    
    metric_func = metric_from_string(metric)
    
    try:
        score = metric_func(y_test, preds)
    except ValueError:
        score = metric_func(y_test, np.round(preds))

    return score

def premodel_check(self):

    if self.all_data is None:
        self.prepare_data()
    
    if self.train_subjects is None:
        print('No train-test set defined! Performing one automatically with default split =.25')
        print('If no test set is intentional, just called train_test_split(test_size=0)')
        self.train_test_split(test_size=.25)

def evaluate_regression_model(self,
                              model_type = 'linear',
                              data_scaler = 'standard',
                              n_splits = 3,
                              n_repeats = 2,
                              int_cv = 3,
                              metric = 'r2', 
                              target_transform = None,
                              random_state = None,
                              extra_params = {}
                              ):

    #Perform check
    self.premodel_check()
    scores = []

    #Setup the desired splits, using the train subjects
    subject_splits = self.CV.repeated_k_fold(subjects=self.train_subjects, n_repeats=n_repeats,
                                n_splits=n_splits, random_state=random_state)

    for train_subjects, test_subjects in subject_splits:

        score = self.test_regression_model(train_subjects,
                                           test_subjects,
                                           model_type, data_scaler,
                                           int_cv, metric, target_transform,
                                           False, random_state, extra_params)
        scores.append(score)

    return compute_macro_micro(scores, n_repeats, n_splits)

def test_regression_model(self,
                          train_subjects = None,
                          test_subjects = None,
                          model_type = 'linear',
                          data_scaler = 'standard',
                          int_cv = 3,
                          metric = 'r2',
                          target_transform = None,
                          return_model = False,
                          random_state = None,
                          extra_params = {}
                          ):

    if train_subjects is None:
        train_subjects = self.train_subjects
    if test_subjects is None:
        test_subjects = self.test_subjects

    train_data = self.all_data.loc[train_subjects]
    test_data = self.all_data.loc[test_subjects]
    
    #if target_transform == 'log':
    #    y = np.log1p(y)

    if data_scaler is not None:

        scaler = get_scaler(data_scaler, extra_params)
        train_data[self.data_keys] = scaler.fit_transform(train_data[self.data_keys])
        test_data[self.data_keys] = scaler.transform(test_data[self.data_keys])

    if type(model_type) == list:
        model = Ensemble_Model(train_data, self.score_key, self.CV, model_type,
                               int_cv, metric, problem_type, random_state, self.n_jobs,
                               extra_params=extra_params)
    else:
        model = train_regression_model(train_data, self.score_key, self.CV,
                                       model_type,
                                       int_cv, metric, random_state, self.n_jobs,
                                       extra_params=extra_params) 

    X_test, y_test = np.array(test_data.drop(self.score_key, axis=1)), np.array(test_data[self.score_key])
    score = get_regression_score(model, X_test, y_test, metric, target_transform)
    return score

def evaluate_binary_model(X, y, model_type='logistic cv', n_splits=3, n_repeats=2, int_cv=3,
                          metric='roc', class_weight='balanced', extra_params={}):
    '''Wrapper function to perform a repeated stratified KFold binary classification analysis'''

    scores = []
    skf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats)

    for train_ind, test_ind in skf.split(X,y):

        X_train, y_train = X[train_ind], y[train_ind]
        X_test, y_test = X[test_ind], y[test_ind]

        scores.append(test_binary_model(X_train, y_train, X_test, y_test, model_type,
                                        int_cv, metric, class_weight, extra_params))

    scores = np.array(scores)
    macro_scores = np.mean(np.reshape(scores, (n_repeats, n_splits)), axis=1)

    return np.mean(macro_scores), np.std(macro_scores)


def test_binary_model(X, y, X_test, y_test, model_type='logistic cv', int_cv=3,
                     metric='roc', class_weight='balanced', extra_params={}):
    '''Wrapper function to perform a single test with explicit X,y and test set for classification'''


    if type(model_type) == list:
        model = Ensemble_Model(X, y, model_type, int_cv, False, class_weight, extra_params)
    else:
        model = train_binary_model(X, y, model_type, int_cv, class_weight, extra_params) 

    score = get_binary_score(model, X_test, y_test, metric)
    return score















