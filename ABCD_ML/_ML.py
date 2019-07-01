'''
ABCD_ML Project

Main wrappers for Machine Learning functions
'''

from sklearn.model_selection import (RepeatedStratifiedKFold, RepeatedKFold, KFold,
RandomizedSearchCV, train_test_split, ParameterSampler)

from sklearn.metrics import (roc_auc_score, mean_squared_error, r2_score, balanced_accuracy_score,
f1_score, log_loss)

from sklearn.preprocessing import (MinMaxScaler,RobustScaler,StandardScaler)
from ABCD_ML.Ensemble_Model import Ensemble_Model
from ABCD_ML.Train_Models import train_regression_model, train_binary_model

import numpy as np
import warnings

#warnings.filterwarnings("ignore", category=DeprecationWarning)

def feature_transformation(feature,method='Standard'):
    ''' Transform continuous features by different scaler so that all values
    would be in the same range'''
    
    if method=='Standard':
        scaler = StandardScaler()
    elif method=='MinMax':
        scaler = MinMaxScaler()
    elif method=='Robust':
        scaler = RobustScaler()
    elif method=='Power':
        pt=PowerTransformer(method='yeo-johnson',standardize=False)
        return pt.fit_transform(feature)
    scaler.fit(feature)
    return scaler.transform(feature)

def metric_from_string(metric):
    ''' Helper function to convert from string input to sklearn metric, 
        can also be passed a metric directly'''

    if callable(metric):
        return metric
    elif metric == 'r2' or metric == 'r2 score':
        return r2_score
    elif metric == 'mse' or metric == 'mean squared error':
        return mean_squared_error
    elif metric == 'roc' or metric == 'roc auc':
        return roc_auc_score
    elif metric == 'bas' or metric == 'balanced acc':
        return balanced_accuracy_score
    elif metric == 'f1':
        return f1_score
    elif metric == 'log loss':
        return log_loss

    print('No metric function defined')
    return None

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

    #Set the data this function to be just the training data
    data = self.all_data.loc[self.train_subjects]

    scores = []

    #Setup the desired splits
    splits = self.CV.repeated_k_fold(subjects=data.index, n_repeats=n_repeats,
                                n_splits=n_splits, random_state=random_state)

    for train_subjects, test_subjects in splits:

        score = test_regression_model(data.loc[train_subjects],
                                      data.loc[test_subjects],
                                      CV, model_type, data_scaler,
                                      int_cv, metric, target_transform,
                                      random_state, extra_params)
        scores.append(score)

    scores = np.array(scores)
    macro_scores = np.mean(np.reshape(scores, (n_repeats, n_splits)), axis=1)

    return np.mean(macro_scores), np.std(macro_scores), np.mean(score), np.std(scores)


def test_regression_model(train_data,
                          test_data,
                          CV,
                          model_type = 'linear',
                          data_scaler = 'standard',
                          int_cv = 3,
                          metric = 'r2',
                          target_transform = None,
                          random_state = None,
                          extra_params = {}
                          ):

    
   

    if target_transform == 'log':
        y = np.log1p(y)

    if type(model_type) == list:
        model = Ensemble_Model(X, y, model_type, int_cv, regresson=True, extra_params=extra_params)
    else:
        model = train_regression_model(X, y, model_type, int_cv, extra_params=extra_params) 

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
















