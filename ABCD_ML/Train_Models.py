'''
ABCD_ML Project

Scripts for training models
'''


from sklearn.linear_model import (LogisticRegressionCV, ElasticNetCV, LinearRegression,
OrthogonalMatchingPursuitCV,LarsCV, RidgeCV)
from sklearn.svm import LinearSVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor, LGBMClassifier
from ABCD_ML.Train_Light_GBM import Train_Light_GBM

def train_regression_model(X, y, model_type='elastic cv', cv=3, extra_params={}):
    '''Wrapper function to train various regression models with X,y input,
       where extra params can be passed to override any default parameters''' 

    if model_type == 'linear':
        model = LinearRegression(fit_intercept=True)
    elif model_type == 'elastic cv':
        model = ElasticNetCV(cv=cv)
    elif model_type == 'omp cv':
        model = OrthogonalMatchingPursuitCV(cv=cv)
    elif model_type == 'lars cv':
        model = LarsCV(cv=cv)
    elif model_type == 'ridge cv':
        model = RidgeCV(cv=cv)
    elif model_type == 'full lightgbm':
        model = Train_Light_GBM(X, y, int_cv=cv, regression=True, **extra_params)
        return model
        
    model.fit(X, y)
    return model

def train_binary_model(X, y, model_type='logistic cv', cv=3, class_weight='balanced', extra_params={}):
    '''Wrapper function to train various binary models with X,y input,
       where extra params can be passed to override any default parameters''' 

    if model_type == 'logistic cv':
        model = LogisticRegressionCV(cv=cv, class_weight=class_weight)
    elif model_type == 'full lightgbm':
        model = Train_Light_GBM(X, y, int_cv=cv, regression=False, **extra_params)
        return model
    
    model.fit(X, y)
    return model