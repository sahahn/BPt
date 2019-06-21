'''
ABCD_ML Project

Scripts for training models
'''

from sklearn.linear_model import (LogisticRegressionCV, ElasticNetCV, LinearRegression,
OrthogonalMatchingPursuitCV,LarsCV, RidgeCV)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor, LGBMClassifier
from ABCD_ML.Train_Light_GBM import Train_Light_GBM

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

def param_search(raw_model, param_grid, search_folds=3, search_scoring='roc_auc', search_type='grid search', **kwargs):
    
    if search_type == 'grid search':
        model = GridSearchCV(raw_model, param_grid, cv=search_folds, scoring=search_scoring)
    elif search_type == 'random search':
        model = RandomizedSearchCV(raw_model, param_grid, cv=search_folds, scoring=search_scoring)
    
    return model

def train_regression_model(X, y, model_type='elastic cv', cv=3, extra_params={}):
    '''Wrapper function to train various regression models with X,y input,
       where extra params can be passed to override any default parameters''' 

    model_type = model_type.lower()

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

    model_type = model_type.lower()

    if model_type == 'logistic cv':
        model = LogisticRegressionCV(cv=cv, class_weight=class_weight, max_iter=1000)
    
    elif model_type == 'nb':
        model = GaussianNB()
    
    elif model_type == 'knn':
        raw_model = KNeighborsClassifier(n_neighbors=1, weights='uniform') 
        param_grid = {n_neighbors : list(range(1,20))}
        model = param_search(raw_model, param_grid, **extra_params)

    elif model_type == 'dtc':
        raw_model = DecisionTreeClassifier()
        param_grid = {max_depth : list(range(1, 20)), min_samples_split: list(range(2, 10))}
        model = param_search(raw_model, param_grid, **extra_params)
    
    elif model_type == 'full lightgbm':
        model = Train_Light_GBM(X, y, int_cv=cv, regression=False, **extra_params)
        return model
    
    model.fit(X, y)
    return model





