'''
ABCD_ML Project

Scripts for training models
'''
import numpy as np
import ABCD_ML.Default_Params as DP

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestRegressor
from sklearn.linear_model import (LogisticRegressionCV, ElasticNetCV, LinearRegression,
                                  OrthogonalMatchingPursuitCV,LarsCV, RidgeCV)

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
#from sklearn.svm import SVC, LinearSVR
#from sklearn.neural_network import MLPClassifier

#from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
#from xgboost import XGBRegressor
#from lightgbm import LGBMRegressor, LGBMClassifier
from ABCD_ML.Train_Light_GBM import Train_Light_GBM
from ABCD_ML.ML_Helpers import metric_from_string

def train_model(problem_type,
                data,
                score_key,
                CV,
                model_type='logistic cv',
                int_cv=3,
                metric='roc',
                class_weight='balanced',
                random_state=None,
                n_jobs=1,
                extra_params={}
                ):

    problem_type = problem_type.lower()
    model_type_lower = model_type.lower()

    base_int_cv = CV.k_fold(data.index, int_cv, random_state=random_state, return_index=True)
    scorer = metric_from_string(metric, return_scorer=True)
    
    params = {'n_jobs': n_jobs}

    if problem_type == 'regression':
        model, params = get_regression_model(model_type_lower, params, base_int_cv, scorer)
    elif problem_type == 'binary':
        model, params = get_binary_model(model_type_lower, params, base_int_cv, scorer, class_weight)
    elif problem_type == 'categorical':
        model, params = get_categorical_model(model_type_lower, params, base_int_cv, scorer, class_weight)

    if model_type in extra_params:
        params.update(extra_params[model_type])

    model = model(**params)

    X, y = np.array(data.drop(score_key, axis=1)), np.array(data[score_key])
    model.fit(X, y)

    return model

def get_regression_model(model_type,
                         params,
                         base_int_cv,
                         scorer
                         ):

    if model_type == 'linear':
        model = LinearRegression
        params.update({'fit_intercept': True})

    elif model_type == 'elastic cv':
        model = ElasticNetCV
        params.update({'cv': base_int_cv, 'max_iter': 5000})
        
    elif model_type == 'omp cv':
        model = OrthogonalMatchingPursuitCV
        params.update({'cv': base_int_cv})
    
    elif model_type == 'lars cv':
        model = LarsCV
        params.update({'cv': base_int_cv})
    
    elif model_type == 'ridge cv':
        model = RidgeCV
        params.update({'cv': base_int_cv})

    elif model_type == 'rf' or model_type == 'random forest':
        raw_model = RandomForestRegressor(n_estimators=100)
        model = RandomizedSearchCV
        params.update({
                  'estimator': raw_model,
                  'param_distributions' : DP.DEFAULT_RF_GRID1,
                  'n_iter': 10,
                  'scoring' : scorer,
                  'cv': base_int_cv,
                  'iid' : False 
                  })
    
    elif model_type == 'gaussian process' or model_type == 'gp':
        model = GaussianProcessRegressor
    
    #elif model_type == 'full lightgbm':
    #    model = Train_Light_GBM(X, y, int_cv=cv, regression=True, **extra_params)
    #    return model

    return model, params

def get_binary_model(model_type,
                     params,
                     base_int_cv,
                     scorer,
                     class_weight
                     ):  

    if model_type == 'logistic cv':
        model = LogisticRegressionCV
        params.update({'cv': base_int_cv, 'class_weight': class_weight, 'max_iter': 5000})
    
    elif model_type == 'nb':
        model = GaussianNB
    
    elif model_type == 'knn':
        raw_model = KNeighborsClassifier(n_neighbors=1)
        model = GridSearchCV
        params.update({
                  'estimator': raw_model,
                  'param_grid' : DP.DEFAULT_KNN_GRID1,
                  'scoring' : scorer,
                  'cv': base_int_cv,
                  'iid' : False 
                  })
        
        model = param_search(raw_model, param_grid, **extra_params)

    elif model_type == 'dtc':
        raw_model = DecisionTreeClassifier(class_weight=class_weight)
        model = GridSearchCV
        params.update({
                  'estimator': raw_model,
                  'param_grid' : DP.DEFAULT_DTC_GRID1,
                  'scoring' : scorer,
                  'cv': base_int_cv,
                  'iid' : False 
                  })
    
    #elif model_type == 'full lightgbm':
    #    model = Train_Light_GBM(X, y, int_cv=cv, regression=False, **extra_params)
    #    return model

    return model, params

def get_categorical_model(model_type,
                          params,
                          base_int_cv,
                          scorer,
                          class_weight
                          ): 
    pass