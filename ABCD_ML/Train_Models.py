'''
ABCD_ML Project

Scripts for training models
'''

from sklearn.linear_model import (LogisticRegressionCV, ElasticNetCV, LinearRegression,
OrthogonalMatchingPursuitCV,LarsCV, RidgeCV)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.svm import LinearSVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor, LGBMClassifier
from ABCD_ML.Train_Light_GBM import Train_Light_GBM
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

from ABCD_ML.ML_Helpers import metric_from_string
import ABCD_ML.Default_Params as P
import numpy as np

def train_regression_model(data,
                           score_key,
                           CV,
                           model_type,
                           int_cv=3,
                           metric='r2',
                           random_state=None,
                           n_jobs=1,
                           extra_params={}
                           ):

    model_type = model_type.lower()
    
    base_int_cv = CV.k_fold(data.index, int_cv, random_state=random_state, return_index=True)
    scorer = metric_from_string(metric, return_scorer=True)

    params = {'n_jobs': n_jobs}

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
        sub_model = RandomForestRegressor(n_estimators=100)
        model = RandomizedSearchCV
        params.update({
                  'estimator': sub_model,
                  'param_distributions' : P.DEFAULT_RF_GRID1,
                  'n_iter': 10,
                  'scoring' : scorer,
                  'cv': base_int_cv,
                  'iid' : False 
                  })

    if model_type in extra_params:
        print('Adding extra user passed params')
        params.update(extra_params[model_type])

    model = model(**params)

    X, y = np.array(data.drop(score_key, axis=1)), np.array(data[score_key])

    print('Training model ', model_type) 
    model.fit(X, y)

    return model

    #elif model_type == 'gaussian process' or model_type == 'gp':
    #    model = GaussianProcessRegressor()
    
    #elif model_type == 'full lightgbm':
    #    model = Train_Light_GBM(X, y, int_cv=cv, regression=True, **extra_params)
    #    return model

def train_binary_model(X, y, model_type='logistic cv', cv=3, class_weight='balanced', extra_params={}):
    '''Wrapper function to train various binary models with X,y input,
       where extra params can be passed to override any default parameters'''

    model_type = model_type.lower()

    if model_type == 'logistic cv':
        model = LogisticRegressionCV(cv=cv, class_weight=class_weight, max_iter=5000)
    
    elif model_type == 'nb':
        model = GaussianNB()
    
    elif model_type == 'knn':
        raw_model = KNeighborsClassifier(n_neighbors=1, weights='uniform') 
        param_grid = {'n_neighbors' : list(range(1,20))}
        model = param_search(raw_model, param_grid, **extra_params)

    elif model_type == 'dtc':
        raw_model = DecisionTreeClassifier()
        param_grid = {'max_depth' : list(range(1, 20)), 'min_samples_split': list(range(2, 50))}
        model = param_search(raw_model, param_grid, **extra_params)
    
    elif model_type == 'full lightgbm':
        model = Train_Light_GBM(X, y, int_cv=cv, regression=False, **extra_params)
        return model
    
    model.fit(X, y)
    return model





