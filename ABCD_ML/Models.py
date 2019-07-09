import ABCD_ML.Default_Params as DP

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.ensemble import (GradientBoostingClassifier, AdaBoostClassifier, RandomForestRegressor, RandomForestClassifier)
from sklearn.linear_model import (LogisticRegression, LogisticRegressionCV, ElasticNetCV, LinearRegression,
                                  OrthogonalMatchingPursuitCV,LarsCV, RidgeCV)

#from sklearn.svm import SVC, LinearSVR
#from sklearn.neural_network import MLPClassifier

#from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
#from xgboost import XGBRegressor
#from lightgbm import LGBMRegressor, LGBMClassifier

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

AVALIABLE = {
            'binary' :
                {
                        'logistic'          : 'logistic',
                        'linear'            : 'logistic',
                        'logistic cv'       : 'logistic cv',
                        'linear cv'         : 'logistic cv',
                        'gaussian nb'       : 'gaussian nb',
                        'knn'               : 'knn classifier',
                        'knn classifier'    : 'knn classifier',
                        'knn gs'            : 'knn classifier gs',
                        'knn classifier gs' : 'knn classifier gs',
                        'dt'                : 'dt classifier',
                        'dt classifier'     : 'dt classifier',
                        'dt gs'             : 'dt classifier gs',
                        'dt classifier gs'  : 'dt classifier gs',
                        'rf'                : 'rf classifier',
                        'rf classifier'     : 'rf classifier',
                        'rf rs'             : 'rf classifier rs',
                        'rf classifier rs'  : 'rf classifier rs'
                },
            'regression' :
                {
                        'linear'            : 'linear regressor',
                        'linear regressor'  : 'linear regressor',
                        'knn'               : 'knn regressor',
                        'knn regressor'     : 'knn regressor',
                        'knn gs'            : 'knn regressor gs',
                        'knn regressor gs'  : 'knn regressor gs',
                        'elastic cv'        : 'elastic net cv',
                        'elastic net cv'    : 'elastic net cv',
                        'omp cv'            : 'omp cv',
                        'lars cv'           : 'lars cv',
                        'ridge cv'          : 'ridge cv',
                        'rf'                : 'rf regressor',
                        'rf regressor'      : 'rf regressor',
                        'rf rs'             : 'rf regressor rs',
                        'rf regressor rs'   : 'rf regressor rs',
                        'gp'                : 'gp regressor',
                        'gp regressor'      : 'gp regressor'
                },
            'categorical' : 
            {
                'multilabel': 
                {
                        'knn'               : 'knn classifier',
                        'knn classifier'    : 'knn classifier',
                        'knn gs'            : 'knn classifier gs',
                        'knn classifier gs' : 'knn classifier gs',
                        'dt'                : 'dt classifier',
                        'dt classifier'     : 'dt classifier',
                        'dt gs'             : 'dt classifier gs',
                        'dt classifier gs'  : 'dt classifier gs',
                        'rf'                : 'rf classifier',
                        'rf classifier'     : 'rf classifier',
                        'rf rs'             : 'rf classifier rs',
                        'rf classifier rs'  : 'rf classifier rs'
                },
                'multiclass':
                {
                        'logistic'          : 'logistic',
                        'linear'            : 'logistic',
                        'logistic cv'       : 'logistic cv',
                        'linear cv'         : 'logistic cv',
                        'gaussian nb'       : 'gaussian nb',
                        'knn'               : 'knn classifier',
                        'knn classifier'    : 'knn classifier',
                        'knn gs'            : 'knn classifier gs',
                        'knn classifier gs' : 'knn classifier gs',
                        'dt'                : 'dt classifier',
                        'dt classifier'     : 'dt classifier',
                        'dt gs'             : 'dt classifier gs',
                        'dt classifier gs'  : 'dt classifier gs',
                        'rf'                : 'rf classifier',
                        'rf classifier'     : 'rf classifier',
                        'rf rs'             : 'rf classifier rs',
                        'rf classifier rs'  : 'rf classifier rs'
                }
            }
        }

MODELS = {

    'logistic': (LogisticRegression, {'n_jobs': 'n_jobs',
                                      'class_weight': 'class_weight'}),

    'logistic cv' : (LogisticRegressionCV, {'cv': 'base_int_cv',
                                            'class_weight': 'class_weight',
                                            'max_iter': 5000,
                                            'n_jobs': 'n_jobs'}),
    
    'gaussian nb' : (GaussianNB, {}),
    
    'knn classifier' : (KNeighborsClassifier, {'n_jobs': 'n_jobs'}),
    
    'knn classifier gs' : (GridSearchCV, {'estimator': 'knn classifier',
                                          'param_grid' : DP.DEFAULT_KNN_GRID1,
                                          'scoring' : 'scorer',
                                          'cv': 'base_int_cv',
                                          'iid' : False,
                                          'n_jobs': 'n_jobs'}),

    'knn regressor' : (KNeighborsRegressor, {'n_jobs': 'n_jobs'}),

    'knn regressor gs' : (GridSearchCV, {'estimator': 'knn regressor',
                                         'param_grid' : DP.DEFAULT_KNN_GRID1,
                                         'scoring' : 'scorer',
                                         'cv': 'base_int_cv',
                                         'iid' : False,
                                         'n_jobs': 'n_jobs'}),

    'dt classifier' : (DecisionTreeClassifier, {'class_weight': 'class_weight'}),

    'dt classifier gs' : (GridSearchCV, {'estimator': 'dt classifier',
                                         'param_grid' : DP.DEFAULT_DTC_GRID1,
                                         'scoring' : 'scorer',
                                         'cv': 'base_int_cv',
                                         'iid' : False,
                                         'n_jobs': 'n_jobs'}),

    'linear regressor' : (LinearRegression, {'fit_intercept': True,
                                             'n_jobs': 'n_jobs'}),

    'elastic net cv' : (ElasticNetCV, {'cv': 'base_int_cv',
                                       'max_iter': 5000,
                                       'n_jobs': 'n_jobs'}),

    'omp cv' : (OrthogonalMatchingPursuitCV, {'cv': 'base_int_cv',
                                              'n_jobs': 'n_jobs'}),

    'lars cv' : (LarsCV, {'cv': 'base_int_cv',
                          'n_jobs': 'n_jobs'}),

    'ridge cv' : (RidgeCV, {'cv': 'base_int_cv',
                            'n_jobs': 'n_jobs'}),

    'rf regressor' : (RandomForestRegressor, {'n_estimators' : 100,
                                              'n_jobs': 'n_jobs'}),

    'rf regressor rs' : (RandomizedSearchCV, {'estimator': 'rf regressor',
                                              'param_distributions' : DP.DEFAULT_RF_GRID1,
                                              'n_iter': 10,
                                              'scoring' : 'scorer',
                                              'cv': 'base_int_cv',
                                              'iid' : False,
                                              'n_jobs' : 'n_jobs'}),

    'rf classifier' : (RandomForestClassifier, {'n_estimators' : 100,
                                                'n_jobs': 'n_jobs',
                                                'class_weight': 'class_weight'}),

    'rf classifier rs' : (RandomizedSearchCV, {'estimator': 'rf classifier',
                                               'param_distributions' : DP.DEFAULT_RF_GRID1,
                                               'n_iter': 10,
                                               'scoring' : 'scorer',
                                               'cv': 'base_int_cv',
                                               'iid' : False,
                                               'n_jobs' : 'n_jobs'}),

    'gp regressor' : (GaussianProcessRegressor, {'n_restarts_optimizer' : 5,
                                                 'normalize_y' : True})
    }
                   
   