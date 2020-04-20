"""
Models.py
====================================
This file contains the different models avaliable for training,
with additional information on which work with which problem types
and default params.
"""

from ..extensions.MLP import MLPRegressor_Wrapper, MLPClassifier_Wrapper
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import (GaussianProcessClassifier,
                                      GaussianProcessRegressor)
from sklearn.ensemble import (GradientBoostingClassifier, AdaBoostClassifier,
                              RandomForestRegressor, RandomForestClassifier)
from sklearn.linear_model import (LogisticRegression, ElasticNet,
                                  LinearRegression, HuberRegressor,
                                  Lasso, Ridge, RidgeClassifier, SGDClassifier)

from sklearn.svm import SVC, LinearSVR, SVR, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from ..helpers.ML_Helpers import show_objects, get_obj_and_params

AVALIABLE = {
        'binary': {
                        'logistic':           'logistic',
                        'linear':             'logistic',
                        'lasso':              'lasso logistic',
                        'ridge':              'ridge logistic',
                        'elastic':            'elastic net logistic',
                        'elastic net':        'elastic net logistic',
                        'gaussian nb':        'gaussian nb',
                        'knn':                'knn classifier',
                        'dt':                 'dt classifier',
                        'random forest':      'random forest classifier',
                        'gp':                 'gp classifier',
                        'svm':                'svm classifier',
                        'linear svm':         'linear svm classifier',
                        'mlp':                'mlp classifier',
                        'sgd':                'sgd classifier',
        },
        'regression': {
                        'linear':             'linear regressor',
                        'knn':                'knn regressor',
                        'dt':                 'dt regressor',
                        'elastic':            'elastic net regressor',
                        'elastic net':        'elastic net regressor',
                        'random forest':      'random forest regressor',
                        'gp':                 'gp regressor',
                        'svm':                'svm regressor',
                        'linear svm':         'linear svm regressor',
                        'mlp':                'mlp regressor',
                        'ridge':              'ridge regressor',
                        'lasso':              'lasso regressor',
        },
}

# Should be the same
AVALIABLE['categorical'] = AVALIABLE['binary'].copy()


MODELS = {
    'logistic': (LogisticRegression, ['base logistic']),

    'lasso logistic': (LogisticRegression, ['base lasso', 'lasso C',
                                            'lasso C extra']),

    'ridge logistic': (LogisticRegression, ['base ridge', 'ridge C',
                                            'ridge C extra']),

    'elastic net logistic': (LogisticRegression, ['base elastic',
                                                  'elastic classifier',
                                                  'elastic classifier extra']),

    'elastic net regressor': (ElasticNet, ['base elastic net',
                                           'elastic regression',
                                           'elastic regression extra']),

    'ridge regressor': (Ridge, ['base ridge regressor',
                                'ridge regressor dist']),
    'lasso regressor': (Lasso, ['base lasso regressor',
                                'lasso regressor dist']),

    'huber': (HuberRegressor, ['base huber']),

    'gaussian nb': (GaussianNB, ['base gnb']),

    'knn classifier': (KNeighborsClassifier, ['base knn', 'knn dist']),
    'knn regressor': (KNeighborsRegressor, ['base knn', 'knn dist']),

    'dt classifier': (DecisionTreeClassifier, ['base dt',
                                               'dt classifier dist']),
    'dt regressor':  (DecisionTreeRegressor, ['base dt', 'dt dist']),

    'linear regressor': (LinearRegression, ['base linear']),

    'random forest regressor': (RandomForestRegressor, ['base rf', 'rf dist']),
    'random forest classifier': (RandomForestClassifier,
                                 ['base rf', 'rf classifier dist']),

    'gp regressor': (GaussianProcessRegressor, ['base gp regressor']),
    'gp classifier': (GaussianProcessClassifier,  ['base gp classifier']),

    'svm regressor': (SVR, ['base svm', 'svm dist']),
    'svm classifier': (SVC, ['base svm classifier', 'svm classifier dist']),

    'mlp regressor': (MLPRegressor_Wrapper, ['base mlp', 'mlp dist 3 layer',
                                             'mlp dist es 3 layer', 'mlp dist 2 layer',
                                             'mlp dist es 2 layer', 'mlp dist 1 layer',
                                             'mlp dist es 1 layer']),
    'mlp classifier': (MLPClassifier_Wrapper, ['base mlp', 'mlp dist 3 layer',
                                               'mlp dist es 3 layer', 'mlp dist 2 layer',
                                               'mlp dist es 2 layer', 'mlp dist 1 layer',
                                               'mlp dist es 1 layer']),

    'linear svm classifier': (LinearSVC, ['base linear svc',
                                          'linear svc dist']),
    'linear svm regressor': (LinearSVR, ['base linear svr',
                                         'linear svr dist']),

    'sgd classifier': (SGDClassifier, ['base sgd', 'sgd classifier']),
    }

try:
        from xgboost import XGBClassifier, XGBRegressor

        AVALIABLE['binary']['xgb'] = 'xgb classifier'
        AVALIABLE['regression']['xgb'] = 'xgb regressor'
        AVALIABLE['categorical']['light gbm'] = 'light gbm classifier'

        MODELS['xgb regressor'] = (XGBRegressor, ['base xgb', 'xgb dist1',
                                                  'xgb dist2', 'xgb dist3'])
        MODELS['xgb classifier'] = (XGBClassifier, ['base xgb classifier',
                                                    'xgb classifier dist1',
                                                    'xgb classifier dist2',
                                                    'xgb classifier dist3'])

except ImportError:
        pass

try:
        from lightgbm import LGBMRegressor, LGBMClassifier

        AVALIABLE['binary']['light gbm'] = 'light gbm classifier'
        AVALIABLE['regression']['light gbm'] = 'light gbm regressor'
        AVALIABLE['categorical']['light gbm'] = 'light gbm classifier'

        MODELS['light gbm regressor'] = (LGBMRegressor, ['base lgbm',
                                                         'lgbm dist1',
                                                         'lgbm dist2'])
        MODELS['light gbm classifier'] = (LGBMClassifier,
                                          ['base lgbm',
                                           'lgbm classifier dist1',
                                           'lgbm classifier dist2'])
except ImportError:
        pass


def get_base_model_and_params(model_type, extra_params, model_type_params,
                              search_type, random_state=None, num_feat_keys=None):

        model, extra_model_params, model_type_params =\
            get_obj_and_params(model_type, MODELS, extra_params,
                               model_type_params, search_type)

        # Init model, w/ any user passed params + class params
        model = model(**extra_model_params)

        return model, model_type_params
