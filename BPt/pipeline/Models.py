"""
Models.py
====================================
This file contains the different models avaliable for training,
with additional information on which work with which problem types
and default params.
"""

from sklearn.experimental import enable_hist_gradient_boosting
from ..extensions.MLP import MLPRegressor_Wrapper, MLPClassifier_Wrapper
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import (GaussianProcessClassifier,
                                      GaussianProcessRegressor)
from sklearn.ensemble import (GradientBoostingClassifier,
                              GradientBoostingRegressor,
                              RandomForestRegressor, RandomForestClassifier,
                              ExtraTreesClassifier, ExtraTreesRegressor,
                              HistGradientBoostingRegressor,
                              HistGradientBoostingClassifier)
from sklearn.linear_model import (LogisticRegression, ElasticNet,
                                  LinearRegression, HuberRegressor,
                                  Lasso, Ridge, SGDClassifier,
                                  SGDRegressor,
                                  PassiveAggressiveClassifier,
                                  BayesianRidge, ARDRegression,
                                  TweedieRegressor)

from sklearn.svm import SVC, LinearSVR, SVR, LinearSVC
from ..helpers.ML_Helpers import get_obj_and_params


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
        'rf':                 'random forest classifier',
        'random forest':      'random forest classifier',
        'gp':                 'gp classifier',
        'svm':                'svm classifier',
        'svc':                'svm classifier',
        'linear svm':         'linear svm classifier',
        'linear svc':         'linear svm classifier',
        'mlp':                'mlp classifier',
        'sgd':                'sgd classifier',
        'gb':                 'gb classifier',
        'hgb':                'hgb classifier',
        'et':                 'et classifier',
        'pa':                 'pa classifier',

    },
    'regression': {
        'linear':             'linear regressor',
        'knn':                'knn regressor',
        'dt':                 'dt regressor',
        'elastic':            'elastic net regressor',
        'elastic net':        'elastic net regressor',
        'rf':                 'random forest regressor',
        'random forest':      'random forest regressor',
        'gp':                 'gp regressor',
        'svm':                'svm regressor',
        'svr':                'svm regressor',
        'linear svm':         'linear svm regressor',
        'linear svr':         'linear svm regressor',
        'mlp':                'mlp regressor',
        'ridge':              'ridge regressor',
        'lasso':              'lasso regressor',
        'gb':                 'gb regressor',
        'hgb':                'hgb regressor',
        'et':                 'et regressor',
        'bayesian ridge':     'bayesian ridge regressor',
        'ard':                'ard regressor',
        'tweedie':            'tweedie regressor',
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
                                                  'elastic clf v2',
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
    'knn regressor': (KNeighborsRegressor, ['base knn regression',
                                            'knn dist regression']),

    'dt classifier': (DecisionTreeClassifier, ['default',
                                               'dt classifier dist']),
    'dt regressor':  (DecisionTreeRegressor, ['default', 'dt dist']),

    'linear regressor': (LinearRegression, ['base linear']),

    'random forest regressor': (RandomForestRegressor, ['base rf', 'rf dist']),
    'random forest classifier': (RandomForestClassifier,
                                 ['base rf regressor', 'rf classifier dist']),

    'gp regressor': (GaussianProcessRegressor, ['base gp regressor']),
    'gp classifier': (GaussianProcessClassifier,  ['base gp classifier']),

    'svm regressor': (SVR, ['base svm', 'svm dist']),
    'svm classifier': (SVC, ['base svm classifier', 'svm classifier dist']),

    'mlp regressor': (MLPRegressor_Wrapper, ['default', 'mlp dist 3 layer',
                                             'mlp dist es 3 layer',
                                             'mlp dist 2 layer',
                                             'mlp dist es 2 layer',
                                             'mlp dist 1 layer',
                                             'mlp dist es 1 layer']),
    'mlp classifier': (MLPClassifier_Wrapper, ['default', 'mlp dist 3 layer',
                                               'mlp dist es 3 layer',
                                               'mlp dist 2 layer',
                                               'mlp dist es 2 layer',
                                               'mlp dist 1 layer',
                                               'mlp dist es 1 layer']),

    'linear svm classifier': (LinearSVC, ['base linear svc',
                                          'linear svc dist']),
    'linear svm regressor': (LinearSVR, ['base linear svr',
                                         'linear svr dist']),

    'sgd classifier': (SGDClassifier, ['base sgd', 'sgd elastic classifier',
                                       'sgd classifier big search']),
    'sgd regressor': (SGDRegressor, ['base sgd', 'sgd elastic']),

    'gb classifier': (GradientBoostingClassifier, ['default']),
    'gb regressor': (GradientBoostingRegressor, ['default']),

    'hgb classifier': (HistGradientBoostingClassifier, ['default']),
    'hgb regressor': (HistGradientBoostingRegressor, ['default']),

    'et classifier': (ExtraTreesClassifier, ['default']),
    'et regressor': (ExtraTreesRegressor, ['default']),

    'pa classifier': (PassiveAggressiveClassifier, ['default']),

    'bayesian ridge regressor': (BayesianRidge, ['default']),
    
    'ard regressor': (ARDRegression, ['default']),

    'tweedie regressor': (TweedieRegressor, ['default']),
}

try:
    from xgboost import XGBClassifier, XGBRegressor

    AVALIABLE['binary']['xgb'] = 'xgb classifier'
    AVALIABLE['regression']['xgb'] = 'xgb regressor'
    AVALIABLE['categorical']['xgb'] = 'xgb classifier'

    MODELS['xgb regressor'] = (XGBRegressor, ['base xgb', 'xgb dist1',
                                              'xgb dist2', 'xgb dist3'])
    MODELS['xgb classifier'] = (XGBClassifier, ['base xgb classifier',
                                                'xgb classifier dist1',
                                                'xgb classifier dist2',
                                                'xgb classifier dist3'])

except ImportError:
    pass

try:
    from ..extensions.BPtLGBM import BPtLGBMRegressor, BPtLGBMClassifier

    AVALIABLE['binary']['light gbm'] = 'light gbm classifier'
    AVALIABLE['binary']['lgbm'] = 'light gbm classifier'
    AVALIABLE['categorical']['light gbm'] = 'light gbm classifier'
    AVALIABLE['categorical']['lgbm'] = 'light gbm classifier'
    AVALIABLE['regression']['light gbm'] = 'light gbm regressor'
    AVALIABLE['regression']['lgbm'] = 'light gbm regressor'

    MODELS['light gbm regressor'] = (BPtLGBMRegressor, ['base lgbm',
                                                        'lgbm dist1',
                                                        'lgbm dist2',
                                                        'lgbm dist3'])
    MODELS['light gbm classifier'] = (BPtLGBMClassifier,
                                      ['base lgbm',
                                       'lgbm classifier dist1',
                                       'lgbm classifier dist2',
                                       'lgbm classifier dist3'])
except ImportError:
    pass

try:
    from ..extensions.AutoGluon import AutoGluon

    AVALIABLE['binary']['auto gluon'] = 'auto gluon binary'
    AVALIABLE['categorical']['auto gluon'] = 'auto gluon categorical'
    AVALIABLE['regression']['auto gluon'] = 'auto gluon regression'

    MODELS['auto gluon binary'] = (AutoGluon, ['pt binary'])
    MODELS['auto gluon categorical'] = (AutoGluon, ['pt multiclass'])
    MODELS['auto gluon regression'] = (AutoGluon, ['pt regression'])

except ImportError:
    pass


def get_base_model_and_params(model_type, extra_params, model_type_params,
                              search_type, random_state=None,
                              num_feat_keys=None):

    model, extra_model_params, model_type_params =\
        get_obj_and_params(model_type, MODELS, extra_params, model_type_params)

    # Init model, w/ any user passed params + class params
    model = model(**extra_model_params)

    # Set random state on model, better to overset this
    if hasattr(model, 'random_state'):
        setattr(model, 'random_state', random_state)

    return model, model_type_params
