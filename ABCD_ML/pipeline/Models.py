"""
Models.py
====================================
This file contains the different models avaliable for training,
with additional information on which work with which problem types
and default params.
"""
from sklearn.neural_network import MLPRegressor, MLPClassifier
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
from ..helpers.ML_Helpers import show_objects

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

        'multilabel': {
                        'knn':                'knn classifier',
                        'dt':                 'dt classifier',
                        'random forest':      'random forest classifier',
                        'mlp':                'mlp classifier',
        }
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

    'ridge regressor': (Ridge, ['base ridge',
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

    'mlp regressor': (MLPRegressor, ['base mlp', 'mlp dist 1 layer',
                                     'mlp dist es 1 layer']),
    'mlp classifier': (MLPClassifier, ['base mlp', 'mlp dist 1 layer',
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


def Show_Models(self, problem_type=None, model=None,
                params_options=False, show_object=False,
                possible_params=False):
        '''Print out the avaliable machine learning models,
        optionally restricted by problem type and other diagnostic args.

        Parameters
        ----------
        problem_type : str, optional
                Where `problem_type` is the underlying ML problem or
                rather type of problem...
                Note enter either 'binary', 'regression',
                'categorical', 'multilabel'
                or None.

                (default = None)

        model : str or list, optional
                If model is passed, will just show the specific
                model, according to the rest of the params passed.
                Note : You must pass the specific model indicator str
                limited preproc will be done on this input!
                If list, will show all models within list

                (default = None)

        params_options : bool, optional
            Flag, if set to True, then will display the ABCD_ML
            param ind options for each model.

                (default = False)

        show_object : bool, optional
                Flag, if set to True, then will print the
                raw model object.

                (default = False)

        possible_params: bool, optional
                Flag, if set to True, then will print all
                possible arguments to the classes __init__

                (default = False)
        '''

        print('Visit the sklearn documentation for more info on most of',
              'the dif. models')
        print('Note: Param distributions with a Rand Distribution')
        print('cannot be used in search_type = "grid"')
        print('More information through this function is avaliable')
        print('By passing optional extra optional params! Please view',
              'the help function for more info!')
        print('Note: the str indicator actually passed during Evaluate / Test')
        print('is listed as ("str indicator")')
        print()

        show_objects(problem_type, model, params_options,
                     show_object, possible_params, AVALIABLE, MODELS)
