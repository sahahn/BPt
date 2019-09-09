"""
Models.py
====================================
This file contains the different models avaliable for training,
with additional information on which work with which problem types
and default params.
"""
from ABCD_ML.ML_Helpers import show_objects
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
                                  Lasso, Ridge)
from sklearn.svm import SVC, LinearSVR, SVR, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMRegressor, LGBMClassifier
from ABCD_ML.Early_Stop import EarlyStopLGBMRegressor

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
                        'light gbm':          'light gbm classifier',
                        'svm':                'svm classifier',
                        'mlp':                'mlp classifier',
                        'xgb':                'xgb classifier',
        },
        'regression': {
                        'linear':             'linear regressor',
                        'knn':                'knn regressor',
                        'dt':                 'dt regressor',
                        'elastic':            'elastic net regressor',
                        'elastic net':        'elastic net regressor',
                        'random forest':      'random forest regressor',
                        'gp':                 'gp regressor',
                        'light gbm':          'light gbm regressor',
                        'light gbm early stop':
                        'light gbm regressor early stop',
                        'svm':                'svm regressor',
                        'mlp':                'mlp regressor',
                        'ridge':              'ridge regressor',
                        'lasso':              'lasso regressor',
                        'xgb':                'xgb regressor',
        },
        'categorical': {
                'multilabel': {
                        'knn':                'knn classifier',
                        'dt':                 'dt classifier',
                        'random forest':      'random forest classifier',
                        'mlp':                'mlp classifier',
                }
        }
}

# Should be the same
AVALIABLE['categorical']['multiclass'] = AVALIABLE['binary'].copy()


MODELS = {
    'logistic': (LogisticRegression, ['base logistic']),

    'lasso logistic': (LogisticRegression, ['base lasso', 'lasso C']),

    'ridge logistic': (LogisticRegression, ['base ridge', 'ridge C']),

    'elastic net logistic': (LogisticRegression, ['base elastic',
                                                  'elastic classifier']),

    'elastic net regressor': (ElasticNet, ['base elastic net',
                                           'elastic regression']),

    'huber': (HuberRegressor, ['base huber']),

    'gaussian nb': (GaussianNB, ['base gnb']),

    'knn classifier': (KNeighborsClassifier, ['base knn', 'knn rs']),
    'knn regressor': (KNeighborsRegressor, ['base knn', 'knn rs']),

    'dt classifier': (DecisionTreeClassifier, ['base dt', 'dt rs']),
    'dt regressor':  (DecisionTreeRegressor, ['base dt', 'dt rs']),

    'linear regressor': (LinearRegression, ['base linear']),

    'random forest regressor': (RandomForestRegressor, ['base rf', 'rf rs']),
    'random forest classifier': (RandomForestClassifier, ['base rf', 'rf rs']),

    'light gbm regressor': (LGBMRegressor, ['base lgbm', 'lgbm rs', 'lgbm 2']),
    'light gbm classifier': (LGBMClassifier, ['base lgbm', 'lgbm rs',
                                              'lgbm 2']),

    'light gbm regressor early stop': (EarlyStopLGBMRegressor, ['base lgbm es',
                                                                'lgbm es rs']),

    'gp regressor': (GaussianProcessRegressor, ['base gp regressor']),
    'gp classifier': (GaussianProcessClassifier,  ['base gp classifier']),

    'svm regressor': (SVR, ['base svm', 'svm rs']),
    'svm classifier': (SVC, ['base svm classifier', 'svm classifier rs']),

    'mlp regressor': (MLPRegressor, ['base mlp', 'mlp rs', 'mlp rs es',
                                     'mlp layers search']),
    'mlp classifier': (MLPClassifier, ['base mlp', 'mlp rs', 'mlp rs es',
                                       'mlp layers search']),
    'ridge regressor': (Ridge, ['base ridge regressor', 'ridge regressor rs']),
    'lasso regressor': (Lasso, ['base lasso regressor', 'lasso regressor rs']),

    'xgb regressor': (XGBRegressor, ['base xgb', 'xgb rs']),
    'xgb classifier': (XGBClassifier, ['base xgb', 'xgb rs']),
    }


def Show_Models(self, problem_type=None, model_type_str=None,
                params_options=False, show_object=False,
                possible_params=False):
        '''Just calls Show_Model_Types.'''

        self.Show_Model_Types(problem_type, model_type_str, params_options,
                              show_object, possible_params)


def Show_Model_Types(self, problem_type=None, model_type_str=None,
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
                'categorical multilabel', 'categorical multiclass'
                or None.

                (default = None)

        model_type_str : str or list, optional
                If model type is passed, will just show the specific
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

        show_objects(problem_type, model_type_str, params_options,
                     show_object, possible_params, AVALIABLE, MODELS)
