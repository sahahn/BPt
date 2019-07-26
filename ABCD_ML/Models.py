"""
Models.py
====================================
This file contains the different models avaliable for training,
with additional information on which work with which problem types
and default params.
"""
from ABCD_ML.ML_Helpers import get_avaliable_by_type
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import (GaussianProcessClassifier,
                                      GaussianProcessRegressor)
from sklearn.ensemble import (GradientBoostingClassifier, AdaBoostClassifier,
                              RandomForestRegressor, RandomForestClassifier)
from sklearn.linear_model import (LogisticRegression, ElasticNet,
                                  LinearRegression, HuberRegressor,
                                  OrthogonalMatchingPursuitCV, LarsCV, RidgeCV)
from sklearn.svm import SVC, LinearSVR, SVR, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

AVALIABLE = {
        'binary': {
                        'user passed':        'user passed',
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
        },
        'regression': {
                        'user passed':        'user passed',
                        'linear':             'linear regressor',
                        'knn':                'knn regressor',
                        'dt':                 'dt regressor',
                        'elastic':            'elastic net regressor'
                        'elastic net':        'elastic net regressor',
                        'random forest':      'random forest regressor',
                        'gp':                 'gp regressor',
                        'light gbm':          'light gbm regressor',
                        'svm':                'svm regressor',
        },
        'categorical': {
                'multilabel': {
                        'user passed':        'user passed',
                        'knn':                'knn classifier',
                        'dt':                 'dt classifier',
                        'random forest':      'random forest classifier',
                }
        }
}

# Should be the same
AVALIABLE['categorical']['multiclass'] = AVALIABLE['binary'].copy()


def get_search_params(grid_name, model_name, gs=False):

        params = {}
        params['iid'] = False
        params['estimator'] = model_name
        params['pre_dispatch'] = 'n_jobs - 1'

        if gs:
                params['param_grid'] = get(grid_name, model_name)
        else:
                params['param_distributions'] = get(grid_name, model_name)

        return params


def get_rs_tuple(grid_name, model_name):
        return (RandomizedSearchCV, get_search_params(grid_name, model_name,
                gs=False))


def get_gs_tuple(grid_name, model_name):
        return (GridSearchCV, get_search_params(grid_name, model_name,
                gs=True))


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

    'light gbm regressor': (LGBMRegressor, ['base lgbm', 'lgbm rs']),
    'light gbm classifier': (LGBMClassifier, ['base lgbm', 'lgbm rs']),

    'gp regressor': (GaussianProcessRegressor, ['base gp regressor']),
    'gp classifier': (GaussianProcessClassifier,  ['base gp classifier']),

    'svm regressor': (SVR, {'kernel': 'rbf', 'gamma': 'scale'}),
    'svm classifier': (SVC, {'kernel': 'rbf', 'gamma': 'scale'}),
    }


def Show_Models(self, problem_type=None, show_model_help=False,
                show_default_params=False, show_grid_params=False):
        '''Just calls Show_Model_Types.'''

        self.Show_Model_Types(problem_type, show_model_help,
                              show_default_params, show_grid_params)


def Show_Model_Types(self, problem_type=None, show_model_help=False,
                     show_default_params=False, show_grid_params=False):
        '''Print out the avaliable machine learning models,
        optionally restricted by problem type + other diagnostic args.

        Parameters
        ----------
        problem_type : {binary, categorical, regression, None}, optional
            Where `problem_type` is the underlying ML problem

            (default = None)

        show_model_help : bool, optional
            Flag, if set to True, then will display the full docstring
            for each model, note: this is pretty terrible to read.

            (default = False)

        show_default_params : bool, optional
            Flag, if set to True, then will display the ABCD_ML
            default parameters for each model.

            (default = False)

        show_grid_params : bool, optional
            Flag, if set to True, and `show_default_params` set to True, then
            when displaying default params for each model will print out the
            grid params also, by default it will skip these as the output is
            messy.

            (default = False)
        '''

        print('Note: gs and rs are  Grid Search and Random Search')
        print('Models with gs or rs will have their hyper-parameters',
              'tuned accordingly.')
        print()

        avaliable_by_type = get_avaliable_by_type(AVALIABLE)

        if problem_type is None:
                for pt in avaliable_by_type:
                        show_type(pt, avaliable_by_type, show_model_help,
                                  show_default_params, show_grid_params)
        else:
                show_type(problem_type, avaliable_by_type, show_model_help,
                          show_default_params, show_grid_params)


def show_type(problem_type, avaliable_by_type, show_model_help,
              show_default_params, show_grid_params):

        print('Problem Type:', problem_type)
        print('----------------------')
        print('Avaliable models: ')
        print()

        for model in avaliable_by_type[problem_type]:
                if 'user passed' not in model:
                        show_model(model, show_model_help, show_default_params,
                                   show_grid_params)
                        print()


def show_model(model, show_model_help, show_default_params, show_grid_params):

        multilabel, multiclass = False, False

        if 'multilabel ' in model:
                multilabel = True
        elif 'multiclass ' in model:
                multiclass = True

        model = model.replace('multilabel ', '')
        model = model.replace('multiclass ', '')

        print('Model str indicator: ', model)

        if multilabel:
                print('(MultiLabel)')
        elif multiclass:
                print('(MultiClass)')

        M = MODELS[model]
        print('Model object: ', M[0])

        if show_model_help:
                print(help(M[0]))

        if show_default_params:
                print('Default Params: ')
                for p in M[1]:
                        if (p == 'param_distributions' or p == 'param_grid') \
                                and show_grid_params is False:
                                print('Param grid not shown.')
                        else:
                                print(p, ':', M[1][p])
