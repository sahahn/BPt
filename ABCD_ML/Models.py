"""
Models.py
====================================
This file contains the different models avaliable for training,
with additional information on which work with which problem types
and default params.
"""
import ABCD_ML.Default_Grids as DG
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import (GaussianProcessClassifier,
                                      GaussianProcessRegressor)
from sklearn.ensemble import (GradientBoostingClassifier, AdaBoostClassifier,
                              RandomForestRegressor, RandomForestClassifier)
from sklearn.linear_model import (LogisticRegression, LogisticRegressionCV,
                                  ElasticNetCV, LinearRegression,
                                  OrthogonalMatchingPursuitCV, LarsCV, RidgeCV)
from sklearn.svm import SVC, LinearSVR
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

AVALIABLE = {
        'binary': {
                        'logistic':           'logistic',
                        'linear':             'logistic',
                        'logistic cv':        'logistic cv',
                        'linear cv':          'logistic cv',
                        'gaussian nb':        'gaussian nb',
                        'knn':                'knn classifier',
                        'knn classifier':     'knn classifier',
                        'knn gs':             'knn classifier gs',
                        'knn classifier gs':  'knn classifier gs',
                        'dt':                 'dt classifier',
                        'dt classifier':      'dt classifier',
                        'dt gs':              'dt classifier gs',
                        'dt classifier gs':   'dt classifier gs',
                        'rf':                 'rf classifier',
                        'rf classifier':      'rf classifier',
                        'rf rs':              'rf classifier rs',
                        'rf classifier rs':   'rf classifier rs',
                        'lgbm classifier':    'lgbm classifier',
                        'lgbm classifier rs': 'lgbm classifier rs'
        },
        'regression': {
                        'linear':             'linear regressor',
                        'linear regressor':   'linear regressor',
                        'knn':                'knn regressor',
                        'knn regressor':      'knn regressor',
                        'knn gs':             'knn regressor gs',
                        'knn regressor gs':   'knn regressor gs',
                        'elastic cv':         'elastic net cv',
                        'elastic net cv':     'elastic net cv',
                        'omp cv':             'omp cv',
                        'lars cv':            'lars cv',
                        'ridge cv':           'ridge cv',
                        'rf':                 'rf regressor',
                        'rf regressor':       'rf regressor',
                        'rf rs':              'rf regressor rs',
                        'rf regressor rs':    'rf regressor rs',
                        'gp':                 'gp regressor',
                        'gp regressor':       'gp regressor',
                        'lgbm regressor':     'lgbm regressor',
        },
        'categorical': {
                'multilabel': {
                        'knn':                'knn classifier',
                        'knn classifier':     'knn classifier',
                        'knn gs':             'knn classifier gs',
                        'knn classifier gs':  'knn classifier gs',
                        'dt':                 'dt classifier',
                        'dt classifier':      'dt classifier',
                        'dt gs':              'dt classifier gs',
                        'dt classifier gs':   'dt classifier gs',
                        'rf':                 'rf classifier',
                        'rf classifier':      'rf classifier',
                        'rf rs':              'rf classifier rs',
                        'rf classifier rs':   'rf classifier rs',
                        'lgbm classifier':    'lgbm classifier',
                        'lgbm classifier rs': 'lgbm classifier rs'
                },
                'multiclass': {
                        'logistic':           'logistic',
                        'linear':             'logistic',
                        'logistic cv':        'logistic cv',
                        'linear cv':          'logistic cv',
                        'gaussian nb':        'gaussian nb',
                        'knn':                'knn classifier',
                        'knn classifier':     'knn classifier',
                        'knn gs':             'knn classifier gs',
                        'knn classifier gs':  'knn classifier gs',
                        'dt':                 'dt classifier',
                        'dt classifier':      'dt classifier',
                        'dt gs':              'dt classifier gs',
                        'dt classifier gs':   'dt classifier gs',
                        'rf':                 'rf classifier',
                        'rf classifier':      'rf classifier',
                        'rf rs':              'rf classifier rs',
                        'rf classifier rs':   'rf classifier rs',
                        'lgbm classifier':    'lgbm classifier',
                        'lgbm classifier rs': 'lgbm classifier rs'
                }
        }
}

# The different models below are contained in a dictionary,
# where each entry has a saved model and default params.
MODELS = {
    'logistic': (LogisticRegression, {'solver': 'lbfgs',
                                      'max_iter': 5000}),

    'logistic cv': (LogisticRegressionCV, {'max_iter': 5000}),

    'gaussian nb': (GaussianNB, {}),

    'knn classifier': (KNeighborsClassifier, {'n_jobs': 'n_jobs'}),

    'knn classifier gs': (GridSearchCV, {'estimator': 'knn classifier',
                                         'param_grid': DG.KNN_GRID1,
                                         'iid': False}),

    'knn regressor': (KNeighborsRegressor, {}),

    'knn regressor gs': (GridSearchCV, {'estimator': 'knn regressor',
                                        'param_grid': DG.KNN_GRID1,
                                        'iid': False}),

    'dt classifier': (DecisionTreeClassifier, {}),

    'dt classifier gs': (GridSearchCV, {'estimator': 'dt classifier',
                                        'param_grid': DG.DTC_GRID1,
                                        'iid': False}),

    'linear regressor': (LinearRegression, {'fit_intercept': True}),

    'elastic net cv': (ElasticNetCV, {'max_iter': 5000}),

    'omp cv': (OrthogonalMatchingPursuitCV, {}),

    'lars cv': (LarsCV, {}),

    'ridge cv': (RidgeCV, {}),

    'rf regressor': (RandomForestRegressor, {'n_estimators': 100}),

    'rf regressor rs': (RandomizedSearchCV, {'estimator': 'rf regressor',
                                             'param_distributions':
                                             DG.RF_GRID1,
                                             'iid': False}),

    'rf classifier': (RandomForestClassifier, {'n_estimators': 100})

    'rf classifier rs': (RandomizedSearchCV, {'estimator': 'rf classifier',
                                              'param_distributions':
                                              DG.RF_GRID1,
                                              'iid': False}),

    'lgbm regressor': (LGBMRegressor, {'silent': True}),

    'lgbm regressor rs': (RandomizedSearchCV, {'estimator':
                                               'lightgbm regressor',
                                               'param_distributions':
                                               DG.LIGHT_GRID1,
                                               'iid': False}),

    'lgbm classifier': (LGBMClassifier, {'silent': True}),

    'lgbm classifier rs': (RandomizedSearchCV, {'estimator':
                                                'lightgbm classifier',
                                                'param_distributions':
                                                DG.LIGHT_GRID1,
                                                'iid': False}),

    'gp regressor': (GaussianProcessRegressor, {'n_restarts_optimizer': 5,
                                                'normalize_y': True})
    }


def show_model_types(self, problem_type=None, show_model_help=False,
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

        avaliable_by_type = get_avaliable_by_type()

        if problem_type is None:
                for pt in avaliable_by_type:
                        show_type(pt, avaliable_by_type, show_model_help,
                                  show_default_params, show_grid_params)
        else:
                show_type(problem_type, avaliable_by_type, show_model_help,
                          show_default_params, show_grid_params)


def get_avaliable_by_type():

        avaliable_by_type = {}

        for pt in AVALIABLE:

                avaliable_by_type[pt] = set()

                if pt == 'categorical':
                        for st in AVALIABLE[pt]:
                                for model in AVALIABLE[pt][st]:
                                        avaliable_by_type[pt].add(st + ' ' +
                                                                  AVALIABLE[pt]
                                                                  [st][model])

                else:
                        for model in AVALIABLE[pt]:
                                avaliable_by_type[pt].add(AVALIABLE[pt][model])

                avaliable_by_type[pt] = list(avaliable_by_type[pt])
                avaliable_by_type[pt].sort()

        return avaliable_by_type


def show_type(problem_type, avaliable_by_type, show_model_help,
              show_default_params, show_grid_params):

        print('Problem Type:', problem_type)
        print('----------------------')
        print('Avaliable models: ')
        print()

        for model in avaliable_by_type[problem_type]:
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
