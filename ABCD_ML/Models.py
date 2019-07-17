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
from sklearn.svm import SVC, LinearSVR, SVR, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV

AVALIABLE = {
        'binary': {
                        'logistic':           'logistic',
                        'linear':             'logistic',
                        'logistic cv':        'logistic cv',
                        'linear cv':          'logistic cv',
                        'gaussian nb':        'gaussian nb',
                        'knn':                'knn classifier',
                        'knn gs':             'knn classifier gs',
                        'dt':                 'dt classifier',
                        'dt gs':              'dt classifier gs',
                        'random forest':      'random forest classifier',
                        'random forest cal':  'random forest classifier cal',
                        'random forest rs':   'random forest classifier rs',
                        'light gbm':          'light gbm classifier',
                        'light gbm rs':       'light gbm classifier rs',
                        'svm':                'svm classifier',
                        'svm gs':             'svm classifier gs',
        },
        'regression': {
                        'linear':             'linear regressor',
                        'knn':                'knn regressor',
                        'knn gs':             'knn regressor gs',
                        'elastic cv':         'elastic net cv',
                        'elastic net cv':     'elastic net cv',
                        'omp cv':             'omp cv',
                        'lars cv':            'lars cv',
                        'ridge cv':           'ridge cv',
                        'random forest':      'random forest regressor',
                        'random forest rs':   'random forest regressor rs',
                        'gp':                 'gp regressor',
                        'light gbm':          'light gbm regressor',
                        'svm':                'svm regressor',
                        'svm gs':             'svm regressor gs',
        },
        'categorical': {
                'multilabel': {
                        'knn':                'knn classifier',
                        'knn gs':             'knn classifier gs',
                        'dt':                 'dt classifier',
                        'dt gs':              'dt classifier gs',
                        'random forest':      'random forest classifier',
                        'random forest rs':   'random forest classifier rs',
                },
                'multiclass': {
                        'logistic':           'logistic',
                        'linear':             'logistic',
                        'logistic cv':        'logistic cv',
                        'linear cv':          'logistic cv',
                        'gaussian nb':        'gaussian nb',
                        'knn':                'knn classifier',
                        'knn gs':             'knn classifier gs',
                        'dt':                 'dt classifier',
                        'dt gs':              'dt classifier gs',
                        'random forest':      'random forest classifier',
                        'random forest cal':  'random forest classifier cal',
                        'random forest rs':   'random forest classifier rs',
                        'light gbm':          'light gbm classifier',
                        'light gbm rs':       'light gbm classifier rs',
                        'svm':                'svm classifier',
                        'svm gs':             'svm classifier gs',
                }
        }
}

# The different models below are contained in a dictionary,
# where each entry has a saved model and default params.
MODELS = {
    'logistic': (LogisticRegression, {'solver': 'lbfgs',
                                      'max_iter': 5000,
                                      'multi_class': 'auto'}),

    'logistic cv': (LogisticRegressionCV, {'max_iter': 5000,
                                           'multi_class': 'auto'}),

    'gaussian nb': (GaussianNB, {}),

    'knn classifier': (KNeighborsClassifier, {'n_jobs': 'n_jobs'}),

    'knn classifier gs': (GridSearchCV, {'estimator': 'knn classifier',
                                         'param_grid': DG.KNN1,
                                         'iid': False}),

    'knn regressor': (KNeighborsRegressor, {}),

    'knn regressor gs': (GridSearchCV, {'estimator': 'knn regressor',
                                        'param_grid': DG.KNN1,
                                        'iid': False}),

    'dt classifier': (DecisionTreeClassifier, {}),

    'dt classifier gs': (GridSearchCV, {'estimator': 'dt classifier',
                                        'param_grid': DG.DTC1,
                                        'iid': False}),

    'linear regressor': (LinearRegression, {'fit_intercept': True}),

    'elastic net cv': (ElasticNetCV, {'max_iter': 5000}),

    'omp cv': (OrthogonalMatchingPursuitCV, {}),

    'lars cv': (LarsCV, {}),

    'ridge cv': (RidgeCV, {}),

    'random forest regressor': (RandomForestRegressor, {'n_estimators': 100}),

    'random forest regressor rs': (RandomizedSearchCV,
                                   {'estimator': 'random forest regressor',
                                    'param_distributions': DG.RF1,
                                    'iid': False}),

    'random forest classifier': (RandomForestClassifier,
                                 {'n_estimators': 100}),

    'random forest classifier cal': (CalibratedClassifierCV,
                                     {'base_estimator':
                                      'random forest classifier'}),

    'random forest classifier rs': (RandomizedSearchCV,
                                    {'estimator': 'random forest classifier',
                                     'param_distributions': DG.RF1,
                                     'iid': False}),

    'light gbm regressor': (LGBMRegressor, {'silent': True}),

    'light gbm regressor rs': (RandomizedSearchCV, {'estimator':
                                                    'light gbm regressor',
                                                    'param_distributions':
                                                    DG.LIGHT1,
                                                    'iid': False}),

    'light gbm classifier': (LGBMClassifier, {'silent': True}),

    'light gbm classifier rs': (RandomizedSearchCV, {'estimator':
                                                     'light gbm classifier',
                                                     'param_distributions':
                                                     DG.LIGHT1,
                                                     'iid': False}),

    'gp regressor': (GaussianProcessRegressor, {'n_restarts_optimizer': 5,
                                                'normalize_y': True}),

    'svm regressor': (SVR, {'kernel': 'rbf'}),

    'svm regressor rs': (RandomizedSearchCV, {'estimator':
                                              'svm regressor',
                                              'param_distributions':
                                              DG.SVM1, 'iid': False}),

    'svm classifier': (SVC, {'kernel': 'rbf'}),

    'svm classifier rs': (RandomizedSearchCV, {'estimator':
                                               'svm classifier',
                                               'param_distributions':
                                               DG.SVM1, 'iid': False}),
    }


def show_model_types(self, problem_type=None, show_model_help=False,
                     show_default_params=False, show__params=False):
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
