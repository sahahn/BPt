"""
Models.py
====================================
This file contains the different models avaliable for training,
with additional information on which work with which problem types
and default params.
"""
from ABCD_ML.Default_Grids import get
from ABCD_ML.ML_Helpers import get_avaliable_by_type
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier
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
from sklearn.calibration import CalibratedClassifierCV

AVALIABLE = {
        'binary': {
                        'user passed':        'user passed',
                        'logistic':           'logistic',
                        'linear':             'logistic',
                        'lasso':              'lasso logistic',
                        'lasso logistic':     'lasso logistic',
                        'lasso rs':           'lasso logistic rs',
                        'lasso logistic':     'lasso logistic',
                        'lasso logistic rs':  'lasso logistic rs',
                        'ridge':              'ridge logistic',
                        'ridge rs':           'ridge logistic rs',
                        'ridge logistic':     'ridge logistic',
                        'ridge logistic rs':  'ridge logistic rs',
                        'elastic net':        'elastic net logistic',
                        'elastic net logistic': 'elastic net logistic',
                        'elastic net rs':     'elastic net logistic rs',
                        'elastic net logistic rs': 'elastic net logistic rs',
                        'gaussian nb':        'gaussian nb',
                        'knn':                'knn classifier',
                        'knn gs':             'knn classifier gs',
                        'dt':                 'dt classifier',
                        'dt gs':              'dt classifier gs',
                        'random forest':      'random forest classifier',
                        'random forest cal':  'random forest classifier cal',
                        'random forest rs':   'random forest classifier rs',
                        'gp':                 'gp classifier',
                        'light gbm':          'light gbm classifier',
                        'light gbm rs':       'light gbm classifier rs',
                        'svm':                'svm classifier',
                        'svm rs':             'svm classifier rs',
        },
        'regression': {
                        'user passed':        'user passed',
                        'linear':             'linear regressor',
                        'knn':                'knn regressor',
                        'knn gs':             'knn regressor gs',
                        'elastic net':        'elastic net',
                        'elastic net rs':     'elastic net rs',
                        'omp cv':             'omp cv',
                        'lars cv':            'lars cv',
                        'ridge cv':           'ridge cv',
                        'random forest':      'random forest regressor',
                        'random forest rs':   'random forest regressor rs',
                        'gp':                 'gp regressor',
                        'light gbm':          'light gbm regressor',
                        'svm':                'svm regressor',
                        'svm rs':             'svm regressor rs',
        },
        'categorical': {
                'multilabel': {
                        'user passed':        'user passed',
                        'knn':                'knn classifier',
                        'knn gs':             'knn classifier gs',
                        'dt':                 'dt classifier',
                        'dt gs':              'dt classifier gs',
                        'random forest':      'random forest classifier',
                        'random forest rs':   'random forest classifier rs',
                }
        }
}

# Should be the same
AVALIABLE['categorical']['multiclass'] = AVALIABLE['binary'].copy()


def get_search_params(grid_name, model_name, gs=False):

        params = {}
        params['iid'] = False
        params['estimator'] = model_name
        params['pre_dispatch'] = 'n_jobs - 2'

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


logistic_params = {'solver': 'saga',
                   'max_iter': 5000,
                   'multi_class': 'auto',
                   'penalty': 'none'}

lasso_params = logistic_params.copy()
lasso_params['penalty'] = 'l1'

ridge_params = logistic_params.copy()
ridge_params['penalty'] = 'l2'

elastic_params = logistic_params.copy()
elastic_params['penalty'] = 'elasticnet'
elastic_params['l1_ratio'] = .5


MODELS = {
    'logistic': (LogisticRegression, logistic_params),

    'lasso logistic': (LogisticRegression, lasso_params),

    'ridge logistic': (LogisticRegression, ridge_params),

    'elastic net logistic': (LogisticRegression, elastic_params),

    'lasso logistic rs': get_rs_tuple('REGRESSION1', 'lasso logistic'),

    'ridge logistic rs': get_rs_tuple('REGRESSION1', 'ridge logistic'),

    'elastic net logistic rs': get_rs_tuple('ELASTIC1',
                                            'elastic net logistic'),

    'elastic net': (ElasticNet, {'max_iter': 5000}),

    'elastic net rs': get_rs_tuple('ELASTIC2', 'elastic net'),

    'huber': (HuberRegressor, {}),

    'gaussian nb': (GaussianNB, {}),

    'knn classifier': (KNeighborsClassifier, {'n_jobs': 'n_jobs'}),

    'knn classifier gs': get_gs_tuple('KNN1', 'knn classifier'),

    'knn regressor': (KNeighborsRegressor, {}),

    'knn regressor gs': get_gs_tuple('KNN1', 'knn regressor'),

    'dt classifier': (DecisionTreeClassifier, {}),

    'dt classifier gs': get_gs_tuple('DTC1', 'dt classifier'),

    'linear regressor': (LinearRegression, {'fit_intercept': True}),

    'omp cv': (OrthogonalMatchingPursuitCV, {}),

    'lars cv': (LarsCV, {}),

    'ridge cv': (RidgeCV, {}),

    'random forest regressor': (RandomForestRegressor, {'n_estimators': 100}),

    'random forest regressor rs': get_rs_tuple('RF1',
                                               'random forest regressor'),

    'random forest classifier': (RandomForestClassifier,
                                 {'n_estimators': 100}),

    'random forest classifier cal': (CalibratedClassifierCV,
                                     {'base_estimator':
                                      'random forest classifier'}),

    'random forest classifier rs': get_rs_tuple('RF1',
                                                'random forest classifier'),

    'light gbm regressor': (LGBMRegressor, {'silent': True}),

    'light gbm regressor rs': get_rs_tuple('LIGHT1', 'light gbm regressor'),

    'light gbm classifier': (LGBMClassifier, {'silent': True}),

    'light gbm classifier rs': get_rs_tuple('LIGHT1', 'light gbm classifier'),

    'gp regressor': (GaussianProcessRegressor, {'n_restarts_optimizer': 5,
                                                'normalize_y': True}),

    'gp classifier': (GaussianProcessClassifier, {'n_restarts_optimizer': 5}),

    'svm regressor': (SVR, {'kernel': 'rbf'}),

    'svm regressor rs': get_rs_tuple('SVM1', 'svm regressor'),

    'svm classifier': (SVC, {'kernel': 'rbf'}),

    'svm classifier rs': get_rs_tuple('SVM1', 'svm classifier'),
    }


def show_models(self, problem_type=None, show_model_help=False,
                show_default_params=False, show_grid_params=False):
        '''Just calls show_model_types.'''

        self.show_model_types(problem_type, show_model_help,
                              show_default_params, show_grid_params)


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
                if model != 'user passed':
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
