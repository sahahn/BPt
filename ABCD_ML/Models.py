"""
Models.py
====================================
This file contains the different models avaliable for training,
with additional information on which work with which problem types
and default params.
"""
from ABCD_ML.ML_Helpers import (get_avaliable_by_type, show_param_options,
                                get_possible_init_params)
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
                                  OrthogonalMatchingPursuitCV, LarsCV, RidgeCV)
from sklearn.svm import SVC, LinearSVR, SVR, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from ABCD_ML.Early_Stop import EarlyStopLGBMRegressor

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
                        'mlp':                'mlp classifier',
        },
        'regression': {
                        'user passed':        'user passed',
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
        },
        'categorical': {
                'multilabel': {
                        'user passed':        'user passed',
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

    'light gbm regressor': (LGBMRegressor, ['base lgbm', 'lgbm rs']),
    'light gbm classifier': (LGBMClassifier, ['base lgbm', 'lgbm rs']),

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
    }


def Show_Models(self, problem_type=None, model_type=None,
                show_param_ind_options=True, show_model_object=False,
                show_all_possible_params=False):
        '''Just calls Show_Model_Types.'''

        self.Show_Model_Types(problem_type, model_type, show_param_ind_options,
                              show_model_object, show_all_possible_params)


def Show_Model_Types(self, problem_type=None, model_type=None,
                     show_param_ind_options=True, show_model_object=False,
                     show_all_possible_params=False):
        '''Print out the avaliable machine learning models,
        optionally restricted by problem type + other diagnostic args.

        Parameters
        ----------
        problem_type : {binary, categorical, regression, None}, optional
                Where `problem_type` is the underlying ML problem

                (default = None)

        model_type : str or list, optional
                If model type is passed, will just show the specific
                model, according to the rest of the params passed.
                Note : You must pass the specific model indicator str
                limited preproc will be done on this input!
                If list, will show all models within list

                (default = None)

        show_param_ind_options : bool, optional
            Flag, if set to True, then will display the ABCD_ML
            param ind options for each model.

                (default = True)

        show_model_object : bool, optional
                Flag, if set to True, then will print the
                raw model object.

                (default = False)

        show_all_possible_params: bool, optional
                Flag, if set to True, then will print all
                possible arguments to the classes __init__

                (default = False)
        '''

        print('Note: Param distributions with a Rand Distribution')
        print('cannot be used in search_type = "grid"')
        print()

        if model_type is not None:
                if isinstance(model_type, str):
                        model_type = [model_type]
                for model_str in model_type:
                        show_model(model_str, show_param_ind_options,
                                   show_model_object, show_all_possible_params)
                return

        avaliable_by_type = get_avaliable_by_type(AVALIABLE)

        if problem_type is None:
                for pt in avaliable_by_type:
                        show_type(pt, avaliable_by_type,
                                  show_param_ind_options, show_model_object,
                                  show_all_possible_params)
        else:
                show_type(problem_type, avaliable_by_type,
                          show_param_ind_options, show_model_object,
                          show_all_possible_params)


def show_type(problem_type, avaliable_by_type, show_param_ind_options,
              show_model_object, show_all_possible_params):

        print('Problem Type:', problem_type)
        print('----------------------------------------')
        print()
        print('Avaliable models: ')
        print()

        for model_str in avaliable_by_type[problem_type]:
                if 'user passed' not in model_str:
                        show_model(model_str, show_param_ind_options,
                                   show_model_object, show_all_possible_params)


def show_model(model_str, show_param_ind_options, show_model_object,
               show_all_possible_params):

        multilabel, multiclass = False, False

        if 'multilabel ' in model_str:
                multilabel = True
        elif 'multiclass ' in model_str:
                multiclass = True

        model_str = model_str.replace('multilabel ', '')
        model_str = model_str.replace('multiclass ', '')

        print('- - - - - - - - - - - - - - - - - - - - ')
        M = MODELS[model_str]
        print(M[0].__name__, end='')
        print(' ("', model_str, '")', sep='')
        print('- - - - - - - - - - - - - - - - - - - - ')
        print()

        if multilabel:
                print('(MultiLabel)')
        elif multiclass:
                print('(MultiClass)')

        if show_model_object:
                print('Model Object: ', M[0])

        print()
        if show_param_ind_options:
                show_param_options(M[1])

        if show_all_possible_params:
                possible_params = get_possible_init_params(M[0])
                print('All Possible Params:', possible_params)
        print()
