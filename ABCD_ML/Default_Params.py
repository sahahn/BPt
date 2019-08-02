"""
Default_Params.py
====================================
File with different saved default parameter grids,
for various classifiers within ABCD_ML.
"""

from scipy.stats import randint, uniform, reciprocal
from sklearn.feature_selection import f_regression, f_classif
from sklearn.svm import SVR

PARAMS = {}

# Models
PARAMS['base logistic'] = {'solver': ['saga'],
                           'max_iter': [5000],
                           'multi_class': ['auto'],
                           'penalty': ['none']}

PARAMS['base lasso'] = PARAMS['base logistic'].copy()
PARAMS['base lasso']['penalty'] = ['l1']

PARAMS['base ridge'] = PARAMS['base logistic'].copy()
PARAMS['base ridge']['penalty'] = ['l2']

PARAMS['base elastic'] = PARAMS['base logistic'].copy()
PARAMS['base elastic']['penalty'] = ['elasticnet']
PARAMS['base elastic']['l1_ratio'] = [.5]

PARAMS['lasso C'] = PARAMS['base lasso'].copy()
PARAMS['lasso C']['C'] = reciprocal(a=1e-4, b=1e+4)

PARAMS['ridge C'] = PARAMS['base ridge'].copy()
PARAMS['ridge C']['C'] = reciprocal(a=1e-4, b=1e+4)

PARAMS['elastic classifier'] = PARAMS['base elastic'].copy()
PARAMS['elastic classifier']['C'] = reciprocal(a=1e-4, b=1e+4)
PARAMS['elastic classifier']['l1_ratio'] = uniform()

PARAMS['base elastic net'] = {'max_iter': [5000]}
PARAMS['elastic regression'] = PARAMS['base elastic net'].copy()
PARAMS['elastic regression']['alpha'] = reciprocal(a=1e-6, b=1e+2)
PARAMS['elastic regression']['l1_ratio'] = uniform()

PARAMS['base huber'] = {'epsilon': [1.35]}
PARAMS['base gnb'] = {'var_smoothing': [1e-9]}

PARAMS['base knn'] = {'n_neighbors': [5]}
PARAMS['knn rs'] = {'weights': ['uniform', 'distance'],
                    'n_neighbors': randint(2, 20)}

PARAMS['base dt'] = {}
PARAMS['dt rs'] = {'max_depth': randint(1, 20),
                   'min_samples_split': randint(2, 50)}

PARAMS['base linear'] = {'fit_intercept': [True]}

PARAMS['base rf'] = {'n_estimators': [100]}
PARAMS['rf rs'] = {'n_estimators': randint(3, 500),
                   'max_depth': randint(2, 200),
                   'max_features': uniform(),
                   'min_samples_split': uniform(),
                   'bootstrap': [True]}

PARAMS['base lgbm'] = {'silent': [True]}
PARAMS['lgbm rs'] = {'silent': [True],
                     'boosting_type': ['gbdt', 'dart', 'goss'],
                     'n_estimators': randint(3, 500),
                     'num_leaves': randint(6, 50),
                     'min_child_samples': randint(100, 500),
                     'min_child_weight': reciprocal(a=1e-5, b=1e+4),
                     'subsample': uniform(loc=0.2, scale=0.8),
                     'colsample_bytree': uniform(loc=0.4, scale=0.6),
                     'reg_alpha': reciprocal(a=1e-1, b=1e+2),
                     'reg_lambda': reciprocal(a=1e-1, b=1e+2)}

PARAMS['base lgbm es'] = {'silent': [True],
                          'val_split_percent': [.1],
                          'early_stop_rounds': [50],
                          }

PARAMS['lgbm es rs'] = PARAMS['lgbm rs'].copy()
PARAMS['lgbm es rs']['val_split_percent'] = uniform(loc=.05, scale=.2)
PARAMS['lgbm es rs']['early_stop_rounds'] = randint(10, 150)


PARAMS['base gp regressor'] = {'n_restarts_optimizer': [5],
                               'normalize_y': [True]}
PARAMS['base gp classifier'] = {'n_restarts_optimizer': [5]}

# probability = True
PARAMS['base svm'] = {'kernel': ['rbf'],
                      'gamma': ['scale']}

PARAMS['svm rs'] = PARAMS['base svm'].copy()
PARAMS['svm rs']['C'] = reciprocal(a=1e-4, b=1e+4)
PARAMS['svm rs']['gamma'] = reciprocal(a=1e-6, b=1e-1)

PARAMS['base svm classifier'] = PARAMS['base svm'].copy()
PARAMS['base svm classifier']['probability'] = [True]

PARAMS['svm classifier rs'] = PARAMS['svm rs'].copy()
PARAMS['svm classifier rs']['probability'] = [True]

PARAMS['base mlp'] = {}

NNs = []
for x in range(2, 150):
    NNs.append((x))
    for y in range(2, 150):
        NNs.append((x, y))
        for z in range(2, 150):
            NNs.append((x, y, z))

PARAMS['mlp rs'] = {'hidden_layer_sizes': NNs,
                    'activation': ['identity', 'logistic', 'tanh', 'relu'],
                    'alpha': reciprocal(a=1e-5, b=1e+2),
                    'batch_size': randint(2, 200),
                    'learning_rate': ['constant', 'invscaling', 'adaptive'],
                    'learning_rate_init': reciprocal(a=1e-5, b=1e-2),
                    'max_iter': randint(100, 500),
                    'beta_1': uniform(loc=0.5, scale=0.5),
                    'beta_2': uniform(loc=0.5, scale=0.5)}

PARAMS['mlp rs es'] = PARAMS['mlp rs'].copy()
PARAMS['mlp rs es']['early_stopping'] = [True]
PARAMS['mlp rs es']['n_iter_no_change'] = randint(5, 50)

PARAMS['mlp layers search'] = {'hidden_layer_sizes': NNs}

# Scalers
PARAMS['base standard'] = {'with_mean': [True],
                           'with_std': [True]}

PARAMS['base minmax'] = {'feature_range': [(0, 1)]}

PARAMS['base robust'] = {'quantile_range': [(5, 95)]}

PARAMS['robust gs'] = {'quantile_range': [(1, 99), (5, 95), (10, 90), (15, 85),
                                          (20, 80), (25, 75), (30, 70),
                                          (35, 65), (40, 60)]}

PARAMS['base power'] = {'method': ['yeo-johnson'],
                        'standardize': [True]}

PARAMS['base pca'] = {}

PARAMS['pca rs'] = {'n_components': uniform()}

# Feat Selectors
PARAMS['base univar fs regression'] = {'score_func': [f_regression],
                                       'percentile': [50]}

PARAMS['univar fs regression gs'] = {'score_func': [f_regression],
                                     'percentile': [10, 20, 30, 40, 50, 60, 70,
                                                    80, 90]}

PARAMS['base univar fs classifier'] = {'score_func': [f_classif],
                                       'percentile': [50]}

PARAMS['univar fs classifier gs'] = {'score_func': [f_classif],
                                     'percentile':  [10, 20, 30, 40, 50, 60,
                                                     70, 80, 90]}

PARAMS['base linear svm rfe regression'] = {'estimator':
                                            [SVR(kernel="linear")],
                                            'n_features_to_select': [None]}


def get(str_indicator, preprend):

        params = PARAMS[str_indicator].copy()
        params = {preprend + '__' + key: params[key] for key in params}

        return params


def show(str_indicator):

        params = PARAMS[str_indicator].copy()

        if len(params) == 0:
                print('None')

        for key in params:
                print(key, ': ', sep='', end='')

                value = params[key]

                # If either rand int or uniform dist
                if 'scipy' in str(type(value)):

                        # Randint distr
                        if isinstance(value.a, int):
                                print('Random Integer Distribution (',
                                      value.a, ', ', value.b, ')', sep='')

                        else:
                                print('Uniform/Reciprocal Distribution Over',
                                      value.interval(1))

                elif len(value) == 1:
                        if callable(value[0]):
                                print(value[0].__name__)
                        else:
                                print(value[0])

                elif len(value) > 50:
                        print('Too many params to print')

                else:
                        print(value)
