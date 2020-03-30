"""
Default_Params.py
====================================
File with different saved default parameter grids,
for various classifiers within ABCD_ML.
"""

from copy import copy, deepcopy
from scipy.stats import randint, uniform, reciprocal
from sklearn.feature_selection import f_regression, f_classif
import nevergrad as ng
from sklearn.svm import SVR
import numpy as np


PARAMS = {}

cls_weight = ng.p.TransitionChoice([None, 'balanced'])
PARAMS['default'] = {}

# Models
PARAMS['base logistic'] =\
        {'max_iter': 5000,
         'multi_class': 'auto',
         'penalty': 'none',
         'class_weight': None,
         'solver': 'lbfgs'}

# Ridge classifier
PARAMS['base ridge'] = {'max_iter': 5000,
                        'penalty': 'l2',
                        'solver': 'saga'}

PARAMS['ridge C'] =\
        {'max_iter': 5000,
         'solver': 'saga',
         'C': ng.p.Log(lower=1e-5, upper=1e3),
         'class_weight': deepcopy(cls_weight)}

PARAMS['ridge C extra'] = PARAMS['ridge C'].copy()
PARAMS['ridge C extra']['max_iter'] =\
        ng.p.Scalar(lower=1000, upper=10000).set_integer_casting()
PARAMS['ridge C extra']['tol'] = ng.p.Log(lower=1e-6, upper=.01)

# Ridge regressor
PARAMS['base ridge regressor'] = {'max_iter': 5000,
                                  'solver': 'lsqr'}

PARAMS['ridge regressor dist'] = PARAMS['base ridge regressor'].copy()
PARAMS['ridge regressor dist']['alpha'] = ng.p.Log(lower=1e-3, upper=1e5)

# Lasso regressor
PARAMS['base lasso regressor'] = {'max_iter': 5000}
PARAMS['lasso regressor dist'] = PARAMS['base lasso regressor'].copy()
PARAMS['lasso regressor dist']['alpha'] = ng.p.Log(lower=1e-5, upper=1e5)

# Lasso classifier
PARAMS['base lasso'] = PARAMS['base logistic'].copy()
PARAMS['base lasso']['solver'] = 'liblinear'
PARAMS['base lasso']['penalty'] = 'l1'

PARAMS['lasso C'] = PARAMS['base lasso'].copy()
PARAMS['lasso C']['C'] = ng.p.Log(lower=1e-5, upper=1e3)
PARAMS['lasso C']['class_weight'] = deepcopy(cls_weight)

PARAMS['lasso C extra'] = PARAMS['lasso C'].copy()
PARAMS['lasso C extra']['max_iter'] =\
        ng.p.Scalar(lower=1000, upper=10000).set_integer_casting()
PARAMS['lasso C extra']['tol'] = ng.p.Log(lower=1e-6, upper=.01)

# Elastic net classifier
PARAMS['base elastic'] = PARAMS['base logistic'].copy()
PARAMS['base elastic']['penalty'] = 'elasticnet'
PARAMS['base elastic']['l1_ratio'] = .5
PARAMS['base elastic']['solver'] = 'saga'

PARAMS['elastic classifier'] = PARAMS['base elastic'].copy()
PARAMS['elastic classifier']['C'] = ng.p.Log(lower=1e-5, upper=1e5)
PARAMS['elastic classifier']['l1_ratio'] = ng.p.Scalar(lower=.01, upper=1)
PARAMS['elastic classifier']['class_weight'] = deepcopy(cls_weight)

PARAMS['elastic classifier extra'] = PARAMS['elastic classifier'].copy()
PARAMS['elastic classifier extra']['max_iter'] =\
        ng.p.Scalar(lower=1000, upper=10000).set_integer_casting()
PARAMS['elastic classifier extra']['tol'] = ng.p.Log(lower=1e-6, upper=.01)

# Elastic net regression
PARAMS['base elastic net'] = {'max_iter': 5000}
PARAMS['elastic regression'] = PARAMS['base elastic net'].copy()
PARAMS['elastic regression']['alpha'] = ng.p.Log(lower=1e-5, upper=1e5)
PARAMS['elastic regression']['l1_ratio'] = ng.p.Scalar(lower=.01, upper=1)

PARAMS['elastic regression extra'] = PARAMS['elastic regression'].copy()
PARAMS['elastic regression extra']['max_iter'] =\
        ng.p.Scalar(lower=1000, upper=10000).set_integer_casting()
PARAMS['elastic regression extra']['tol'] = ng.p.Log(lower=1e-6, upper=.01)


PARAMS['base huber'] = {'epsilon': 1.35}
PARAMS['base gnb'] = {'var_smoothing': 1e-9}

PARAMS['base knn'] = {'n_neighbors': 5}
PARAMS['knn dist'] = {'weights': ng.p.TransitionChoice(['uniform', 'distance']),
                      'n_neighbors': ng.p.Scalar(lower=2, upper=25).set_integer_casting()}

PARAMS['base dt'] = {}
PARAMS['dt dist'] = {'max_depth': ng.p.Scalar(lower=1, upper=30).set_integer_casting(),
                     'min_samples_split': ng.p.Scalar(lower=2, upper=50).set_integer_casting()}

PARAMS['dt classifier dist'] = PARAMS['dt dist'].copy()
PARAMS['dt classifier dist']['class_weight'] = deepcopy(cls_weight)

PARAMS['base linear'] = {'fit_intercept': True}



PARAMS['base rf'] = {'n_estimators': 100}

n_estimators = ng.p.Scalar(init=100, lower=3, upper=500).set_integer_casting()
var_depths = ng.p.Scalar(init=25, lower=2, upper=200).set_integer_casting()
depths = ng.p.TransitionChoice([None, var_depths])

PARAMS['rf dist'] = {'n_estimators': deepcopy(n_estimators),
                     'max_depth': deepcopy(depths),
                     'max_features': ng.p.Scalar(lower=0, upper=1),
                     'min_samples_split': ng.p.Scalar(lower=0, upper=1),
                     'bootstrap': True}

PARAMS['rf classifier dist'] = PARAMS['rf dist'].copy()
PARAMS['rf classifier dist']['class_weight'] = deepcopy(cls_weight)

# Light gbm params
PARAMS['base lgbm'] = {'silent': True}

reg = ng.p.TransitionChoice([0, ng.p.Log(lower=1e-5, upper=1)])

PARAMS['lgbm dist1'] = {'silent': True,
                        'boosting_type':
                        ng.p.TransitionChoice(['gbdt', 'dart', 'goss']),
                        'n_estimators': deepcopy(n_estimators),
                        'num_leaves': ng.p.Scalar(init=20, lower=6, upper=80).set_integer_casting(),
                        'min_child_samples': ng.p.Scalar(lower=10, upper=500).set_integer_casting(),
                        'min_child_weight': ng.p.Log(lower=1e-5, upper=1e4),
                        'subsample': ng.p.Scalar(lower=.3, upper=.95),
                        'colsample_bytree':  ng.p.Scalar(lower=.3, upper=.95),
                        'reg_alpha': deepcopy(reg),
                        'reg_lambda': deepcopy(reg)}

PARAMS['lgbm dist2'] = {'silent': True,
                        'lambda_l2': 0.001,
                        'boosting_type':
                        ng.p.TransitionChoice(['gbdt', 'dart']),
                        'min_child_samples':
                        ng.p.TransitionChoice([1, 5, 7, 10, 15, 20, 35, 50,
                                              100, 200, 500, 1000]),
                        'num_leaves':
                        ng.p.TransitionChoice([2, 4, 7, 10, 15, 20, 25, 30,
                                               35, 40, 50, 65, 80, 100, 125,
                                               150, 200, 250]),
                        'colsample_bytree':
                        ng.p.TransitionChoice([0.7, 0.9, 1.0]),
                        'subsample':
                        ng.p.Scalar(lower=.3, upper=1),
                        'learning_rate':
                        ng.p.TransitionChoice([0.01, 0.05, 0.1]),
                        'n_estimators':
                        ng.p.TransitionChoice([5, 20, 35, 50, 75, 100, 150,
                                                200, 350, 500, 750, 1000])}

PARAMS['lgbm classifier dist1'] = PARAMS['lgbm dist1'].copy()
PARAMS['lgbm classifier dist1']['class_weight'] = deepcopy(cls_weight)

PARAMS['lgbm classifier dist2'] = PARAMS['lgbm dist2'].copy()
PARAMS['lgbm classifier dist2']['class_weight'] = deepcopy(cls_weight)

PARAMS['base xgb'] = {'verbosity': 0,
                      'objective': 'reg:squarederror'}

PARAMS['base xgb classifier'] = PARAMS['base xgb'].copy()
PARAMS['base xgb classifier']['objective'] = 'binary:logistic'

PARAMS['xgb dist1'] =\
        {'verbosity': 0,
         'objective': 'reg:squarederror',
         'n_estimators': deepcopy(n_estimators),
         'min_child_weight': ng.p.Log(lower=1e-5, upper=1e4),
         'subsample': ng.p.Scalar(lower=.3, upper=.95),
         'colsample_bytree': ng.p.Scalar(lower=.3, upper=.95),
         'reg_alpha': deepcopy(reg),
         'reg_lambda': deepcopy(reg)}

PARAMS['xgb dist2'] =\
        {'verbosity': 0,
         'objective': 'reg:squarederror',
         'max_depth': deepcopy(depths),
         'learning_rate': ng.p.Scalar(lower=.01, upper=.5),
         'n_estimators': ng.p.Scalar(lower=3, upper=500).set_integer_casting(),
         'min_child_weight': ng.p.TransitionChoice([1, 5, 10, 50]),
         'subsample': ng.p.Scalar(lower=.5, upper=1),
         'colsample_bytree': ng.p.Scalar(lower=.4, upper=.95)}

PARAMS['xgb dist3'] =\
        {'verbosity': 0,
         'objective': 'reg:squarederror',
         'learning_rare': ng.p.Scalar(lower=.005, upper=.3),
         'min_child_weight': ng.p.Scalar(lower=.5, upper=10),
         'max_depth': ng.p.TransitionChoice([np.arange(3, 10)]),
         'subsample': ng.p.Scalar(lower=.5, upper=1),
         'colsample_bytree': ng.p.Scalar(lower=.5, upper=1),
         'reg_alpha': ng.p.Log(lower=.00001, upper=1)}

PARAMS['xgb classifier dist1'] = PARAMS['xgb dist1'].copy()
PARAMS['xgb classifier dist1']['objective'] = 'binary:logistic'

PARAMS['xgb classifier dist2'] = PARAMS['xgb dist2'].copy()
PARAMS['xgb classifier dist2']['objective'] = 'binary:logistic'

PARAMS['xgb classifier dist3'] = PARAMS['xgb dist3'].copy()
PARAMS['xgb classifier dist3']['objective'] = 'binary:logistic'

PARAMS['base gp regressor'] = {'n_restarts_optimizer': 5,
                               'normalize_y': True}
PARAMS['base gp classifier'] = {'n_restarts_optimizer': 5}

# probability = True
PARAMS['base svm'] = {'kernel': 'rbf',
                      'gamma': 'scale'}

PARAMS['svm dist'] = PARAMS['base svm'].copy()
PARAMS['svm dist']['C'] = ng.p.Log(lower=1e-4, upper=1e4)
PARAMS['svm dist']['gamma'] = ng.p.Log(lower=1e-6, upper=1)

PARAMS['base svm classifier'] = PARAMS['base svm'].copy()
PARAMS['base svm classifier']['probability'] = True

PARAMS['svm classifier dist'] = PARAMS['svm dist'].copy()
PARAMS['svm classifier dist']['probability'] = True
PARAMS['svm classifier dist']['class_weight'] = deepcopy(cls_weight)


# Define different choices for the mlp
PARAMS['base mlp'] = {}

batch_size =\
        ng.p.TransitionChoice(['auto',
                               ng.p.Scalar(init=200,
                                           lower=50,
                                           upper=400).set_integer_casting()])

PARAMS['mlp dist 1 layer'] =\
        {'hidden_layer_sizes':
         ng.p.Scalar(init=100, lower=2, upper=300),
         'activation':
         ng.p.TransitionChoice(['identity', 'logistic',
                                'tanh', 'relu']),
         'alpha': ng.p.Log(lower=1e-5, upper=1e2),
         'batch_size': deepcopy(batch_size),
         'learning_rate':
         ng.p.TransitionChoice(['constant', 'invscaling', 'adaptive']),
         'learning_rate_init': ng.p.Log(lower=1e-5, upper=1e2),
         'max_iter': ng.p.Scalar(init=200, lower=100, upper=500),
         'beta_1': ng.p.Scalar(init = .9, lower=.1, upper=.99),
         'beta_2': ng.p.Scalar(init=.999, lower=.1, upper=.9999)}

PARAMS['mlp dist es 1 layer'] = PARAMS['mlp dist 1 layer'].copy()
PARAMS['mlp dist es 1 layer']['early_stopping'] = True
PARAMS['mlp dist es 1 layer']['n_iter_no_change'] =\
        ng.p.Scalar(lower=5, upper=50)

two_layer = ng.p.Array(init=(100, 100)).set_mutation(sigma=50)
two_layer.set_bounds(lower=1, upper=300).set_integer_casting()

PARAMS['mlp dist 2 layer'] = PARAMS['mlp dist 1 layer'].copy()
PARAMS['mlp dist 2 layer']['hidden_layer_sizes'] = deepcopy(two_layer)

PARAMS['mlp dist es 2 layer'] = PARAMS['mlp dist es 1 layer'].copy()
PARAMS['mlp dist 2 layer']['hidden_layer_sizes'] = deepcopy(two_layer)

three_layer = ng.p.Array(init=(100, 100, 100)).set_mutation(sigma=50)
three_layer.set_bounds(lower=1, upper=300).set_integer_casting()

PARAMS['mlp dist 3 layer'] = PARAMS['mlp dist 1 layer'].copy()
PARAMS['mlp dist 3 layer']['hidden_layer_sizes'] = deepcopy(three_layer)

PARAMS['mlp dist es 3 layer'] = PARAMS['mlp dist es 1 layer'].copy()
PARAMS['mlp dist 3 layer']['hidden_layer_sizes'] = deepcopy(three_layer)



PARAMS['base linear svc'] = {'max_iter': 5000}

PARAMS['linear svc dist'] = PARAMS['base linear svc'].copy()
PARAMS['linear svc dist']['C'] = ng.p.Log(lower=1e-4, upper=1e4)
PARAMS['linear svc dist']['class_weight'] = deepcopy(cls_weight)

PARAMS['base linear svr'] = {'loss': 'epsilon_insensitive',
                             'max_iter': 5000}

PARAMS['linear svr dist'] = PARAMS['base linear svr'].copy()
PARAMS['linear svr dist']['C'] = ng.p.Log(lower=1e-4, upper=1e4)

PARAMS['base sgd'] = {'loss': 'hinge'}
PARAMS['sgd classifier'] =\
        {'loss': ng.p.TransitionChoice(['hinge', 'log', 'modified_huber',
                                           'squared_hinge', 'perceptron']),
         'penalty': ng.p.TransitionChoice(['l2', 'l1', 'elasticnet']),
         'alpha': ng.p.Log(lower=1e-5, upper=1e2),
         'l1_ratio': ng.p.Scalar(lower=0, upper=1),
         'max_iter': 5000,
         'learning_rate': ng.p.TransitionChoice(['optimal', 'invscaling', 'adaptive', 'constant']),
         'eta0': ng.p.Log(lower=1e-6, upper=1e3),
         'power_t': ng.p.Scalar(lower=.1, upper=.9),
         'early_stopping': ng.p.TransitionChoice([False, True]),
         'validation_fraction': ng.p.Scalar(lower=.05, upper=.5),
         'n_iter_no_change': ng.p.TransitionChoice([np.arange(2, 20)]),
         'class_weight': deepcopy(cls_weight)}


# Transformers
PARAMS['pca var search'] = {'n_components' : ng.p.Scalar(init=.75,
                                                         lower=.1,
                                                         upper=.99),
                            'svd_solver' : 'full'}

# Scalers
PARAMS['base standard'] = {'with_mean': True,
                           'with_std': True}

PARAMS['base minmax'] = {'feature_range': (0, 1)}

PARAMS['base robust'] = {'quantile_range': (5, 95)}

PARAMS['base winsorize'] = {'quantile_range': (1, 99)}

PARAMS['robust gs'] =\
        {'quantile_range': ng.p.TransitionChoice([(x, 100-x) for x in np.arange(1, 40)])}

PARAMS['winsorize gs'] =\
        {'quantile_range': ng.p.TransitionChoice([(x, 100-x) for x in np.arange(1, 40)])}

PARAMS['base yeo'] = {'method': 'yeo-johnson',
                      'standardize': True}

PARAMS['base boxcox'] = {'method': 'box-cox',
                         'standardize': True}

PARAMS['base quant norm'] = {'output_distribution': 'normal'}

PARAMS['base quant uniform'] = {'output_distribution': 'uniform'}

# Feat Selectors
PARAMS['base univar fs regression'] = {'score_func': f_regression,
                                       'percentile': 50}

PARAMS['univar fs regression dist'] = {'score_func': f_regression,
                                       'percentile':
                                       ng.p.Scalar(init=50, lower=1, upper=99)}


PARAMS['base univar fs classifier'] = {'score_func': f_classif,
                                       'percentile': 50}

PARAMS['univar fs classifier dist'] = {'score_func': f_classif,
                                       'percentile':
                                       ng.p.Scalar(init=50, lower=1, upper=99)}


PARAMS['base rfe'] = {'n_features_to_select': None}

PARAMS['rfe num feats dist'] = {'n_features_to_select':
                                ng.p.Scalar(init=.5, lower=.1, upper=.99)}

PARAMS['random'] = {'mask': 'sets as random features'}
PARAMS['searchable'] = {'mask': 'sets as hyperparameters'}

# Imputers
PARAMS['mean imp'] = {'strategy': 'mean'}
PARAMS['median imp'] = {'strategy': 'median'}
PARAMS['most freq imp'] = {'strategy': 'most_frequent'}
PARAMS['constant imp'] = {'strategy': 'constant'}
PARAMS['iterative imp'] = {'initial_strategy': 'mean'}

# Samplers
PARAMS['base no change sampler'] = {'sampler_type': 'no change',
                                    'regression_bins': 3,
                                    'regression_bin_strategy': 'uniform'}

PARAMS['base special sampler'] = {'sampler_type': 'special',
                                  'regression_bins': 3,
                                  'regression_bin_strategy': 'uniform'}

PARAMS['base change sampler'] = {'sampler_type': 'change',
                                 'regression_bins': 3,
                                 'regression_bin_strategy': 'uniform'}


PARAMS['rus binary ratio'] = PARAMS['base no change sampler'].copy()
PARAMS['rus binary ratio']['sampling_strategy'] =\
        ng.p.Scalar(init=.8, lower=.1, upper=1.2)


# Ensemblers
PARAMS['des default'] = {'needs_split': True,
                         'single_estimator': False}

PARAMS['single default'] = {'needs_split': False,
                            'single_estimator': True}

PARAMS['bb default'] = PARAMS['single default'].copy()

PARAMS['stacking default'] = {'needs_split': False,
                              'single_estimator': False,
                              'cv': 3}


# Feat Importances
PARAMS['base shap'] =\
        {'shap__global__avg_abs': False,
         'shap__linear__feature_dependence': 'independent',
         'shap__linear__nsamples': 1000,
         'shap__tree__feature_perturbation': 'tree_path_dependent',
         'shap__tree__model_output': 'margin',
         'shap__tree__tree_limit': None,
         'shap__kernel__nkmean': 10,
         'shap__kernel__nsamples': 'auto',
         'shap__kernel__l1_reg': 'aic'}

PARAMS['base perm'] = {'perm__n_perm': 10}


def get_base_params(str_indicator):

        base_params = deepcopy(PARAMS[str_indicator])
        return base_params


def proc_params(base_params, prepend=None):

        params = {prepend + '__' + key: base_params[key] for key in
                  base_params}

        return params


def show(str_indicator):

        params = PARAMS[str_indicator].copy()

        if len(params) == 0:
                print('None')

        for key in params:
                print(key, ': ', sep='', end='')

                value = params[key]
                print(value)
