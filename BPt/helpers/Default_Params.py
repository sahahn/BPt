"""
Default_Params.py
====================================
File with different saved default parameter grids,
for various classifiers within BPt.
"""

from copy import deepcopy
from sklearn.feature_selection import (f_regression, f_classif,
                                       mutual_info_classif,
                                       mutual_info_regression, chi2)
import nevergrad as ng
import numpy as np


P = {}

cls_weight = "ng.p.TransitionChoice([None, 'balanced'])"
P['default'] = {}

# Models
P['base logistic'] =\
        {'max_iter': "1000",
         'multi_class': "'auto'",
         'penalty': "'none'",
         'class_weight': "None",
         'solver': "'lbfgs'"}

# Ridge classifier
P['base ridge'] = {'max_iter': "1000",
                   'penalty': "'l2'",
                   'solver': "'saga'"}

P['ridge C'] =\
        {'max_iter': "1000",
         'solver': "'saga'",
         'C': "ng.p.Log(lower=1e-5, upper=1e3)",
         'class_weight': cls_weight}

P['ridge C extra'] = P['ridge C'].copy()
P['ridge C extra']['max_iter'] =\
        "ng.p.Scalar(lower=1000, upper=10000).set_integer_casting()"
P['ridge C extra']['tol'] = "ng.p.Log(lower=1e-6, upper=.01)"

# Ridge regressor
P['base ridge regressor'] = {'max_iter': "1000",
                             'solver': "'lsqr'"}

P['ridge regressor dist'] = P['base ridge regressor'].copy()
P['ridge regressor dist']['alpha'] = "ng.p.Log(lower=1e-3, upper=1e5)"

# Lasso regressor
P['base lasso regressor'] = {'max_iter': "1000"}
P['lasso regressor dist'] = P['base lasso regressor'].copy()
P['lasso regressor dist']['alpha'] = "ng.p.Log(lower=1e-5, upper=1e5)"

# Lasso classifier
P['base lasso'] = P['base logistic'].copy()
P['base lasso']['solver'] = "'liblinear'"
P['base lasso']['penalty'] = "'l1'"

P['lasso C'] = P['base lasso'].copy()
P['lasso C']['C'] = "ng.p.Log(lower=1e-5, upper=1e3)"
P['lasso C']['class_weight'] = cls_weight

P['lasso C extra'] = P['lasso C'].copy()
P['lasso C extra']['max_iter'] =\
        "ng.p.Scalar(lower=1000, upper=10000).set_integer_casting()"
P['lasso C extra']['tol'] = "ng.p.Log(lower=1e-6, upper=.01)"

# Elastic net classifier
P['base elastic'] = P['base logistic'].copy()
P['base elastic']['penalty'] = "'elasticnet'"
P['base elastic']['l1_ratio'] = ".5"
P['base elastic']['solver'] = "'saga'"

P['elastic classifier'] = P['base elastic'].copy()
P['elastic classifier']['C'] = "ng.p.Log(lower=1e-5, upper=1e5)"
P['elastic classifier']['l1_ratio'] = "ng.p.Scalar(lower=.01, upper=1)"
P['elastic classifier']['class_weight'] = cls_weight

P['elastic clf v2'] = P['elastic classifier'].copy()
P['elastic clf v2']['C'] = "ng.p.Log(lower=1e-2, upper=1e5)"

P['elastic classifier extra'] = P['elastic classifier'].copy()
P['elastic classifier extra']['max_iter'] =\
        "ng.p.Scalar(lower=1000, upper=10000).set_integer_casting()"
P['elastic classifier extra']['tol'] = "ng.p.Log(lower=1e-6, upper=.01)"

# Elastic net regression
P['base elastic net'] = {'max_iter': "1000"}
P['elastic regression'] = P['base elastic net'].copy()
P['elastic regression']['alpha'] = "ng.p.Log(lower=1e-5, upper=1e5)"
P['elastic regression']['l1_ratio'] = "ng.p.Scalar(lower=.01, upper=1)"

P['elastic regression extra'] = P['elastic regression'].copy()
P['elastic regression extra']['max_iter'] =\
        "ng.p.Scalar(lower=1000, upper=10000).set_integer_casting()"
P['elastic regression extra']['tol'] = "ng.p.Log(lower=1e-6, upper=.01)"


P['base huber'] = {'epsilon': "1.35"}
P['base gnb'] = {'var_smoothing': "1e-9"}

P['base knn'] = {'n_neighbors': "5"}
P['knn dist'] = {'weights': "ng.p.TransitionChoice(['uniform', 'distance'])",
                 'n_neighbors':
                 "ng.p.Scalar(lower=2, upper=25).set_integer_casting()"}
P['base knn regression'] = P['base knn'].copy()
P['knn dist regression'] = P['knn dist'].copy()

P['base dt'] = {}
P['dt dist'] = {'max_depth':
                "ng.p.Scalar(lower=1, upper=30).set_integer_casting()",
                'min_samples_split':
                "ng.p.Scalar(lower=2, upper=50).set_integer_casting()"}

P['dt classifier dist'] = P['dt dist'].copy()
P['dt classifier dist']['class_weight'] = cls_weight

P['base linear'] = {'fit_intercept': "True"}

P['base rf'] = {'n_estimators': "100"}
P['base rf regressor'] = P['base rf'].copy()

n_estimators = "ng.p.Scalar(init=100, lower=3, upper=500).set_integer_casting()"
depths = "ng.p.TransitionChoice([None, ng.p.Scalar(init=25, lower=2, upper=200).set_integer_casting()])"

P['rf dist'] = {'n_estimators': n_estimators,
                'max_depth': depths,
                'max_features': "ng.p.Scalar(lower=.1, upper=1.0)",
                'min_samples_split': "ng.p.Scalar(lower=.1, upper=1.0)",
                'bootstrap': "True"}

P['rf classifier dist'] = P['rf dist'].copy()
P['rf classifier dist']['class_weight'] = cls_weight

# Light gbm params
P['base lgbm'] = {'silent': "True"}

reg = "ng.p.TransitionChoice([0, ng.p.Log(lower=1e-5, upper=1)])"

P['lgbm dist1'] =\
        {'silent': "True",
         'boosting_type': "ng.p.TransitionChoice(['gbdt', 'dart', 'goss'])",
         'n_estimators': n_estimators,
         'num_leaves':
         "ng.p.Scalar(init=20, lower=6, upper=80).set_integer_casting()",
         'min_child_samples':
         "ng.p.Scalar(lower=10, upper=500).set_integer_casting()",
         'min_child_weight':
         "ng.p.Log(lower=1e-5, upper=1e4)",
         'subsample':
         "ng.p.Scalar(lower=.3, upper=.95)",
         'colsample_bytree':
         "ng.p.Scalar(lower=.3, upper=.95)",
         'reg_alpha': reg,
         'reg_lambda': reg}

P['lgbm dist2'] =\
        {'silent': "True",
         'lambda_l2': "0.001",
         'boosting_type': "ng.p.TransitionChoice(['gbdt', 'dart'])",
         'min_child_samples':
         "ng.p.TransitionChoice([1, 5, 7, 10, 15, 20, 35, 50, 100, 200, 500, 1000])",
         'num_leaves':
         "ng.p.TransitionChoice([2, 4, 7, 10, 15, 20, 25, 30, 35, 40, 50, 65, 80, 100, 125, 150, 200, 250])",
         'colsample_bytree':
         "ng.p.TransitionChoice([0.7, 0.9, 1.0])",
         'subsample':
         "ng.p.Scalar(lower=.3, upper=1)",
         'learning_rate':
         "ng.p.TransitionChoice([0.01, 0.05, 0.1])",
         'n_estimators':
         "ng.p.TransitionChoice([5, 20, 35, 50, 75, 100, 150, 200, 350, 500, 750, 1000])"}

P['lgbm dist3'] = {'silent': "True",
                   'n_estimators': "1000",
                   'early_stopping_rounds': "150",
                   'eval_split': ".2",
                   'boosting_type': '"gbdt"',
                   'learning_rate':
                   'ng.p.Log(lower=5e-3, upper=.2, init=.1)',
                   'colsample_bytree':
                   'ng.p.Scalar(lower=.75, upper=1, init=1)',
                   'min_child_samples':
                   "ng.p.Scalar(lower=2, upper=30, init=20).set_integer_casting()",
                   'num_leaves':
                   "ng.p.Scalar(lower=16, upper=96, init=31).set_integer_casting()"}

P['lgbm classifier dist1'] = P['lgbm dist1'].copy()
P['lgbm classifier dist1']['class_weight'] = cls_weight

P['lgbm classifier dist2'] = P['lgbm dist2'].copy()
P['lgbm classifier dist2']['class_weight'] = cls_weight

P['lgbm classifier dist3'] = P['lgbm dist3'].copy()
P['lgbm classifier dist3']['class_weight'] = cls_weight

P['base xgb'] = {'verbosity': "0",
                 'objective': "'reg:squarederror'"}

P['base xgb classifier'] = P['base xgb'].copy()
P['base xgb classifier']['objective'] = "'binary:logistic'"

P['xgb dist1'] =\
        {'verbosity': "0",
         'objective': "'reg:squarederror'",
         'n_estimators': n_estimators,
         'min_child_weight': "ng.p.Log(lower=1e-5, upper=1e4)",
         'subsample': "ng.p.Scalar(lower=.3, upper=.95)",
         'colsample_bytree': "ng.p.Scalar(lower=.3, upper=.95)",
         'reg_alpha': reg,
         'reg_lambda': reg}

P['xgb dist2'] =\
        {'verbosity': "0",
         'objective': "'reg:squarederror'",
         'max_depth': depths,
         'learning_rate': "ng.p.Scalar(lower=.01, upper=.5)",
         'n_estimators': "ng.p.Scalar(lower=3, upper=500).set_integer_casting()",
         'min_child_weight': "ng.p.TransitionChoice([1, 5, 10, 50])",
         'subsample': "ng.p.Scalar(lower=.5, upper=1)",
         'colsample_bytree': "ng.p.Scalar(lower=.4, upper=.95)"}

P['xgb dist3'] =\
        {'verbosity': "0",
         'objective': "'reg:squarederror'",
         'learning_rare': "ng.p.Scalar(lower=.005, upper=.3)",
         'min_child_weight': "ng.p.Scalar(lower=.5, upper=10)",
         'max_depth': "ng.p.TransitionChoice(np.arange(3, 10))",
         'subsample': "ng.p.Scalar(lower=.5, upper=1)",
         'colsample_bytree': "ng.p.Scalar(lower=.5, upper=1)",
         'reg_alpha': "ng.p.Log(lower=.00001, upper=1)"}

P['xgb classifier dist1'] = P['xgb dist1'].copy()
P['xgb classifier dist1']['objective'] = "'binary:logistic'"

P['xgb classifier dist2'] = P['xgb dist2'].copy()
P['xgb classifier dist2']['objective'] = "'binary:logistic'"

P['xgb classifier dist3'] = P['xgb dist3'].copy()
P['xgb classifier dist3']['objective'] = "'binary:logistic'"

P['base gp regressor'] = {'n_restarts_optimizer': "5",
                          'normalize_y': "True"}
P['base gp classifier'] = {'n_restarts_optimizer': "5"}

# probability = True
P['base svm'] = {'kernel': "'rbf'",
                 'gamma': "'scale'"}

P['svm dist'] = P['base svm'].copy()
P['svm dist']['C'] = "ng.p.Log(lower=1e-4, upper=1e4)"
P['svm dist']['gamma'] = "ng.p.Log(lower=1e-6, upper=1)"

P['base svm classifier'] = P['base svm'].copy()
P['base svm classifier']['probability'] = "True"

P['svm classifier dist'] = P['svm dist'].copy()
P['svm classifier dist']['probability'] = "True"
P['svm classifier dist']['class_weight'] = cls_weight


# Define different choices for the mlp
P['base mlp'] = {}

batch_size =\
        "ng.p.TransitionChoice(['auto', ng.p.Scalar(init=200, lower=50," +\
        " upper=400).set_integer_casting()])"

P['mlp dist 1 layer'] =\
        {'hidden_layer_sizes':
         "ng.p.Scalar(init=100, lower=2, upper=300).set_integer_casting()",
         'activation':
         "ng.p.TransitionChoice(['identity', 'logistic', 'tanh', 'relu'])",
         'alpha': "ng.p.Log(lower=1e-5, upper=1e2)",
         'batch_size': batch_size,
         'learning_rate':
         "ng.p.TransitionChoice(['constant', 'invscaling', 'adaptive'])",
         'learning_rate_init': "ng.p.Log(lower=1e-5, upper=1e2)",
         'max_iter':
         "ng.p.Scalar(init=200, lower=100, upper=1000).set_integer_casting()",
         'beta_1': "ng.p.Scalar(init=.9, lower=.1, upper=.99)",
         'beta_2': "ng.p.Scalar(init=.999, lower=.1, upper=.9999)"}

P['mlp dist es 1 layer'] = P['mlp dist 1 layer'].copy()
P['mlp dist es 1 layer']['early_stopping'] = "True"
P['mlp dist es 1 layer']['n_iter_no_change'] =\
        "ng.p.Scalar(lower=5, upper=50)"

two_layer = "ng.p.Array(init=(100, 100)).set_mutation(sigma=50)" +\
            ".set_bounds(lower=1, upper=300).set_integer_casting()"

P['mlp dist 2 layer'] = P['mlp dist 1 layer'].copy()
P['mlp dist 2 layer']['hidden_layer_sizes'] = two_layer

P['mlp dist es 2 layer'] = P['mlp dist es 1 layer'].copy()
P['mlp dist 2 layer']['hidden_layer_sizes'] = two_layer

three_layer = "ng.p.Array(init=(100, 100, 100)).set_mutation(sigma=50)" +\
              ".set_bounds(lower=1, upper=300).set_integer_casting()"

P['mlp dist 3 layer'] = P['mlp dist 1 layer'].copy()
P['mlp dist 3 layer']['hidden_layer_sizes'] = three_layer

P['mlp dist es 3 layer'] = P['mlp dist es 1 layer'].copy()
P['mlp dist 3 layer']['hidden_layer_sizes'] = three_layer

P['base linear svc'] = {'max_iter': "1000"}

P['linear svc dist'] = P['base linear svc'].copy()
P['linear svc dist']['C'] = "ng.p.Log(lower=1e-4, upper=1e4)"
P['linear svc dist']['class_weight'] = cls_weight

P['base linear svr'] = {'loss': "'epsilon_insensitive'",
                        'max_iter': "1000"}

P['linear svr dist'] = P['base linear svr'].copy()
P['linear svr dist']['C'] = "ng.p.Log(lower=1e-4, upper=1e4)"

P['base sgd'] = {'loss': "'squared_loss'"}

loss_choice = "ng.p.TransitionChoice(['hinge', 'log', " +\
              "'modified_huber', 'squared_hinge', 'perceptron'])"

lr_choice = "ng.p.TransitionChoice(['optimal', 'invscaling', " +\
            "'adaptive', 'constant'])"

P['sgd classifier big search'] =\
        {'loss': loss_choice,
         'penalty': "ng.p.TransitionChoice(['l2', 'l1', 'elasticnet'])",
         'alpha': "ng.p.Log(lower=1e-5, upper=1e2)",
         'l1_ratio': "ng.p.Scalar(lower=.01, upper=1)",
         'max_iter': "1000",
         'learning_rate': lr_choice,
         'eta0': "ng.p.Log(lower=1e-6, upper=1e3)",
         'power_t': "ng.p.Scalar(lower=.1, upper=.9)",
         'early_stopping': "ng.p.TransitionChoice([False, True])",
         'validation_fraction': "ng.p.Scalar(lower=.05, upper=.5)",
         'n_iter_no_change': "ng.p.TransitionChoice(np.arange(2, 20))",
         'class_weight': cls_weight}

# Make elastic net version
P['sgd elastic'] =\
        {'loss': "'squared_epsilon_insensitive'",
         'penalty': "'elasticnet'",
         'alpha': "ng.p.Log(lower=1e-5, upper=1e5)",
         'l1_ratio': "ng.p.Scalar(lower=.01, upper=1)"}

P['sgd elastic classifier'] = P['sgd elastic'].copy()
P['sgd elastic classifier']['class_weight'] = cls_weight

# Auto gluon
P['pt binary'] = {'problem_type': "'binary'"}
P['pt multiclass'] = {'problem_type': "'multiclass'"}
P['pt regression'] = {'problem_type': "'regression'"}


# Transformers
P['pca var search'] =\
        {'n_components': "ng.p.Scalar(init=.75, lower=.1, upper=.99)",
         'svd_solver': "'full'"}

P['ohe'] =\
        {'sparse': "False",
         'handle_unknown': "'ignore'"}

# Scalers
P['base standard'] = {'with_mean': "True",
                      'with_std': "True"}

P['base minmax'] = {'feature_range': "(0, 1)"}

P['base robust'] = {'quantile_range': "(5, 95)"}

P['base winsorize'] = {'quantile_range': "(1, 99)"}

P['robust gs'] =\
        {'quantile_range':
         "ng.p.TransitionChoice([(x, 100-x) for x in np.arange(1, 40)])"}

P['winsorize gs'] =\
        {'quantile_range':
         "ng.p.TransitionChoice([(x, 100-x) for x in np.arange(1, 40)])"}

P['base yeo'] = {'method': "'yeo-johnson'",
                 'standardize': "True"}

P['base boxcox'] = {'method': "'box-cox'",
                    'standardize': "True"}

P['base quant norm'] = {'output_distribution': "'normal'"}

P['base quant uniform'] = {'output_distribution': "'uniform'"}

# Feat Selectors
P['base univar fs regression'] = {'score_func': "f_regression",
                                  'percentile': "50"}

P['univar fs regression dist'] = {'score_func': "f_regression",
                                  'percentile':
                                  "ng.p.Scalar(init=50, lower=1, upper=99)"}

P['univar fs regression dist2'] = {'score_func': "f_regression",
                                   'percentile':
                                   "ng.p.Scalar(init=75, lower=50, upper=99)"}


P['base univar fs classifier'] = {'score_func': "f_classif",
                                  'percentile': "50"}

P['univar fs classifier dist'] = {'score_func': "f_classif",
                                  'percentile':
                                  "ng.p.Scalar(init=50, lower=1, upper=99)"}

P['univar fs classifier dist2'] = {'score_func': "f_classif",
                                   'percentile':
                                   "ng.p.Scalar(init=75, lower=50, upper=99)"}


P['base rfe'] = {'n_features_to_select': "None"}

P['rfe num feats dist'] = {'n_features_to_select':
                           "ng.p.Scalar(init=.5, lower=.1, upper=.99)"}

P['random'] = {'mask': "'sets as random features'"}
P['searchable'] = {'mask': "'sets as hyperparameters'"}

# Imputers
P['mean imp'] = {'strategy': "'mean'"}
P['median imp'] = {'strategy': "'median'"}
P['most freq imp'] = {'strategy': "'most_frequent'"}
P['constant imp'] = {'strategy': "'constant'"}
P['iterative imp'] = {'initial_strategy': "'mean'",
                      'skip_complete': "True"}

# Samplers
P['base no change sampler'] = {'sampler_type': "'no change'"}

P['base special sampler'] = {'sampler_type': "'special'"}

P['base change sampler'] = {'sampler_type': "'change'"}


P['rus binary ratio'] = P['base no change sampler'].copy()
P['rus binary ratio']['sampling_strategy'] =\
        "ng.p.Scalar(init=.8, lower=.1, upper=1.2)"


# Ensemblers
P['stacking default'] = {'cv': "3"}

P['voting classifier'] = {'voting': "'soft'"}

# Feat Importances
P['base shap'] =\
        {'shap__global__avg_abs': "False",
         'shap__linear__feature_dependence': "'independent'",
         'shap__linear__nsamples': "1000",
         'shap__tree__feature_perturbation': "'tree_path_dependent'",
         'shap__tree__model_output': "'margin'",
         'shap__tree__tree_limit': "None",
         'shap__kernel__nkmean': "10",
         'shap__kernel__nsamples': "'auto'",
         'shap__kernel__l1_reg': "'aic'"}

P['base perm'] = {'perm__n_perm': "10"}

PARAMS = {}
for param in P:
    PARAMS[param] = {}
    for p in P[param]:
        try:
            PARAMS[param][p] = eval(P[param][p])
        except TypeError:
            PARAMS[param][p] = P[param][p]


def get_base_params(str_indicator):

    base_params = deepcopy(PARAMS[str_indicator])
    return base_params


def proc_params(base_params, prepend=None):

    if isinstance(base_params, int):
        return {}

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
