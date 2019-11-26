"""
Default_Params.py
====================================
File with different saved default parameter grids,
for various classifiers within ABCD_ML.
"""

from scipy.stats import randint, uniform, reciprocal
from sklearn.feature_selection import f_regression, f_classif
import nevergrad as ng
from sklearn.svm import SVR


PARAMS = {}

PARAMS['default'] = {}

# Models
PARAMS['base logistic'] =\
        {'solver': 'saga',
         'max_iter': 5000,
         'multi_class': 'auto',
         'penalty': 'none',
         'class_weight': None}

PARAMS['base lasso'] = PARAMS['base logistic'].copy()
PARAMS['base lasso']['penalty'] = 'l1'

PARAMS['base ridge'] = PARAMS['base logistic'].copy()
PARAMS['base ridge']['penalty'] = 'l2'

PARAMS['base elastic'] = PARAMS['base logistic'].copy()
PARAMS['base elastic']['penalty'] = 'elasticnet'
PARAMS['base elastic']['l1_ratio'] = .5

PARAMS['lasso C'] = PARAMS['base lasso'].copy()
PARAMS['lasso C']['C'] =\
        ng.var.Scalar().bounded(-4, 4).exponentiated(base=10, coeff=-1)
PARAMS['lasso C']['class_weight'] =\
        ng.var.OrderedDiscrete([None, 'balanced'])

PARAMS['ridge C'] = PARAMS['base ridge'].copy()
PARAMS['ridge C']['C'] =\
        ng.var.Scalar().bounded(-4, 4).exponentiated(base=10, coeff=-1)
PARAMS['ridge C']['class_weight'] =\
        ng.var.OrderedDiscrete([None, 'balanced'])

PARAMS['elastic classifier'] = PARAMS['base elastic'].copy()
PARAMS['elastic classifier']['C'] =\
        ng.var.Scalar().bounded(-4, 4).exponentiated(base=10, coeff=-1)
PARAMS['elastic classifier']['l1_ratio'] = ng.var.Scalar().bounded(0, 1)
PARAMS['elastic classifier']['class_weight'] =\
        ng.var.OrderedDiscrete([None, 'balanced'])

PARAMS['base elastic net'] = {'max_iter': 5000}
PARAMS['elastic regression'] = PARAMS['base elastic net'].copy()
PARAMS['elastic regression']['alpha'] =\
        ng.var.Scalar().bounded(-2, 5).exponentiated(base=10, coeff=-1)
PARAMS['elastic regression']['l1_ratio'] = ng.var.Scalar().bounded(0, 1)

PARAMS['base huber'] = {'epsilon': 1.35}
PARAMS['base gnb'] = {'var_smoothing': 1e-9}

PARAMS['base knn'] = {'n_neighbors': 5}
PARAMS['knn dist'] = {'weights':
                      ng.var.OrderedDiscrete(['uniform', 'distance']),
                      'n_neighbors': ng.var.Scalar(int).bounded(2, 25)}

PARAMS['base dt'] = {}
PARAMS['dt dist'] = {'max_depth': ng.var.Scalar(int).bounded(1, 30),
                     'min_samples_split': ng.var.Scalar(int).bounded(2, 50)}

PARAMS['dt classifier dist'] = PARAMS['dt dist'].copy()
PARAMS['dt classifier dist']['class_weight'] =\
        ng.var.OrderedDiscrete([None, 'balanced'])

PARAMS['base linear'] = {'fit_intercept': True}

PARAMS['base rf'] = {'n_estimators': 100}
PARAMS['rf dist'] = {'n_estimators': ng.var.Scalar(int).bounded(3, 500),
                     'max_depth': ng.var.Scalar(int).bounded(2, 200),
                     'max_features': ng.var.Scalar().bounded(0, 1),
                     'min_samples_split': ng.var.Scalar().bounded(0, 1),
                     'bootstrap': True}

PARAMS['rf classifier dist'] = PARAMS['rf dist'].copy()
PARAMS['rf classifier dist']['class_weight'] =\
        ng.var.OrderedDiscrete([None, 'balanced'])

PARAMS['base lgbm'] = {'silent': True}
PARAMS['lgbm dist1'] = {'silent': True,
                        'boosting_type':
                        ng.var.OrderedDiscrete(['gbdt', 'dart', 'goss']),
                        'n_estimators': ng.var.Scalar(int).bounded(3, 500),
                        'num_leaves': ng.var.Scalar(int).bounded(6, 80),
                        'min_child_samples': ng.var.Scalar(int).bounded(10,
                                                                        500),
                        'min_child_weight':
                        ng.var.Scalar().bounded(-4, 5).exponentiated(base=10,
                                                                     coeff=-1),
                        'subsample': ng.var.Scalar().bounded(.3, .95),
                        'colsample_bytree': ng.var.Scalar().bounded(.3, .95),
                        'reg_alpha':
                        ng.var.Scalar().bounded(-2, 1).exponentiated(base=10,
                                                                     coeff=-1),
                        'reg_lambda':
                        ng.var.Scalar().bounded(-2, 1).exponentiated(base=10,
                                                                     coeff=-1)}

PARAMS['lgbm dist2'] = {'silent': True,
                        'lambda_l2': 0.001,
                        'boosting_type':
                        ng.var.OrderedDiscrete(['gbdt', 'dart']),
                        'min_child_samples':
                        ng.var.OrderedDiscrete([1, 5, 7, 10, 15, 20, 35, 50,
                                                100, 200, 500, 1000]),
                        'num_leaves':
                        ng.var.OrderedDiscrete([2, 4, 7, 10, 15, 20, 25, 30,
                                                35, 40, 50, 65, 80, 100, 125,
                                                150, 200, 250]),
                        'colsample_bytree':
                        ng.var.OrderedDiscrete([0.7, 0.9, 1.0]),
                        'subsample':
                        ng.var.Scalar().bounded(.3, 1),
                        'learning_rate':
                        ng.var.OrderedDiscrete([0.01, 0.05, 0.1]),
                        'n_estimators':
                        ng.var.OrderedDiscrete([5, 20, 35, 50, 75, 100, 150,
                                                200, 350, 500, 750, 1000])}

PARAMS['lgbm classifier dist1'] = PARAMS['lgbm dist1'].copy()
PARAMS['lgbm classifier dist1']['class_weight'] =\
        ng.var.OrderedDiscrete([None, 'balanced'])

PARAMS['lgbm classifier dist2'] = PARAMS['lgbm dist2'].copy()
PARAMS['lgbm classifier dist2']['class_weight'] =\
        ng.var.OrderedDiscrete([None, 'balanced'])

PARAMS['base xgb'] = {'verbosity': 0}

PARAMS['xgb dist'] =\
        {'verbosity': 0,
         'max_depth': ng.var.Scalar(int).bounded(2, 50),
         'learning_rate': ng.var.Scalar().bounded(.01, .5),
         'n_estimators': ng.var.Scalar(int).bounded(3, 500),
         'min_child_weight': ng.var.OrderedDiscrete([1, 5, 10, 50]),
         'subsample': ng.var.Scalar().bounded(.5, 1),
         'colsample_bytree': ng.var.Scalar().bounded(.4, .95)}

PARAMS['base gp regressor'] = {'n_restarts_optimizer': 5,
                               'normalize_y': True}
PARAMS['base gp classifier'] = {'n_restarts_optimizer': 5}

# probability = True
PARAMS['base svm'] = {'kernel': 'rbf',
                      'gamma': 'scale'}

PARAMS['svm dist'] = PARAMS['base svm'].copy()
PARAMS['svm dist']['C'] =\
        ng.var.Scalar().bounded(-4, 4).exponentiated(base=10,
                                                     coeff=-1)
PARAMS['svm dist']['gamma'] =\
        ng.var.Scalar().bounded(1, 6).exponentiated(base=10,
                                                    coeff=-1)

PARAMS['base svm classifier'] = PARAMS['base svm'].copy()
PARAMS['base svm classifier']['probability'] = True

PARAMS['svm classifier dist'] = PARAMS['svm dist'].copy()
PARAMS['svm classifier dist']['probability'] = True
PARAMS['svm classifier dist']['class_weight'] =\
        ng.var.OrderedDiscrete([None, 'balanced'])

PARAMS['base mlp'] = {}

PARAMS['base lasso regressor'] = {'max_iter': 5000}
PARAMS['lasso regressor dist'] =\
        {'alpha': ng.var.Scalar().bounded(-4, 5).exponentiated(base=10,
                                                               coeff=-1)}

PARAMS['base ridge regressor'] = PARAMS['base lasso regressor'].copy()
PARAMS['ridge regressor dist'] =\
        {'alpha': ng.var.Scalar().bounded(-4, 5).exponentiated(base=10,
                                                               coeff=-1)}


PARAMS['mlp dist'] = {'hidden_layer_sizes':
                      ng.var.Array(1, 1, 1).bounded(2, 100),
                      'activation':
                      ng.var.OrderedDiscrete(['identity', 'logistic',
                                              'tanh', 'relu']),
                      'alpha':
                      ng.var.Scalar().bounded(-2, 5).exponentiated(base=10,
                                                                   coeff=-1),
                      'batch_size': ng.var.Scalar(int).bounded(2, 200),
                      'learning_rate':
                      ng.var.OrderedDiscrete(['constant', 'invscaling',
                                              'adaptive']),
                      'learning_rate_init':
                      ng.var.Scalar().bounded(-2, 5).exponentiated(base=10,
                                                                   coeff=-1),
                      'max_iter': ng.var.Scalar(int).bounded(100, 500),
                      'beta_1': ng.var.Scalar().bounded(.1, .95),
                      'beta_2': ng.var.Scalar().bounded(.1, .95)}

PARAMS['mlp dist es'] = PARAMS['mlp dist'].copy()
PARAMS['mlp dist es']['early_stopping'] = True
PARAMS['mlp dist es']['n_iter_no_change'] = ng.var.Scalar(int).bounded(5, 50)

PARAMS['mlp layers search'] = {'hidden_layer_sizes':
                               ng.var.Array(1, 1, 1).bounded(2, 100)}

# Scalers
PARAMS['base standard'] = {'with_mean': True,
                           'with_std': True}

PARAMS['base minmax'] = {'feature_range': (0, 1)}

PARAMS['base robust'] = {'quantile_range': (5, 95)}

PARAMS['base winsorize'] = {'quantile_range': (1, 99)}

PARAMS['robust gs'] =\
        {'quantile_range': ng.var.OrderedDiscrete(
                [(1, 99), (3, 97), (5, 95), (10, 90), (15, 85),
                 (20, 80), (25, 75), (30, 70), (35, 65), (40, 60)])}

PARAMS['winsorize gs'] =\
        {'quantile_range': ng.var.OrderedDiscrete(
                [(.1, 99.9), (.5, 99.5), (1, 99), (1.5, 98.5), (2, 98),
                 (2.5, 97.5), (3, 97), (3.5, 96.5), (4, 96), (4.5, 95.5),
                 (5, 95)])}

PARAMS['base power'] = {'method': 'yeo-johnson',
                        'standardize': True}

# Feat Selectors
PARAMS['base univar fs regression'] = {'score_func': f_regression,
                                       'percentile': 50}

PARAMS['univar fs regression dist'] = {'score_func': f_regression,
                                       'percentile':
                                       ng.var.Scalar(int).bounded(1, 99)}


PARAMS['base univar fs classifier'] = {'score_func': f_classif,
                                       'percentile': 50}

PARAMS['univar fs classifier dist'] = {'score_func': f_classif,
                                       'percentile':
                                       ng.var.Scalar(int).bounded(1, 99)}


PARAMS['base rfe'] = {'n_features_to_select': None}

PARAMS['rfe num feats dist'] = {'n_features_to_select':
                                ng.var.Scalar().bounded(.01, .99)}

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

# Ensemblers
PARAMS['des default'] = {'needs_split': True,
                         'single_estimator': False}

PARAMS['single default'] = {'needs_split': False,
                            'single_estimator': True}

PARAMS['bb default'] = PARAMS['single default'].copy()


# Feat Importances
PARAMS['base shap'] =\
        {'shap__global__avg_abs': False,
         'shap__linear__feature_dependence': 'independent',
         'shap__linear__nsamples': 1000,
         'shap__tree__feature_dependence': 'tree_path_dependent',
         'shap__tree__model_output': 'margin',
         'shap__tree__tree_limit': None,
         'shap__kernel__nkmean': 10,
         'shap__kernel__nsamples': 'auto',
         'shap__kernel__l1_reg': 'aic'}

PARAMS['base perm'] = {'perm__n_perm': 10}


def get_base_params(str_indicator):

        base_params = PARAMS[str_indicator].copy()
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