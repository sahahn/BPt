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
         'class_weight': ng.var.SoftmaxCategorical([None, 'balanced'])}

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

PARAMS['ridge C'] = PARAMS['base ridge'].copy()
PARAMS['ridge C']['C'] =\
        ng.var.Scalar().bounded(-4, 4).exponentiated(base=10, coeff=-1)

PARAMS['elastic classifier'] = PARAMS['base elastic'].copy()
PARAMS['elastic classifier']['C'] =\
        ng.var.Scalar().bounded(-4, 4).exponentiated(base=10, coeff=-1)
PARAMS['elastic classifier']['l1_ratio'] = ng.var.Scalar().bounded(0, 1)

PARAMS['base elastic net'] = {'max_iter': 5000}
PARAMS['elastic regression'] = PARAMS['base elastic net'].copy()
PARAMS['elastic regression']['alpha'] =\
        ng.var.Scalar().bounded(-2, 5).exponentiated(base=10, coeff=-1)
PARAMS['elastic regression']['l1_ratio'] = ng.var.Scalar().bounded(0, 1)

PARAMS['base huber'] = {'epsilon': 1.35}
PARAMS['base gnb'] = {'var_smoothing': 1e-9}

PARAMS['base knn'] = {'n_neighbors': 5}
PARAMS['knn rs'] = {'weights':
                    ng.var.SoftmaxCategorical(['uniform', 'distance']),
                    'n_neighbors': ng.var.Scalar(int).bounded(2, 25)}

PARAMS['base dt'] = {}
PARAMS['dt rs'] = {'max_depth': ng.var.Scalar(int).bounded(1, 30),
                   'min_samples_split': ng.var.Scalar(int).bounded(2, 50)}

PARAMS['dt classifier rs'] = PARAMS['dt rs'].copy()
PARAMS['dt classifier rs']['class_weight'] =\
        ng.var.SoftmaxCategorical([None, 'balanced'])

PARAMS['base linear'] = {'fit_intercept': True}

PARAMS['base rf'] = {'n_estimators': 100}
PARAMS['rf rs'] = {'n_estimators': ng.var.Scalar(int).bounded(3, 500),
                   'max_depth': ng.var.Scalar(int).bounded(2, 200),
                   'max_features': ng.var.Scalar().bounded(0, 1),
                   'min_samples_split': ng.var.Scalar().bounded(0, 1),
                   'bootstrap': True}

PARAMS['rf classifier rs'] = PARAMS['rf rs'].copy()
PARAMS['rf classifier rs']['class_weight'] =\
        ng.var.SoftmaxCategorical([None, 'balanced'])

PARAMS['base lgbm'] = {'silent': True}
PARAMS['lgbm rs1'] = {'silent': True,
                      'boosting_type':
                      ng.var.OrderedDiscrete(['gbdt', 'dart', 'goss']),
                      # ng.var.SoftmaxCategorical(),
                      'n_estimators': ng.var.Scalar(int).bounded(3, 500),
                      'num_leaves': ng.var.Scalar(int).bounded(6, 80),
                      'min_child_samples': ng.var.Scalar(int).bounded(10, 500),
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

PARAMS['lgbm rs2'] = {'silent': True,
                      'lambda_l2': 0.001,
                      'boosting_type':
                      ng.var.SoftmaxCategorical(['gbdt', 'dart']),
                      'min_child_samples':
                      ng.var.OrderedDiscrete([1, 5, 7, 10, 15, 20, 35, 50, 100,
                                              200, 500, 1000]),
                      'num_leaves':
                      ng.var.OrderedDiscrete([2, 4, 7, 10, 15, 20, 25, 30, 35,
                                              40, 50, 65, 80, 100, 125, 150,
                                              200, 250]),
                      'colsample_bytree':
                      ng.var.OrderedDiscrete([0.7, 0.9, 1.0]),
                      'subsample':
                      ng.var.Scalar().bounded(.3, 1),
                      'learning_rate':
                      ng.var.OrderedDiscrete([0.01, 0.05, 0.1]),
                      'n_estimators':
                      ng.var.OrderedDiscrete([5, 20, 35, 50, 75, 100, 150, 200,
                                              350, 500, 750, 1000])}

PARAMS['lgbm classifier rs1'] = PARAMS['lgbm rs1'].copy()
PARAMS['lgbm classifier rs1']['class_weight'] =\
        ng.var.OrderedDiscrete([None, 'balanced'])
        # ng.var.SoftmaxCategorical([None, 'balanced'])

PARAMS['lgbm classifier rs2'] = PARAMS['lgbm rs2'].copy()
PARAMS['lgbm classifier rs2']['class_weight'] =\
        ng.var.SoftmaxCategorical([None, 'balanced'])

PARAMS['base lgbm es'] = {'silent': True,
                          'val_split_percent': .1,
                          'early_stop_rounds': 50}

PARAMS['lgbm es rs'] = PARAMS['lgbm rs1'].copy()
PARAMS['lgbm es rs']['val_split_percent'] = ng.var.Scalar().bounded(.05, .2)
PARAMS['lgbm es rs']['early_stop_rounds'] = ng.var.Scalar(int).bounded(10, 150)

PARAMS['base xgb'] = {'verbosity': 0}

PARAMS['xgb rs'] =\
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

PARAMS['svm rs'] = PARAMS['base svm'].copy()
PARAMS['svm rs']['C'] =\
        ng.var.Scalar().bounded(-4, 4).exponentiated(base=10,
                                                     coeff=-1)
PARAMS['svm rs']['gamma'] =\
        ng.var.Scalar().bounded(1, 6).exponentiated(base=10,
                                                    coeff=-1)

PARAMS['base svm classifier'] = PARAMS['base svm'].copy()
PARAMS['base svm classifier']['probability'] = True

PARAMS['svm classifier rs'] = PARAMS['svm rs'].copy()
PARAMS['svm classifier rs']['probability'] = True
PARAMS['svm classifier rs']['class_weight'] =\
        ng.var.SoftmaxCategorical([None, 'balanced'])

PARAMS['base mlp'] = {}

PARAMS['base lasso regressor'] = {'max_iter': 5000}
PARAMS['lasso regressor rs'] =\
        {'alpha': ng.var.Scalar().bounded(-4, 5).exponentiated(base=10,
                                                               coeff=-1)}

PARAMS['base ridge regressor'] = PARAMS['base lasso regressor'].copy()
PARAMS['ridge regressor rs'] =\
        {'alpha': ng.var.Scalar().bounded(-4, 5).exponentiated(base=10,
                                                               coeff=-1)}


PARAMS['mlp rs'] = {'hidden_layer_sizes':
                    ng.var.Array(1, 1, 1).bounded(2, 100),
                    'activation':
                    ng.var.SoftmaxCategorical(['identity', 'logistic',
                                               'tanh', 'relu']),
                    'alpha':
                    ng.var.Scalar().bounded(-2, 5).exponentiated(base=10,
                                                                 coeff=-1),
                    'batch_size': ng.var.Scalar(int).bounded(2, 200),
                    'learning_rate':
                    ng.var.SoftmaxCategorical(['constant', 'invscaling',
                                               'adaptive']),
                    'learning_rate_init':
                    ng.var.Scalar().bounded(-2, 5).exponentiated(base=10,
                                                                 coeff=-1),
                    'max_iter': ng.var.Scalar(int).bounded(100, 500),
                    'beta_1': ng.var.Scalar().bounded(.1, .95),
                    'beta_2': ng.var.Scalar().bounded(.1, .95)}

PARAMS['mlp rs es'] = PARAMS['mlp rs'].copy()
PARAMS['mlp rs es']['early_stopping'] = True
PARAMS['mlp rs es']['n_iter_no_change'] = ng.var.Scalar(int).bounded(5, 50)

PARAMS['mlp layers search'] = {'hidden_layer_sizes':
                               ng.var.Array(1, 1, 1).bounded(2, 100)}

# Scalers
PARAMS['base standard'] = {'with_mean': True,
                           'with_std': True}

PARAMS['base minmax'] = {'feature_range': (0, 1)}

PARAMS['base robust'] = {'quantile_range': (5, 95)}

PARAMS['robust gs'] =\
        {'quantile_range': ng.var.OrderedDiscrete(
                [(1, 99), (5, 95), (10, 90), (15, 85),
                 (20, 80), (25, 75), (30, 70),
                 (35, 65), (40, 60)])}

PARAMS['base power'] = {'method': 'yeo-johnson',
                        'standardize': True}

# Feat Selectors
PARAMS['base univar fs regression'] = {'score_func': f_regression,
                                       'percentile': 50}

PARAMS['univar fs regression rs'] = {'score_func': f_regression,
                                     'percentile':
                                     ng.var.Scalar(int).bounded(1, 99)}


PARAMS['base univar fs classifier'] = {'score_func': [f_classif],
                                       'percentile': [50]}

PARAMS['univar fs classifier rs'] = {'score_func': [f_classif],
                                     'percentile':
                                     ng.var.Scalar(int).bounded(1, 99)}


PARAMS['base rfe'] = {'n_features_to_select': None}

PARAMS['rfe num feats rs'] = {'n_features_to_select':
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


def get_base_params(str_indicator):

        base_params = PARAMS[str_indicator].copy()
        return base_params


def proc_params(base_params, prepend=None):

        # Return dict with prepend on all keys
        if prepend is not None:
                params = {prepend + '__' + key: base_params[key] for key in
                          base_params}

        # Grabs param grid as single set of fixed params
        else:
                params = {}
                for key in base_params:
                        params[key] = base_params[key]

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
                                print('Distribution Over',
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
