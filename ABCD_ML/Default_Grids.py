"""
Default_Grids.py
====================================
File with different saved default parameter grids,
for various classifiers within ABCD_ML.
"""

from scipy.stats import (randint as sp_randint, uniform as sp_uniform)
from sklearn.feature_selection import f_regression, f_classif

GRIDS = {}

# Models
GRIDS['base logistic'] = {'solver': ['saga'],
                          'max_iter': [5000],
                          'multi_class': ['auto'],
                          'penalty': ['none']}

GRIDS['base lasso'] = GRIDS['base logistic'].copy()
GRIDS['base lasso']['penalty'] = ['l1']

GRIDS['base ridge'] = GRIDS['base logistic'].copy()
GRIDS['base ridge']['penalty'] = ['l2']

GRIDS['base elastic'] = GRIDS['base logistic'].copy()
GRIDS['base elastic']['penalty'] = ['elasticnet']
GRIDS['base elastic']['l1_ratio'] = [.5]

GRIDS['lasso C'] = GRIDS['base lasso'].copy()
GRIDS['lasso C']['C'] = sp_uniform(loc=1e-4, scale=1e+4)

GRIDS['ridge C'] = GRIDS['base ridge'].copy()
GRIDS['ridge C']['C'] = sp_uniform(loc=1e-4, scale=1e+4)

GRIDS['elastic classifier'] = GRIDS['base elastic'].copy()
GRIDS['elastic classifier']['C'] = sp_uniform(loc=1e-4, scale=1e+4)
GRIDS['elastic classifier']['l1_ratio'] = sp_uniform()

GRIDS['base elastic net'] = {'max_iter': [5000]}
GRIDS['elastic regression'] = GRIDS['base elastic net'].copy()
GRIDS['elastic regression']['alpha'] = sp_uniform(loc=1e-4, scale=1e+4)
GRIDS['elastic regression']['l1_ratio'] = sp_uniform()

GRIDS['base huber'] = {'epsilon': [1.35]}
GRIDS['base gnb'] = {'var_smoothing': [1e-9]}

GRIDS['base knn'] = {'n_neighbors': [5]}
GRIDS['knn rs'] = {'weights': ['uniform', 'distance'],
                   'n_neighbors': sp_randint(2, 20)}

GRIDS['base dt'] = {}
GRIDS['dt rs'] = {'max_depth': sp_randint(1, 20),
                  'min_samples_split': sp_randint(2, 50)}

GRIDS['base linear'] = {'fit_intercept': [True]}

GRIDS['base rf'] = {'n_estimators': [100]}
GRIDS['rf rs'] = {'n_estimators': sp_randint(3, 500),
                  'max_depth': sp_randint(2, 200),
                  'max_features': sp_uniform(),
                  'min_samples_split': sp_uniform(),
                  'bootstrap': [True]},

GRIDS['base lgbm'] = {'silent': [True]}
GRIDS['lgbm rs'] = {'silent': [True],
                    'boosting_type': ['gbdt', 'dart', 'goss'],
                    'n_estimators': sp_randint(3, 500),
                    'num_leaves': sp_randint(6, 50),
                    'min_child_samples': sp_randint(100, 500),
                    'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1,
                                         1e2, 1e3, 1e4],
                    'subsample': sp_uniform(loc=0.2, scale=0.8),
                    'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
                    'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
                    'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]}

GRIDS['base gp regressor'] = {'n_restarts_optimizer': [5],
                              'normalize_y': [True]}
GRIDS['base gp classifier'] = {'n_restarts_optimizer': [5]}


GRIDS['base svm'] = {'kernel': 'rbf',
                     'gamma': 'scale'}

GRIDS['svm rs'] = GRIDS['base svm'].copy()
GRIDS['svm rs']['C'] = [.0001, .001, .10, .1, 1, 5, 10, 25, 50, 100, 500,
                        1000, 5000, 10000]
GRIDS['svm rs']['gamma'] = ['auto', 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

# Scalers
GRIDS['base standard'] = {'with_mean': [True],
                          'with_std': [True]}

GRIDS['base minmax'] = {'feature_range': [(0, 1)]}

GRIDS['base robust'] = {'quantile_range': [(5, 95)]}

GRIDS['base power'] = {'method': ['yeo-johnson'],
                       'standardize': [True]}

GRIDS['base pca'] = {}

# Feat Selectors
GRIDS['base univar fs regression'] = {'score_func': [f_regression],
                                      'percentile': [50]}

GRIDS['base univar fs classifier'] = {'score_func': [f_classif],
                                      'percentile': [50]}


def get(grid_name, preprend):

        params = GRIDS[grid_name].copy()
        params = {preprend + '__' + key: params[key] for key in params}

        return params
