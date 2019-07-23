"""
Default_Grids.py
====================================
File with different saved default parameter grids,
for various classifiers within ABCD_ML.
"""

from scipy.stats import (randint as sp_randint, uniform as sp_uniform)

GRIDS = {
        'REGRESSION1': {'C': sp_uniform(loc=1e-4, scale=1e+4)},

        'ELASTIC1': {'C': sp_uniform(loc=1e-4, scale=1e+4),
                     'l1_ratio': sp_uniform()},

        'ELASTIC2': {'alpha': sp_uniform(loc=1e-4, scale=1e+4),
                     'l1_ratio': sp_uniform()},

        'RF1': {'n_estimators': list(range(3, 500)),
                'max_depth': [None] + list(range(2, 100, 5)),
                'max_features': sp_uniform(),
                'min_samples_split': sp_uniform(),
                'bootstrap': [True]},

        'DTC1': {'max_depth': list(range(1, 20)),
                 'min_samples_split': list(range(2, 50))},

        'KNN1': {'weights': ['uniform', 'distance'],
                 'n_neighbors': list(range(1, 20))},

        'SVM1': {'C': [.0001, .001, .10, .1, 1, 5, 10, 25, 50, 100, 500, 1000,
                 5000, 10000],
                 'gamma': ['auto', 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]},

        'LIGHT1': {'boosting_type': ['gbdt', 'dart', 'goss'],
                   'n_estimators': sp_randint(3, 500),
                   'num_leaves': sp_randint(6, 50),
                   'min_child_samples': sp_randint(100, 500),
                   'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1,
                                        1e2, 1e3, 1e4],
                   'subsample': sp_uniform(loc=0.2, scale=0.8),
                   'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
                   'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
                   'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]},
}


def get(grid_name, preprend):

        params = GRIDS[grid_name].copy()
        params = {preprend + '__' + key: params[key] for key in params}

        return params
