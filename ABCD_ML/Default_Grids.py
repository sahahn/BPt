"""
Default_Grids.py
====================================
File with different saved default parameter grids,
for various classifiers within ABCD_ML.
"""

from scipy.stats import (randint as sp_randint, uniform as sp_uniform)

RF_GRID1 = {'n_estimators': list(range(3, 500)),
            'max_depth': [None] + list(range(2, 100, 5)),
            'max_features': sp_uniform(),
            'min_samples_split': sp_uniform(),
            'bootstrap': [True]}

DTC_GRID1 = {'max_depth': list(range(1, 20)),
             'min_samples_split': list(range(2, 50))}

KNN_GRID1 = {'weights': ['uniform', 'distance'],
             'n_neighbors': list(range(1, 20))}

LIGHT_GRID1 = {'boosting_type': ['gbdt', 'dart', 'goss'],
               'n_estimators': sp_randint(3, 500),
               'num_leaves': sp_randint(6, 50),
               'min_child_samples': sp_randint(100, 500),
               'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1,
                                    1e2, 1e3, 1e4],
               'subsample': sp_uniform(loc=0.2, scale=0.8),
               'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
               'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
               'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]}
