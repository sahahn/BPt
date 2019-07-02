from scipy.stats import (randint as sp_randint, uniform as sp_uniform)

DEFAULT_RF_GRID1 = {
                   "n_estimators": list(range(3,500)),
                   "max_depth": [None] + list(range(2,100,5)),
                   "max_features": sp_uniform(),
                   "min_samples_split": sp_uniform(),
                   "bootstrap": [True],
                   "criterion": ["mse"]
                   }