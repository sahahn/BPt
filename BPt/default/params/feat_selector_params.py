
from sklearn.feature_selection import (f_regression, f_classif)
from .Params import Scalar

P = {}

# Feat Selectors
P['base univar fs regression'] = {'score_func': f_regression,
                                  'percentile': 50}

P['univar fs regression dist'] = {'score_func': f_regression,
                                  'percentile':
                                  Scalar(init=50, lower=1, upper=99)}

P['univar fs regression dist2'] = {'score_func': f_regression,
                                   'percentile':
                                   Scalar(init=75, lower=50, upper=99)}


P['base univar fs classifier'] = {'score_func': f_classif,
                                  'percentile': 50}

P['univar fs classifier dist'] = {'score_func': f_classif,
                                  'percentile':
                                  Scalar(init=50, lower=1, upper=99)}

P['univar fs classifier dist2'] = {'score_func': f_classif,
                                   'percentile':
                                   Scalar(init=75, lower=50, upper=99)}


P['base rfe'] = {'n_features_to_select': None}

P['rfe num feats dist'] = {'n_features_to_select':
                           Scalar(init=.5, lower=.1, upper=.99)}

P['random'] = {'mask': 'sets as random features'}
P['searchable'] = {'mask': 'sets as hyperparameters'}
