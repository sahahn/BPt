from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from ..helpers.ML_Helpers import get_obj_and_params


IMPUTERS = {
    'mean': (SimpleImputer, ['mean imp']),
    'median': (SimpleImputer, ['median imp']),
    'most frequent': (SimpleImputer, ['most freq imp']),
    'constant': (SimpleImputer, ['constant imp']),
    'iterative': (IterativeImputer, ['iterative imp']),
}


def get_imputer_and_params(imputer_str, extra_params, params, search_type,
                           random_state=None, num_feat_keys=None):

    imputer, extra_imputer_params, imputer_params =\
        get_obj_and_params(imputer_str, IMPUTERS, extra_params, params)

    return imputer(**extra_imputer_params), imputer_params
