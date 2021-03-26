from ..helpers import get_obj_and_params, all_from_avaliable

from sklearn.ensemble import (BaggingClassifier, BaggingRegressor,
                              AdaBoostRegressor, AdaBoostClassifier)
from ...pipeline.ensemble_wrappers import (BPtStackingRegressor,
                                           BPtVotingRegressor,
                                           BPtStackingClassifier,
                                           BPtVotingClassifier)

AVALIABLE = {
        'binary': {
            'basic ensemble': 'voting classifier',
            None: 'voting classifier',
            'voting': 'voting classifier',
            'bagging': 'bagging classifier',
            'adaboost': 'adaboost classifier',
            'stacking': 'stacking classifier',
        },
        'regression': {
                'basic ensemble': 'voting regressor',
                None: 'voting regressor',
                'voting': 'voting regressor',
                'bagging': 'bagging regressor',
                'stacking': 'stacking regressor',
                'adaboost': 'adaboost regressor',
        },
}


ENSEMBLES = {
    'bagging classifier': (BaggingClassifier, ['default']),
    'bagging regressor': (BaggingRegressor, ['default']),
    'adaboost classifier': (AdaBoostClassifier, ['default']),
    'adaboost regressor': (AdaBoostRegressor, ['default']),
    'stacking regressor': (BPtStackingRegressor,
                           ['default']),
    'stacking classifier': (BPtStackingClassifier,
                            ['default']),
    'voting classifier': (BPtVotingClassifier,
                          ['voting classifier']),
    'voting regressor': (BPtVotingRegressor,
                         ['default']),
}

try:

    from imblearn.ensemble import BalancedBaggingClassifier

    AVALIABLE['binary']['balanced bagging'] = 'balanced bagging classifier'

    ENSEMBLES['balanced bagging classifier'] =\
        (BalancedBaggingClassifier, ['default'])

except ImportError:
    pass

# Should be the same
AVALIABLE['categorical'] = AVALIABLE['binary'].copy()


def get_ensemble_and_params(ensemble_str, extra_params, params,
                            **kwargs):

    if ensemble_str == 'basic ensemble' or ensemble_str is None:
        return None, {}

    ensemble, extra_ensemble_params, ensemble_params =\
        get_obj_and_params(ensemble_str, ENSEMBLES, extra_params, params)

    # Slight tweak here, return tuple ensemble, extra_params
    return (ensemble, extra_ensemble_params), ensemble_params


all_obj_keys = all_from_avaliable(AVALIABLE)
