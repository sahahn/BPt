from ..helpers import get_obj_and_params

from deslib.dcs import (
    APosteriori, APriori, LCA,
    MCB, MLA, OLA, Rank)

from deslib.des import (
    METADES, DESClustering, DESP,
    DESKNN, KNOP, KNORAE, KNORAU, DESMI,
    RRC,  DESKL, MinimumDifference,
    Exponential, Logarithmic)

from deslib.static import SingleBest
from deslib.static.stacked import StackedClassifier

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
            'aposteriori': 'aposteriori',
            'apriori': 'apriori',
            'lca': 'lca',
            'mcb': 'mcb',
            'mla': 'mla',
            'ola': 'ola',
            'rank': 'rank',
            'metades': 'metades',
            'des clustering': 'des clustering',
            'desp': 'desp',
            'des knn': 'des knn',
            'knop': 'knop',
            'knorae': 'knorae',
            'knrau': 'knrau',
            'desmi': 'desmi',
            'rrc': 'rrc',
            'deskl': 'deskl',
            'min dif': 'min dif',
            'exponential': 'exponential',
            'logarithmic': 'logarithmic',
            'single best': 'single best',
            'stacked': 'stacked',
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
    'aposteriori': (APosteriori, ['default']),
    'apriori': (APriori, ['default']),
    'lca': (LCA, ['default']),
    'mcb': (MCB, ['default']),
    'mla': (MLA, ['default']),
    'ola': (OLA, ['default']),
    'rank': (Rank, ['default']),
    'metades': (METADES, ['default']),
    'des clustering': (DESClustering, ['default']),
    'desp': (DESP, ['default']),
    'des knn': (DESKNN, ['default']),
    'knop': (KNOP, ['default']),
    'knorae': (KNORAE, ['default']),
    'knrau': (KNORAU, ['default']),
    'desmi': (DESMI, ['default']),
    'rrc': (RRC, ['default']),
    'deskl': (DESKL, ['default']),
    'min dif': (MinimumDifference, ['default']),
    'exponential': (Exponential, ['default']),
    'logarithmic': (Logarithmic, ['default']),
    'single best': (SingleBest, ['default']),
    'stacked': (StackedClassifier, ['default']),
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
