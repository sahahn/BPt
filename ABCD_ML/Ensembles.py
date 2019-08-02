from deslib.dcs.a_posteriori import APosteriori
from deslib.dcs.a_priori import APriori
from deslib.dcs.lca import LCA
from deslib.dcs.mcb import MCB
from deslib.dcs.mla import MLA
from deslib.dcs.ola import OLA
from deslib.dcs.rank import Rank

from deslib.des.meta_des import METADES
from deslib.des.des_clustering import DESClustering
from deslib.des.des_p import DESP
from deslib.des.des_knn import DESKNN
from deslib.des.knop import KNOP
from deslib.des.knora_e import KNORAE
from deslib.des.knora_u import KNORAU
from deslib.des.des_mi import DESMI

from deslib.des.probabilistic import RRC
from deslib.des.probabilistic import DESKL
from deslib.des.probabilistic import MinimumDifference
from deslib.des.probabilistic import Exponential
from deslib.des.probabilistic import Logarithmic

from deslib.static.single_best import SingleBest
from deslib.static.stacked import StackedClassifier

AVALIABLE = {
        'binary': {
            'basic ensemble': 'basic ensemble',
            None: 'basic ensemble',
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
        },
        'regression': {
                'basic ensemble': 'basic ensemble',
                None: 'basic ensemble',
        },
        'categorical': {
            'multilabel': {
                'basic ensemble': 'basic ensemble',
                None: 'basic ensemble',
            },
        }
}

# Should be the same
AVALIABLE['categorical']['multiclass'] = AVALIABLE['binary'].copy()

ENSEMBLES = {
    'aposteriori': (APosteriori, {}),
    'apriori': (APriori, {}),
    'lca': (LCA, {}),
    'mcb': (MCB, {}),
    'mla': (MLA, {}),
    'rank': (Rank, {}),
    'metades': (METADES, {}),
    'des clustering': (DESClustering, {}),
    'desp': (DESP, {}),
    'des knn': (DESKNN, {}),
    'knop': (KNOP, {}),
    'knorae': (KNORAE, {}),
    'knrau': (KNORAU, {}),
    'desmi': (DESMI, {}),
    'rrc': (RRC, {}),
    'deskl': (DESKL, {}),
    'min dif': (MinimumDifference, {}),
    'exponential': (Exponential, {}),
    'logarithmic': (Logarithmic, {}),
    'single best': (SingleBest, {}),
    'stacked': (StackedClassifier, {}),
}


def get_ensemble_and_params(ensemble_str, extra_params):

    if ensemble_str == 'basic ensemble' or ensemble_str is None:
        return None, None

    try:
        ensemble, params = ENSEMBLES[ensemble_str]
    except KeyError:
        print('Requested ensemble str:', ensemble_str, 'does not exist!')

    # Update params if any user passed in extra params
    if ensemble_str in extra_params:
        params.update(extra_params[ensemble_str])

    return ensemble, params
