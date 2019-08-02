from deslib.des.knora_e import KNORAE
from deslib.dcs.a_posteriori import APosteriori
from deslib.dcs.a_priori import APriori
from deslib.dcs.lca import LCA
from deslib.dcs.mcb import MCB
from deslib.dcs.mla import MLA


AVALIABLE = {
        'binary': {
            'basic ensemble': 'basic ensemble',
            None: 'basic ensemble',
            'knorae': 'knorae',
            'aposteriori': 'aposteriori',
            'apriori': 'apriori',
            'lca': 'lca',
            'mcb': 'mcb',
            'mla': 'mla',
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
    'knorae': (KNORAE, {}),
    'aposteriori': (APosteriori, {}),
    'apriori': (APriori, {}),
    'lca': (LCA, {}),
    'mcb': (MCB, {}),
    'mla': (MLA, {}),
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
