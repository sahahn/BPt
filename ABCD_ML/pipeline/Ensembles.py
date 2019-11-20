import numpy as np

from ..helpers.ML_Helpers import get_avaliable_by_type, get_obj_and_params
from ..helpers.ML_Helpers import show_objects

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

from sklearn.ensemble.voting import _BaseVoting
from copy import deepcopy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from imblearn.ensemble import BalancedBaggingClassifier


class Basic_Ensemble(_BaseVoting):

    def __init__(self, estimators, weights=None, n_jobs=1):
        self.estimators = estimators
        self.weights = None
        self.n_jobs = n_jobs

    def fit(self, X, y, sample_weight=None):

        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        return super().fit(X, y, sample_weight)

    def predict(self, X):
        '''Calls predict on each model and
        returns the averaged prediction. Handling
        different problem types cases.

        Parameters
        ----------
        X : array_like
            The input data to be passed to each model
            for prediction. Should be the right shape.
        '''

        preds = np.array([model.predict(X) for model in self.estimators_])

        try:
            self.estimators_[0].predict_proba([0])
        except AttributeError:
            regression = True
        except ValueError:
            regression = False

        if len(preds[0].shape) == 1:

            if regression:
                return np.mean(preds, axis=0)

            else:
                preds = preds.astype(int)

                class_preds = [np.argmax(np.bincount(preds[:, i]))
                               for i in range(preds.shape[1])]
                class_preds = np.array(class_preds)
                return class_preds

        else:
            return np.round(np.mean(preds, axis=0))

    def predict_proba(self, X):
        '''Calls predict_proba on each model and
        returns the averaged prediction.

        Parameters
        ----------
        X : array_like
            The input data to be passed to each model
            for prediction. Should be the right shape.
        '''

        preds = np.array([model.predict_proba(X) for model in
                         self.estimators_])
        mean_preds = np.mean(preds, axis=0)

        # Multi-label case
        if len(mean_preds.shape) > 2:
            return list(mean_preds)

        return mean_preds

    def get_params(self, deep=True):
        return self._get_params('estimators', deep=deep)

    def set_params(self, **kwargs):
        self._set_params('estimators', **kwargs)
        return self


class DES_Ensemble(_BaseVoting):

    def __init__(self, estimators, ensemble, ensemble_name, ensemble_split,
                 ensemble_params={}, random_state=None, weights=None,
                 n_jobs=1):

        self.estimators = estimators
        self.ensemble = ensemble
        self.ensemble_name = ensemble_name
        self.ensemble_split = ensemble_split
        self.ensemble_params = ensemble_params
        self.random_state = random_state
        self.weights = weights
        self.n_jobs = n_jobs

    def fit(self, X, y, sample_weight=None):
        '''Assume y is multi-class'''

        X_train, X_ensemble, y_train, y_ensemble =\
            train_test_split(X, y, test_size=self.ensemble_split,
                             random_state=self.random_state,
                             stratify=y)

        # Fit estimators
        super().fit(X_train, y_train, sample_weight)

        self.ensemble_ = deepcopy(self.ensemble)
        self.ensemble_.set_params(pool_classifiers=self.estimators_)
        self.ensemble_.set_params(**self.ensemble_params)

        self.ensemble_.fit(X_ensemble, y_ensemble)

        return self

    def predict(self, X):
        return self.ensemble_.predict(X)

    def predict_proba(self, X):
        return self.ensemble_.predict_proba(X)

    def get_params(self, deep=True):
        return self._get_params('estimators', deep=deep)

    def set_params(self, **kwargs):

        ensemble_params = {}

        keys = list(kwargs.keys())
        for param in keys:
            if self.ensemble_name + '__' in param:

                nm = param.split('__')[-1]
                ensemble_params[nm] = kwargs.pop(param)

        self.ensemble_params.update(ensemble_params)

        self._set_params('estimators', **kwargs)
        return self


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
            'bagging': 'bagging classifier',
            'balanced bagging': 'balanced bagging classifier',
        },
        'regression': {
                'basic ensemble': 'basic ensemble',
                None: 'basic ensemble',
                'bagging': 'bagging regressor',
        },
        'multilabel': {
                'basic ensemble': 'basic ensemble',
                None: 'basic ensemble',
        },
}

# Should be the same
AVALIABLE['categorical'] = AVALIABLE['binary'].copy()

ENSEMBLES = {
    'aposteriori': (APosteriori, ['des default']),
    'apriori': (APriori, ['des default']),
    'lca': (LCA, ['des default']),
    'mcb': (MCB, ['des default']),
    'mla': (MLA, ['des default']),
    'ola': (OLA, ['des default']),
    'rank': (Rank, ['des default']),
    'metades': (METADES, ['des default']),
    'des clustering': (DESClustering, ['des default']),
    'desp': (DESP, ['des default']),
    'des knn': (DESKNN, ['des default']),
    'knop': (KNOP, ['des default']),
    'knorae': (KNORAE, ['des default']),
    'knrau': (KNORAU, ['des default']),
    'desmi': (DESMI, ['des default']),
    'rrc': (RRC, ['des default']),
    'deskl': (DESKL, ['des default']),
    'min dif': (MinimumDifference, ['des default']),
    'exponential': (Exponential, ['des default']),
    'logarithmic': (Logarithmic, ['des default']),
    'single best': (SingleBest, ['des default']),
    'stacked': (StackedClassifier, ['des default']),
    'bagging classifier': (BaggingClassifier, ['single default']),
    'bagging regressor': (BaggingRegressor, ['single default']),
    'balanced bagging classifier': (BalancedBaggingClassifier,
                                    ['bb default']),
}


def move_keys(ensemble_params, extra_ensemble_params):

    to_move = ['needs_split', 'single_estimator']

    keys = list(ensemble_params.keys())
    for key in keys:
        for move_key in to_move:
            if move_key in key:
                extra_ensemble_params[move_key] = ensemble_params[key]
                ensemble_params.pop(key)

    return ensemble_params, extra_ensemble_params


def get_ensemble_and_params(ensemble_str, extra_params, params, search_type):

    if ensemble_str == 'basic ensemble' or ensemble_str is None:
        return None, {}

    ensemble, extra_ensemble_params, ensemble_params =\
        get_obj_and_params(ensemble_str, ENSEMBLES, extra_params,
                           params, search_type)

    ensemble_params, extra_ensemble_params =\
        move_keys(ensemble_params, extra_ensemble_params)

    # Slight tweak here, return tuple ensemble, extra_params
    return (ensemble, extra_ensemble_params), ensemble_params


def Show_Ensembles(self, problem_type=None, ensemble=None,
                   show_object=False, possible_params=False):
    '''Print out the avaliable ensemble types,
    optionally restricted by problem type

    Parameters
    ----------
    problem_type : str or None, optional
        Where `problem_type` is the underlying ML problem

        (default = None)

    ensemble : str or list
        Where ensemble is a specific str indicator

        (default = None)

    show_object : bool, optional
        Flag, if set to True, then will print the
        raw sampler object.

        (default = False)

    possible_params: bool, optional
        Flag, if set to True, then will print all
        possible arguments to the classes __init__

        (default = False)
    '''
    print('Visit: ')
    print('https://deslib.readthedocs.io/en/latest/api.html')
    print('For actual descriptions about the different ensemble types,')
    print('as this is the base library used for this functionality!')
    print('More information through this function is avaliable')
    print('By passing optional extra optional params! Please view',
          'the help function for more info!')
    print('Note: the str indicator actually passed during Evaluate / Test')
    print('is listed as ("str indicator")')
    print()

    show_objects(problem_type, ensemble, False, show_object,
                 possible_params, AVALIABLE, ENSEMBLES)
