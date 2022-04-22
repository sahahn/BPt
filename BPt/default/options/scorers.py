from sklearn.metrics import SCORERS
import sklearn.metrics as M
from ..helpers import proc_type_dep_str
from ...util import conv_to_list
from joblib import wrap_non_picklable_objects

AVAILABLE = {
    'binary': {
        'accuracy': 'accuracy',
        'roc_auc': 'roc_auc',
        'roc_auc_ovr': 'roc_auc_ovr',
        'roc_auc_ovo': 'roc_auc_ovo',
        'roc_auc_ovr_weighted': 'roc_auc_ovr_weighted',
        'roc_auc_ovo_weighted': 'roc_auc_ovo_weighted',
        'balanced_accuracy': 'balanced_accuracy',
        'average_precision': 'average_precision',
        'neg_log_loss': 'neg_log_loss',
        'neg_brier_score': 'neg_brier_score',
        'precision': 'precision',
        'precision_macro': 'precision_macro',
        'precision_micro': 'precision_micro',
        'precision_samples': 'precision_samples',
        'precision_weighted': 'precision_weighted',
        'recall': 'recall',
        'recall_macro': 'recall_macro',
        'recall_micro': 'recall_micro',
        'recall_samples': 'recall_samples',
        'recall_weighted': 'recall_weighted',
        'f1': 'f1',
        'f1_macro': 'f1_macro',
        'f1_micro': 'f1_micro',
        'f1_samples': 'f1_samples',
        'f1_weighted': 'f1_weighted',
        'jaccard': 'jaccard',
        'jaccard_macro': 'jaccard_macro',
        'jaccard_micro': 'jaccard_micro',
        'jaccard_samples': 'jaccard_samples',
        'jaccard_weighted': 'jaccard_weighted',
        'neg_hamming': 'neg_hamming',
        'matthews': 'matthews',
    },
    'regression': {
        'explained_variance': 'explained_variance',
        'explained_variance score': 'explained_variance',
        'r2': 'r2',
        'max_error': 'max_error',
        'neg_median_absolute_error': 'neg_median_absolute_error',
        'median_absolute_error': 'neg_median_absolute_error',
        'neg_mean_absolute_error': 'neg_mean_absolute_error',
        'mean_absolute_error': 'neg_mean_absolute_error',
        'neg_mean_squared_error': 'neg_mean_squared_error',
        'mean_squared_error': 'neg_mean_squared_error',
        'neg_mean_squared_log_error': 'neg_mean_squared_log_error',
        'mean_squared_log_error': 'neg_mean_squared_log_error',
        'neg_root_mean_squared_error': 'neg_root_mean_squared_error',
        'root_mean_squared_error': 'neg_root_mean_squared_error',
        'neg_mean_poisson_deviance': 'neg_mean_poisson_deviance',
        'mean_poisson_deviance': 'neg_mean_poisson_deviance',
        'neg_mean_gamma_deviance': 'neg_mean_gamma_deviance',
        'mean_gamma_deviance': 'neg_mean_gamma_deviance',
    },
}
AVAILABLE['categorical'] = AVAILABLE['binary'].copy()

# Set defaults
AVAILABLE['binary']['default'] = 'roc_auc'
AVAILABLE['regression']['default'] = 'r2'
AVAILABLE['categorical']['default'] = 'roc_auc_ovr'

SCORERS.update({

    'neg_hamming': M.make_scorer(score_func=M.hamming_loss,
                                 greater_is_better=False,
                                 needs_threshold=False),

    'matthews': M.make_scorer(score_func=M.matthews_corrcoef,
                              greater_is_better=True)

})

default_scorers = {'regression': ['explained_variance',
                                  'neg_mean_squared_error'],
                   'binary': ['matthews', 'roc_auc',
                              'balanced_accuracy'],
                   'categorical': ['matthews', 'roc_auc_ovr',
                                   'balanced_accuracy']}


def get_scorer_from_str(scorer_str):

    return SCORERS[scorer_str]


def get_scorers_by_type(problem_type):

    objs = []
    for scorer_str in AVAILABLE[problem_type]:
        conv = AVAILABLE[problem_type][scorer_str]
        score_func = get_scorer_from_str(conv)._score_func
        objs.append((scorer_str, score_func))

    return objs


def _proc_scorer(scorer_str, problem_type, cnt=0):

    if isinstance(scorer_str, str):
        name = proc_type_dep_str(scorer_str, AVAILABLE, problem_type)
        return get_scorer_from_str(name), name, cnt
    else:
        name = 'Custom Scorer ' + str(cnt)
        cnt += 1
        return wrap_non_picklable_objects(scorer_str), name, cnt
        

def process_scorers(in_scorers, problem_type):

    # TODO add sklean style check scorer?

    # If already correct scorers dict.
    # Check each item in dict
    if isinstance(in_scorers, dict):

        scorers = {}
        for key in in_scorers:

            # Use passed name
            scorer, _, _ = _proc_scorer(in_scorers[key], problem_type)
            scorers[key] = scorer

        return scorers

    # Check for default case
    if in_scorers == 'default':
        in_scorers = default_scorers[problem_type]

    # get scorer_strs as initially list
    scorer_strs = conv_to_list(in_scorers)

    scorers, cnt = {}, 0
    for m in range(len(scorer_strs)):
        scorer, name, cnt = _proc_scorer(scorer_strs[m], problem_type, cnt=cnt)
        scorers[name] = scorer

    return scorers


def process_scorer(scorer_str, problem_type):

    # Process scorer strs if default
    if scorer_str == 'default':
        scorer_str = default_scorers[problem_type][0]

    # If passed as str
    if isinstance(scorer_str, str):
        scorer_str = proc_type_dep_str(scorer_str, AVAILABLE, problem_type)
        return get_scorer_from_str(scorer_str)

    # Otherwise assume function and wrap
    return wrap_non_picklable_objects(scorer_str)
