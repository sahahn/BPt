from sklearn.metrics import SCORERS
import sklearn.metrics as M

AVALIABLE = {
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
        'r2': 'r2',
        'max_error': 'max_error',
        'neg_median_absolute_error': 'neg_median_absolute_error',
        'neg_mean_absolute_error': 'neg_mean_absolute_error',
        'neg_mean_squared_error': 'neg_mean_squared_error',
        'neg_mean_squared_log_error': 'neg_mean_squared_log_error',
        'neg_root_mean_squared_error': 'neg_root_mean_squared_error',
        'neg_mean_poisson_deviance': 'neg_mean_poisson_deviance',
        'neg_mean_gamma_deviance': 'neg_mean_gamma_deviance'
    },
}
AVALIABLE['categorical'] = AVALIABLE['binary'].copy()

SCORERS.update({

    'neg_hamming': M.make_scorer(score_func=M.hamming_loss,
                                 greater_is_better=False,
                                 needs_threshold=False),

    'matthews': M.make_scorer(score_func=M.matthews_corrcoef,
                              greater_is_better=True)

})


def get_scorer_from_str(scorer_str):

    return SCORERS[scorer_str]


def get_scorers_by_type(problem_type):

    objs = []
    for scorer_str in AVALIABLE[problem_type]:
        score_func = get_scorer_from_str(scorer_str)._score_func
        objs.append((scorer_str, score_func))

    return objs
