'''
File with functions related to calculating score from ML models
'''
import numpy as np
from sklearn.metrics import (roc_auc_score, mean_squared_error, r2_score,
                             balanced_accuracy_score, f1_score, log_loss,
                             make_scorer)


def roc_auc_score_wrapper(y_true, y_score, average='macro', sample_weight=None,
                          max_fpr=None):
    '''Wrapper around sklearn roc_auc_score to support multilabel'''

    y_score = mutlilabel_compat(y_score)
    return roc_auc_score(y_true, y_score, average, sample_weight, max_fpr)


def f1_score_wrapper(y_true, y_pred, labels=None, pos_label=1,
                     average='binary'):
    '''Wrapper around sklearn f1_score to support multilabel'''

    y_score = mutlilabel_compat(y_score)
    return f1_score(y_true, y_pred, labels, pos_label, average)


def log_loss_wrapper(y_true, y_pred, eps=1e-15, normalize=True,
                     sample_weight=None):
    '''Wrapper around sklearn log_loss to support multilabel'''

    y_score = mutlilabel_compat(y_score)
    return log_loss(y_true, y_pred, eps, normalize, sample_weight)


def mutlilabel_compat(y_score):
    '''Quick wrapper to help support multilabel output'''

    if type(y_score) == list:
        y_score = np.stack([s[:, 1] for s in y_score], axis=1)

    return y_score


AVALIABLE = {
        'binary': {
                        'f1': 'f1',
                        'roc auc': 'macro roc auc',
                        'macro roc auc': 'macro roc auc',
                        'bas': 'bas',
                        'log': 'log'
            },
        'regression': {
                        'r2': 'r2'
            },
        'categorical': {
            'multilabel': {
                        'weighted roc auc': 'weighted roc auc'
            },
            'multiclass': {
                        'bas': 'bas'
            }
        }
    }

SCORERS = {
    'r2': {'score_func': r2_score, 'greater_is_better': True},
    'mse': {'score_func': mean_squared_error, 'greater_is_better': False},
    'macro roc auc': {'score_func': roc_auc_score_wrapper,
                      'greater_is_better': True, 'needs_proba': True},
    'micro roc auc': {'score_func': roc_auc_score_wrapper,
                      'greater_is_better': True, 'needs_proba': True,
                      'average': 'micro'},
    'weighted roc auc': {'score_func': roc_auc_score_wrapper,
                         'greater_is_better': True, 'needs_proba': True,
                         'average': 'weighted'},
    'bas': {'score_func': balanced_accuracy_score, 'greater_is_better': True},
    'f1': {'score_func': f1_score_wrapper, 'greater_is_better': True},
    'weighted f1': {'score_func': f1_score_wrapper, 'greater_is_better': True,
                    'average': 'weighted'},
    'log': {'score_func': log_loss_wrapper, 'greater_is_better': True,
            'needs_proba': True}
}


def get_scorer(scorer_str):
    '''From a final scorer str indicator, return the
    scorer object.'''

    scorer_params = SCORERS[scorer_str]
    scorer = make_scorer(**scorer_params)
    return scorer
