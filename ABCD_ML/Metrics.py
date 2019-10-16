"""
Metrics.py
====================================
File with functions related to calculating metrics/scores from ML models
"""
import numpy as np
import sklearn.metrics as M
from sklearn.preprocessing import label_binarize
from ABCD_ML.ML_Helpers import get_avaliable_by_type


def roc_auc_score_wrapper(y_true, y_score, average='macro', sample_weight=None,
                          max_fpr=None, multiclass=False):
    '''Wrapper around sklearn roc_auc_score to support multilabel and
       multiclass.'''

    if multiclass:
        y_true = binarize(y_true)

    y_score = mutlilabel_compat(y_score)

    return M.roc_auc_score(y_true, y_score, average, sample_weight, max_fpr)


def f1_score_wrapper(y_true, y_pred, labels=None, pos_label=1,
                     average='binary'):
    '''Wrapper around sklearn f1_score to support multilabel'''

    y_pred = mutlilabel_compat(y_pred)
    return M.f1_score(y_true, y_pred, labels, pos_label, average)


def recall_score_wrapper(y_true, y_pred, label=None, pos_label=1,
                         average='binary', sample_weight=None):
    '''Wrapper around sklearn recall to support multilabel'''

    y_pred = mutlilabel_compat(y_pred)
    return M.recall_score(y_true, y_pred, label, pos_label, average,
                          sample_weight)


def precision_score_wrapper(y_true, y_pred, label=None, pos_label=1,
                            average='binary', sample_weight=None):
    '''Wrapper around sklearn precision_score to support multilabel'''

    y_pred = mutlilabel_compat(y_pred)
    return M.precision_score(y_true, y_pred, label, pos_label, average,
                             sample_weight)


def log_loss_wrapper(y_true, y_pred, eps=1e-15, normalize=True,
                     sample_weight=None):
    '''Wrapper around sklearn log_loss to support multilabel'''

    y_pred = mutlilabel_compat(y_pred)
    return M.log_loss(y_true, y_pred, eps, normalize, sample_weight)


def average_precision_score_wrapper(y_true, y_score, average="macro",
                                    pos_label=1, sample_weight=None,
                                    multiclass=False):
    '''Wrapper around sklearn average_precision score to support multilabel'''

    if multiclass:
        y_true = binarize(y_true)

    y_score = mutlilabel_compat(y_score)
    return M.average_precision_score(y_true, y_score, average, pos_label,
                                     sample_weight)


def jaccard_score_wrapper(y_true, y_pred, labels=None, pos_label=1,
                          average='binary', sample_weight=None):
    '''Wrapper around sklearn jaccard_score to support multilabel'''

    y_pred = mutlilabel_compat(y_pred)
    return M.jaccard_score(y_true, y_pred, labels, pos_label, average,
                           sample_weight)


def mutlilabel_compat(y_score):
    '''Quick wrapper to help support multilabel output'''

    if type(y_score) == list:
        y_score = np.stack([s[:, 1] for s in y_score], axis=1)

    return y_score


def binarize(y_true):

    classes = np.unique(y_true)
    y_true = label_binarize(y_true, classes)
    return y_true


AVALIABLE = {
        'binary': {
                        'f1': 'f1',
                        'recall': 'recall',
                        'precision': 'precision',
                        'roc auc': 'macro roc auc',
                        'macro roc auc': 'macro roc auc',
                        'average precision': 'macro average precision',
                        'macro average precision': 'macro average precision',
                        'balanced accuracy': 'balanced accuracy',
                        'log': 'log',
                        'accuracy': 'accuracy',
                        'brier': 'brier',
                        'hamming': 'hamming',
                        'jaccard': 'jaccard',
                        'matthews': 'matthews',
            },
        'regression': {
                        'r2': 'r2',
                        'mean squared error': 'mean squared error',
                        'explained variance': 'explained variance',
                        'max error': 'max error',
                        'mean absolute error': 'mean absolute error',
                        'mean squared log error': 'mean squared log error',
                        'median absolute error': 'median absolute error',
            },
        'multilabel': {
                        'weighted roc auc': 'weighted roc auc',
                        'macro roc auc': 'macro roc auc',
                        'micro roc auc': 'micro roc auc',
                        'samples roc auc': 'samples roc auc',
                        'by class roc auc': 'by class roc auc',
                        'weighted f1': 'weighted f1',
                        'macro f1': 'macro f1',
                        'micro f1': 'micro f1',
                        'samples f1': 'samples f1',
                        'by class f1': 'by class f1',
                        'weighted recall': 'weighted recall',
                        'macro recall': 'macro recall',
                        'micro recall': 'micro recall',
                        'samples recall': 'samples recall',
                        'by class recall': 'by class recall',
                        'weighted precision': 'weighted precision',
                        'macro precision': 'macro precision',
                        'micro precision': 'micro precision',
                        'samples precision': 'samples precision',
                        'by class precision': 'by class precision',
                        'log': 'log',
                        'accuracy': 'accuracy',
                        'macro average precision': 'macro average precision',
                        'micro average precision': 'micro average precision',
                        'weighted average precision':
                        'weighted average precision',
                        'samples average precision':
                        'samples average precision',
                        'by class average precision':
                        'by class average precision',
                        'hamming': 'hamming',
                        'weighted jaccard': 'weighted jaccard',
                        'macro jaccard': 'macro jaccard',
                        'micro jaccard': 'micro jaccard',
                        'samples jaccard': 'samples jaccard',
                        'by class jaccard': 'by class jaccard',
            },
        'categorical': {
                        'weighted roc auc': 'multiclass weighted roc auc',
                        'macro roc auc': 'multiclass macro roc auc',
                        'micro roc auc': 'multiclass micro roc auc',
                        'samples roc auc': 'multiclass samples roc auc',
                        'by class roc auc': 'multiclass by class roc auc',
                        'balanced accuracy': 'balanced accuracy',
                        'accuracy': 'accuracy',
                        'macro average precision':
                        'multiclass macro average precision',
                        'micro average precision':
                        'multiclass micro average precision',
                        'weighted average precision':
                        'multiclass weighted average precision',
                        'samples average precision':
                        'multiclass samples average precision',
                        'by class average precision':
                        'multiclass by class average precision',
                        'log': 'log',
                        'hamming': 'hamming',
                        'weighted f1': 'weighted f1',
                        'macro f1': 'macro f1',
                        'micro f1': 'micro f1',
                        'by class f1': 'by class f1',
                        'weighted recall': 'weighted recall',
                        'macro recall': 'macro recall',
                        'micro recall': 'micro recall',
                        'by class recall': 'by class recall',
                        'weighted precision': 'weighted precision',
                        'macro precision': 'macro precision',
                        'micro precision': 'micro precision',
                        'by class precision': 'by class precision',
                        'weighted jaccard': 'weighted jaccard',
                        'macro jaccard': 'macro jaccard',
                        'micro jaccard': 'micro jaccard',
                        'by class jaccard': 'by class jaccard',
                        'matthews': 'matthews',
        }
    }

METRICS = {
    'r2': {'score_func': M.r2_score, 'greater_is_better': True},

    'mean squared error': {'score_func': M.mean_squared_error,
                           'greater_is_better': False},

    'explained variance': {'score_func': M.explained_variance_score,
                           'greater_is_better': True},

    'max error': {'score_func': M.max_error, 'greater_is_better': False},

    'mean absolute error': {'score_func': M.mean_absolute_error,
                            'greater_is_better': False},

    'mean squared log error': {'score_func': M.mean_squared_log_error,
                               'greater_is_better': False},

    'median absolute error': {'score_func': M.median_absolute_error,
                              'greater_is_better': False},

    'macro roc auc': {'score_func': roc_auc_score_wrapper,
                      'greater_is_better': True, 'needs_proba': True,
                      'average': 'macro'},

    'micro roc auc': {'score_func': roc_auc_score_wrapper,
                      'greater_is_better': True, 'needs_proba': True,
                      'average': 'micro'},

    'weighted roc auc': {'score_func': roc_auc_score_wrapper,
                         'greater_is_better': True, 'needs_proba': True,
                         'average': 'weighted'},

    'samples roc auc': {'score_func': roc_auc_score_wrapper,
                        'greater_is_better': True, 'needs_proba': True,
                        'average': 'samples'},

    'by class roc auc': {'score_func': roc_auc_score_wrapper,
                         'greater_is_better': True, 'needs_proba': True,
                         'average': None},

    'multiclass weighted roc auc': {'score_func': roc_auc_score_wrapper,
                                    'greater_is_better': True,
                                    'needs_proba': True, 'average': 'weighted',
                                    'multiclass': True},

    'multiclass macro roc auc': {'score_func': roc_auc_score_wrapper,
                                 'greater_is_better': True,
                                 'needs_proba': True, 'average': 'macro',
                                 'multiclass': True},

    'multiclass micro roc auc': {'score_func': roc_auc_score_wrapper,
                                 'greater_is_better': True,
                                 'needs_proba': True, 'average': 'micro',
                                 'multiclass': True},

    'multiclass samples roc auc': {'score_func': roc_auc_score_wrapper,
                                   'greater_is_better': True,
                                   'needs_proba': True, 'average': 'samples',
                                   'multiclass': True},

    'multiclass by class roc auc': {'score_func': roc_auc_score_wrapper,
                                    'greater_is_better': True,
                                    'needs_proba': True, 'average': None,
                                    'multiclass': True},

    'balanced accuracy': {'score_func': M.balanced_accuracy_score,
                          'greater_is_better': True},

    'f1': {'score_func': f1_score_wrapper, 'greater_is_better': True},

    'weighted f1': {'score_func': f1_score_wrapper, 'greater_is_better': True,
                    'average': 'weighted'},

    'macro f1': {'score_func': f1_score_wrapper, 'greater_is_better': True,
                 'average': 'macro'},

    'micro f1': {'score_func': f1_score_wrapper, 'greater_is_better': True,
                 'average': 'micro'},

    'samples f1': {'score_func': f1_score_wrapper, 'greater_is_better': True,
                   'average': 'samples'},

    'by class f1': {'score_func': f1_score_wrapper, 'greater_is_better': True,
                    'average': None},

    'recall': {'score_func': recall_score_wrapper, 'greater_is_better': True},

    'weighted recall': {'score_func': recall_score_wrapper,
                        'greater_is_better': True, 'average': 'weighted'},

    'macro recall': {'score_func': recall_score_wrapper,
                     'greater_is_better': True, 'average': 'macro'},

    'micro recall': {'score_func': recall_score_wrapper,
                     'greater_is_better': True, 'average': 'micro'},

    'samples recall': {'score_func': recall_score_wrapper,
                       'greater_is_better': True, 'average': 'samples'},

    'by class recall': {'score_func': recall_score_wrapper,
                        'greater_is_better': True, 'average': None},

    'precision': {'score_func': precision_score_wrapper,
                  'greater_is_better': True},

    'weighted precision': {'score_func': precision_score_wrapper,
                           'greater_is_better': True, 'average': 'weighted'},

    'macro precision': {'score_func': precision_score_wrapper,
                        'greater_is_better': True, 'average': 'macro'},

    'micro precision': {'score_func': precision_score_wrapper,
                        'greater_is_better': True, 'average': 'micro'},

    'samples precision': {'score_func': precision_score_wrapper,
                          'greater_is_better': True, 'average': 'samples'},

    'by class precision': {'score_func': precision_score_wrapper,
                           'greater_is_better': True, 'average': None},

    'log': {'score_func': log_loss_wrapper, 'greater_is_better': True,
            'needs_proba': True},

    'accuracy': {'score_func': M.accuracy_score, 'greater_is_better': True},

    'macro average precision': {
        'score_func': average_precision_score_wrapper,
        'greater_is_better': True, 'needs_proba': True, 'average': 'macro'},

    'micro average precision': {
        'score_func': average_precision_score_wrapper,
        'greater_is_better': True, 'needs_proba': True, 'average': 'micro'},

    'weighted average precision': {
        'score_func': average_precision_score_wrapper,
        'greater_is_better': True, 'needs_proba': True, 'average': 'weighted'},

    'samples average precision': {
        'score_func': average_precision_score_wrapper,
        'greater_is_better': True, 'needs_proba': True, 'average': 'samples'},

    'by class average precision': {
        'score_func': average_precision_score_wrapper,
        'greater_is_better': True, 'needs_proba': True, 'average': None},

    'multiclass macro average precision': {
        'score_func': average_precision_score_wrapper,
        'greater_is_better': True, 'needs_proba': True, 'average': 'macro',
        'multiclass': True},

    'multiclass micro average precision': {
        'score_func': average_precision_score_wrapper,
        'greater_is_better': True, 'needs_proba': True, 'average': 'micro',
        'multiclass': True},

    'multiclass weighted average precision': {
        'score_func': average_precision_score_wrapper,
        'greater_is_better': True, 'needs_proba': True, 'average': 'weighted',
        'multiclass': True},

    'multiclass samples average precision': {
        'score_func': average_precision_score_wrapper,
        'greater_is_better': True, 'needs_proba': True, 'average': 'samples',
        'multiclass': True},

    'multiclass by class average precision': {
        'score_func': average_precision_score_wrapper,
        'greater_is_better': True, 'needs_proba': True, 'average': None,
        'multiclass': True},

    'brier': {'score_func': M.brier_score_loss,
              'greater_is_better': False,
              'needs_proba': True},

    'hamming': {'score_func': M.hamming_loss, 'greater_is_better': False,
                'needs_proba': False},

    'jaccard': {'score_func': jaccard_score_wrapper,
                'greater_is_better': True},

    'weighted jaccard': {'score_func': jaccard_score_wrapper,
                         'greater_is_better': True, 'average': 'weighted'},

    'macro jaccard': {'score_func': jaccard_score_wrapper,
                      'greater_is_better': True, 'average': 'macro'},

    'micro jaccard': {'score_func': jaccard_score_wrapper,
                      'greater_is_better': True, 'average': 'micro'},

    'samples jaccard': {'score_func': jaccard_score_wrapper,
                        'greater_is_better': True, 'average': 'samples'},

    'by class jaccard': {'score_func': jaccard_score_wrapper,
                         'greater_is_better': True, 'average': None},

    'matthews': {'score_func': M.matthews_corrcoef,
                 'greater_is_better': True},
    }


def get_metric(metric_str):
    '''From a final metric str indicator, return the
    metric object.'''

    metric_params = METRICS[metric_str]
    metric = M.make_scorer(**metric_params)
    return metric


def get_metrics_by_type(problem_type):
    '''problem_type must be binary, regression or categorical multilabel or
    categorical multiclass'''

    avaliable_by_type = get_avaliable_by_type(AVALIABLE)

    objs = []
    for metric_str in avaliable_by_type[problem_type]:

        score_func = METRICS[metric_str]['score_func']
        objs.append((metric_str, score_func))

    return objs


def Show_Metrics(self, problem_type=None, metric=None):
    '''Print out the avaliable metrics
    optionally restricted by problem type

    Parameters
    ----------
    problem_type : str or None, optional
        Where `problem_type` is the underlying ML problem

        (default = None)

    metric : str or list
        Where metric is the specific metric indicator str
    '''
    print('Visit: ')
    print('https://scikit-learn.org/stable/modules/model_evaluation.html')
    print('For more detailed information on different metrics.')
    print('The str indicators avaliable are listed below.')
    print()

    if metric is not None:
        if isinstance(metric, str):
                metric = [metric]
        for metric_str in metric:
                print(metric_str)
        return

    avaliable_by_type = get_avaliable_by_type(AVALIABLE)

    if problem_type is not None:
        problem_types = [problem_type]

    else:
        problem_types = list(avaliable_by_type)

    for pt in problem_types:
        show_type(pt, avaliable_by_type)


def show_type(problem_type, avaliable_by_type):

        print('Problem Type:', problem_type)
        print('----------------------------------------')
        print()
        print('Avaliable metrics: ')
        print()

        for metric_str in avaliable_by_type[problem_type]:
            if 'user passed' not in metric_str:
                print(metric_str)

        print()
