''' 
File with functions related to calculating score from ML models
'''
import numpy as np
from sklearn.metrics import (roc_auc_score, mean_squared_error, r2_score, balanced_accuracy_score,
                             f1_score, log_loss, make_scorer)

def scorer_from_string(scorer):
    ''' 
    Helper function to convert from string input to sklearn scorer/metric, 
    '''

    scorer = scorer.lower()
    scorer = scorer.replace('_',' ')

    #Correct for common conversions
    conv_dict = {'r2 score':'r2','r2score':'r2', 'mean squared error':'mse',
    'roc':'macro roc auc', 'roc auc':'macro roc auc', 'balanced acc':'bac', 
    'f1 score': 'f1', 'log loss':'log', 'logistic':'log', 'logistic loss':'log',
    'cross entropy loss': 'log', 'crossentropy loss':'log'}

    if scorer in conv_dict:
        scorer = conv_dict[scorer]

    print('using scorer / metric: ', scorer)

    #Regression
    if scorer == 'r2':
        return make_scorer(r2_score, greater_is_better=True)

    #Regression
    elif scorer == 'mse':
        return make_scorer(mean_squared_error, greater_is_better=False)

    #Binary, multilabel, multiclass?
    elif scorer == 'macro roc auc':
        return make_scorer(roc_auc_score_wrapper, greater_is_better=True, needs_proba=True)

    #Binary, multilabel, multiclass?
    elif scorer == 'micro roc auc':
        return make_scorer(roc_auc_score_wrapper, greater_is_better=True, needs_proba=True, average='micro')

    # Binary, multilabel, multiclass?
    elif scorer == 'weighted roc auc':
        return make_scorer(roc_auc_score_wrapper, greater_is_better=True, needs_proba=True, average='weighted')

    # Binary, multiclass
    elif scorer == 'bas':
        return make_scorer(balanced_accuracy_score, greater_is_better=True)

    elif scorer == 'f1':
        return make_scorer(f1_score_wrapper, greater_is_better=True)

    elif scorer == 'weighted f1':
        return make_scorer(f1_score_wrapper, greater_is_better=True, average='weighted')

    elif scorer == 'log':
        return make_scorer(log_loss_wrapper, greater_is_better=False, needs_proba=True)
        
    print('No scorer function defined')
    return None


def roc_auc_score_wrapper(y_true, y_score, average='macro', sample_weight=None, max_fpr=None):
    y_score = mutlilabel_compat(y_score)
    return roc_auc_score(y_true, y_score, average, sample_weight, max_fpr)

def f1_score_wrapper(y_true, y_pred, labels=None, pos_label=1, average='binary'):
    y_score = mutlilabel_compat(y_score)
    return f1_score(y_true, y_pred, labels, pos_label, average)

def log_loss_wrapper(y_true, y_pred, eps=1e-15, normalize=True, sample_weight=None):
    y_score = mutlilabel_compat(y_score)
    return log_loss(y_true, y_pred, eps, normalize, sample_weight)

def mutlilabel_compat(y_score):

    if type(y_score) == list:
        y_score = np.stack([s[:,1] for s in y_score], axis=1)

    return y_score

def get_score(problem_type,
              model,
              test_data,
              score_key,
              metric,
              score_encoder=None
              ):
    '''
    Computes a score for a given model with given test data.

    Parameters
    ----------
    problem_type : str, either 'regression', 'binary' or 'categorical'

    model : sklearn model,
        A trained model must be passed in to be evaluated.

    test_data : pandas DataFrame,
        ABCD_ML formatted df.

    score_key : str or numpy array,
        The score column key or keys within test_data

    metric : str,
        Indicator for which metric to use for parameter selection
        and model evaluation. For a full list of supported metrics,
        call self.show_metrics(problem_type=problem_type)

    score_encoder : sklearn encoder, optional (default=None)
        A sklearn api encoder, for optionally transforming the target
        variable. Used in the case of categorical data in converting from
        one-hot encoding to ordinal.

    Returns
    -------
    score : returns a score for whatever metric was passed
    '''

    X_test = np.array(test_data.drop(score_key, axis=1))
    y_test = np.array(test_data[score_key])

    #If a score encoder is passed, transform the encoded y back for testing
    if score_encoder is not None:
        y_test = score_encoder.inverse_transform(y_test).squeeze()

    score_func = scorer_from_string(metric)
    return score_func(model, X_test, y_test)
