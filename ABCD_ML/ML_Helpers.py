''' 
File with various ML helper functions for ABCD_ML
Specifically, these are non-class functions that are used in _ML.py and Scoring.py
'''
import numpy as np

from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.metrics import (roc_auc_score, mean_squared_error, r2_score, balanced_accuracy_score,
                             f1_score, log_loss, make_scorer)

def get_scaler(method='standard', extra_params=None):
    ''' 
    Returns a scaler based on the method passed,
    Likewise, if a dictionary exists in extra_params with the same key as the
    method string, then that will be passed as arguments to the scaler instead
    '''

    method_lower = method.lower()
    params = {}
    
    if method_lower == 'standard':
        scaler = StandardScaler
    
    elif method_lower == 'minmax':
        scaler = MinMaxScaler
    
    elif method_lower == 'robust':
        scaler = RobustScaler
        params = {'quantile_range': (5,95)}
    
    elif method_lower == 'power':
        scaler = PowerTransformer
        params = {'method': 'yeo-johnson', 'standardize': True}

    #Check to see if user passed in params, otherwise params will remain default
    if method in extra_params:
        params.update(extra_params[method])
    
    scaler = scaler(**params)
    return scaler

def scale_data(train_data, test_data, data_scaler, data_keys, extra_params):
    '''
    Wrapper function to take in train/test data and if applicable fit + transform
    a data scaler on the train data, and then transform the test data
    '''

    if data_scaler is not None:

        scaler = get_scaler(data_scaler, extra_params)
        train_data[data_keys] = scaler.fit_transform(train_data[data_keys])
        test_data[data_keys] = scaler.transform(test_data[data_keys])

    return train_data, test_data

def metric_from_string(metric, return_needs_proba=False, return_scorer=False):
    ''' 
    Helper function to convert from string input to sklearn metric, 
    can also be passed a metric directly. 
    
    For a user passed metric, will return that it needs proba by default, and that greater is better.

    If pass return_needs_proba = True, will return metric, and if that metric needs a probability,
    for all classification / categorical valid metrics

    For a metric that requires a special parameter, the metric will be returned in a list as [metric, param_dict]
    '''

    #If a user passed metric
    if callable(metric):
        if return_needs_proba:
            return metric, True
        elif return_scorer:
            return make_scorer(metric, greater_is_better=True, needs_proba=True)
        return metric

    metric = metric.lower()
    metric = metric.replace('_',' ')

    #Correct for common conversions
    conv_dict = {'r2_score':'r2','r2score':'r2', 'mean squared error':'mse',
    'roc':'macro roc auc', 'roc auc':'macro roc auc', 'balanced acc':'bac', 
    'f1 score': 'f1', 'log loss':'log', 'logistic':'log', 'logistic loss':'log',
    'cross entropy loss': 'log', 'crossentropy loss':'log'}

    if metric in conv_dict:
        metric = conv_dict[metric]

    elif metric == 'r2':
        
        m = r2_score
        if return_scorer:
            return make_scorer(m, greater_is_better=True)
        return m

    elif metric == 'mse':
        
        m = mean_squared_error
        if return_scorer:
            return make_scorer(m, greater_is_better=False)
        return m

    elif metric == 'macro roc auc':
        
        m = roc_auc_score
        if return_needs_proba:
            return m, True
        elif return_scorer:
            return make_scorer(m, greater_is_better=True, needs_proba=True)
        return m

    elif metric == 'micro roc auc':
        
        m = roc_auc_score
        metric_params = {'average': 'micro'}
        if return_needs_proba:
            return [m, metric_params], True
        elif return_scorer:
            return make_scorer(m, greater_is_better=True, needs_proba=True, average='micro')
        return [m, metric_params]

    elif metric == 'weighted roc auc':
        
        m = roc_auc_score
        metric_params = {'average': 'weighted'}
        if return_needs_proba:
            return [m, metric_params], True
        elif return_scorer:
            return make_scorer(m, greater_is_better=True, needs_proba=True, average='weighted')
        return [m, metric_params]
        
    elif metric == 'bas':
        
        m = balanced_accuracy_score
        if return_needs_proba:
            return m, False
        if return_scorer:
            return make_scorer(m, greater_is_better=True)
        return m

    elif metric == 'f1':

        m = f1_score
        if return_needs_proba:
            return m, False
        if return_scorer:
            return make_scorer(m, greater_is_better=True)
        return m

    elif metric == 'weighted f1':

        m = f1_score
        metric_params = {'average': 'weighted'}
        if return_needs_proba:
            return [m, metric_params], False
        if return_scorer:
            return make_scorer(m, greater_is_better=True, average='weighted')
        return [m, metric_params]

    elif metric == 'log':

        m = log_loss
        if return_needs_proba:
            return m, True
        if return_scorer:
            return make_scorer(m, greater_is_better=False, needs_proba=True)
        return m

    print('No metric function defined')
    return None

def compute_macro_micro(scores, n_repeats, n_splits):
    '''Compute and return the macro mean and std, as well as the micro mean and std'''

    scores = np.array(scores)
    macro_scores = np.mean(np.reshape(scores, (n_repeats, n_splits)), axis=1)

    return np.mean(macro_scores), np.std(macro_scores), np.mean(scores), np.std(scores)
