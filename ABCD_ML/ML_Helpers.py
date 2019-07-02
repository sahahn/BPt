''' 
File with various ML helper functions for ABCD_ML
Specifically, these are non-class functions that are used in _ML.py.
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

def metric_from_string(metric, return_scorer=False):
    ''' 
    Helper function to convert from string input to sklearn metric, 
    can also be passed a metric directly (though can't make scorer for user passed metric)
    '''

    if callable(metric):
        return metric

    elif metric == 'r2' or metric == 'r2 score':
        
        m = r2_score
        if return_scorer:
            return make_scorer(m, greater_is_better=True)
        return m

    elif metric == 'mse' or metric == 'mean squared error':
        
        m = mean_squared_error
        if return_scorer:
            return make_scorer(m, greater_is_better=False)
        return m

    elif metric == 'roc' or metric == 'roc auc':
        
        m = roc_auc_score

        if return_scorer:
            return make_scorer(m, greater_is_better=True, needs_proba=True)
        return m
        
    elif metric == 'bas' or metric == 'balanced acc':
        m = balanced_accuracy_score

        if return_scorer:
            return make_scorer(m, greater_is_better=True)
        return m

    elif metric == 'f1':

        m = f1_score

        if return_scorer:
            return make_scorer(m, greater_is_better=True)
        return m

    elif metric == 'log loss':

        m = log_loss

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
