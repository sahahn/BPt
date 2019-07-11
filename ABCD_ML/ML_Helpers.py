''' 
File with various ML helper functions for ABCD_ML
Specifically, these are non-class functions that are used in _ML.py and Scoring.py
'''
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from ABCD_ML.Models import AVALIABLE

def process_model_type(problem_type, model_type, extra_params):

    #Only in the case of categorical, with model types that just support multiclass and not multilabel
    #Will this flag be set to true, which means score must be converted to ordinal
    cat_conv_flag = False

    if type(model_type) == list:
        conv_model_types = [m.replace('_', ' ').lower() for m in model_type]

        if problem_type == 'categorical':

            #Check first to see if all model names are in multilabel
            if np.array([m in AVALIABLE['categorical']['multilabel'] for m in conv_model_types]).all():
                conv_model_types = [AVALIABLE['categorical']['multilabel'][m] for m in conv_model_types]
            
            #Then check for multiclass, if multilabel not avaliable
            elif np.array([m in AVALIABLE['categorical']['multiclass'] for m in conv_model_types]).all():
                conv_model_types = [AVALIABLE['categorical']['multiclass'][m] for m in conv_model_types]
                cat_conv_flag = True
            
            else:
                assert 0 == 1, "Selected model type not avaliable for type of problem"

        else:

            #Ensure for binary/regression the models passed exist, and change names
            assert np.array([m in AVALIABLE[problem_type] for m in conv_model_types]).all(), "Selected model type not avaliable for type of problem"
            conv_model_types = [AVALIABLE[problem_type][m] for m in conv_model_types]

        #If any extra params passed for the model, change to conv'ed name
        for m in range(len(conv_model_types)):
            if model_type[m] in extra_params:
                extra_params[conv_model_types[m]] = extra_params[model_type[m]]

        return conv_model_types, extra_params, cat_conv_flag

    else:
        conv_model_type = model_type.replace('_', ' ').lower()

        if problem_type == 'categorical':
            
            #Check multilabel first
            if conv_model_type in AVALIABLE['categorical']['multilabel']:
                conv_model_type = AVALIABLE['categorical']['multilabel'][conv_model_type]
            
            #Then multi class
            elif conv_model_type in AVALIABLE['categorical']['multiclass']:
                conv_model_type = AVALIABLE['categorical']['multiclass'][conv_model_type]
                cat_conv_flag = True

            else:
                assert 0 == 1, "Selected model type not avaliable for type of problem"

        else:

            #Ensure for binary/regression the model passed exist, and change name
            assert conv_model_type in AVALIABLE[problem_type], "Selected model type not avaliable for type of problem"
            conv_model_type = AVALIABLE[problem_type][conv_model_type]

            if conv_model_type in extra_params:
                extra_params[conv_model_type] = extra_params[model_type]

        return conv_model_type, extra_params, cat_conv_flag

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

def compute_macro_micro(scores, n_repeats, n_splits):
    '''Compute and return the macro mean and std, as well as the micro mean and std'''

    scores = np.array(scores)
    macro_scores = np.mean(np.reshape(scores, (n_repeats, n_splits)), axis=1)

    return np.mean(macro_scores), np.std(macro_scores), np.mean(scores), np.std(scores)
