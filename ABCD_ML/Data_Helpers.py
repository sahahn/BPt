'''
Various helper functions for loading and processing data for ABCD_ML
'''
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from operator import add
from functools import reduce

def process_binary_input(data, key, verbose=True):
    '''Helper function to perform processing on binary input'''
        
    unique_vals, counts = np.unique(data[key], return_counts=True)

    #Preform check for mistaken values (assuming should be binary, so 2 unique values)
    if len(unique_vals) != 2:
        
        #Select top two scores by count
        keep_inds = np.argpartition(counts, -2)[-2:] 
        keep_vals = unique_vals[keep_inds]
        keep_vals.sort()

        #Only keep rows with those scores
        data.drop(data.index[~data[key].isin(keep_vals)], inplace=True)
        
        if verbose:
            print('More than two unique score values found, filtered all but', keep_vals)

    #Perform actual binary encoding
    encoder = LabelEncoder()
    data[key] = encoder.fit_transform(np.array(data[key]))
    
    assert len(np.unique(data[key])) == 2, "Error: Binary type, but more than two unique values"
    return data, encoder

def process_categorical_input(data, key, drop=None, verbose=True):
    '''Helper function to perform processing on categorical input'''

    vals = np.array(data[key]).reshape(-1, 1)

    #If drop is set to 'first', then performs dummy coding
    encoder = OneHotEncoder(categories='auto', sparse=False, drop=drop)

    vals = encoder.fit_transform(vals)
    categories = encoder.categories_[0]

    if drop == 'first':
        categories = categories[1:]
    
    new_keys = []

    #Add the encoded features to the dataframe in their own columns
    for i in range(len(categories)):
        k = key + '_' + str(categories[i])
        data[k] = vals[:,i]
        new_keys.append(k)

    #Remove the original key column from the dataframe
    data = data.drop(key, axis=1)

    if verbose:
        print('Encoded to', len(categories), 'categories')

    return data, new_keys, encoder

def filter_float_by_outlier(data, key, filter_outlier_percent, in_place, verbose=True):
        '''
        Helper function to perform filtering on a dataframe,
        setting values outside of the range to be np.nan
        '''
        
        if verbose:
            print('Filtering for outliers, dropping rows with params: ', filter_outlier_percent)
            print('Min-Max Score (before outlier filtering):', np.nanmin(data[key]), np.nanmax(data[key]))
        
        if type(filter_outlier_percent) != tuple:

            #If provided as just % number, divide by 100
            if filter_outlier_percent >= 1:
                filter_outlier_percent /= 100

            filter_outlier_percent = (filter_outlier_percent, 1-filter_outlier_percent)
        
        elif filter_outlier_percent[0] >= 1:
            filter_outlier_percent = tuple([f/100 for f in filter_outlier_percent])

        if in_place:
            data = data[data[key] > data[key].quantile(filter_outlier_percent[0])]
            data = data[data[key] < data[key].quantile(filter_outlier_percent[1])]
        else:
            data.loc[data[key] < data[key].quantile(filter_outlier_percent[0]), key] = np.nan
            data.loc[data[key] > data[key].quantile(filter_outlier_percent[1]), key] = np.nan

        if verbose:
            print('Min-Max Score (post outlier filtering):', np.nanmin(data[key]), np.nanmax(data[key]))

        return data

def get_unique_combo(data, keys):
    '''
    Get unique label combinations from a dataframe (data) and multiple column names,
    returns the combined unique values.
    '''

    combo = [data[k].astype(str) + '***' for k in keys]
    combo = reduce(add, combo).dropna()

    label_encoder = LabelEncoder()
    combo[data.index] = label_encoder.fit_transform(combo)

    return combo
    