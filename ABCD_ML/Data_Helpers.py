"""
Data_Helpers.py
====================================
Various helper functions for loading and processing data for ABCD_ML.
Specifically, these are non-class functions used in _Data.py and ABCD_ML.py.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from operator import add
from functools import reduce


def process_binary_input(data, key, _print=print):
    '''Helper function to perform processing on binary input

    Parameters
    ----------
    data : pandas DataFrame
        ABCD_ML formatted df. Must contain a column with `key`

    key : str
        Column key of the column to process within `data` input.

    _print : print func, optional
        Either python print statement or overriden print
        func.

        (default = print)

    Returns
    ----------
    pandas DataFrame
        The post-processed ABCD_ML formatted input df.

    sklearn LabelEncoder
        The sklearn labelencoder object mapping input to
        transformed binary label.
    '''

    unique_vals, counts = np.unique(data[key], return_counts=True)

    assert len(unique_vals) != 1, \
        "Binary input passed with only 1 unique value!"

    # Preform check for mistaken values
    # Assuming should be binary, so 2 unique values
    if len(unique_vals) != 2:

        # Select top two scores by count
        keep_inds = np.argpartition(counts, -2)[-2:]
        keep_vals = unique_vals[keep_inds]
        keep_vals.sort()

        # Only keep rows with those scores
        data.drop(data.index[~data[key].isin(keep_vals)], inplace=True)

        _print('More than two unique score values found,',
               'filtered all but', keep_vals)

    # Perform actual binary encoding
    encoder = LabelEncoder()
    data[key] = encoder.fit_transform(np.array(data[key]))
    data[key] = data[key].astype('category')

    assert len(np.unique(data[key])) == 2, \
        "Error: Binary type, but more than two unique values"
    return data, encoder


def process_ordinal_input(data, key, drop_percent=None, _print=print):
    '''Helper function to perform processing on ordinal input,
    where note this definition of ordinal means categorical ordinal...
    so converting input to ordinal, but from categorical!

    Parameters
    ----------
    data : pandas DataFrame
        ABCD_ML formatted df. Must contain a column with `key`

    key : str
        Column key of the column to process within `data` input.

    Returns
    ----------
    pandas DataFrame
        The post-processed ABCD_ML formatted input df.

    sklearn LabelEncoder
        The sklearn labelencoder object mapping input to
        transformed ordinal label
    '''

    if drop_percent:

        unique_vals, counts = np.unique(data[key], return_counts=True)

        drop_inds = np.where(counts / len(data) < drop_percent)
        drop_vals = unique_vals[drop_inds]

        data.drop(data.index[data[key].isin(drop_vals)], inplace=True)

        _print('Dropping', drop_vals, 'according to passed drop percent of',
               drop_percent)

    label_encoder = LabelEncoder()
    data[key] = label_encoder.fit_transform(data[key])
    data[key] = data[key].astype('category')

    return data, label_encoder


def process_categorical_input(data, key, drop='one hot', drop_percent=None,
                              _print=print):
    '''Helper function to perform processing on categorical input

    Parameters
    ----------
    data : pandas DataFrame
        ABCD_ML formatted df. Must contain a column with `key`

    key : str
        Column key of the column to process within `data` input.

    drop : 'dummy' or 'one hot', optional
        If 'dummy', then dummy code categorical variable,
        Otherwise if None or one hot (default), perform one-hot encoding
        (default = 'one hot')

    _print : print func, optional
        Either python print statement or overriden print
        func.

        (default = print)

    Returns
    ----------
    pandas DataFrame
        The post-processed ABCD_ML formatted input df.

    list
        The new score keys to the encoded columns.

    (sklearn LabelEncoder, sklearn OneHotEncoder)
        The sklearn labelencoder object mapping input to
        transformed ordinal label, in index 0 of the tuple,
        and the sklearn onehotencoder obect mapping ordinal
        input to sparsely/encoded.
    '''

    # First convert to label encoder style,
    # want to be able to do reverse transform w/ 1-hot encoder
    # between label encoded results.
    data, label_encoder = process_ordinal_input(data, key, drop_percent,
                                                _print)

    # Now convert to one hot or dummy encoded
    vals = np.array(data[key]).reshape(-1, 1)

    encoder = OneHotEncoder(categories='auto', sparse=False)

    vals = encoder.fit_transform(vals)
    categories = encoder.categories_[0]

    _print('Found', len(categories), 'categories')

    new_keys = []

    # Add the encoded features to the dataframe in their own columns
    for i in range(len(categories)):
        k = key + '_' + str(categories[i])
        data[k] = vals[:, i]
        data[k] = data[k].astype(int)
        data[k] = data[k].astype('category')
        new_keys.append(k)

    # Remove the original key column from the dataframe
    data = data.drop(key, axis=1)

    if drop == 'dummy':
        max_col = np.sum(data[new_keys]).idxmax()
        data = data.drop(max_col, axis=1)
        _print('dummy coding by dropping col', max_col)

    return data, new_keys, (label_encoder, encoder)


def filter_float_by_outlier(data, key, filter_outlier_percent, in_place,
                            _print=print):
    '''Helper function to perform filtering on a dataframe,
    by setting values to be NaN, then optionally removing rows inplace

    Parameters
    ----------
    data : pandas DataFrame
        ABCD_ML formatted df. Must contain a column with `key`

    key : str
        Column key of the column to process within `data` input.

    filter_outlier_percent : int, float, tuple or None
        A percent of values to exclude from either end of the
        score distribution, provided as either 1 number,
        or a tuple (% from lower, % from higher).
        set `filter_outlier_percent` to None for no filtering.
        If over 1, then treated as a percent.

    in_place : bool
        Defines if rows with float outliers should be removed right away,
        or just set to NaN. If performing outlier removal on multiple
        columns, this should be set to False, as you would only want to
        remove rows with missing values after all columns have been checked.
        If only filtering one column, this can be set to True.

    _print : print func, optional
        Either python print statement or overriden print
        func.

        (default = print)

    Returns
    ----------
    pandas DataFrame
        The post-processed ABCD_ML formatted input df.
    '''

    # For length of code / readability
    fop = filter_outlier_percent

    _print('Filtering for outliers, dropping rows with params: ', fop)
    _print('Min-Max Score (before outlier filtering):',
           np.nanmin(data[key]), np.nanmax(data[key]))

    if type(fop) != tuple:

        # If provided as just % number, divide by 100
        if fop >= 1:
            fop /= 100

        fop = (fop, 1-fop)

    elif fop[0] >= 1 or fop[1] >= 1:
        fop = tuple([f/100 for f in fop])

    if in_place:
        data = data[data[key] > data[key].quantile(fop[0])]
        data = data[data[key] < data[key].quantile(fop[1])]
    else:
        data.loc[data[key] < data[key].quantile(fop[0]), key] = np.nan
        data.loc[data[key] > data[key].quantile(fop[1]), key] = np.nan

    _print('Min-Max Score (post outlier filtering):',
           np.nanmin(data[key]), np.nanmax(data[key]))

    return data


def get_unique_combo(data, keys):
    '''Get the unique label combinations from a dataframe (data)
    given multiple column names.

    Parameters
    ----------
    data : pandas DataFrame
        ABCD_ML formatted df. Must contain columns with `keys`.

    keys : list
        Column keys within `data`, to merge unique values for.

    Returns
    ----------
    pandas Series
        Series as indexed by subject (ABCD_ML) format,
        containing the merged unique values.
    '''

    combo = [data[k].astype(str) + '***' for k in keys]
    combo = reduce(add, combo).dropna()

    label_encoder = LabelEncoder()
    combo[data.index] = label_encoder.fit_transform(combo)

    return combo, label_encoder


def drop_col_duplicates(data, corr_thresh):
    '''Drop duplicates columns within data based on
    if two data columns are >= to a certain correlation threshold.

    Parameters
    ----------
    data : pandas DataFrame
        ABCD_ML formatted df.

    corr_thresh : float
        A value between 0 and 1, where if two columns within .data
        are correlated >= to `corr_thresh`, the second column is removed.
        A value of 1 acts like dropping exact repeats.

    Returns
    ----------
    pandas DataFrame
        ABCD_ML formatted df with duplicates removes

    list
        The list of columns which were dropped from `data`
    '''

    dropped = []

    for col1 in data:
        for col2 in data:

            if col1 != col2 and col1 in list(data) and col2 in list(data):

                corr = np.corrcoef(data[col1], data[col2])[0][1]
                if corr >= corr_thresh:
                    data = data.drop(col2, axis=1)
                    dropped.append(col2)

    return data, dropped


def get_original_cat_names(names, encoder, original_key):

    if isinstance(names[0], (np.integer, int)):
        base = names
    else:
        base = [int(name.replace(original_key + '_', '')) for name in names]

    original = encoder.inverse_transform(base)

    return original
