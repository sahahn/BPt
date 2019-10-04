"""
Data_Helpers.py
====================================
Various helper functions for loading and processing data for ABCD_ML.
Specifically, these are non-class functions used in _Data.py and ABCD_ML.py.
"""
import numpy as np
import numpy.ma as ma
import pandas as pd
import random
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from operator import add
from functools import reduce


def get_non_drop(data, key, in_place, drop_val):

    if in_place:
        non_drop_subjects = data.index

    else:

        # If drop_val np.nan, means treating dropped as missing value
        if drop_val is np.nan:
            non_drop_subjects = data[~(data[key].isna())].index

        # Otherwise
        else:
            non_drop_subjects = data[~(data[key] == drop_val)].index

    non_drop_data = data.loc[non_drop_subjects]
    return non_drop_data, non_drop_subjects


def put_non_drop_back(data, non_drop_subjects, non_drop_data):

    data.loc[non_drop_subjects] = non_drop_data

    # Make sure col types are right
    for dtype, key in zip(non_drop_data.dtypes, list(data)):
        data[key] = data[key].astype(dtype.name)

    return data


def scale_input(data, key, standardize, normalize, in_place=True,
                drop_val=np.nan):

    if not standardize and not normalize:
        return data

    non_drop_data, non_drop_subjects =\
        get_non_drop(data, key, in_place, drop_val)

    scaler = None
    if standardize:
        scaler = StandardScaler()
    if normalize:
        scaler = MinMaxScaler()
    if scaler is not None:
        vals = np.array(non_drop_data[key]).reshape(-1, 1)
        scaled_vals = scaler.fit_transform(vals)
        non_drop_data[key] = np.squeeze(scaled_vals)

    data = put_non_drop_back(data, non_drop_subjects, non_drop_data)
    return data


def process_binary_input(data, key, in_place=True, drop_val=np.nan,
                         _print=print):
    '''Helper function to perform processing on binary input

    Parameters
    ----------
    data : pandas DataFrame
        ABCD_ML formatted df. Must contain a column with `key`

    key : str
        Column key of the column to process within `data` input.

    in_place : bool
        If True, drop in place, otherwise replace with drop_val

    drop_val : NaN or int
        If a row needs to be dropped, replace with drop_val

        (default = np.nan)

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

        if in_place:
            data.drop(data.index[~data[key].isin(keep_vals)], inplace=True)
        else:
            to_drop = data.index[~data[key].isin(keep_vals)]
            data.loc[to_drop, key] = drop_val

        _print('More than two unique score values found,',
               'filtered all but', keep_vals)

    # Work on only non_dropped data / non NaN data if applicable
    non_drop_data, non_drop_subjects =\
        get_non_drop(data, key, in_place, drop_val)

    # Perform actual binary encoding
    encoder = LabelEncoder()
    non_drop_data[key] = encoder.fit_transform(np.array(non_drop_data[key]))
    non_drop_data[key] = non_drop_data[key].astype('category')

    assert len(np.unique(non_drop_data[key])) == 2, \
        "Error: Binary type, but more than two unique values"

    data = put_non_drop_back(data, non_drop_subjects, non_drop_data)

    return data, encoder


def process_ordinal_input(data, key, drop_percent=None, in_place=True,
                          drop_val=np.nan, _print=print):
    '''Helper function to perform processing on ordinal input,
    where note this definition of ordinal means categorical ordinal...
    so converting input to ordinal, but from categorical!

    Parameters
    ----------
    data : pandas DataFrame
        ABCD_ML formatted df. Must contain a column with `key`

    key : str
        Column key of the column to process within `data` input.

    drop_percent : float
        % to drop

    in_place : bool
        If True, drop in place, otherwise replace with drop_val

    drop_val : NaN or int
        If a row needs to be dropped, replace with drop_val

        (default = np.nan)

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

        if in_place:
            data.drop(data.index[data[key].isin(drop_vals)], inplace=True)
        else:
            to_drop = data.index[data[key].isin(drop_vals)]
            data.loc[to_drop, key] = drop_val

        _print('Dropping', drop_vals, 'according to passed drop percent of',
               drop_percent)

    # Work on only non_dropped data / non NaN data if applicable
    non_drop_data, non_drop_subjects =\
        get_non_drop(data, key, in_place, drop_val)

    # Encode ordinally
    label_encoder = LabelEncoder()
    non_drop_data[key] = label_encoder.fit_transform(non_drop_data[key])
    non_drop_data[key] = non_drop_data[key].astype('category')

    data = put_non_drop_back(data, non_drop_subjects, non_drop_data)

    return data, label_encoder


def process_categorical_input(data, key, drop='one hot', drop_percent=None,
                              in_place=True, drop_val=np.nan, _print=print):
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
                                                in_place=in_place,
                                                drop_val=drop_val,
                                                _print=_print)

    # Work on only non_dropped data / non NaN data if applicable
    non_drop_data, non_drop_subjects =\
        get_non_drop(data, key, in_place, drop_val)

    vals = np.array(non_drop_data[key]).reshape(-1, 1)

    encoder = OneHotEncoder(categories='auto', sparse=False)
    vals = encoder.fit_transform(vals).astype(int)
    categories = encoder.categories_[0]

    _print('Found', len(categories), 'categories')

    new_keys = []

    # Add the encoded features to the dataframe in their own columns
    for i in range(len(categories)):
        k = key + '_' + str(categories[i])

        # First create new col in ordiginal df w/ all drop val
        data[k] = drop_val

        # Set non drop subjects to new vals
        data.loc[non_drop_subjects, k] = vals[:, i]

        data[k] = data[k].astype('category')
        new_keys.append(k)

    # Remove the original key column from the dataframe
    data = data.drop(key, axis=1)
    ind = None

    if drop == 'dummy':
        max_col = data[new_keys].sum().idxmax()
        data = data.drop(max_col, axis=1)
        _print('dummy coding by dropping col', max_col)

        ind = new_keys.index(max_col)

    return data, new_keys, (label_encoder, encoder, ind)


def process_float_input(data, key, bins, strategy):

    encoder = KBinsDiscretizer(n_bins=bins, encode='ordinal',
                               strategy=strategy)

    vals = np.array(data[key]).reshape(-1, 1)
    vals = np.squeeze(encoder.fit_transform(vals))
    data[key] = vals

    return data, encoder


def get_unused_drop_val(data):

    drop_val = random.randint(-100000, 100000)

    while (data == drop_val).any().any():
        drop_val = random.randint(-100000, 100000)

    return drop_val


def filter_float_by_outlier(data, key, filter_outlier_percent, in_place,
                            drop_val=999, _print=print):
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
        or just set to drop_val. If performing outlier removal on multiple
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
        q1 = data[key].quantile(fop[0])
        q2 = data[key].quantile(fop[1])
        data.loc[data[key] < q1, key] = drop_val
        data.loc[data[key] > q2, key] = drop_val

    _print('Min-Max Score (post outlier filtering):',
           np.nanmin(data[key]), np.nanmax(data[key]))

    return data


def get_unique_combo_df(data, keys):
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


def get_unique_combo(to_join):

    as_str = ['***'.join(to_join[i].astype(str)) for i in range(len(to_join))]
    le = LabelEncoder()

    unique_combo = le.fit_transform(as_str)
    return unique_combo, le


def reverse_unique_combo(unique_combo, le):

    reverse = le.inverse_transform(unique_combo)
    seperate = np.array([np.array(reverse[i].split('***')).astype(float)
                        for i in range(len(reverse))])

    return seperate


def reverse_unique_combo_df(unique_combo, le):

    reverse = le.inverse_transform(unique_combo)
    col_split = np.array([r.split('***')[:-1] for r in reverse])
    col_split = col_split.astype(float).astype(int)

    return col_split


def drop_duplicate_cols(data, corr_thresh):
    '''Drop duplicates columns within data based on
    if two data columns are >= to a certain correlation threshold.

    Parameters
    ----------
    data : pandas DataFrame
        ABCD_ML formatted df.

    corr_thresh : float
        A value between 0 and 1, where if two columns within .data
        are correlated >= to `corr_thresh`, the second column is removed.

        A value of 1 will instead make a quicker direct =='s comparison.

    Returns
    ----------
    pandas DataFrame
        ABCD_ML formatted df with duplicates removes

    list
        The list of columns which were dropped from `data`
    '''

    dropped = []

    for col1 in data:

        if col1 in data:

            A = data[col1]
            a = ma.masked_invalid(A)

        for col2 in data:
            if col1 != col2 and col1 in list(data) and col2 in list(data):

                B = data[col2]
                b = ma.masked_invalid(B)

                overlap = (~a.mask & ~b.mask)

                if corr_thresh == 1:

                    if (np.array(A[overlap]) == np.array(B[overlap])).all():
                        data = data.drop(col2, axis=1)
                        dropped.append(col2)

                else:

                    corr = np.corrcoef(A[overlap], B[overlap])[0][1]
                    if corr >= corr_thresh:
                        data = data.drop(col2, axis=1)
                        dropped.append(col2)

    return data, dropped


def get_original_cat_names(names, encoder, original_key):

    try:
        float(names[0])
        base = names
    except ValueError:
        base = [int(name.replace(original_key + '_', '')) for name in names]

    if isinstance(encoder, dict):
        original = [encoder[name] for name in names]
    else:
        original = encoder.inverse_transform(base)

    return original


def substrs(x):
    return {x[i:i+j] for i in range(len(x)) for j in range(len(x) - i + 1)}


def find_substr(data):

    s = substrs(data[0])

    for val in data[1:]:
        s.intersection_update(substrs(val))

    return max(s, key=len)


def get_top_substrs(keys):

    found = []
    top = find_substr(keys)

    while len(top) > 1:
        found.append(top)

        keys = [k.replace(top, '') for k in keys]
        top = find_substr(keys)

    return found


def proc_datatypes(data_types, col_names):

    if not isinstance(data_types, list):
        data_types = list([data_types])

    if len(data_types) != len(col_names):
        raise RuntimeError('The same number of datatypes were not passed as',
                           'columns!')

    return data_types
