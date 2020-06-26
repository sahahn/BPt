"""
Data_Helpers.py
====================================
Various helper functions for loading and processing data for ABCD_ML.
Specifically, these are non-class functions used in _Data.py and ABCD_ML.py.
"""
import numpy as np
import numpy.ma as ma
import random
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from operator import add
from functools import reduce
import warnings
import pandas as pd
from ..helpers.Data_File import load_data_file_proxies


def get_non_drop(data, key, drop_val):

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


def process_binary_input(data, key, drop_val=np.nan,
                         _print=print):
    '''Helper function to perform processing on binary input

    Parameters
    ----------
    data : pandas DataFrame
        ABCD_ML formatted df. Must contain a column with `key`

    key : str
        Column key of the column to process within `data` input.

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

    if len(unique_vals) == 1:
        warnings.warn('Binary input ' + repr(key) + ' was '
                      'passed with only 1 unique value')

    # Preform check for mistaken values
    # Assuming should be binary, so 2 unique values
    if len(unique_vals) > 2:

        # Select top two scores by count
        keep_inds = np.argpartition(counts, -2)[-2:]
        keep_vals = unique_vals[keep_inds]
        keep_vals.sort()

        to_drop = data.index[~data[key].isin(keep_vals)]
        data.loc[to_drop, key] = drop_val

        _print('More than two unique score values found,',
               'filtered all but', keep_vals)

    # Work on only non_dropped data / non NaN data if applicable
    non_drop_data, non_drop_subjects =\
        get_non_drop(data, key, drop_val)

    # Perform actual binary encoding
    encoder = LabelEncoder()
    non_drop_data[key] = encoder.fit_transform(np.array(non_drop_data[key]))
    non_drop_data[key] = non_drop_data[key].astype('category')

    if len(np.unique(non_drop_data[key])) > 2:
        raise ValueError('Binary type, but more than two unique values found '
                         'for input' + repr(key))

    data = put_non_drop_back(data, non_drop_subjects, non_drop_data)

    return data, encoder


def process_ordinal_input(data, key, drop_percent=None,
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

        to_drop = data.index[data[key].isin(drop_vals)]
        data.loc[to_drop, key] = drop_val

        _print('Dropped value(s)', drop_vals, 'according to passed drop',
               'percent of', drop_percent)

    # Work on only non_dropped data / non NaN data if applicable
    non_drop_data, non_drop_subjects =\
        get_non_drop(data, key, drop_val)

    # Encode ordinally
    label_encoder = LabelEncoder()
    non_drop_data[key] = label_encoder.fit_transform(non_drop_data[key])
    non_drop_data[key] = non_drop_data[key].astype('category')

    data = put_non_drop_back(data, non_drop_subjects, non_drop_data)

    return data, label_encoder


def process_categorical_input(data, key, drop='one hot', drop_percent=None,
                              drop_val=np.nan, _print=print):
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
                                                drop_val=drop_val,
                                                _print=_print)

    # Work on only non_dropped data / non NaN data if applicable
    non_drop_data, non_drop_subjects =\
        get_non_drop(data, key, drop_val)

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
        _print('Dummy coding by dropping col', max_col)

        ind = new_keys.index(max_col)

    return data, (label_encoder, encoder, ind)


def process_float_input(data, key, bins, strategy):

    encoder = KBinsDiscretizer(n_bins=bins, encode='ordinal',
                               strategy=strategy)

    vals = np.array(data[key]).reshape(-1, 1)
    vals = np.squeeze(encoder.fit_transform(vals))
    data[key] = vals

    return data, encoder


def get_unused_drop_val(data):

    drop_val = random.random()
    while (data == drop_val).any().any():
        drop_val = random.random()
    return drop_val


def proc_fop(fop):

    # If provided as just % number, divide by 100
    if not isinstance(fop, tuple):
        fop /= 100
        fop = (fop, 1-fop)

    else:
        fop = tuple([f/100 for f in fop])

    return fop


def filter_float_by_outlier(data, key, filter_outlier_percent,
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
    fop = proc_fop(filter_outlier_percent)

    _print('Filtering for outliers, dropping rows with params: ', fop)
    _print('Min-Max value (pre-filter):',
           np.nanmin(data[key]), np.nanmax(data[key]))

    q1 = data[key].quantile(fop[0])
    q2 = data[key].quantile(fop[1])
    data.loc[data[key] < q1, key] = drop_val
    data.loc[data[key] > q2, key] = drop_val

    _print('Min-Max value (post outlier filtering):',
           np.nanmin(data[key][data[key] != drop_val]),
           np.nanmax(data[key][data[key] != drop_val]))

    return data


def filter_float_by_std(data, key, n_std,
                        drop_val=999, _print=print):

    if not isinstance(n_std, tuple):
        n_std = (n_std, n_std)

    _print('Filtering for outliers by stds:', n_std)
    _print('Min-Max value (post-filter):',
           np.nanmin(data[key]), np.nanmax(data[key]))

    mean = data[key].mean()
    std = data[key].std()

    if n_std[0] is not None:
        l_scale = n_std[0] * std
        data.loc[data[key] < mean - l_scale, key] = drop_val

    if n_std[1] is not None:
        u_scale = n_std[1] * std
        data.loc[data[key] > mean + u_scale, key] = drop_val

    _print('Min-Max Score (post outlier filtering):',
           np.nanmin(data[key][data[key] != drop_val]),
           np.nanmax(data[key][data[key] != drop_val]))

    return data


def filter_float_df_by_outlier(data, filter_outlier_percent,
                               drop_val=999):

    # For length of code / readability
    fop = proc_fop(filter_outlier_percent)

    data[data < data.quantile(fop[0])] = drop_val
    data[data > data.quantile(fop[1])] = drop_val

    return data


def filter_float_df_by_std(data, n_std,
                           drop_val=999):

    if not isinstance(n_std, tuple):
        n_std = (n_std, n_std)

    mean = data.mean()
    std = data.std()

    if n_std[0] is not None:
        l_scale = n_std[0] * std
        data[data < mean - l_scale] = drop_val

    if n_std[1] is not None:
        u_scale = n_std[1] * std
        data[data > mean + u_scale] = drop_val

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


def drop_duplicate_cols(data, corr_thresh, _print=print):
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

    if corr_thresh is not None and corr_thresh is not False:

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

                        A_o, B_o = np.array(A[overlap]), np.array(B[overlap])
                        if (A_o == B_o).all():

                            data = data.drop(col2, axis=1)
                            dropped.append(col2)

                    else:

                        corr = np.corrcoef(A[overlap], B[overlap])[0][1]
                        if corr >= corr_thresh:
                            data = data.drop(col2, axis=1)
                            dropped.append(col2)

        _print('Dropped', len(dropped), 'columns as duplicate cols!')

    return data


def get_original_cat_names(names, encoder, original_key):

    if isinstance(encoder, dict):
        original = [encoder[name] for name in names]

    else:

        try:
            float(names[0])
            base = names
        except ValueError:
            base = [int(float(name.replace(original_key + '_', '')))
                    for name in names]

        try:
            original = encoder.inverse_transform(base)
        except IndexError:
            original = encoder.inverse_transform([int(b) for b in base])

    return original


def substrs(x):
    return {x[i:i+j] for i in range(len(x)) for j in range(len(x) - i + 1)}


def find_substr(data):

    s = substrs(data[0])

    for val in data[1:]:
        s.intersection_update(substrs(val))

    try:
        mx = max(s, key=len)

    except ValueError:
        mx = ''

    return mx


def get_top_substrs(keys):

    found = []
    top = find_substr(keys)

    while len(top) > 1:
        found.append(top)

        keys = [k.replace(top, '') for k in keys]
        top = find_substr(keys)

    return found


def proc_datatypes(data_types, col_names):

    # For multi-label case
    if data_types == 'multilabel' or data_types == 'm':
        col_names = [col_names]

    if not isinstance(data_types, list):
        data_types = list([data_types])

    if len(data_types) == 1:
        data_types = [data_types[0] for i in range(len(col_names))]

    if len(data_types) != len(col_names):
        raise RuntimeError('The same number of datatypes were not passed as',
                           'columns!')

    return data_types, col_names


def proc_args(args, data_types):

    if not isinstance(args, list):
        args = list([args])

    # Set to same arg for all if only one passed
    if len(args) == 1:
        args = [args[0] for i in range(len(data_types))]

    if len(args) != len(data_types):
        raise RuntimeError('The length of', args, 'must match length of input',
                           'cols!')

    return args


def process_multilabel_input(keys):

    if not isinstance(keys, list):
        raise RuntimeError(keys, 'must be a list for',
                           'multilabel type')

    if len(keys) < 2:
        raise RuntimeError(keys, 'must be read from multiple',
                           'columns for multilabel type')

    return get_common_name(keys)


def get_common_name(keys):

    top_strs = get_top_substrs(keys)
    if len(top_strs) == 0:
        return keys[0]
    else:
        return top_strs[0]


def filter_data_cols(data, filter_outlier_percent, filter_outlier_std,
                     drop_or_na='drop', seperate_keys=None,
                     subject_id='subject_id', _print=print):

    # Seperate data from data files if applicable
    if seperate_keys is not None:
        sep_data = data[seperate_keys]
        data = data.drop(seperate_keys, axis=1)

    if filter_outlier_percent is None and filter_outlier_std is None:
        return data

    if drop_or_na == 'na':
        drop_val = np.nan
    else:
        drop_val = get_unused_drop_val(data)

    # Filter based on outlier percent
    if filter_outlier_percent is not None:

        data = filter_float_df_by_outlier(data, filter_outlier_percent,
                                          drop_val=drop_val)

    # Filter by std
    if filter_outlier_std is not None:

        data = filter_float_df_by_std(data, filter_outlier_std,
                                      drop_val=drop_val)

    # Only remove if not NaN
    if drop_val is not np.nan:
        data = drop_from_filter(data, drop_val, _print=_print)

    # Re-merge, if seperated data files
    if seperate_keys is not None:
        data = pd.merge(data, sep_data, on=subject_id)

    return data


def filter_data_file_cols(data, reduce_funcs, filter_outlier_percent,
                          filter_outlier_std, data_file_keys,
                          file_mapping, subject_id='subject_id',
                          n_jobs=1, _print=print):

    if not isinstance(reduce_funcs, list):
        reduce_funcs = [reduce_funcs]

    data_file_proxies = load_data_file_proxies(data, reduce_funcs,
                                               data_file_keys,
                                               file_mapping, n_jobs)
    valid_subjects = set(data.index)
    for proxy in data_file_proxies:

        proxy = filter_data_cols(proxy, filter_outlier_percent,
                                 filter_outlier_std, drop_or_na='drop',
                                 seperate_keys=None, subject_id=subject_id,
                                 _print=_print)
        valid_subjects = valid_subjects.intersection(proxy.index)

    filtered_data = data.loc[list(valid_subjects)]

    # Remove any unused file mapping entries
    all_keys = list(file_mapping)
    remaining_keys = np.unique(filtered_data)
    for key in all_keys:
        if key not in remaining_keys:
            file_mapping.pop(key)

    return filtered_data, file_mapping


def drop_from_filter(data, drop_val=999, _print=print):

    to_drop = data[(data == drop_val).any(axis=1)].index

    if len(to_drop) > 0:
        data = data.drop(to_drop)
        _print('Dropped', len(to_drop), 'rows based on filter input',
               'params, e.g. filter outlier percent, drop cat, ect...')
    return data


def proc_file_input(files, file_to_subject, df, subject_id):

    if files is None:
        return df

    if not isinstance(files, dict):
        raise ValueError('files must be passed as a python dict')

    if file_to_subject is None:
        raise RuntimeError('file_to_subject must be specified!')

    if not isinstance(file_to_subject, dict):
        file_to_subject = {key: file_to_subject for key in files}

    for key in files:
        if key not in file_to_subject:
            raise ValueError('If passing file_to_subject as a dict '
                             'then a value must be passed for all '
                             'keys in files. ' + repr(key) + 'was '
                             'not passed in this case.')

    # Compute pd series for each passed
    files_series, all_subjects = dict(), []
    for key in files:

        file_paths = files[key]
        subjects = [file_to_subject[key](fp) for fp in file_paths]
        files_series[key] = pd.Series(file_paths, index=subjects)
        all_subjects += subjects

    # Compute the overlap of subjects
    subjects = sorted(list(set(all_subjects)))

    # Make a dataframe + fill
    files_df = pd.DataFrame(index=subjects)
    files_df.index.name = subject_id
    for key in files_series:
        files_df[key] = files_series[key]

    # If no df, then return
    if df is None:
        return files_df

    # Merge with passed df, and return
    if not isinstance(df, list):
        df = [df]
    df.append(files_df)

    return df
