"""
_Data.py
====================================
Main class extension file for the Data loading functionality methods of
the ABCD_ML class.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.stats.mstats import winsorize
from ABCD_ML.Data_Helpers import (process_binary_input,
                                  process_categorical_input,
                                  filter_float_by_outlier)


def load_name_mapping(self,
                      loc,
                      existing_name_col="NDAR name",
                      changed_name_col="REDCap name/NDA alias"):
    '''Loads a mapping dictionary for loading column names

    Parameters
    ----------
    loc : str, Path or None
        The location of the csv file which contains the mapping.

    existing_name_col : str, optional (default = "NDAR name")
        The column name with the file which lists names to be changed,

    changed_name_col : str, optional (default = "REDCap name/NDA alias")
        The column name within the file which lists the new name.
    '''

    mapping = pd.read_csv(loc)

    try:
        self.name_map = dict(zip(mapping[existing_name_col],
                                 mapping[changed_name_col]))
    except KeyError:
        print('Error: One or both provided column names do not exist!')

    self._print('Loaded map file')


def load_data(self, loc, drop_keys=[], filter_outlier_percent=None,
              winsorize_val=None):
    """Load a 2.0_ABCD_Data_Explorer release formatted
    neuroimaging dataset of ROI's

    Parameters
    ----------
    loc : str, Path or None
        The location of the csv file to load data load from.

    drop_keys : list, optional (default = [])
        A list of keys to drop columns by, where if any key given in a columns
        name, then that column will be dropped.
        (Note: if a name mapping exists, this drop step will be
        conducted after renaming)

    filter_outlier_percent : int, tuple or None, optional (default = None)
        For float / ordinal data only.
        A percent of values to exclude from either end of the
        score distribution, provided as either 1 number,
        or a tuple (% from lower, % from higher).
        set `filter_outlier_percent` to None for no filtering.

    winsorize_val : int, tuple or None, optional (default = None)
        The (limits[0])th lowest values are set to
        the (limits[0])th percentile, and the (limits[1])th highest values
        are set to the (1 - limits[1])th percentile.
        If one value passed, used for both ends.
        If None, then no winsorization performed.
        Note: Winsorizing will be performed after
        filtering for outliers if values are passed for both.
    """

    self._print('Loading', loc)
    data = pd.read_csv(loc, na_values=self.default_na_values)

    # Drop the first two columns by default
    # typically specific id, then dataset id for NDA tyle csv
    first_cols = list(data)[:2]
    data = data.drop(first_cols, axis=1)
    self._print('dropped', first_cols, 'columns by default')

    # Perform common operations
    # (check subject id, drop duplicate subjects ect...)
    data = self._proc_df(data)

    # Drop any columns if any of the drop keys occur in the column name
    column_names = list(data)
    to_drop = [name for name in column_names for drop_key in drop_keys
               if drop_key in name]
    data = data.drop(to_drop, axis=1)
    self._print('Dropped', len(to_drop), 'columns, per drop_keys argument')

    # Drop any rows with missing data
    data = self._drop_na(data)
    self._print('Dropped rows with missing data')

    data_keys = list(data)

    if filter_outlier_percent is not None:
        for key in data_keys:
            data = filter_float_by_outlier(data, key, filter_outlier_percent,
                                           in_place=False, verbose=False)

        data = data.dropna()  # To actually remove the outliers
        self._print('Filtered data for outliers with value: ',
                    filter_outlier_percent)

    if winsorize_val is not None:
        if type(winsorize_val) != tuple:
            winsorize_val = (winsorize_val)

        data[data_keys] = winsorize(data[data_keys], winsorize_val, axis=0)
        self._print('Winsorized data with value: ', winsorize_val)

    self._print('loaded shape: ', data.shape)

    # If other data is already loaded,
    # merge this data with existing loaded data
    self.data = self._merge_existing(self.data, data)
    self._process_new()


def load_covars(self, loc, col_names, data_types, dummy_code_categorical=True,
                filter_float_outlier_percent=None, standardize=True,
                normalize=False):
    '''Load a covariate or covariates from a 2.0_ABCD_Data_Explorer
    release formatted csv.

    Parameters
    ----------
    loc : str, Path or None
        The location of the csv file to load co-variates load from.

    col_names : str or list
        The name(s) of the column(s) to load.
        Note: Must be in the same order as data types passed in.

    data_types : {'binary', 'categorical', 'ordinal', 'float'} or list of
        The data types of the different columns to load,
        in the same order as the column names passed in.
        Short hand options for datatypes are
        'b' for 'binary',
        'c' for 'categorical'
        'o' for 'ordinal'
        'f' for 'float'

    dummy_code_categorical: bool, optional (default = True)
        If True, then categorical variables are dummy coded.
        If False, then categorical variables are one-hot encoded.

    filter_float_outlier_percent, tuple or None, optional (default = None)
        For float datatypes only.
        A percent of values to exclude from either end of the
        score distribution, provided as either 1 number,
        or a tuple (% from lower, % from higher).
        set `filter_float_outlier_percent` to None for no filtering.

    standardize : bool, optional (default = True)
        If True, scales any float/ordinal covariate loaded to have
        a mean of 0 and std of 1.
        Note: Computed before normalization, both set to True.

    normalize : bool, optional (default = False)
        If True, scales any float/ordinal covariates loaded
        to be between 0 and 1.
        Note: Computed after standardization, if both set to True.
    '''

    drop = None
    if dummy_code_categorical:
        drop = 'first'

    self._print('Reading covariates from', loc)
    covars, col_names = self._common_load(loc, col_names=col_names)

    if not isinstance(data_types, list):
        data_types = list([data_types])

    assert len(data_types) == len(col_names), \
        "You must provide the same # of datatypes as column names!"

    for key, d_type in zip(col_names, data_types):

        self._print('load:', key)

        if d_type == 'binary' or d_type == 'b':
            covars, encoder = process_binary_input(covars, key, self.verbose)
            self.covar_encoders[key] = encoder

        elif d_type == "categorical" or d_type == 'c':
            covars, new_keys, encoder = process_categorical_input(covars, key,
                                                                  drop,
                                                                  self.verbose)
            self.covar_encoders[key] = encoder

        elif (d_type == 'float' or d_type == 'ordinal' or
                d_type == 'f' or d_type == 'o'):

            if (filter_float_outlier_percent is not None) and \
                    (d_type == 'float' or d_type == 'f'):

                covars = filter_float_by_outlier(covars, key,
                                                 filter_outlier_percent,
                                                 in_place=False,
                                                 verbose=self.verbose)

            if standardize:
                covars[key] -= np.mean(covars[key])
                covars[key] /= np.std(covars[key])

            if normalize:
                min_val, max_val = np.min(covars[key]), np.max(covars[key])
                covars[key] = (covars[key] - min_val) / (max_val - min_val)

    # Filter float by outlier just replaces with NaN, so actually remove here.
    covars = covars.dropna()

    # If other data is already loaded,
    # merge this data with existing loaded data.
    self.covars = self._merge_existing(self.covars, covars)
    self._process_new()


def load_scores(self, loc, col_name, data_type='float',
                filter_outlier_percent=None):
    '''Loads in a set of subject ids and associated scores from a
    2.0_ABCD_Data_Explorer release formatted csv.
    See Notes for more info.

    Parameters
    ----------
    loc : str, Path or None
        The location of the csv file to load scores load from.

    col_name : str
        The name of the column to load.

    data_type : {'binary', 'categorical', 'ordinal', 'float'}
        The data type of the score column.
        Short hand options for datatypes are,
        'b' for 'binary',
        'c' for 'categorical'
        'o' for 'ordinal'
        'f' for 'float'
        Datatypes are explained further in Notes.

    filter_outlier_percent: tuple or None, optional (default = None)
        For float or ordinal datatypes only.
        A percent of values to exclude from either end of the
        score distribution, provided as either 1 number,
        or a tuple (% from lower, % from higher).
        set `filter_outlier_percent` to None for no filtering.

    Notes
    ----------
    Scores can be either 'binary', 'categorical', 'ordinal' or 'float',
    where ordinal and float are treated the same.

    For binary: scores are read in and label encoded to be 0 or 1,
        Will also work if passed column of unique string also, e.g. 'M' and 'F'

    For categorical: scores are read in and by default one-hot encoded,
        Note: This function is designed only to work with categorical scores
        read in from one column!
        Reading multiple scores from multiple places is not
        supported as of now.

    For ordinal and float: scores are read in as a floating point number,
        and optionally then filtered for outliers with the
        filter_outlier_percent flag.
    '''

    # By default set the score key to be the class original score key
    self.score_key = self.original_score_key

    self._print('Loading ', loc)
    scores = self._common_load(loc, col_name=col_name)

    # Rename the column with score to default score key name
    scores = scores.rename({col_name: self.score_key}, axis=1)

    if data_type == 'binary' or data_type == 'b':

        # Processing for binary, with some tolerance to funky input
        scores, self.score_encoder = process_binary_input(scores,
                                                          self.score_key,
                                                          self.verbose)

    elif data_type == 'categorical' or data_type == 'c':

        # Processing for categorical input,
        # score encoder is a tuple with encoder to ordinal
        # then encoder from ordinal to sparse.
        scores, self.score_key, self.score_encoder = \
            process_categorical_input(scores, self.score_key, drop=None,
                                      verbose=self.verbose)

    elif (data_type == 'float' or data_type == 'ordinal' or
            data_type == 'f' or data_type == 'o'):

        if filter_outlier_percent is not None:
            scores = filter_float_by_outlier(scores, self.score_key,
                                             filter_outlier_percent,
                                             in_place=True,
                                             verbose=self.verbose)

    self._print('Final shape: ', scores.shape)

    # By default only load one set of scores note now, so no merging
    self.scores = scores
    self._process_new()


def load_strat_values(self, loc, col_names):
    '''Load stratification values from a 2.0_ABCD_Data_Explorer
    release formatted csv.
    See Notes for more details on what stratification values are.

    Parameters
    ----------
    loc : str, Path or None
        The location of the csv file to load stratification vals from.

    col_names : str or list
        The name(s) of the column(s) to load.

    Notes
    ----------
    Stratification values are categorical variables which are loaded for the
    purpose of defining custom validation behavior.

    For example: Sex might be loaded here, and used later to ensure
    that any validation splits retain the same distribution of each sex.
    '''

    self._print('Reading stratification values from', loc)
    strat, col_names = self._common_load(loc, col_names=col_names)

    # Encode each column into unique values
    for col in col_names:

        label_encoder = LabelEncoder()
        strat[col] = label_encoder.fit_transform(strat[col])
        self.strat_encoders[col] = label_encoder

    self.strat = self._merge_existing(self.strat, strat)
    self._process_new()


def load_exclusions(self, loc=None, exclusions=None):
    '''Loads in a set of excluded subjects,
    from either a file or as directly passed in.

    Parameters
    ----------
    loc : str, Path or None, optional (default = None)
        Location of a file to load in excluded subjects from.
        The file should be formatted as one subject per line.

    exclusions : list, set, array-like or None, optional (default=None)
        An explicit list of subjects to add to exclusions.

    Notes
    ----------
    If default subject id behavior is set to False,
    reading subjects from a exclusion loc might not
    function as expected.
    '''

    self.exclusions.update(self._load_set_of_subjects(loc=loc,
                                                      subjects=exclusions))
    self._print('Total excluded subjects: ', len(self.exclusions))
    self._process_new()


def clear_exclusions(self):
    '''Resets exclusions to be an empty set'''

    self.exclusions = set()


def _common_load(self, loc, col_name=None, col_names=None):
    '''Internal helper function to perform set of commonly used loading functions,
    on 2.0_ABCD_Data_Explorer release formatted csv'

    Parameters
    ----------
    loc : str, Path or None, optional (default = None)
        Location of a csv file to load in selected columns from.

    col_name : str
        The name of the column to load.

    col_names : str or list
        The name(s) of the column(s) to load.

    Returns
    -------
    pandas DataFrame and list
        If col name is passed, returns just the DataFrame,
        If col names is passed, returns the Dataframe
        and processed column names.
    '''

    # Read csv data from loc
    data = pd.read_csv(loc, na_values=self.default_na_values)

    # Perform common df corrections
    data = self._proc_df(data)

    if col_name is not None:
        data = data[[col_name]].dropna()
        return data

    if not isinstance(col_names, list):
        col_names = list([col_names])

    # Drop rows with NaN
    data = data[col_names].dropna()
    return data, col_names


def _merge_existing(self, class_data, local_data):
    '''Internal helper function to handle either merging dataframes
    after loading, or if not loaded then setting class data.

    Parameters
    ----------
    class_data : pandas DataFrame
        The df stored in the class attribute.
    local_data : pandas DataFrame
        The df of just loaded local data.

    Returns
    ----------
    pandas DataFrame
        Return the class data, either as a copy of the local data,
        or as the clas data merged with local data.
    '''

    # If other data is already loaded, merge with it
    if len(class_data) > 0:
        class_data = pd.merge(class_data, local_data, on='src_subject_id')
        self._print('Merged with existing!')
        return class_data
    else:
        return local_data


def _proc_df(self, data):
    '''Internal helper function, when passed a 2.0_ABCD_Data_Explorer
    release formated dataframe, perform common processing steps.

    Parameters
    ----------
    data : pandas DataFrame
        A df formatted by the ABCD_ML class / 2.0_ABCD_Data_Explorer format

    Returns
    ----------
    pandas DataFrame
        The df post-processing
    '''

    assert 'src_subject_id' in data, "Missing subject id column!"

    # Perform corrects on subject ID
    data.src_subject_id = data.src_subject_id.apply(self._process_subject_name)

    # Filter by eventname is applicable
    data = self._filter_by_eventname(data)

    # Drop any duplicate subjects, default behavior for now
    # though, could imagine a case where you wouldn't want to when
    # there are future releases.
    data = data.drop_duplicates(subset='src_subject_id')

    # Rename columns if loaded name map
    if self.name_map:
        data = data.rename(self.name_map, axis=1)

    data = data.set_index('src_subject_id')

    return data


def _load_set_of_subjects(self, loc=None, subjects=None):
    '''Internal helper function, to load in a set of subjects from either
    a saved location, or directly passed in as a set or list of subjects.

    Parameters
    ----------
    loc : str, Path or None, optional (default = None)
        Location of a file to load in subjects from.
        The file should be formatted as one subject per line.
    subjects : list, set, array-like or None, optional (default=None)
        An explicit list of subjects to add to exclusions.

    Returns
    ----------
    set
        The loaded subjects.
    '''

    loaded_subjects = set()

    if loc is not None:
        with open(loc, 'r') as f:
            lines = f.readlines()

            for line in lines:
                subject = line.rstrip()
                loaded_subjects.add(self._process_subject_name(subject))

    if subjects is not None:
        loaded_subjects = set([self._process_subject_name(s)
                               for s in subjects])

    return loaded_subjects


def _process_subject_name(self, subject):
    '''Internal helper function, to be applied to subject name
    applying standard formatting.

    Parameters
    ----------
    subject : str
        An input subject name

    Returns
    ----------
    str
        Formatted subject name, or input
    '''

    if self.use_default_subject_ids:
        subject = subject.strip().upper()

        if 'NDAR_' not in subject:
            subject = 'NDAR_' + subject
        return subject

    else:
        return subject


def _drop_na(self, data):
    '''Wrapper function to drop rows with NaN values.

    Parameters
    ----------
    data : pandas DataFrame
        ABCD_ML formatted.

    Returns
    ----------
    pandas DataFrame
        Input df, with dropped rows for NaN values
    '''

    missing_values = data.isna().any(axis=1)
    data = data.dropna()
    self._print('Dropped', sum(missing_values), 'rows for missing values')

    return data


def _filter_by_eventname(self, data):
    '''Internal helper function, to simply filter a dataframe by eventname,
    and then return the dataframe.

    Parameters
    ----------
    data : pandas DataFrame
        ABCD_ML formatted.

    Returns
    ----------
    pandas DataFrame
        Input df, with only valid eventname rows
    '''

    # Filter data by eventname
    if self.eventname:
        data = data[data['eventname'] == self.eventname]
        data = data.drop('eventname', axis=1)

    # If eventname none, but still exists, take out the column
    else:
        if 'eventname' in list(data):
            data = data.drop('eventname', axis=1)

    return data


def _process_new(self):
    '''Internal helper function to handle keeping an overlapping subject list,
    with additional useful print statements.
    '''

    valid_subjects = []

    if len(self.data) > 0:
        valid_subjects.append(set(self.data.index))
    if len(self.covars) > 0:
        valid_subjects.append(set(self.covars.index))
    if len(self.scores) > 0:
        valid_subjects.append(set(self.scores.index))
    if len(self.strat) > 0:
        valid_subjects.append(set(self.strat.index))

    overlap = set.intersection(*valid_subjects)
    overlap = overlap - self.exclusions

    self._print('Removing non overlapping + excluded subjects')

    if len(self.data) > 0:
        self.data = self.data[self.data.index.isin(overlap)]
    if len(self.covars) > 0:
        self.covars = self.covars[self.covars.index.isin(overlap)]
    if len(self.scores) > 0:
        self.scores = self.scores[self.scores.index.isin(overlap)]
    if len(self.strat) > 0:
        self.strat = self.strat[self.strat.index.isin(overlap)]

    self._print('Total subjects = ', len(overlap))
    self._print()


def _prepare_data(self):
    '''Helper function to prepare all loaded data,
    from different sources, self.data, self.covars, ect...
    into self.all_data for use directly in ML.
    '''

    dfs = []

    assert len(self.scores > 0), \
        'Scores must be loaded!'
    assert len(self.data) > 0 or len(self.covars) > 0, \
        'Some data must be loaded!'

    if len(self.data) > 0:
        dfs.append(self.data)
        self.data_keys = list(self.data)

    if len(self.covars) > 0:
        dfs.append(self.covars)
        self.covars_keys = list(self.covars)

    dfs.append(self.scores)

    self.all_data = dfs[0]
    for i in range(1, len(dfs)):
        self.all_data = pd.merge(self.all_data, dfs[i], on='src_subject_id')
