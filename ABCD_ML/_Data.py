"""
_Data.py
====================================
Main class extension file for the Data loading functionality methods of
the ABCD_ML class.
"""
import numpy as np
import pandas as pd
from scipy.stats.mstats import winsorize
from ABCD_ML.Data_Helpers import (process_binary_input,
                                  process_ordinal_input,
                                  process_categorical_input,
                                  process_float_input,
                                  process_multilabel_input,
                                  scale_input,
                                  get_unused_drop_val,
                                  filter_float_by_outlier,
                                  drop_duplicate_cols,
                                  get_top_substrs,
                                  proc_datatypes,
                                  get_common_name)


def Load_Name_Map(self, loc, dataset_type='default',
                  source_name_col="NDAR name",
                  target_name_col="REDCap name/NDA alias"):
    '''Loads a mapping dictionary for loading column names

    Parameters
    ----------
    loc : str, Path or None
        The location of the csv file which contains the mapping.

    dataset_type : {'default', 'basic', 'explorer', 'custom'}, optional
        The type of file to load from.
        Dataset types are,

        - 'default' : Use the class defined default dataset type,\
            if not set by the user this is 'basic'.

        - 'basic' : ABCD2p0NDA style (.txt and tab seperated)

        - 'explorer' : 2.0_ABCD_Data_Explorer style (.csv and comma seperated)

        - 'custom' : A user-defined custom dataset. Right now this is only\
            supported as a comma seperated file, with the subject names in a\
            column called self.subject_id.

        (default = 'default')

    source_name_col : str, optional
        The column name with the file which lists names to be changed.
        (default = "NDAR name")

    target_name_col : str, optional
        The column name within the file which lists the new name.
        (default = "REDCap name/NDA alias")
    '''

    # Load mapping based on dataset type
    mapping = self._load(loc, dataset_type)

    try:
        self.name_map = dict(zip(mapping[source_name_col],
                                 mapping[target_name_col]))
        self._print('Loaded map file')

    except KeyError:
        print('Error: One or both provided column names do not exist!')
        print('Name map not loaded!')


def Load_Data(self, loc, dataset_type='default', overlap_subjects=False,
              drop_keys=[], drop_nan='default', filter_outlier_percent=None,
              winsorize_val=None, unique_val_drop_thresh=2,
              unique_val_warn_percent=.2, drop_col_duplicates=None,
              drop_or_nan='drop', clear_existing=False):
    """Load a ABCD2p0NDA (default) or 2.0_ABCD_Data_Explorer (explorer)
    release formatted neuroimaging dataset - of derived ROI level info.

    Parameters
    ----------
    loc : str, Path or list of
        The location of the csv file to load data load from.
        If passed a list, then will load each loc in the list,
        and will assume them all to be of the same dataset_type if one
        dataset_type is passed, or if they differ in type, a list must be
        passed to dataset_type with the different types in order.

        Note: some proc will be done on each loaded dataset before merging
        with the rest (duplicate subjects, proc for eventname ect...), but
        other dataset loading behavior won't occur until after the merge,
        e.g., dropping cols by key, filtering for outlier, ect...

    dataset_type : {'default', 'basic', 'explorer', 'custom'} or list, optional
        The type of dataset to load from. If a list is passed, then loc must
        also be a list, and the indices should correspond.
        Likewise, if loc is a list and dataset_type is not,
        it is assumed all datasets are the same type.
        Where each dataset type is,

        - 'default' : Use the class defined default dataset type,\
            if not set by the user this is 'basic'.

        - 'basic' : ABCD2p0NDA style (.txt and tab seperated)\
            Typically the default columns, and therefore not neuroimaging\
            data, will be dropped, also not including the eventname column.

        - 'explorer' : 2.0_ABCD_Data_Explorer style (.csv and comma seperated)\
            The first 2 columns before self.subject_id\
            (typically the default columns, and therefore not neuroimaging\
            data - also not including the eventname column), will be dropped.\

        - 'custom' : A user-defined custom dataset. Right now this is only\
            supported as a comma seperated file, with the subject names in a\
            column called self.subject_id, and can optionally have\
            'eventname'. No columns will be dropped,\
            (except eventname) or unless specific drop keys are passed.

        (default = 'default')

    overlap_subjects : bool, optional
        This parameter dictates if the loaded data,
        (after initial basic proc and merge w/ other passed loc's),
        should be restricted to only the overlapping subjects from previously
        loaded data, targets, covars or strat. If False, then all subjects will
        be kept throughout the rest of the optional processing - and only
        merged at the end with only the existing loaded data (if any).

        Note: Inclusions and Exclusions are always applied regardless of this
        parameter.

        (default = False)

    drop_keys : list, optional
        A list of keys to drop columns by, where if any key given in a columns
        name, then that column will be dropped.
        (Note: if a name mapping exists, this drop step will be
        conducted after renaming)

        (default = [])

    drop_nan : bool, int, float or 'default', optional
        If set to True, then will drop any row within the loaded
        data if there are any NaN! If False, the will not drop any
        rows for missing values.

        If set to 'default' will use
        the saved value within the class.

        If an int or float, then this means some NaN entries
        will potentially be preserved! Missing data imputation
        will therefore be required later on!

        If an int > 1, then will drop any row with more than drop_nan
        NaN values. If a float, will determine the drop threshold as
        a percentage of the possible values, where 1 would not drop any rows
        as it would require the number of columns + 1 NaN, and .5 would require
        that more than half the column entries are NaN in order to drop that
        row.

        (default = 'default')

    filter_outlier_percent : int, float, tuple or None, optional
        *For float / ordinal data only.*
        A percent of values to exclude from either end of the
        targets distribution, provided as either 1 number,
        or a tuple (% from lower, % from higher).
        set `filter_outlier_percent` to None for no filtering.
        If over 1 then treated as a percent, if under 1, then
        used directly.

        If drop_or_nan == 'drop', then all rows/subjects with >= 1
        value(s) found outside of the percent will be dropped.
        Otherwise, if drop_or_nan = 'nan', then any outside values
        will be set to NaN.

        (default = None)

    winsorize_val : float, tuple or None, optional
        The (winsorize_val[0])th lowest values are set to
        the (winsorize_val[0])th percentile,
        and the (winsorize_val[1])th highest values
        are set to the (1 - winsorize_val[1])th percentile.
        If one value passed, used for both ends.
        If None, then no winsorization performed.

        Note: Winsorizing will be performed after
        filtering for outliers if values are passed for both.

        (default = None)

    unique_val_drop_thresh : int, optional
        This parameter allows you to drop a column within data,
        or rather a feature, if there are under the passed
        `unique_val_drop_thresh` number of unique values.
        By default this is 2, so any feature with only 1 unique value
        will be dropped.

        (default = 2)

    unique_val_warn_percent : int or float, optional
        This value controls the warn threshold for printing
        out potential data columns with a suspiciously low
        number of unique values. Specifically, it refers to
        the percentage of the total data that a features number of
        unique values has to be under. For example, if 5 is passed,
        that any column / feature with under .05 * len(data) unique
        vals.

        If float, then pass a value between 0 and 1, otherwise if greater
        than 1, will turn to a percent.

        (default = .2)

    drop_col_duplicates : float or None/False, optional
        If set to None, will not drop any.
        If float, then pass a value between 0 and 1,
        where if two columns within data
        are correlated >= to `corr_thresh`, the second column is removed.

        A value of 1 will instead make a quicker direct =='s comparison.

        Note: This param just drops duplicated within the just loaded data.
        You can call self.Drop_Data_Duplicates() to drop duplicates across
        all loaded data.

        Be advised, this functionality runs rather slow when there are ~500+
        columns to compare!

        (default = None)

    drop_or_nan : {'drop', 'nan'}, optional

        filter_outlier_percent by default drops rows with extreme values, if
        drop_or_nan is set to 'nan', then these values will be replaced with
        NaN instead.

        Otherwise, will continue to drop with default param == 'drop'

        (default = 'drop')

    clear_existing : bool, optional
        If this parameter is set to True, then any existing
        loaded data will first be cleared before loading new data!

        .. WARNING::
            If any subjects have been dropped from a different place,
            e.g. targets, then simply reloading / clearing existing data
            might result in computing a misleading overlap of final
            valid subjects. Reloading should therefore be best used
            right after loading the original data, or if not possible,
            then reloading the notebook or re-running the script.

        (default = False)

    Notes
    ----------
    For loading a truly custom dataset, an advanced user can
    load all the data themselves into a pandas DataFrame.
    They will need to have the DataFrame indexed by self.subject_id
    e.g., ::

        data = data.set_index(self.subject_id)

    and subject ids will need to be in the correct style...
    but if they do all this, then they can just set ::

        self.data = data

    """
    if clear_existing:
        self.Clear_Data()
        self._print()

    # Load in the dataset & proc. dataset type
    if isinstance(loc, list):
        data = self._load_datasets(loc, dataset_type)
    else:
        data = self._load_dataset(loc, dataset_type)

    if overlap_subjects:
        data = data[data.index.isin(self._get_overlapping_subjects())]
        self._print('Set to overlap subjects only')

    # Drop any columns if any of the drop keys occur in the column name
    column_names = list(data)

    assert isinstance(drop_keys, list), "drop_keys must be list!"

    to_drop = [name for name in column_names for drop_key in drop_keys
               if drop_key in name]
    data = data.drop(to_drop, axis=1)
    self._print('Dropped', len(to_drop), 'columns, per drop_keys argument')

    # Drop any cols with all missing and rows by drop_nan argument
    data = self._drop_na(data, drop_nan)
    self._print()

    data_keys = list(data)

    # Filter based on outlier percent
    if filter_outlier_percent is not None:

        if drop_or_nan == 'nan':
            drop_val = np.nan
        else:
            drop_val = get_unused_drop_val(data)

        for key in data_keys:
            data = filter_float_by_outlier(data, key, filter_outlier_percent,
                                           in_place=False, drop_val=drop_val,
                                           _print=self._print_nothing)

        # Only remove if not NaN
        if drop_val is not np.nan:
            data = self._drop_from_filter(data, drop_val)

    if winsorize_val is not None:
        if type(winsorize_val) != tuple:
            winsorize_val = (winsorize_val)

        for key in data_keys:
            non_nan_subjects = ~data[key].isna()
            data.loc[non_nan_subjects, key] =\
                winsorize(data.loc[non_nan_subjects, key], winsorize_val)

        self._print('Winsorized data with value: ', winsorize_val)

    # Check for questionable loaded columns & print warning
    if unique_val_warn_percent > 1:
        unique_val_warn_percent /= 100

    warn_thresh = unique_val_warn_percent * len(data)

    unique_counts = [np.sum(~np.isnan(np.unique(data[x]))) for x in data]

    if (np.array(unique_counts) < unique_val_drop_thresh).any():

        self._print()
        self._print('The following columns were dropped due to being under',
                    'the passed unique_val_drop_thresh of: ',
                    unique_val_drop_thresh)

        for col, count in zip(list(data), unique_counts):
            if count < unique_val_drop_thresh:
                data = data.drop(col, axis=1)
                self._print('Dropped', col, 'unique vals:', count)

        self._print()
        unique_counts = [np.sum(~np.isnan(np.unique(data[x]))) for x in data]

    if (np.array(unique_counts) < warn_thresh).any():

        self._print()
        self._print('The following columns have a questionable number',
                    'of unique values (under the warn thresh of', warn_thresh,
                    '): ')
        for col, count in zip(list(data), unique_counts):
            if count < warn_thresh:
                self._print(col, 'unique vals:', count)
        self._print()

    if drop_col_duplicates is False:
        drop_col_duplicates = None
    if drop_col_duplicates is not None:
        data, dropped = drop_duplicate_cols(data, drop_col_duplicates)
        self._print('Dropped', len(dropped), 'columns as duplicate cols!')

    self._show_nan_info(data)

    self._print('loaded shape: ', data.shape)

    # If other data is already loaded,
    # merge this data with existing loaded data
    self.data = self._merge_existing(self.data, data)
    self._process_new(self.low_memory_mode)


def Load_Covars(self, loc, col_name, data_type, dataset_type='default',
                overlap_subjects=False, drop_nan='default',
                code_categorical_as='dummy',
                categorical_drop_percent=None,
                filter_float_outlier_percent=None, standardize=False,
                normalize=False, drop_or_nan='drop', clear_existing=False):
    '''Load a covariate or covariates, type data.

    Parameters
    ----------
    loc : str, Path or None
        The location of the csv file to load co-variates load from.

    col_name : str or list
        The name(s) of the column(s) to load.

        Note: Must be in the same order as data types passed in.

    data_type : {'binary', 'categorical', 'multilabel', 'ordinal', 'float'}
        The data types of the different columns to load,
        in the same order as the column names passed in.
        Shorthands for datatypes can be used as well.

        - 'binary' or 'b' : Binary input
        - 'categorical' or 'c' : Categorical input
        - 'multilabel' or 'm' : Multilabel categorical input
        - 'ordinal' or 'o' : Ordinal input
        - 'float' or 'f' : Float numerical input

        .. WARNING::
            If 'multilabel' datatype is specified, then the associated col name
            should be a list of columns, and will be assumed to be.
            For example, if loading multiple targets and one is multilabel,
            a nested list should be passed to col_name.

    drop_nan : bool, int, float or 'default', optional
        If set to True, then will drop any row within the loaded
        data if there are any NaN! If False, the will not drop any
        rows for missing values.

        If set to 'default' will use
        the saved value within the class.

        If an int or float, then this means some NaN entries
        will potentially be preserved! Missing data imputation
        will therefore be required later on!

        If an int > 1, then will drop any row with more than drop_nan
        NaN values. If a float, will determine the drop threshold as
        a percentage of the possible values, where 1 would not drop any rows
        as it would require the number of columns + 1 NaN, and .5 would require
        that more than half the column entries are NaN in order to drop that
        row.

        (default = 'default')

    dataset_type : {'default', 'basic', 'explorer', 'custom'}, optional
        The type of file to load from.
        Dataset types are,

        - 'default' : Use the class defined default dataset type,\
            if not set by the user this is 'basic'.

        - 'basic' : ABCD2p0NDA style (.txt and tab seperated)

        - 'explorer' : 2.0_ABCD_Data_Explorer style (.csv and comma seperated)

        - 'custom' : A user-defined custom dataset. Right now this is only\
            supported as a comma seperated file, with the subject names in a\
            column called self.subject_id.

        (default = 'default')

     overlap_subjects : bool, optional
        This parameter dictates if the loaded covars,
        (after initial basic proc)
        should be restricted to only the overlapping subjects from previously
        loaded data, targets, covars or strat. If False, then all subjects will
        be kept throughout the rest of the optional processing - and only
        merged at the end with only the existing loaded covars (if any).

        Note: Inclusions and Exclusions are always applied regardless of this
        parameter.

        (default = False)

    code_categorical_as: {'dummy', 'one hot', 'ordinal'}
        How to code categorical data,

        - 'dummy' : perform dummy coding, to len(unique classes)-1 columns

        - 'one hot' : for one hot encoding, to len(unique classes) columns

        - 'ordinal' : for just ordinal, one column values 0 to \
                      len(unique classes) -1

        (default = 'dummy')

    categorical_drop_percent: float or None, optional
        Optional percentage threshold for dropping categories when
        loading categorical data. If a float is given, then a category
        will be dropped if it makes up less than that % of the data points.
        E.g. if .01 is passed, then any datapoints with a category with less
        then 1% of total valid datapoints is dropped.

        (default = None)

    filter_float_outlier_percent : float, int, tuple or None, optional
        For float datatypes only.
        A percent of values to exclude from either end of the
        targets distribution, provided as either 1 number,
        or a tuple (% from lower, % from higher).
        set `filter_float_outlier_percent` to None for no filtering.

        (default = None)

    standardize : bool, optional
        If True, scales any float/ordinal covariate loaded to have
        a mean of 0 and std of 1.

        Note: Computed before normalization, if both set to True,
        and both computed after filter_float_outlier_percent.

        (default = False)

    normalize : bool, optional
        If True, scales any float/ordinal covariates loaded
        to be between 0 and 1.

        Note: Computed after standardization, if both set to True.

        (default = False)

    drop_or_nan : {'drop', 'nan'}, optional
        When loading a binary variable and more than two classed are present,
        or when using filter_float_outlier_percent or categorical_drop_percent,
        by default rows are dropped.
        This parameter allows alternatively to instead specify those
        values that would be dropped as NaN instead.

        (default = 'drop')

    clear_existing : bool, optional
        If this parameter is set to True, then any existing
        loaded covars will first be cleared before loading new covars!

        .. WARNING::
            If any subjects have been dropped from a different place,
            e.g. targets or data, then simply reloading / clearing existing
            covars might result in computing a misleading overlap of final
            valid subjects. Reloading should therefore be best used
            right after loading the original data, or if not possible,
            then reloading the notebook or re-running the script.

        (default = False)
    '''

    if clear_existing:
        self.Clear_Covars()

    self._print('Loading Covars!')
    covars, col_names = self._common_load(loc, dataset_type,
                                          overlap_subjects,
                                          col_names=col_name,
                                          drop_nan=drop_nan)

    data_types, col_names = proc_datatypes(data_type, col_names)

    if drop_or_nan == 'nan':
        drop_val = np.nan
    else:
        drop_val = get_unused_drop_val(covars)

    for key, d_type in zip(col_names, data_types):

        self._print('loading:', key)

        non_nan_subjects = covars[~covars[key].isna()].index
        non_nan_covars = covars.loc[non_nan_subjects]

        if d_type == 'binary' or d_type == 'b':

            non_nan_covars, self.covars_encoders[key] =\
                process_binary_input(non_nan_covars, key, in_place=False,
                                     drop_val=drop_val, _print=self._print)

        elif d_type == "categorical" or d_type == 'c':

            if code_categorical_as == 'ordinal':
                non_nan_covars, self.covars_encoders[key] =\
                    process_ordinal_input(non_nan_covars, key,
                                          categorical_drop_percent,
                                          in_place=False, drop_val=drop_val,
                                          _print=self._print)

            # If using one hot or dummy coding, encoder will
            # contain 2 transformers
            else:
                non_nan_covars, new_keys, self.covars_encoders[key] =\
                    process_categorical_input(non_nan_covars, key,
                                              code_categorical_as,
                                              categorical_drop_percent,
                                              in_place=False,
                                              drop_val=drop_val,
                                              _print=self._print)

                # Add any new cols
                covars = covars.reindex(columns=list(non_nan_covars))

        elif (d_type == 'float' or d_type == 'ordinal' or
              d_type == 'f' or d_type == 'o'):

            if (filter_float_outlier_percent is not None) and \
                    (d_type == 'float' or d_type == 'f'):

                non_nan_covars =\
                    filter_float_by_outlier(non_nan_covars, key,
                                            filter_float_outlier_percent,
                                            in_place=False, drop_val=drop_val,
                                            _print=self._print)

            # Perform scaling if needed
            non_nan_covars = scale_input(non_nan_covars, key, standardize,
                                         normalize, in_place=False,
                                         drop_val=drop_val)

            self.covars_encoders[key] = None

        elif d_type == 'multilabel' or d_type == 'm':
            common_name = process_multilabel_input(key)

            self.covars_encoders[common_name] = key
            self._print('Base str indicator/name for loaded multilabel =',
                        common_name)

        # Now update the changed values within covars
        covars.loc[non_nan_subjects] = non_nan_covars

        # Make sure col types are right
        for dtype, key in zip(non_nan_covars.dtypes, list(covars)):
            covars[key] = covars[key].astype(dtype.name)

    # Have to remove rows with drop_val if drop_val not NaN
    if drop_val is not np.nan:
        covars = self._drop_from_filter(covars, drop_val)

    self._print('loaded shape: ', covars.shape)

    # If other data is already loaded,
    # merge this data with existing loaded data.
    self.covars = self._merge_existing(self.covars, covars)
    self._process_new(self.low_memory_mode)


def Load_Targets(self, loc, col_name, data_type,
                 dataset_type='default', overlap_subjects=False,
                 filter_outlier_percent=None,
                 categorical_drop_percent=None,
                 clear_existing=False):
    '''Loads in a set of subject ids and associated targets from a
    2.0_ABCD_Data_Explorer release formatted csv.
    See Notes for more info.

    Parameters
    ----------
    loc : str, Path or None
        The location of the csv file to load targets load from.

    col_name : str or list
        The name(s) of the column(s) to load.

        Note: Must be in the same order as data types passed in.

    data_type : {'binary', 'categorical', 'multilabel', 'ordinal', 'float'}
        The data types of the different columns to load,
        in the same order as the column names passed in.
        Shorthands for datatypes can be used as well.

        - 'binary' or 'b' : Binary input
        - 'categorical' or 'c' : Categorical input
        - 'multilabel' or 'm' : Multilabel categorical input
        - 'ordinal' or 'o' : Ordinal input
        - 'float' or 'f' : Float numerical input

        .. WARNING::
            If 'multilabel' datatype is specified, then the associated col name
            should be a list of columns, and will be assumed to be.
            For example, if loading multiple targets and one is multilabel,
            a nested list should be passed to col_name.

        Datatypes are explained further in Notes.

    dataset_type : {'default', 'basic', 'explorer', 'custom'}, optional
        The type of file to load from.
        Dataset types are,

        - 'default' : Use the class defined default dataset type,\
            if not set by the user this is 'basic'.\

        - 'basic' : ABCD2p0NDA style, (.txt and tab seperated)

        - 'explorer' : 2.0_ABCD_Data_Explorer style (.csv and comma seperated)

        - 'custom' : A user-defined custom dataset. Right now this is only\
            supported as a comma seperated file, with the subject names in a\
            column called self.subject_id.

        (default = 'default')

     overlap_subjects : bool, optional
        This parameter dictates if the loaded targets,
        (after initial basic proc)
        should be restricted to only the overlapping subjects from previously
        loaded data, targets, covars or strat. If False, then all subjects will
        be kept throughout the rest of the optional processing - and only
        merged at the end with only the existing loaded targets (if any).

        Note: Inclusions and Exclusions are always applied regardless of this
        parameter.

        (default = False)

    filter_outlier_percent : float, tuple or None, optional
        For float or ordinal datatypes only.
        A percent of values to exclude from either end of the
        target distribution, provided as either 1 number,
        or a tuple (% from lower, % from higher).
        set `filter_outlier_percent` to None for no filtering.

        (default = None).

    categorical_drop_percent: float or None, optional
        Optional percentage threshold for dropping categories when
        loading categorical data. If a float is given, then a category
        will be dropped if it makes up less than that % of the data points.
        E.g. if .01 is passed, then any datapoints with a category with less
        then 1% of total valid datapoints is dropped.

        (default = None)

    clear_existing : bool, optional
        If this parameter is set to True, then any existing
        loaded targets will first be cleared before loading new targets!

        .. WARNING::
            If any subjects have been dropped from a different place,
            e.g. covars or data, then simply reloading / clearing existing
            covars might result in computing a misleading overlap of final
            valid subjects. Reloading should therefore be best used
            right after loading the original data, or if not possible,
            then reloading the notebook or re-running the script.

        (default = False)

    Notes
    ----------
    Targets can be either 'binary', 'categorical', 'multilabel',
    'ordinal' or 'float',
    where ordinal and float are generally treated the same.

    - binary : Targets are read in and label encoded to be 0 or 1, \
    Will also work if passed column of unique string also, e.g. 'M' and 'F'

    - categorical : Targets are treated as taking on one fixed value from a\
    limited set of possible values.

    - multilabel : Simmilar to categorical, but targets can take on multiple\
    values, and therefore cannot be reduced to an ordinal representation.

    - ordinal and float : Targets are read in as a floating point number,\
    and optionally then filtered.
    '''

    if clear_existing:
        self.Clear_Targets()

    targets, col_names = self._common_load(loc, dataset_type,
                                           overlap_subjects,
                                           col_names=col_name)

    data_types, col_names = proc_datatypes(data_type, col_names)

    for key, d_type in zip(col_names, data_types):

        self._print('loading:', key)
        targets_key = key

        # Processing for binary, with some tolerance to funky input
        if d_type == 'binary' or d_type == 'b':

            targets, self.targets_encoders[key] =\
                process_binary_input(targets, targets_key, self._print)

        # Categorical proc, defaults to ordinal encoding
        elif d_type == 'categorical' or d_type == 'c':

            targets, self.targets_encoders[key] =\
                process_ordinal_input(targets, targets_key,
                                      categorical_drop_percent,
                                      in_place=True, _print=self._print)

        # For float or ordinal, just optionally apply outlier percent filter
        elif (d_type == 'float' or d_type == 'ordinal' or
              d_type == 'f' or d_type == 'o'):

            if filter_outlier_percent is not None:
                targets = filter_float_by_outlier(targets, targets_key,
                                                  filter_outlier_percent,
                                                  in_place=True,
                                                  _print=self._print)

            self.targets_encoders[key] = None

        # Multilabel type must be read in from multiple columns
        elif d_type == 'multilabel' or d_type == 'm':
            common_name = process_multilabel_input(targets_key)

            self.targets_encoders[common_name] = targets_key
            self._print('Base str indicator/name for loaded multilabel =',
                        common_name)

        # Keep track of each loaded target in targets_keys
        if targets_key not in self.targets_keys:
            self.targets_keys.append(targets_key)

    self._print('Final shape: ', targets.shape)

    # By default only load one set of targets note now, so no merging
    self.targets = self._merge_existing(self.targets, targets)
    self._process_new(self.low_memory_mode)

    self._print('All loaded targets')
    for i in range(len(self.targets_keys)):
        self._print(i, ':', self.targets_keys[i])


def Load_Strat(self, loc, col_name, dataset_type='default',
               overlap_subjects=False, binary_col_inds=None,
               float_col_inds=None, float_bins=10,
               float_bin_strategy='uniform', clear_existing=False):
    '''Load stratification values from a file.
    See Notes for more details on what stratification values are.

    Parameters
    ----------
    loc : str, Path or None
        The location of the csv file to load stratification vals from.

    col_name : str or list
        The name(s) of the column(s) to load.

    dataset_type : {'default', 'basic', 'explorer', 'custom'}, optional
        The type of file to load from.
        Dataset types are,

        - 'default' : Use the class defined default dataset type,\
            if not set by the user this is 'basic'.

        - 'basic' : ABCD2p0NDA style (.txt and tab seperated)

        - 'explorer' : 2.0_ABCD_Data_Explorer style (.csv and comma seperated)

        - 'custom' : A user-defined custom dataset. Right now this is only\
            supported as a comma seperated file, with the subject names in a\
            column called self.subject_id.

        (default = 'default')

     overlap_subjects : bool, optional
        This parameter dictates if the loaded strat,
        (after initial basic proc)
        should be restricted to only the overlapping subjects from previously
        loaded data, targets, covars or strat. If False, then all subjects will
        be kept throughout the rest of the optional processing - and only
        merged at the end with only the existing loaded strat (if any).

        Note: Inclusions and Exclusions are always applied regardless of this
        parameter.

        (default = False)

    binary_col_inds : int, list or None, optional
        Strat values are loaded as ordinal categorical, but there still
        exists the case where the user would like to load a binary set of
        values, and would like to ensure they are binary (filtering out
        all values but the top 2 most frequent). This input should be
        either None, for just loading in all cols as ordinal,
        or an int or list of ints, where each int refers to the
        numerical index within the passed `col_names` of a column
        which should be loaded explicitly as binary.

        (default = None)

    float_col_inds : int, list or None, optional
        Strat values are loaded as ordinal categorical, but one
        could also want to load a float value, and bin it into according
        to some strategy into ordinal categorical. This input should be
        either None, for just loading in all cols as ordinal,
        or an int or list of ints, where each int refers to the
        numerical index within the passed `col_names` of a column
        which should be loaded as a float.

        (default = None)

    float_bins : int, optional
        If any float_col_inds are specified, then the float
        input must be discretized into bins. This param controls
        the number of bins to create. If the user desires different
        settings for making bins for different loaded values, they should
        just call Load_Strat seperately for each unique set of `float_bins`
        and `float_bin_strategy` params.

        (default = 10)

    float_bin_strategy : {'uniform', 'quantile', 'kmeans'}, optional
        The strategy used to define the bins if any `float_col_inds`
        are specified.

        - 'uniform' : All bins in each feature have identical widths.
        - 'quantile' : All bins in each feature have the same number of points.
        - 'kmeans' : Values in each bin have the same nearest center of a 1D \
                     k-means cluster.

        (default = 'uniform')

    clear_existing : bool, optional
        If this parameter is set to True, then any existing
        loaded strat will first be cleared before loading new strat!

        .. WARNING::
            If any subjects have been dropped from a different place,
            e.g. targets or data, then simply reloading / clearing existing
            strat might result in computing a misleading overlap of final
            valid subjects. Reloading should therefore be best used
            right after loading the original strat, or if not possible,
            then reloading the notebook or re-running the script.

        (default = False)

    Notes
    ----------
    Stratification values are categorical variables which are loaded for the
    purpose of defining custom validation behavior.

    For example: Sex might be loaded here, and used later to ensure
    that any validation splits retain the same distribution of each sex.

    There is a reason strat is loaded after data, covars and targets,
    If you re-load any of them for whatever reason, after strat is already
    loaded, it could potentially lead to weird bugs.
    The easiest option is to just load strat last.
    '''
    if clear_existing:
        self.Clear_Strat()

    self._print('Reading strat/stratification values!')
    strat, col_names = self._common_load(loc, dataset_type,
                                         overlap_subjects,
                                         col_names=col_name)

    binary_col_names = []
    if binary_col_inds is not None:

        if isinstance(binary_col_inds, int):
            binary_col_inds = [binary_col_inds]

        binary_col_names = [col_names[i] for i in binary_col_inds]

    float_col_names = []
    if float_col_inds is not None:

        if isinstance(float_col_inds, int):
            float_col_inds = [float_col_inds]

        float_col_names = [col_names[i] for i in float_col_inds]

    # Encode each column into unique values
    for col in col_names:

        if col in binary_col_names:
            strat, self.strat_encoders[col + self.strat_u_name] =\
                process_binary_input(strat, col, self._print)

        elif col in float_col_names:
            strat, self.strat_encoders[col + self.strat_u_name] =\
                process_float_input(strat, col, float_bins, float_bin_strategy)

        else:
            strat, self.strat_encoders[col + self.strat_u_name] =\
                process_ordinal_input(strat, col)

    col_mapping = {col: col + self.strat_u_name for col in strat}
    strat = strat.rename(col_mapping, axis=1)

    self.strat = self._merge_existing(self.strat, strat)
    self._process_new(self.low_memory_mode)


def Load_Exclusions(self, loc=None, subjects=None, clear_existing=False):
    '''Loads in a set of excluded subjects,
    from either a file or as directly passed in.

    Parameters
    ----------
    loc : str, Path or None, optional
        Location of a file to load in excluded subjects from.
        The file should be formatted as one subject per line.
        (default = None)

    subjects : list, set, array-like or None, optional
        An explicit list of subjects to add to exclusions.
        (default = None)

    clear_existing : bool, optional
        If this parameter is set to True, then any existing
        loaded exclusions will first be cleared before loading new exclusions!

        .. WARNING::
            If any subjects have been dropped from a different place,
            e.g. targets or data, then simply reloading / clearing existing
            exclusions might result in computing a misleading overlap of final
            valid subjects. Reloading should therefore be best used
            right after loading the original exclusions, or if not possible,
            then reloading the notebook or re-running the script.

        (default = False)

    Notes
    ----------
    For best/most reliable performance across all data loading cases,
    exclusions should be loaded before data, covars and targets.

    If default subject id behavior is set to False,
    reading subjects from a exclusion loc might not
    function as expected.
    '''
    if clear_existing:
        self.Clear_Exclusions()

    self.exclusions.update(self._load_set_of_subjects(loc=loc,
                                                      subjects=subjects))
    self._print('Total excluded subjects: ', len(self.exclusions))
    self._filter_excluded()


def Load_Inclusions(self, loc=None, subjects=None, clear_existing=False):
    '''Loads in a set of subjects such that only these subjects
    can be loaded in, and any subject not as an inclusion is dropped,
    from either a file or as directly passed in.

    Parameters
    ----------
    loc : str, Path or None, optional
        Location of a file to load in inclusion subjects from.
        The file should be formatted as one subject per line.
        (default = None)

    subjects : list, set, array-like or None, optional
        An explicit list of subjects to add to inclusions.
        (default = None)

    clear_existing : bool, optional
        If this parameter is set to True, then any existing
        loaded inclusions will first be cleared before loading new inclusions!

        .. WARNING::
            If any subjects have been dropped from a different place,
            e.g. targets or data, then simply reloading / clearing existing
            inclusions might result in computing a misleading overlap of final
            valid subjects. Reloading should therefore be best used
            right after loading the original inclusions, or if not possible,
            then reloading the notebook or re-running the script.

        (default = False)

    Notes
    ----------
    For best/most reliable performance across all data loading cases,
    inclusions should be loaded before data, covars and targets.

    If default subject id behavior is set to False,
    reading subjects from a inclusion loc might not
    function as expected.
    '''
    if clear_existing:
        self.Clear_Inclusions()

    self.inclusions.update(self._load_set_of_subjects(loc=loc,
                                                      subjects=subjects))
    self._print('Total inclusion subjects: ', len(self.inclusions))
    self._filter_included()


def Clear_Name_Map(self):
    '''Reset name mapping'''
    self.name_map = {}
    self._print('cleared name map.')


def Clear_Data(self):
    '''Resets any loaded data.

    .. WARNING::
            If any subjects have been dropped from a different place,
            e.g. targets, then simply clearing data might result
            in computing a misleading overlap of final valid subjects.
            Reloading should therefore be best used right after loading
            the original data, or if not possible, then reloading the
            notebook or re-running the script.

    '''
    self.data = pd.DataFrame()
    self._print('Cleared loaded data.')


def Clear_Covars(self):
    '''Reset any loaded covars.

    .. WARNING::
            If any subjects have been dropped from a different place,
            e.g. targets or data, then simply reloading / clearing existing
            covars might result in computing a misleading overlap of final
            valid subjects. Reloading should therefore be best used
            right after loading the original covars, or if not possible,
            then reloading the notebook or re-running the script.

    '''
    self.covars = pd.DataFrame()
    self.covars_encoders = {}
    self._print('cleared covars.')


def Clear_Targets(self):
    '''Resets targets'''
    self.targets = pd.DataFrame()
    self.targets_encoders = {}
    self.targets_keys = []
    self._print('cleared targets.')


def Clear_Strat(self):
    '''Reset any loaded strat

    .. WARNING::
            If any subjects have been dropped from a different place,
            e.g. targets or data, then simply reloading / clearing existing
            strat might result in computing a misleading overlap of final
            valid subjects. Reloading should therefore be best used
            right after loading the original strat, or if not possible,
            then reloading the notebook or re-running the script.

    '''
    self.strat = pd.DataFrame()
    self.strat_encoders = {}
    self._print('cleared strat.')


def Clear_Exclusions(self):
    '''Resets exclusions to be an empty set.

    .. WARNING::
            If any subjects have been dropped from a different place,
            e.g. targets or data, then simply reloading / clearing existing
            exclusions might result in computing a misleading overlap of final
            valid subjects. Reloading should therefore be best used
            right after loading the original exclusions, or if not possible,
            then reloading the notebook or re-running the script.

    '''
    self.exclusions = set()
    self._print('cleared exclusions.')


def Clear_Inclusions(self):
    '''Resets inclusions to be an empty set.

    .. WARNING::
        If any subjects have been dropped from a different place,
        e.g. targets or data, then simply reloading / clearing existing
        inclusions might result in computing a misleading overlap of final
        valid subjects. Reloading should therefore be best used
        right after loading the original inclusions, or if not possible,
        then reloading the notebook or re-running the script.

    '''
    self.inclusions = set()
    self._print('cleared inclusions.')


def Drop_Data_Duplicates(self, corr_thresh):
    '''Drop duplicates columns within self.data based on
    if two data columns are >= to a certain correlation threshold.

    Parameters
    ----------
    corr_thresh : float
        A value between 0 and 1, where if two columns within self.data
        are correlated >= to `corr_thresh`, the second column is removed.

        A value of 1 will instead make a quicker direct =='s comparison.
    '''

    self.data, dropped = drop_duplicate_cols(self.data, corr_thresh)
    self._print('Dropped', len(dropped), 'columns as duplicate cols!')


def Binarize_Target(self, lower, upper, target=0):
    '''This function binarizes a loaded target variable,
    assuming that an ordinal or float type target is loaded,
    otherwise this function will break!

    Parameters
    ----------
    lower : float
        Any value that is greater than lower will be set to 1,
        and any value <= upper and >= lower will be dropped.

    upper : float
        Any value that is less than upper will be set to 0,
        and any value <= upper and >= lower will be dropped.

    target : int or str, optional
        The loaded target in which to Binarize. This can be
        the int index, or the name of the target column.
        If only one target is loaded, just leave as default.

        (default = 0)
    '''

    targets_key = self._get_targets_key(target)

    self._print('Binarizing', targets_key)

    target_values = self.targets[targets_key]
    self._print('Keeping:', (target_values > upper).sum(), 'as 1.')
    self._print('Keeping:', (target_values < lower).sum(), 'as 0.')

    # Drop out the middle
    to_drop = target_values[(target_values <= upper) &
                            (target_values >= lower)].index
    self.targets = self.targets.drop(to_drop)
    drop_middle = self.targets[targets_key]

    # Binarize remaining
    binarize = drop_middle.where(drop_middle > lower, 0)
    binarize = binarize.where(binarize < upper, 1)

    # Fill back into targets
    self.targets[targets_key] = binarize

    # Conv to categorical data type
    self.targets[targets_key] =\
        self.targets[targets_key].astype('category')

    # Global proc.
    self._process_new(self.low_memory_mode)

    # Save new encoder
    self.targets_encoders[self._get_targets_key(target, base_key=True)] =\
        {0: '<' + str(lower), 1: '>' + str(upper)}


def Get_Overlapping_Subjects(self):
    '''This function will return the set of valid
    overlapping subjects currently loaded across data,
    targets, covars, strat ect... respecting any inclusions
    and exclusions.

    Returns
    ----------
    set
        The set of valid overlapping subjects.
    '''
    return self._get_overlapping_subjects()


def _get_targets_key(self, key, base_key=False):

    targets_base_keys = self._get_base_targets_names()

    if isinstance(key, int):
        ind = key
    else:
        ind = targets_base_keys.index(key)

    if base_key:
        return targets_base_keys[ind]
    return self.targets_keys[ind]


def _load_datasets(self, locs, dataset_types):
    '''Helper function to load in multiple datasets with default
    load and drop behavior based on type. And calls proc_df on each
    before merging.

    Parameters
    ----------
    locs : list of str, Path
        The location of the csv files to load data load from.

    dataset_types : {'default', 'basic' 'explorer', 'custom'} or list
        The type of dataset to load from. If a list is passed, then locs must
        also be a list, and the indices should correspond.
        Likewise, if locs is a list and dataset_type is not,
        it is assumed all datasets are the same type.
        Where each dataset type is,

        - 'default': Use the class defined default dataset type,
            if not set by the user this is 'basic'.

        - 'basic' : ABCD2p0NDA style, (.txt and tab seperated)
            Typically the default columns, and therefore not neuroimaging
            data, will be dropped, also not including the eventname column.

        - 'explorer' : 2.0_ABCD_Data_Explorer tyle (.csv and comma seperated)
            The first 2 columns before self.subject_id
            (typically the default columns, and therefore not neuroimaging
            data - also not including the eventname column), will be dropped.

        - 'custom' : A user-defined custom dataset. Right now this is only
            supported as a comma seperated file, with the subject names in a
            column called self.subject_id. No columns will be dropped,
            unless specific drop keys are passed.

    Returns
    ----------
    pandas DataFrame
        ABCD ML formatted pd DataFrame, with the loaded
        and merged minimally proc'ed data.
    '''

    # If only one dataset type, use it for all
    if not isinstance(dataset_types, list):
        dataset_types = [dataset_types for i in range(len(locs))]

    # Load the first loc
    data = self._load_dataset(locs[0], dataset_types[0])

    # For the rest
    for loc, dataset_type in zip(locs[1:], dataset_types[1:]):

        # Load & Merge
        more_data = self._load_dataset(loc, dataset_type)

        repeat_col_names = set(list(data)).intersection(set(list(more_data)))

        if len(repeat_col_names) > 0:
            self._print('Warning,', repeat_col_names,
                        'exist in both dataframes!')
            self._print('By default repeats will be added as new unique',
                        'columns within merged data.')

        data = pd.merge(data, more_data, on=self.subject_id)

    return data


def _load_dataset(self, loc, dataset_type):
    '''Helper function to load in a dataset with default
    load and drop behavior based on type. And calls proc_df.

    Parameters
    ----------
    loc : str, Path or None
        The location of the csv/txt file to load data load from.

    dataset_type : {'default', 'basic', 'explorer', 'custom'}
        The type of dataset to load from. Where,

        - 'default' : Use the class defined default dataset type,
            if not set by the user this is 'basic'.

        - 'basic' : ABCD2p0NDA style, (.txt and tab seperated)
            Typically the default columns, and therefore not neuroimaging
            data, will be dropped, also not including the eventname column.

        - 'explorer' : 2.0_ABCD_Data_Explorer tyle (.csv and comma seperated)
            The first 2 columns before self.subject_id
            (typically the default columns, and therefore not neuroimaging
            data - also not including the eventname column), will be dropped.

        - 'custom' : A user-defined custom dataset. Right now this is only
            supported as a comma seperated file, with the subject names in a
            column called self.subject_id. No columns will be dropped,
            unless specific drop keys are passed.

    Returns
    ----------
    pandas DataFrame
        ABCD ML formatted pd DataFrame, with the loaded
        minimally proc'ed data.
    '''

    if dataset_type == 'default':
        dataset_type = self.default_dataset_type

    data = self._load(loc, dataset_type)

    # If dataset type is basic or explorer, drop some cols by default
    if dataset_type == 'basic' or dataset_type == 'explorer':

        if dataset_type == 'basic':
            non_data_cols = list(data)[:4] + list(data)[5:8]
        else:
            non_data_cols = list(data)[:2]

        data = data.drop(non_data_cols, axis=1)

        # Drop extra by presence of extra drop keys
        extra_drop_keys = ['visitid', 'collection_title', 'study_cohort_name']
        to_drop = [name for name in list(data) for drop_key in extra_drop_keys
                   if drop_key in name]
        data = data.drop(to_drop, axis=1)

        self._print('dropped', non_data_cols + to_drop, 'columns by default',
                    ' due to dataset type')

    self._print()

    # Perform common operations
    # (check subject id, drop duplicate subjects ect...)
    data = self._proc_df(data)

    return data


def _common_load(self, loc, dataset_type, overlap_subjects,
                 col_name=None, col_names=None, drop_nan=True):

    # Read csv data from loc
    data = self._load(loc, dataset_type)

    # Perform common df processing, on subject ids,
    # duplicate subjects + eventname
    data = self._proc_df(data)

    if overlap_subjects:
        valid_subjects = self._get_overlapping_subjects()

        if len(valid_subjects) > 0:
            data = data[data.index.isin(valid_subjects)]
            self._print('Set to overlap subjects only')

    if col_name is not None:

        if col_name in self.name_map:
            col_name = self.name_map[col_name]

        data = self._drop_na(data[[col_name]], drop_nan)
        return data

    if not isinstance(col_names, list):
        col_names = list([col_names])

    for i in range(len(col_names)):
        if col_names[i] in self.name_map:
            col_names[i] = self.name_map[col_names[i]]

    # Drop rows with NaN
    data = self._drop_na(data[col_names], drop_nan)
    return data, col_names


def _load(self, loc, dataset_type):
    '''Base load helper function, for simply loading file
    into memory based on dataset type.

    Parameters
    ----------
    loc : str, Path or None
        The location of the csv/txt file to load data load from.

    dataset_type : {'default', 'basic', 'explorer', 'custom'}
        The type of file to load from.
        Dataset types are,

        - 'default' : Use the class defined default dataset type,
            if not set by the user this is 'basic'.

        - 'basic' : ABCD2p0NDA style, (.txt and tab seperated)

        - 'explorer' : 2.0_ABCD_Data_Explorer style (.csv and comma seperated)

        - 'custom' : A user-defined custom dataset. Right now this is only
            supported as a comma seperated file, with the subject names in a
            column called self.subject_id.

    Returns
    ----------
    pandas DataFrame
        Loaded DataFrame.
    '''

    if dataset_type == 'default':
        dataset_type = self.default_dataset_type

    self._print('Loading', loc, 'assumed to be dataset type:', dataset_type)

    if dataset_type == 'basic':
        data = pd.read_csv(loc, sep='\t', skiprows=[1],
                           na_values=self.default_na_values,
                           low_memory=self.low_memory_mode)
    else:
        data = pd.read_csv(loc, na_values=self.default_na_values,
                           low_memory=self.low_memory_mode)

    return data


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

        repeat_col_names = set(class_data).intersection(set(local_data))
        if len(repeat_col_names) > 0:
            raise RuntimeError('These col_names appear in both dfs:',
                               repeat_col_names)

        class_data = pd.merge(class_data, local_data, on=self.subject_id)
        self._print('Merged with existing!')
        self._print('New combined shape:', class_data.shape)
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

    assert self.subject_id in data, "Missing subject id column!"

    # Perform corrects on subject ID
    data[self.subject_id] =\
        data[self.subject_id].apply(self._process_subject_name)

    # Filter by eventname is applicable
    data = self._filter_by_eventname(data)

    # Drop any duplicate subjects, default behavior for now
    # though, could imagine a case where you wouldn't want to when
    # there are future releases.
    before = data.shape[0]
    data = data.drop_duplicates(subset=self.subject_id)
    after = data.shape[0]

    if before != after:
        self._print(before - after, 'duplicate subjects dropped.')

    # Rename columns if loaded name map
    if self.name_map:
        data = data.rename(self.name_map, axis=1)

    data = data.set_index(self.subject_id)

    # Drop excluded subjects, if any
    data = self._drop_excluded(data)
    data = self._drop_included(data)

    return data


def _load_set_of_subjects(self, loc=None, subjects=None):
    '''Internal helper function, to load in a set of subjects from either
    a saved location, or directly passed in as a set or list of subjects.

    Parameters
    ----------
    loc : str, Path or None, optional
        Location of a file to load in subjects from.
        The file should be formatted as one subject per line.
        (default = None)

    subjects : list, set, array-like or None, optional
        An explicit list of subjects to add to exclusions.
        (default = None)

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

        if isinstance(subjects, str):
            self._print('Subjects passed as str, assuming just one subject')
            subjects = [subjects]

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


def _drop_na(self, data, drop_nan=True):
    '''Wrapper function to drop rows with NaN values.

    Parameters
    ----------
    data : pandas DataFrame
        ABCD_ML formatted.

    drop_nan : drop_nan param
        ABCD_ML drop_nan param

    Returns
    ----------
    pandas DataFrame
        Input df, with dropped rows for NaN values
    '''

    # First drop any columns with all NaN
    missing_values = data.isna().all(axis=0)
    data = data.dropna(axis=1, how='all')
    self._print('Dropped', sum(missing_values), 'cols for all missing values')

    # Handle dropping rows based on passed drop_nan param
    if drop_nan == 'default':
        drop_nan = self.drop_nan

    if drop_nan is not False:

        if drop_nan is True:
            nan_thresh = 0
        elif drop_nan <= 1:
            nan_thresh = int(drop_nan * data.shape[1])
        else:
            nan_thresh = drop_nan

        to_drop = data[data.isna().sum(axis=1) > nan_thresh].index
        data = data.drop(to_drop)
        self._print('Dropped', len(to_drop), 'rows for missing values, based',
                    'on the provided drop_nan param:', drop_nan,
                    'with actual nan_thresh:', nan_thresh)

    remaining_na_rows = data.isna().any(axis=1).sum()
    self._print('Loaded rows with NaN remaining:', remaining_na_rows)

    return data


def _drop_from_filter(self, data, drop_val=999):

    to_drop = data[(data == drop_val).any(axis=1)].index

    if len(to_drop) > 0:
        data = data.drop(to_drop)
        self._print('Dropped', len(to_drop), 'rows based on filter input',
                    'params, e.g. filter outlier percent, drop cat, ect...')
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

        if 'eventname' in list(data):

            before = data.shape[0]
            data = data[data['eventname'] == self.eventname]
            after = data.shape[0]

            if before != after:
                self._print(before - after, 'subjects dropped for eventname')

            data = data.drop('eventname', axis=1)

        else:
            self._print('Warning: filter by eventname =', self.eventname,
                        'is set but this data does not have a column with',
                        'eventname.')

    # If eventname None, but still exists, take out the column
    else:
        if 'eventname' in list(data):
            data = data.drop('eventname', axis=1)

    return data


def _show_nan_info(self, data):

    na_counts = data.isna().sum().sort_values(ascending=False)

    if na_counts.sum() > 0:
        self._print('Loaded NaN Info:')
        self._print('There are:', na_counts.sum(), 'total missing values')

        u_counts, c_counts = np.unique(na_counts, return_counts=True)
        u_counts, c_counts = u_counts[1:], c_counts[1:]

        inds = c_counts.argsort()
        u_counts = u_counts[inds[::-1]]
        c_counts = c_counts[inds[::-1]]

        for u, c in zip(u_counts, c_counts):
            if c > 1:

                keys = list(na_counts[na_counts == u].index)
                substrs = get_top_substrs(keys)

                self._print(c, ' columns found with ', u, ' missing values',
                            ' (column name overlap: ', substrs, ')', sep='')

        self._print()


def _drop_excluded(self, data):
    '''Wrapper to drop subjects from a df,
    if any overlap with existing.

    Parameters
    ----------
    data : pandas DataFrame
        ABCD_ML formatted.

    Returns
    ----------
    pandas DataFrame
        Input df, with dropped excluded subjects
    '''

    overlap = set(data.index).intersection(self.exclusions)
    data = data.drop(overlap)

    return data


def _drop_included(self, data):
    '''Wrapper to drop subjects from a df,
    if any do not overlap with inclusions.

    Parameters
    ----------
    data : pandas DataFrame
        ABCD_ML formatted.

    Returns
    ----------
    pandas DataFrame
        Input df, with dropped non included subjects
    '''

    if len(self.inclusions) > 0:

        to_drop = set(data.index) - self.inclusions
        data = data.drop(to_drop)

    return data


def _filter_excluded(self):
    '''This method is called after loading any new
    exclusions, it retroactively removes subjects from any
    loaded sources.'''

    if len(self.data) > 0:
        self.data = self._drop_excluded(self.data)
    if len(self.covars) > 0:
        self.covars = self._drop_excluded(self.covars)
    if len(self.targets) > 0:
        self.targets = self._drop_excluded(self.targets)
    if len(self.strat) > 0:
        self.strat = self._drop_excluded(self.strat)

    self._print('Removed excluded subjects from loaded dfs')


def _filter_included(self):
    '''This method is called after loading any new
    inclusions, it retroactively removes subjects from any
    loaded sources.'''

    if len(self.data) > 0:
        self.data = self._drop_included(self.data)
    if len(self.covars) > 0:
        self.covars = self._drop_included(self.covars)
    if len(self.targets) > 0:
        self.targets = self._drop_included(self.targets)
    if len(self.strat) > 0:
        self.strat = self._drop_included(self.strat)

    if len(self.inclusions) > 0:
        self._print('Removed subjects from loaded dfs based on inclusions')


def _get_overlapping_subjects(self):

    valid_subjects = []

    if len(self.data) > 0:
        valid_subjects.append(set(self.data.index))
    if len(self.covars) > 0:
        valid_subjects.append(set(self.covars.index))
    if len(self.targets) > 0:
        valid_subjects.append(set(self.targets.index))
    if len(self.strat) > 0:
        valid_subjects.append(set(self.strat.index))

    if len(valid_subjects) > 0:
        overlap = set.intersection(*valid_subjects)

        return overlap

    return set()


def _process_new(self, remove=False):
    '''Internal helper function to handle keeping an overlapping subject list,
    with additional useful print statements.

    Parameters
    ----------
    remove : bool, optional
        If True, remove non overlapping subjects - exclusions
        from all data in place.

    '''

    overlap = self._get_overlapping_subjects()

    if len(overlap) > 0:
        self._print()
        self._print('Total valid overlapping subjects =', len(overlap))
        self._print()

        if remove:
            self._print('Removing non overlapping subjects')

            if len(self.data) > 0:
                self.data = self.data[self.data.index.isin(overlap)]
            if len(self.covars) > 0:
                self.covars = self.covars[self.covars.index.isin(overlap)]
            if len(self.targets) > 0:
                self.targets = self.targets[self.targets.index.isin(overlap)]
            if len(self.strat) > 0:
                self.strat = self.strat[self.strat.index.isin(overlap)]


def _prepare_data(self):
    '''Helper function to prepare all loaded data,
    from different sources, self.data, self.covars, ect...
    into self.all_data for use directly in ML.
    '''

    self._print('Preparing final data, in self.all_data')
    self._print('Any changes to loaded data, covars or strat will not be',
                'included.')

    dfs = []

    assert len(self.data) > 0 or len(self.covars) > 0, \
        'Some data must be loaded!'

    if len(self.data) > 0:
        dfs.append(self.data)

    if len(self.covars) > 0:
        dfs.append(self.covars)

    if len(self.strat) > 0:
        dfs.append(self.strat)

    assert len(self.targets) > 0, \
        'Targets must be loaded!'

    dfs.append(self.targets)

    self.all_data = dfs[0]
    for i in range(1, len(dfs)):

        overlap_c_names = np.array([c_name in list(self.all_data) for
                                    c_name in list(dfs[i])])

        if overlap_c_names.any():

            self._print('Col names from data, covars, targets and strat',
                        'must be unique!')

            overlap = np.array(list(dfs[i]))[overlap_c_names]
            assert 1 == 2, str(overlap) + ' col(s) overlap!'

        self.all_data = pd.merge(self.all_data, dfs[i], on=self.subject_id)

    # Set data keys, covars, strat, ect...
    self._set_all_data_keys()

    self._print('Final data (w/ target) for modeling loaded shape:',
                self.all_data.shape)

    if self.low_memory_mode:
        self._print('Low memory mode is on!')
        self._print('Clearing self.data, self.covars, self.targets',
                    'and self.strat from memory!')
        self._print('Note: Final data, self.all_data, the',
                    'merged dataframe is still in memory')

        self.Clear_Data()
        self.Clear_Targets()
        self.Clear_Covars()
        self.Clear_Strat()


def _get_cat_keys(self):
    '''Determines and sets the column for
    all categorical features if any. Also sets the class
    cat_keys attribute.'''

    # First determine which columns contain categorical
    cat_keys = list(self.all_data.select_dtypes(include='category'))

    # If any target is categorical exclude it
    for targets_key in self.targets_keys:

        try:
            if isinstance(targets_key, list):
                for t_key in targets_key:
                    cat_keys.remove(t_key)

            else:
                cat_keys.remove(targets_key)

        except ValueError:
            pass

    return cat_keys


def _set_all_data_keys(self):

    self.all_data_keys['data_keys'] = list(self.data)
    self.all_data_keys['covars_keys'] = list(self.covars)
    self.all_data_keys['strat_keys'] = list(self.strat)
    self.all_data_keys['cat_keys'] = self._get_cat_keys()


def _get_base_targets_names(self):

    targets_base_keys = []

    for targets_key in self.targets_keys:

        if isinstance(targets_key, list):
            base_targets_key = get_common_name(targets_key)
        else:
            base_targets_key = targets_key

        targets_base_keys.append(base_targets_key)

    return targets_base_keys


def _get_covar_scopes(self):

    # categorical also includes multilabel

    covar_scopes = {'float': [],
                    'categorical': [],
                    'ordinal categorical': []}
    cat_encoders = []

    for base_covar in list(self.covars_encoders):

        cov_encoder = self.covars_encoders[base_covar]

        # One-hot or dummy
        if isinstance(cov_encoder, tuple):

            one_hot_encoder = cov_encoder[1]
            cat_encoders.append(cov_encoder)

            categories = one_hot_encoder.categories_[0]
            covar_df_names = [base_covar + '_' + str(c) for
                              c in categories]
            valid_df_names = [c for c in covar_df_names if
                              c in self.all_data]

            covar_scopes['categorical'].append(valid_df_names)

        # Multilabel
        elif isinstance(cov_encoder, list):
            cat_encoders.append(None)
            covar_scopes['categorical'].append(cov_encoder)

        # Float
        elif cov_encoder is None:
            covar_scopes['float'].append(base_covar)

        # Binary/ordinal
        else:
            covar_scopes['ordinal categorical'].append(base_covar)

    return covar_scopes, cat_encoders
