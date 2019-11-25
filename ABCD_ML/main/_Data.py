"""
_Data.py
====================================
Main class extension file for the Loading functionality methods of
the ABCD_ML class.
"""
import pandas as pd
import numpy as np

from ..helpers.Data_Helpers import (process_binary_input,
                                    process_ordinal_input,
                                    process_categorical_input,
                                    process_float_input,
                                    process_multilabel_input,
                                    get_unused_drop_val,
                                    filter_float_by_outlier,
                                    filter_float_by_std,
                                    filter_float_df_by_outlier,
                                    filter_float_df_by_std,
                                    drop_duplicate_cols,
                                    get_top_substrs,
                                    proc_datatypes,
                                    proc_args,
                                    get_common_name)


def Set_Default_Load_Params(self, dataset_type='default', subject_id='default',
                            eventname='default', eventname_col='default',
                            overlap_subjects='default', na_values='default',
                            drop_na='default', drop_or_na='default'):
    ''' This function is used to define default values for a series of
    params accessible to all or most of the different loading functions.
    By setting common values here, it reduces the need to repeat params within
    each loader (e.g. Load_Data, Load_Targets, ect...)

    Parameters
    ----------
    dataset_type : {'basic', 'explorer', 'custom'}, optional
        The dataset_type / file-type to load from.
        Dataset types are,

        - 'basic'
            ABCD2p0NDA style (.txt and tab seperated).
            Typically the default columns, and therefore not neuroimaging
            data, will be dropped, also not including the eventname column.

        - 'explorer'
            2.0_ABCD_Data_Explorer style (.csv and comma seperated).
            The first 2 columns before self.subject_id
            (typically the default columns, and therefore not neuroimaging
            data - also not including the eventname column), will be dropped.

        - 'custom'
            A user-defined custom dataset. Right now this is only.
            supported as a comma seperated file, with the subject names in a
            column called self.subject_id, and can optionally have
            'eventname'. No columns will be dropped,
            (except eventname) or unless specific drop keys are passed.

        If loading multiple locs as a list, dataset_type can be a list with
        inds corresponding to which datatype for each loc.

        if 'default', and not already defined, set to 'basic'
        (default = 'default')

    subject_id : str, optional
        The name of the column with unique subject ids in different
        dataset, for default ABCD datasets this is 'src_subject_id',
        but if a user wanted to load and work with a different dataset,
        they just need to change this accordingly
        (in addition to setting eventname most likely to None and
        use_default_subject_ids to False)

        if 'default', and not already defined, set to 'src_subject_id'.
        (default = 'default')

    eventname : value, list of values or None, optional
        Optional value to provide, specifying to optional keep certain rows
        when reading data based on the eventname flag, where eventname
        is the value and eventname_col is the name of the value.

        If a list of values are passed, then it will be treated as keeping a
        row if that row's value within the eventname_col is equal to ANY
        of the passed eventname values.

        As ABCD is a longitudinal study, this flag lets you select only
        one specific time point, or if set to None, will load everything.

        For selecting only baseline imagine data one might consider
        setting this param to 'baseline_year_1_arm_1'.

        if 'default', and not already defined, set to None.
        (default = 'default')

    eventname_col : str or None, optional
        If an eventname is provided, this param refers to
        the column name containing the eventname. This could
        also be used along with eventname to be set to any
        arbitrary value, in order to perform selection by specific
        column value.

        Note: The eventname col is dropped after proc'ed!

        if 'default', and not already defined, set to 'eventname'
        (default = 'default')

    overlap_subjects : bool, optional
        This parameter dictates when loading data, covars, targets or strat
        (after initial basic proc and/or merge w/ other passed loc's),
        if the loaded data should be restricted to only the
        overlapping subjects from previously loaded data, targets, covars
        or strat - important when performing intermediate proc.
        If False, then all subjects will
        be kept throughout the rest of the optional processing - and only
        merged at the end AFTER processing has been done.

        Note: Inclusions and Exclusions are always applied regardless of this
        parameter.

        if 'default', and not already defined, set to False
        (default = 'default')

    na_values : list, optional
        Additional values to treat as NaN, by default ABCD specific
        values of '777' and '999' are treated as NaN,
        and those set to default by pandas 'read_csv' function.
        Note: if new values are passed here,
        it will override these default '777' and '999' NaN values,
        so if it desired to keep these, they should be passed explicitly,
        along with any new values.

        if 'default', and not already defined, set to ['777', '999']
        (default = 'default')

    drop_na : bool, int, float or 'default', optional
        This setting sets the value for drop_na,
        which is used when loading data and covars only!

        If set to True, then will drop any row within the loaded
        data if there are any NaN! If False, the will not drop any
        rows for missing values.

        If an int or float, then this means some NaN entries
        will potentially be preserved! Missing data imputation
        will therefore be required later on!

        If an int > 1, then will drop any row with more than drop_na
        NaN values. If a float, will determine the drop threshold as
        a percentage of the possible values, where 1 would not drop any
        rows as it would require the number of columns + 1 NaN, and .5
        would require that more than half the column entries are NaN in
        order to drop that row.

        if 'default', and not already defined, set to True
        (default = 'default')

    drop_or_na : {'drop', 'na'}, optional

        This setting sets the value for drop_na,
        which is used when loading data and covars only!

        filter_outlier_percent, or when loading a binary variable
        in load covars and more then two classes are present - are both
        instances where rows/subjects are by default dropped.
        If drop_or_na is set to 'na', then these values will instead be
        set to 'na' rather then the whole row dropped!

        Otherwise, if left as default value of 'drop', then rows will be
        dropped!

        if 'default', and not already defined, set to 'drop'
        (default = 'default')
    '''

    if dataset_type != 'default':
        self.default_load_params['dataset_type'] = dataset_type
    elif 'dataset_type' not in self.default_load_params:
        self.default_load_params['dataset_type'] = 'basic'
        self._print('No default dataset_type passed, set to "basic"')

    if subject_id != 'default':
        self.default_load_params['subject_id'] = subject_id
    elif 'subject_id' not in self.default_load_params:
        self.default_load_params['subject_id'] = 'src_subject_id'
        self._print('No default subject_id passed, set to "src_subject_id"')

    if eventname != 'default':
        self.default_load_params['eventname'] = eventname
    elif 'eventname' not in self.default_load_params:
        self.default_load_params['eventname'] = None
        self._print('No default eventname passed,',
                    'set to None')

    if eventname_col != 'default':
        self.default_load_params['eventname_col'] = eventname_col
    elif 'eventname_col' not in self.default_load_params:
        self.default_load_params['eventname_col'] = 'eventname'
        self._print('No default eventname_col passed,',
                    'set to "eventname"')

    if overlap_subjects != 'default':
        self.default_load_params['overlap_subjects'] = overlap_subjects
    elif 'overlap_subjects' not in self.default_load_params:
        self.default_load_params['overlap_subjects'] = False
        self._print('No default overlap_subjects passed, set to False')

    if na_values != 'default':
        self.default_load_params['na_values'] = na_values
    elif 'na_values' not in self.default_load_params:
        self.default_load_params['na_values'] = ['777', '999']
        self._print('No default na_values passed, set to ["777", "999"]')

    if drop_na != 'default':
        self.default_load_params['drop_na'] = drop_na
    elif 'drop_na' not in self.default_load_params:
        self.default_load_params['drop_na'] = True
        self._print('No default drop_na passed, set to True')

    if drop_or_na != 'default':
        self.default_load_params['drop_or_na'] = drop_or_na
    elif 'drop_or_na' not in self.default_load_params:
        self.default_load_params['drop_or_na'] = 'drop'
        self._print('No default drop_or_na passed, set to "drop"')

    self._print('Default load params set within self.default_load_params.')
    self._print()

    # subject_id='src_subject_id',
    # eventname='baseline_year_1_arm_1',
    # default_dataset_type='basic',
    # drop_nan=True,
    # default_na_values=['777', '999'],


def _make_load_params(self, args):

    if len(self.default_load_params) == 0:

        # Set default load params with default vals
        self._print('Setting default load params, as they have not been set!')
        self._print()
        self.Set_Default_Load_Params()
        self._print('To change the default load params, call',
                    'self.Set_Default_Load_Params()')
        self._print()

    load_params = self.default_load_params.copy()

    for key in args:
        if key in load_params:
            if args[key] != 'default' and args[key] != 'self':
                load_params[key] = args[key]

    return load_params


def Load_Name_Map(self, name_map=None, loc=None, dataset_type='default',
                  source_name_col="NDAR name",
                  target_name_col="REDCap name/NDA alias",
                  na_values='default',
                  clear_existing=False):
    '''Loads a mapping dictionary for loading column names. Either a loc
    or name_map must be passed! Note: If both a name_map and loc are passed,
    the name_map will be loaded first, then updated with values from the loc.

    Parameters
    ----------
    name_map : dict or None, optional
        A dictionary containing the mapping to be passed directly.
        Set to None if using loc instead!

        (default = None)

    loc : str, Path or None, optional
        The location of the csv file which contains the mapping.

        (default = None)

    dataset_type :

    source_name_col : str, optional
        The column name with the file which lists names to be changed.

        (default = "NDAR name")

    target_name_col : str, optional
        The column name within the file which lists the new name.

        (default = "REDCap name/NDA alias")

    na_values :

    clear_existing : bool, optional
        If set to True, will clear the existing loaded name_map, otherwise the
        name_map dictionary will be updated if already loaded!
    '''

    if clear_existing:
        self.Clear_Name_Map()

    if name_map is not None:

        if len(self.name_map) > 0:
            self._print('Updating existing name_map with new!')
        else:
            self._print('Loading new name_map')

        self.name_map.update(name_map)

    if loc is not None:

        load_params = self._make_load_params(args=locals())

        # Load mapping based on dataset type
        mapping = self._load(loc, load_params['dataset_type'],
                             load_params['na_values'])

        try:
            name_map_from_loc = dict(zip(mapping[source_name_col],
                                         mapping[target_name_col]))

            if len(self.name_map) > 0:
                self._print('Updating existing name_map with new from file!')
            else:
                self._print('Loading new name_map from file!')

            self.name_map.update(name_map_from_loc)

        except KeyError:
            print('Error: One or both provided column names do not exist!')
            print('Name map not loaded from loc!')


def Load_Data(self, loc=None, df=None, dataset_type='default', drop_keys=None,
              inclusion_keys=None, subject_id='default', eventname='default',
              eventname_col='default', overlap_subjects='default',
              na_values='default', drop_na='default', drop_or_na='default',
              filter_outlier_percent=None, filter_outlier_std=None,
              unique_val_drop=None, unique_val_warn=.05,
              drop_col_duplicates=None,
              clear_existing=False):
    """Class method for loading ROI-style data, assuming all loaded
    columns are continuous / float datatype.

    Parameters
    ----------
    loc : str Path, list of or None, optional
        The location of the file to load data load from.
        If passed a list, then will load each loc in the list,
        and will assume them all to be of the same dataset_type if one
        dataset_type is passed, or if they differ in type, a list must be
        passed to dataset_type with the different types in order.

        Note: some proc will be done on each loaded dataset before merging
        with the rest (duplicate subjects, proc for eventname ect...), but
        other dataset loading behavior won't occur until after the merge,
        e.g., dropping cols by key, filtering for outlier, ect...

    df : pandas DataFrame or None, optional
        This parameter represents the option for the user to pass in
        a raw custom dataframe. A loc and/or a df must be passed.

        When pasing a raw DataFrame, the loc and dataset_type
        param will be ignored, as those are for loading data from a file.
        Otherwise, it will be treated the same as
        if loading from a file, which means, there should be a column within
        the passed dataframe with subject_id, and e.g. if eventname params are
        passed, they will be applied along with any other proc. specified.

    dataset_type :
    drop_keys : str, list or None, optional
        A list of keys to drop columns by, where if any key given in a columns
        name, then that column will be dropped. If a str, then same behavior,
        just with one col.
        (Note: if a name mapping exists, this drop step will be
        conducted after renaming)

        (default = None)

    inclusion_keys : str, list or None, optional
        A list of keys in which to only keep a loaded data
        column if ANY of the passed inclusion_keys are present
        within that column name.

        If passed only with drop_keys will be proccessed second.

        (Note: if a name mapping exists, this drop step will be
        conducted after renaming)

        (default = None)

    subject_id :
    eventname :
    eventname_col :
    overlap_subjects :
    na_values :
    drop_na :
    drop_or_na :
    filter_outlier_percent : int, float, tuple or None, optional
        *For float data only.*
        A percent of values to exclude from either end of the
        targets distribution, provided as either 1 number,
        or a tuple (% from lower, % from higher).
        set `filter_outlier_percent` to None for no filtering.
        If over 1 then treated as a percent, if under 1, then
        used directly.

        If drop_or_na == 'drop', then all rows/subjects with >= 1
        value(s) found outside of the percent will be dropped.
        Otherwise, if drop_or_na = 'na', then any outside values
        will be set to NaN.

        (default = None)

    filter_outlier_std : int, float, tuple or None, optional
        *For float data only.*
        Determines outliers as data points within each column where their
        value is less than the mean of the column - `filter_outlier_std[0]`
        * the standard deviation of the column,
        and greater than the mean of the column + `filter_outlier_std[1]`
        * the standard deviation of the column.

        If a singler number is passed, that number is applied to both the lower
        and upper range. If a tuple with None on one side is passed, e.g.
        (None, 3), then nothing will be taken off that lower or upper bound.

        If drop_or_na == 'drop', then all rows/subjects with >= 1
        value(s) found will be dropped. Otherwise, if drop_or_na = 'na',
        then any outside values will be set to NaN.

        (default = None)

    unique_val_drop : int, float None, optional
        This parameter allows you to drops columns within loaded data
        where there are under a certain threshold of unique values.

        The threshold is determined by the passed value as first converted
        to a float between 0 and 1 (e.g. if passed 5, to .05), and then
        computed as unique_val_drop * len(data). Any column with less unique
        values then this threshold will be dropped

        (default = None)

    unique_val_warn : int or float, optional
        This parameter is simmilar to unique_val_drop, but only
        warns about columns with under the threshold (see unique_val_drop for
        how the threshold is computed) unique vals.

        (default = .05)

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
    """

    # Clear existing if requested, otherwise append to
    if clear_existing:
        self.Clear_Data()

    # Get the common load params as a mix of user-passed + default values
    load_params = self._make_load_params(args=locals())

    # Load in the raw dataframe - based on dataset type and/or passed user df
    data = self._load_datasets(loc, df, load_params)

    # Set to only overlap subjects if passed
    data = self._set_overlap(data, load_params['overlap_subjects'])

    # Drop cols by drop keys and inclusion_keys
    data = self._drop_data_cols(data, drop_keys, inclusion_keys)

    # Handle missing data
    data = self._drop_na(data, load_params['drop_na'])

    # Filter based on passed filter_outlier_percent
    data = self._filter_data_cols(data, filter_outlier_percent,
                                  filter_outlier_std,
                                  load_params['drop_or_na'])

    # Drop/warn about number of unique cols
    data = self._proc_data_unique_cols(data, unique_val_drop, unique_val_warn)

    # Drop column duplicates if param passed
    data = drop_duplicate_cols(data, drop_col_duplicates)

    # Show final na info after all proc
    self._show_na_info(data)

    # Display final shape
    self._print('loaded shape: ', data.shape)

    # Merge self.data with new loaded data
    self.data = self._merge_existing(self.data, data)

    # Process new loaded subjects
    self._process_new(self.low_memory_mode)


def Load_Targets(self, loc=None, df=None, col_name=None, data_type=None,
                 dataset_type='default', subject_id='default',
                 eventname='default', eventname_col='default',
                 overlap_subjects='default', filter_outlier_percent=None,
                 filter_outlier_std=None, categorical_drop_percent=None,
                 na_values='default', clear_existing=False):
    '''Loads in targets, the outcome / variable(s) to predict.

    Parameters
    ----------
    loc : str, Path or None, optional
        The location of the file to load targets load from.

        Either loc or df must be set, but they both cannot be set!

        (default = None)

    df : pandas DataFrame or None, optional
        This parameter represents the option for the user to pass in
        a raw custom dataframe. A loc and/or a df must be passed.

        When pasing a raw DataFrame, the loc and dataset_type
        param will be ignored, as those are for loading from a file.
        Otherwise, it will be treated the same as
        if loading from a file, which means, there should be a column within
        the passed dataframe with subject_id, and e.g. if eventname params are
        passed, they will be applied along with any other proc. specified.

        Either loc or df must be set, but they both cannot be set!

    col_name : str, list, optional
        The name(s) of the column(s) to load.

        Note: Must be in the same order as data types passed in.
        (default = None)

    data_type : {'b', 'c', 'm', 'o', 'f'}, optional
        The data types of the different columns to load,
        in the same order as the column names passed in.
        Shorthands for datatypes can be used as well.

        - 'binary' or 'b'
            Binary input

        - 'categorical' or 'c'
            Categorical input

        - 'multilabel' or 'm'
            Multilabel categorical input

        - 'float' or 'f'
            Float numerical input

        .. WARNING::
            If 'multilabel' datatype is specified, then the associated col name
            should be a list of columns, and will be assumed to be.
            For example, if loading multiple targets and one is multilabel,
            a nested list should be passed to col_name.

        Datatypes are explained further in Notes.

        (default = None)

    dataset_type :
    subject_id :
    eventname :
    eventname_col :
    overlap_subjects :

    filter_outlier_percent : float, tuple, list of or None, optional
        For float datatypes only.
        A percent of values to exclude from either end of the
        target distribution, provided as either 1 number,
        or a tuple (% from lower, % from higher).
        set `filter_outlier_percent` to None for no filtering.

        A list of values can also be passed in the case that
        multiple col_names / targets are being loaded. In this
        case, the index should correspond. If a list is not passed
        then the same value is used for all targets.

        (default = None).

    filter_outlier_std : int, float, tuple, None or list of, optional
        For float datatypes only.
        Determines outliers as data points within each column
        (target distribution) where their
        value is less than the mean of the column - `filter_outlier_std[0]`
        * the standard deviation of the column,
        and greater than the mean of the column + `filter_outlier_std[1]`
        * the standard deviation of the column.

        If a single number is passed, that number is applied to both the lower
        and upper range.  If a tuple with None on one side is passed, e.g.
        (None, 3), then nothing will be taken off that lower or upper bound.

        A list of values can also be passed in the case that
        multiple col_names / covars are being loaded. In this
        case, the index should correspond. If a list is not passed
        here, then the same value is used when loading all targets.

        (default = None)

    categorical_drop_percent: float, list of or None, optional
        Optional percentage threshold for dropping categories when
        loading categorical data. If a float is given, then a category
        will be dropped if it makes up less than that % of the data points.
        E.g. if .01 is passed, then any datapoints with a category with less
        then 1% of total valid datapoints is dropped.

        A list of values can also be passed in the case that
        multiple col_names / targets are being loaded. In this
        case, the index should correspond. If a list is not passed
        then the same value is used for all targets.

        (default = None)

    na_values :

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
    or 'float',

    - binary
        Targets are read in and label encoded to be 0 or 1,
        Will also work if passed column of unique string also,
        e.g. 'M' and 'F'.

    - categorical
        Targets are treated as taking on one fixed value from a
        limited set of possible values.

    - multilabel
        Simmilar to categorical, but targets can take on multiple
        values, and therefore cannot be reduced to an ordinal representation.

    - float
        Targets are read in as a floating point number,
        and optionally then filtered.
    '''

    if clear_existing:
        self.Clear_Targets()

    # Get the common load params as a mix of user-passed + default values
    load_params = self._make_load_params(args=locals())

    # For targets, set these regardless
    load_params['drop_na'] = True
    load_params['drop_or_na'] = 'drop'

    # Load in the targets w/ basic pre-processing
    targets, col_names = self._common_load(loc, df, dataset_type,
                                           load_params,
                                           col_names=col_name)

    # Proccess the passed in data_types - get right number and in list
    data_types, col_names = proc_datatypes(data_type, col_names)

    # Process pass in other args to be list of len(datatypes)
    fops = proc_args(filter_outlier_percent, data_types)
    foss = proc_args(filter_outlier_std, data_types)
    cdps = proc_args(categorical_drop_percent, data_types)

    # Get drop val, no option for keeping NaN for targets
    drop_val = get_unused_drop_val(targets)

    self._print()

    # Process each target to load
    for key, d_type, fop, fos, cdp in zip(col_names, data_types,
                                          fops, foss, cdps):
        targets =\
            self._proc_target(targets, key, d_type, fop, fos, cdp, drop_val)

    self._print()

    # Drop rows set to drop
    targets = self._drop_from_filter(targets, drop_val)

    self._print('Final shape: ', targets.shape)

    # Merge with existing and set self.targets
    self.targets = self._merge_existing(self.targets, targets)

    # Process new subjects
    self._process_new(self.low_memory_mode)

    # Print out info on all loaded targets, w/ index names / keys
    self._print('All loaded targets')
    for i in range(len(self.targets_keys)):
        self._print(i, ':', self.targets_keys[i])
    self._print()


def _proc_target(self, targets, key, d_type, fop, fos, cdp, drop_val):

    self._print('loading:', key)

    targets_key = key
    d_type = d_type[0]

    # Processing for binary, with some tolerance to funky input
    if d_type == 'b':

        targets, self.targets_encoders[key] =\
            process_binary_input(targets, targets_key, drop_val, self._print)

    # Proc. Categoirical as ordinal
    elif d_type == 'c':

        targets, self.targets_encoders[key] =\
            process_ordinal_input(targets, targets_key, cdp, drop_val,
                                  _print=self._print)

    # For float, just optionally apply outlier percent filter
    elif d_type == 'f':

        # Encoder set to None for targets
        self.targets_encoders[key] = None

        if fop is not None:
            targets = filter_float_by_outlier(targets, targets_key,
                                              fop, drop_val=drop_val,
                                              _print=self._print)

        if fos is not None:
            targets = filter_float_by_std(targets, targets_key,
                                          fos, drop_val=drop_val,
                                          _print=self._print)

    # Multilabel type must be read in from multiple columns
    elif d_type == 'multilabel' or d_type == 'm':
        common_name = process_multilabel_input(targets_key)

        self.targets_encoders[common_name] = targets_key
        self._print('Base str indicator/name for loaded multilabel =',
                    common_name)

    # Keep track of each loaded target in targets_keys
    if targets_key not in self.targets_keys:
        self.targets_keys.append(targets_key)

    return targets


def Load_Covars(self, loc=None, df=None, col_name=None, data_type=None,
                dataset_type='default', subject_id='default',
                eventname='default', eventname_col='default',
                overlap_subjects='default',
                na_values='default', drop_na='default', drop_or_na='default',
                code_categorical_as='dummy', categorical_drop_percent=None,
                filter_outlier_percent=None,
                filter_outlier_std=None,
                clear_existing=False):
    '''Load a covariate or covariates, type data.

    Parameters
    ----------
    loc : str, Path or None, optional
        The location of the file to load co-variates load from.

        Either loc or df must be set, but they both cannot be set!

        (default = None)

    df : pandas DataFrame or None, optional
        This parameter represents the option for the user to pass in
        a raw custom dataframe. A loc and/or a df must be passed.

        When pasing a raw DataFrame, the loc and dataset_type
        param will be ignored, as those are for loading from a file.
        Otherwise, it will be treated the same as
        if loading from a file, which means, there should be a column within
        the passed dataframe with subject_id, and e.g. if eventname params are
        passed, they will be applied along with any other proc. specified.

        Either loc or df must be set, but they both cannot be set!

    col_name : str or list, optional
        The name(s) of the column(s) to load.

        Note: Must be in the same order as data types passed in.

        (default = None)

    data_type : {'b', 'c', 'm', 'f'} or None, optional
        The data types of the different columns to load,
        in the same order as the column names passed in.
        Shorthands for datatypes can be used as well.

        - 'binary' or 'b'
            Binary input

        - 'categorical' or 'c'
            Categorical input

        - 'multilabel' or 'm'
            Multilabel categorical input

        - 'float' or 'f'
            Float numerical input


        .. WARNING::
            If 'multilabel' datatype is specified, then the associated col name
            should be a list of columns, and will be assumed to be.
            For example, if loading multiple targets and one is multilabel,
            a nested list should be passed to col_name.

        (default = None)

    dataset_type :
    subject_id :
    eventname :
    eventname_col :
    overlap_subjects :
    na_values :
    drop_na :
    drop_or_na :

    code_categorical_as: {'dummy', 'one hot', 'ordinal', list of}, optional
        How to code categorical data,

        - 'dummy' : perform dummy coding, to len(unique classes)-1 columns

        - 'one hot' : for one hot encoding, to len(unique classes) columns

        - 'ordinal' : one column values 0 to len(unique classes) -1

        A list of values can also be passed in the case that
        multiple col_names / covars are being loaded. In this
        case, the index should correspond. If a list is not passed
        here, then the same value is used when loading all covars.

        (default = 'dummy')

    categorical_drop_percent: float, None or list of, optional
        Optional percentage threshold for dropping categories when
        loading categorical data. If a float is given, then a category
        will be dropped if it makes up less than that % of the data points.
        E.g. if .01 is passed, then any datapoints with a category with less
        then 1% of total valid datapoints is dropped.

        A list of values can also be passed in the case that
        multiple col_names / covars are being loaded. In this
        case, the index should correspond. If a list is not passed
        here, then the same value is used when loading all covars.

        (default = None)

    filter_outlier_percent : int, float, tuple, None or list of, optional
        For float datatypes only.
        A percent of values to exclude from either end of the
        covars distribution, provided as either 1 number,
        or a tuple (% from lower, % from higher).
        set `filter_outlier_percent` to None for no filtering.

        A list of values can also be passed in the case that
        multiple col_names / covars are being loaded. In this
        case, the index should correspond. If a list is not passed
        here, then the same value is used when loading all covars.

        (default = None)

    filter_outlier_std : int, float, tuple, None or list of, optional
        For float datatypes only.
        Determines outliers as data points within each column where their
        value is less than the mean of the column - `filter_outlier_std[0]`
        * the standard deviation of the column,
        and greater than the mean of the column + `filter_outlier_std[1]`
        * the standard deviation of the column.

        If a single number is passed, that number is applied to both the lower
        and upper range. If a tuple with None on one side is passed, e.g.
        (None, 3), then nothing will be taken off that lower or upper bound.

        A list of values can also be passed in the case that
        multiple col_names / covars are being loaded. In this
        case, the index should correspond. If a list is not passed
        here, then the same value is used when loading all covars.

        (default = None)

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

    # Get the common load params as a mix of user-passed + default values
    load_params = self._make_load_params(args=locals())

    # Load in covars w/ basic pre-proc
    covars, col_names = self._common_load(loc, df, dataset_type,
                                          load_params,
                                          col_names=col_name)

    # Proccess the passed in data_types / get right number and in list
    data_types, col_names = proc_datatypes(data_type, col_names)

    # Process pass in other args to be list of len(datatypes)
    ccas = proc_args(code_categorical_as, data_types)
    cdps = proc_args(categorical_drop_percent, data_types)
    fops = proc_args(filter_outlier_percent, data_types)
    foss = proc_args(filter_outlier_std, data_types)

    # Set the drop_val
    if load_params['drop_or_na'] == 'na':
        drop_val = np.nan
    else:
        drop_val = get_unused_drop_val(covars)

    # Load in each covar
    for key, d_type, cca, cdp, fop, fos in zip(col_names, data_types,
                                               ccas, cdps, fops, foss):
        covars =\
            self._proc_covar(covars, key, d_type, cca, cdp, fop, fos, drop_val)

    # Have to remove rows with drop_val if drop_val not NaN
    if drop_val is not np.nan:
        covars = self._drop_from_filter(covars, drop_val)

    self._print('loaded shape: ', covars.shape)

    # If other data is already loaded,
    # merge this data with existing loaded data.
    self.covars = self._merge_existing(self.covars, covars)
    self._process_new(self.low_memory_mode)


def _proc_covar(self, covars, key, d_type, cca, cdp, fop, fos, drop_val):

    self._print('loading:', key)
    d_type = d_type[0]

    # Set to only the non-Nan subjects for this column
    non_nan_subjects = covars[~covars[key].isna()].index
    non_nan_covars = covars.loc[non_nan_subjects]

    # Binary
    if d_type == 'b':

        non_nan_covars, self.covars_encoders[key] =\
            process_binary_input(non_nan_covars, key, drop_val=drop_val,
                                 _print=self._print)

    # Categorical
    elif d_type == 'c':

        # Cat. ordinal
        if cca == 'ordinal':
            non_nan_covars, self.covars_encoders[key] =\
                process_ordinal_input(non_nan_covars, key, cdp,
                                      drop_val=drop_val, _print=self._print)

        # Dummy or one-hot
        else:
            non_nan_covars, self.covars_encoders[key] =\
                process_categorical_input(non_nan_covars, key, cca, cdp,
                                          drop_val=drop_val,
                                          _print=self._print)

            # Add any new cols from encoding to base covars (removing old)
            covars = covars.reindex(columns=list(non_nan_covars))

    # Float
    elif d_type == 'f':

        # Float type need an encoder set to None
        self.covars_encoders[key] = None

        # If filter float outlier percent
        if fop is not None:
            non_nan_covars = filter_float_by_outlier(non_nan_covars, key,
                                                     fop, drop_val=drop_val,
                                                     _print=self._print)

        # if filter float by std
        if fos is not None:
            non_nan_covars = filter_float_by_std(non_nan_covars, key, fos,
                                                 drop_val=drop_val,
                                                 _print=self._print)

    # Multilabel
    elif d_type == 'm':

        common_name = process_multilabel_input(key)

        self.covars_encoders[common_name] = key
        self._print('Base str indicator/name for loaded multilabel =',
                    common_name)

    # Now update the changed values within covars
    covars.loc[non_nan_subjects] = non_nan_covars

    # Update col datatype
    for dtype, key in zip(non_nan_covars.dtypes, list(covars)):
        covars[key] = covars[key].astype(dtype.name)

    return covars


def Load_Strat(self, loc=None, df=None, col_name=None, dataset_type='default',
               subject_id='default', eventname='default',
               eventname_col='default', overlap_subjects='default',
               binary_col=False, float_col=False,
               float_bins=10, float_bin_strategy='uniform',
               categorical_drop_percent=None,
               na_values='default', clear_existing=False):
    '''Load stratification values from a file.
    See Notes for more details on what stratification values are.

    Parameters
    ----------
    loc : str, Path or None, optional
        The location of the file to load stratification vals load from.

        Either loc or df must be set, but they both cannot be set!

        (default = None)

    df : pandas DataFrame or None, optional
        This parameter represents the option for the user to pass in
        a raw custom dataframe. A loc and/or a df must be passed.

        When pasing a raw DataFrame, the loc and dataset_type
        param will be ignored, as those are for loading from a file.
        Otherwise, it will be treated the same as
        if loading from a file, which means, there should be a column within
        the passed dataframe with subject_id, and e.g. if eventname params are
        passed, they will be applied along with any other proc. specified.

        Either loc or df must be set, but they both cannot be set!

    col_name : str or list, optional
        The name(s) of the column(s) to load. Any datatype can be
        loaded with the exception of multilabel, but for float variables
        in particular, they should be specified with the `float_col` and
        corresponding `float_bins` and `float_bin_strategy` params. Noisy
        binary cols can also be specified with the `binary_col` param.

        (default = None)

    dataset_type :
    subject_id :
    eventname :
    eventname_col :
    overlap_subjects :

    binary_col : bool or list of, optional
        Strat values are loaded as ordinal categorical, but there still
        exists the case where the user would like to load a binary set of
        values, and would like to ensure they are binary (filtering out
        all values but the top 2 most frequent).

        This input should either be one boolean True False value,
        or a list of values corresponding the the length of col_name if
        col_name is a list.

        If col_name is a list and only one value for binary_col is
        passed, then that value is applied to all loaded cols.

        (default = False)

    float_col : int, list or None, optional
        Strat values are loaded as ordinal categorical, but one
        could also want to load a float value, and bin it into according
        to some strategy into ordinal categorical.

        This input should either be one boolean True False value,
        or a list of values corresponding the the length of col_name if
        col_name is a list.

        If col_name is a list and only one value for binary_col is
        passed, then that value is applied to all loaded cols.

        (default = None)

    float_bins : int or list of, optional
        If any float_col are set to True, then the float
        input must be discretized into bins. This param controls
        the number of bins to create. As with float_col, if one value
        is passed, it is applied to all columns, but if different values
        per column loaded are desired, a list of ints (with inds correponding)
        should be pased.

        (default = 10)

    float_bin_strategy : {'uniform', 'quantile', 'kmeans'}, optional
        If any float_col are set to True, then the float
        input must be discretized into bins. This param controls
        the strategy used to define the bins. Options are,

        - 'uniform'
            All bins in each feature have identical widths.

        - 'quantile'
            All bins in each feature have the same number of points.

        - 'kmeans'
            Values in each bin have the same nearest center of a 1D
            k-means cluster.

        As with float_col and float_bins, if one value
        is passed, it is applied to all columns, but if different values
        per column loaded are desired, a list of choices
        (with inds correponding) should be pased.

        (default = 'uniform')

    categorical_drop_percent: float, None or list of, optional
        Optional percentage threshold for dropping categories when
        loading categorical data (so for strat these are any column that are
        not specified as float or binary). If a float is given, then a category
        will be dropped if it makes up less than that % of the data points.
        E.g. if .01 is passed, then any datapoints with a category with less
        then 1% of total valid datapoints is dropped.

        A list of values can also be passed in the case that
        multiple col_names / strat vals are being loaded. In this
        case, the indices should correspond. If a list is not passed
        here, then the same value is used when loading all non float non binary
        strat cols.

        (default = None)

    na_values :

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
    See :func:`Define_Validation_Strategy`, and some arguments within
    :func:`Evaluate` (sample_on and subjects_to_use).

    For most relaible split behavior based off strat values, make sure to load
    strat values after data, targets and covars.
    '''

    if clear_existing:
        self.Clear_Strat()

    # Get the common load params as a mix of user-passed + default values
    load_params = self._make_load_params(args=locals())

    # For strat, set these regardless
    load_params['drop_na'] = True
    load_params['drop_or_na'] = 'drop'

    # Load in strat w/ basic pre-processing
    strat, col_names = self._common_load(loc, df, dataset_type,
                                         load_params,
                                         col_names=col_name)

    # Add strat unique name to end of each col name
    col_mapping = {col: col + self.strat_u_name for col in strat}
    strat = strat.rename(col_mapping, axis=1)

    # Proc list optional args to right length
    bcs = proc_args(binary_col, col_names)
    fcs = proc_args(float_col, col_names)
    fbs = proc_args(float_bins, col_names)
    fbss = proc_args(float_bin_strategy, col_names)
    cdps = proc_args(categorical_drop_percent, col_names)

    # Get drop val, no option for keeping NaN for strat
    drop_val = get_unused_drop_val(strat)

    # Load in each strat w/ passed args
    for key, bc, fc, fb, fbs, cdp in zip(col_names, bcs, fcs, fbs, fbss, cdps):
        strat = self._proc_strat(strat, key, bc, fc, fb, fbs, cdp, drop_val)

    # Drop rows set to drop
    strat = self._drop_from_filter(strat, drop_val)

    # Merge with existing if any, and process new overlap of global subjects
    self.strat = self._merge_existing(self.strat, strat)
    self._process_new(self.low_memory_mode)


def _proc_strat(self, strat, key, bc, fc, fb, fbs, cdp, drop_val):

    key = key + self.strat_u_name

    # Binary
    if bc:
        strat, self.strat_encoders[key] =\
            process_binary_input(strat, key, drop_val, self._print)

    # Float
    elif fc:
        strat, self.strat_encoders[key] =\
            process_float_input(strat, key, bins=fb, strategy=fbs)

    # Categorical
    else:
        strat, self.strat_encoders[key] =\
            process_ordinal_input(strat, key, cdp, drop_val, self._print)

    return strat


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
    For best/most reliable performance across all Loading cases,
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
    For best/most reliable performance across all Loading cases,
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


def Drop_Data_Cols(self, drop_keys=None, inclusion_keys=None):
    '''Function to drop columns within loaded data by drop_keys
    or inclusion_keys.

    Parameters
    ----------
    drop_keys : str, list or None, optional
        A list of keys to drop columns within loaded data by,
        where if ANY key given in a columns
        name, then that column will be dropped.
        If a str, then same behavior, just with one col.

        If passed along with inclusion_keys will be processed first.

        (Note: if a name mapping exists, this drop step will be
        conducted after renaming)

        (default = None)

    inclusion_keys : str, list or None, optional
        A list of keys in which to only keep a loaded data
        column if ANY of the passed inclusion_keys are present
        within that column name.

        If passed only with drop_keys will be proccessed second.

        (Note: if a name mapping exists, this drop step will be
        conducted after renaming)

        (default = None)
    '''

    self.data = self._drop_data_cols(self.data, drop_keys, inclusion_keys)


def _drop_data_cols(self, data, drop_keys, inclusion_keys):

    if drop_keys is not None:

        column_names = list(data)
        if isinstance(drop_keys, str):
            drop_keys = [drop_keys]

        to_drop = [name for name in column_names for drop_key in drop_keys
                   if drop_key in name]

        data = data.drop(to_drop, axis=1)
        self._print('Dropped', len(to_drop), 'columns',
                    'per passed drop_keys argument')

    if inclusion_keys is not None:

        column_names = list(data)
        if isinstance(inclusion_keys, str):
            inclusion_keys = [inclusion_keys]

        to_keep = [name for name in column_names for
                   inclusion_key in inclusion_keys
                   if inclusion_key in name]

        data = data[to_keep]
        self._print('Keeping', len(to_keep), 'columns',
                    'per passed inclusion_keys argument')

    return data


def Filter_Data_Cols(self, filter_outlier_percent=None,
                     filter_outlier_std=None, overlap_subjects='default',
                     drop_or_na='default'):
    '''Perform filtering on all loaded data based on an outlier percent,
    either dropping outlier rows or setting specific outliers to NaN.

    Note, if overlap_subject is set to True here, only the overlap will
    be saved after proc within self.data.

    Parameters
    ----------
    filter_outlier_percent : int, float, tuple or None
        *For float data only.*
        A percent of values to exclude from either end of the
        targets distribution, provided as either 1 number,
        or a tuple (% from lower, % from higher).
        set `filter_outlier_percent` to None for no filtering.
        If over 1 then treated as a percent, if under 1, then
        used directly.

        If drop_or_na == 'drop', then all rows/subjects with >= 1
        value(s) found outside of the percent will be dropped.
        Otherwise, if drop_or_na = 'na', then any outside values
        will be set to NaN.

        (default = None)

    filter_outlier_std : int, float, tuple or None, optional
        *For float data only.*
        Determines outliers as data points within each column where their
        value is less than the mean of the column - `filter_outlier_std[0]`
        * the standard deviation of the column,
        and greater than the mean of the column + `filter_outlier_std[1]`
        * the standard deviation of the column.

        If a single number is passed, that number is applied to both the lower
        and upper range.  If a tuple with None on one side is passed, e.g.
        (None, 3), then nothing will be taken off that lower or upper bound.

        (default = None)

    overlap_subjects :
    drop_or_na :

    '''

    load_params = self._make_load_params(args=locals())
    data = self._set_overlap(self.data, load_params['overlap_subjects'])

    self.data = self._filter_data_cols(self.data, filter_outlier_percent,
                                       filter_outlier_std,
                                       load_params['drop_or_na'])


def _filter_data_cols(self, data, filter_outlier_percent, filter_outlier_std,
                      drop_or_na):

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
        data = self._drop_from_filter(data, drop_val)

    return data


def Proc_Data_Unique_Cols(self, unique_val_drop=None, unique_val_warn=.05,
                          overlap_subjects='default'):
    ''' This function performs proccessing on all loaded data based on
    the number of unique values loaded within each column
    (allowing users to drop or warn!).

    Note, if overlap_subjects is set to True here, only the overlap will
    be saved after proc within self.data.

    Parameters
    ----------
    unique_val_drop : int, float None, optional
        This parameter allows you to drops columns within loaded data
        where there are under a certain threshold of unique values.

        The threshold is determined by the passed value as first converted
        to a float between 0 and 1 (e.g. if passed 5, to .05), and then
        computed as unique_val_drop * len(data). Any column with less unique
        values then this threshold will be dropped

        (default = None)

    unique_val_warn : int or float, optional
        This parameter is simmilar to unique_val_drop, but only
        warns about columns with under the threshold (see unique_val_drop for
        how the threshold is computed) unique vals.

        (default = .05)

    overlap_subjects :

    '''

    load_params = self._make_load_params(args=locals())
    data = self._set_overlap(self.data, load_params['overlap_subjects'])

    self.data =\
        self._proc_data_unique_cols(data, unique_val_drop, unique_val_warn)


def _proc_data_unique_cols(self, data, unique_val_drop, unique_val_warn):

    if unique_val_drop is None:
        unique_val_drop = 0
    if unique_val_warn is None:
        unique_val_warn = 0

    if unique_val_drop > 1:
        unique_val_drop /= 100
    if unique_val_warn > 1:
        unique_val_warn /= 100

    drop_thresh = unique_val_drop * len(data)
    warn_thresh = unique_val_warn * len(data)

    unique_counts = [np.sum(~np.isnan(np.unique(data[x]))) for x in data]
    unique_counts = np.array(unique_counts)

    self._print()

    # If any valid for warn or drop
    if ((unique_counts < drop_thresh).any()) or (
     (unique_counts < warn_thresh).any()):

        self._print('Processing unique col values with drop threshold:',
                    drop_thresh, '- warn threshold:', warn_thresh, '- out of',
                    len(data), 'rows')

        for col, count in zip(list(data), unique_counts):

            if count < drop_thresh:
                data = data.drop(col, axis=1)
                self._print('Dropped -', col, 'with unique vals:', count)

            elif count < warn_thresh:
                self._print('Warn -', col, 'has unique vals:', count)

        self._print()

    return data


def Drop_Data_Duplicates(self, corr_thresh, overlap_subjects='default'):
    '''Drop duplicates columns within self.data based on
    if two data columns are >= to a certain correlation threshold.

    Note, if overlap_subjects is set to True here, only the overlap will
    be saved after proc within self.data.

    Parameters
    ----------
    corr_thresh : float
        A value between 0 and 1, where if two columns within self.data
        are correlated >= to `corr_thresh`, the second column is removed.

        A value of 1 will instead make a quicker direct =='s comparison.

    overlap_subjects
    '''

    load_params = self._make_load_params(args=locals())
    data = self._set_overlap(self.data, load_params['overlap_subjects'])

    self.data = drop_duplicate_cols(data, corr_thresh)


def Binarize_Target(self, lower, upper, target=0):
    '''This function binarizes a loaded target variable,
    assuming that a float type target is loaded,
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


def Clear_Name_Map(self):
    '''Reset name mapping'''
    self.name_map = {}
    self._print('cleared name map.')
    self._print()


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
    self._print()


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
    self._print()


def Clear_Targets(self):
    '''Resets targets'''
    self.targets = pd.DataFrame()
    self.targets_encoders = {}
    self.targets_keys = []
    self._print('cleared targets.')
    self._print()


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
    self._print()


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
    self._print()


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
    self._print()


def Get_Nan_Subjects(self):
    '''Retrieves all subjects with any loaded NaN data, returns
    their pandas index.'''

    if self.all_data is None:
        self._prepare_data()

    return self.all_data[pd.isnull(self.all_data).any(axis=1)].index


def _get_targets_key(self, key, base_key=False):

    targets_base_keys = self._get_base_targets_names()

    if isinstance(key, int):
        ind = key

    else:
        if key in self.name_map:
            key = self.name_map[key]

        ind = targets_base_keys.index(key)

    if base_key:
        return targets_base_keys[ind]
    return self.targets_keys[ind]


def _load_datasets(self, locs, df, load_params):
    '''Helper function to load in multiple datasets with default
    load and drop behavior based on type. And calls proc_df on each
    before merging.

    Parameters
    ----------
    locs : list of str, Path
        The location of the  files to load data load from.

    df : pandas df
        User passed

    load_params : dict
        load params

    Returns
    ----------
    pandas DataFrame
        ABCD ML formatted pd DataFrame, with the loaded
        and merged minimally proc'ed data.
    '''

    dfs = []

    if not isinstance(locs, list) and locs is not None:
        locs = [locs]

    # Load from file
    if locs is not None:

        # If only one dataset type, use it for all
        if not isinstance(load_params['dataset_type'], list):
            dataset_types = [load_params['dataset_type']
                             for i in range(len(locs))]
        else:
            dataset_types = load_params['dataset_type']

        dfs = [self._load_dataset(locs[i], dataset_types[i],
               load_params) for i in range(len(locs))]

    # Load from user-passed df
    if df is not None:

        df = self._load_user_passed(df, load_params['na_values'])
        df = self._proc_df(df, load_params)
        dfs.append(df)

    # Set first df
    data = dfs[0]

    # For each additional
    for more_data in dfs[1:]:
        repeat_col_names = set(list(data)).intersection(set(list(more_data)))

        if len(repeat_col_names) > 0:
            self._print('Warning,', repeat_col_names,
                        'exist in both dataframes!')
            self._print('By default repeats will be added as new unique',
                        'columns within merged data.')

        data = pd.merge(data, more_data, on=self.subject_id)

    return data


def _load_user_passed(self, df, na_values):

    self._print('Loading user passed df')

    if len(df.index.name) > 0:
        df = df.reset_index()

    df = df.replace(na_values, np.nan)
    return df


def _load_dataset(self, loc, dataset_type, load_params):
    '''Helper function to load in a dataset with default
    load and drop behavior based on type. And calls proc_df.

    Parameters
    ----------
    loc : str, Path or None
        The location of the csv/txt file to load data load from.

    dataset_type : {'default', 'basic', 'explorer', 'custom'}
        The type of dataset to load from.

    Returns
    ----------
    pandas DataFrame
        ABCD ML formatted pd DataFrame, with the loaded
        minimally proc'ed data.
    '''

    data = self._load(loc, dataset_type, load_params['na_values'])

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

    data = self._proc_df(data, load_params)
    self._print()

    return data


def _common_load(self, loc, df, dataset_type, load_params,
                 col_names=None):

    if loc is not None and df is not None:
        raise AssertionError('Both loc and df cannot be set!')
    if loc is None and df is None:
        raise AssertionError('Either loc or df must be passed!')

    # Reads raw data based on dataset type
    if loc is not None:
        data = self._load(loc, load_params['dataset_type'],
                          load_params['na_values'])

    # User passed
    if df is not None:
        data = self._load_user_passed(df, load_params['na_values'])

    # Perform proc common operations
    data = self._proc_df(data, load_params)

    # Set to only overlap subjects if passed
    data = self._set_overlap(data, load_params['overlap_subjects'])

    # Proc input col_names
    if not isinstance(col_names, list):
        col_names = list([col_names])
    for i in range(len(col_names)):
        if col_names[i] in self.name_map:
            col_names[i] = self.name_map[col_names[i]]

    # Set data to only the requested cols and drop_na
    data = self._drop_na(data[col_names], load_params['drop_na'])

    return data, col_names


def _load(self, loc, dataset_type, na_values):
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

    na_values

    Returns
    ----------
    pandas DataFrame
        Loaded DataFrame.
    '''

    self._print('Loading', loc, ' with dataset type:', dataset_type)

    if dataset_type == 'basic':
        data = pd.read_csv(loc, sep='\t', skiprows=[1],
                           na_values=na_values,
                           low_memory=self.low_memory_mode)
    else:
        data = pd.read_csv(loc, na_values=na_values,
                           low_memory=self.low_memory_mode)

    return data


def _set_overlap(self, data, overlap_subjects):

    if overlap_subjects:
        data = data[data.index.isin(self._get_overlapping_subjects())]
        self._print('Set to overlapping loaded subjects.')

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


def _proc_df(self, data, load_params):
    '''Internal helper function to perform common processing steps.

    Parameters
    ----------
    data : pandas DataFrame
        a semi-raw df

    load_params : dict
        dict containing load param values

    Returns
    ----------
    pandas DataFrame
        The df post-processing
    '''

    # Rename columns with loaded name map
    data = data.rename(self.name_map, axis=1)

    # If passed subject id is different then int name, change it
    if load_params['subject_id'] in self.name_map:
        load_params['subject_id'] = self.name_map[load_params['subject_id']]

    if load_params['subject_id'] != self.subject_id:
        data = data.rename({load_params['subject_id']: self.subject_id},
                           axis=1)

    # Make sure a column w/ subject id exists
    if self.subject_id not in data:
        raise IndexError('No valid subject id column found!')

    # If using default subject ids, proc for
    data[self.subject_id] =\
        data[self.subject_id].apply(self._process_subject_name)

    # Filter by eventname is applicable.
    if load_params['eventname_col'] in self.name_map:
        load_params['eventname_col'] =\
            self.name_map[load_params['eventname_col']]

    data = self._filter_by_eventname(data, load_params['eventname'],
                                     load_params['eventname_col'])

    # Drop any duplicate subjects, longitudinal data should just change col
    # name.
    before = data.shape[0]
    data = data.drop_duplicates(subset=self.subject_id)
    after = data.shape[0]

    if before != after:
        self._print('Note: ABCD_ML does not currently support',
                    'duplicate subjects loaded as seperate rows!',
                    before - after, 'subjects have been dropped',
                    'accordingly.')

    # Set the subject_id column as the index column
    data = data.set_index(self.subject_id)

    # Drop excluded and/or included subjects, if any
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


def _drop_na(self, data, drop_na=True):
    '''Wrapper function to drop rows with NaN values.

    Parameters
    ----------
    data : pandas DataFrame
        ABCD_ML formatted.

    drop_na : drop_na param
        ABCD_ML drop_na param

    Returns
    ----------
    pandas DataFrame
        Input df, with dropped rows for NaN values
    '''

    # First drop any columns with all NaN
    missing_values = data.isna().all(axis=0)
    data = data.dropna(axis=1, how='all')
    self._print('Dropped', sum(missing_values), 'cols for all missing values')

    if drop_na is not False:

        if drop_na is True:
            na_thresh = 0
        elif drop_na <= 1:
            na_thresh = int(drop_na * data.shape[1])
        else:
            na_thresh = drop_na

        to_drop = data[data.isna().sum(axis=1) > na_thresh].index
        data = data.drop(to_drop)
        self._print('Dropped', len(to_drop), 'rows for missing values, based',
                    'on the provided drop_na param:', drop_na,
                    'with actual na_thresh:', na_thresh)

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


def _filter_by_eventname(self, data, eventname, eventname_col):
    '''Internal helper function, to filter a dataframe by eventname,
    and then return the dataframe.

    Parameters
    ----------
    data : pandas DataFrame
        ABCD_ML formatted.

    eventname : value or list of values
        The value that if eventname_col is equal to,
        if a list, then treates as row is kept if equal to any in the list

    eventname_col : str
        The name of the eventname column

    Returns
    ----------
    pandas DataFrame
        Input df, with only valid eventname rows
    '''

    if eventname is not None:
        if eventname_col in list(data):

            if not isinstance(eventname, list):
                eventname = [eventname]

            before = data.shape[0]
            data = data[data[eventname_col].isin(eventname)]
            after = data.shape[0]

            if before != after:
                self._print(before - after, 'subjects have been dropped',
                            'based on passed eventname params.')

        else:
            self._print('Warning: filter by eventname_col:',
                        eventname_col, 'specfied, with value:',
                        eventname, 'but the column was not found!')

    # Remove eventname column
    if eventname_col in list(data):
        data = data.drop('eventname', axis=1)

    return data


def _show_na_info(self, data):

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

        if remove:
            self._print('Removing non overlapping subjects from loaded data,',
                        'covars, ect...')

            if len(self.data) > 0:
                self.data = self.data[self.data.index.isin(overlap)]
            if len(self.covars) > 0:
                self.covars = self.covars[self.covars.index.isin(overlap)]
            if len(self.targets) > 0:
                self.targets = self.targets[self.targets.index.isin(overlap)]
            if len(self.strat) > 0:
                self.strat = self.strat[self.strat.index.isin(overlap)]

        self._print()


def _prepare_data(self):
    '''Helper function to prepare all loaded data,
    from different sources, self.data, self.covars, ect...
    into self.all_data for use directly in ML.
    '''

    self._print('Preparing final data, in self.all_data')
    self._print('Any changes to loaded data, covars or strat will not be',
                'included, from now on.')
    self._print()

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
