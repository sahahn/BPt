"""
_Data.py
====================================
Main class extension file for the Loading functionality methods of
the ML class.
"""
import pandas as pd
import numpy as np
import os
from joblib import wrap_non_picklable_objects

from ..helpers.VARS import is_f2b
from ..helpers.Data_Scopes import Data_Scopes
from ..helpers.Data_File import Data_File
from ..helpers.Data_Helpers import (auto_data_type,
                                    process_binary_input,
                                    process_ordinal_input,
                                    process_float_input,
                                    process_multilabel_input,
                                    get_unused_drop_val,
                                    filter_float_by_outlier,
                                    filter_float_by_std,
                                    drop_duplicate_cols,
                                    get_top_substrs,
                                    proc_datatypes,
                                    proc_args,
                                    get_common_name,
                                    filter_data_cols,
                                    filter_data_file_cols,
                                    drop_from_filter,
                                    proc_file_input,
                                    fill_df_with)


def Set_Default_Load_Params(self, dataset_type='default', subject_id='default',
                            eventname='default', eventname_col='default',
                            overlap_subjects='default', merge='default',
                            na_values='default',
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

        ::

            default = 'default'

    subject_id : str, optional
        The name of the column with unique subject ids in different
        dataset, for default ABCD datasets this is 'src_subject_id',
        but if a user wanted to load and work with a different dataset,
        they just need to change this accordingly
        (in addition to setting eventname most likely to None and
        use_abcd_subject_ids to False)

        if 'default', and not already defined, set to 'src_subject_id'.

        ::

            default = 'default'

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

    merge : {'inner' or 'outer'}
        Simmilar to overlap subjects, this parameter controls the merge
        behavior between different df's. i.e., when calling Load_Data twice,
        a local dataframe is merged with the class self.data on the second
        call.
        There are two behaviors that make sense here, one is 'inner'
        which says, only take the overlapping subjects from each dataframe,
        and the other is 'outer' which will keep all subjects from both,
        and set any missing subjects values to NaN.

        if 'default', and not already defined, set to 'inner'
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

    if subject_id != 'default':
        self.default_load_params['subject_id'] = subject_id
    elif 'subject_id' not in self.default_load_params:
        self.default_load_params['subject_id'] = 'src_subject_id'

    if eventname != 'default':
        self.default_load_params['eventname'] = eventname
    elif 'eventname' not in self.default_load_params:
        self.default_load_params['eventname'] = None

    if eventname_col != 'default':
        self.default_load_params['eventname_col'] = eventname_col
    elif 'eventname_col' not in self.default_load_params:
        self.default_load_params['eventname_col'] = 'eventname'

    if overlap_subjects != 'default':
        self.default_load_params['overlap_subjects'] = overlap_subjects
    elif 'overlap_subjects' not in self.default_load_params:
        self.default_load_params['overlap_subjects'] = False

    if merge != 'default':
        self.default_load_params['merge'] = merge
    elif 'merge' not in self.default_load_params:
        self.default_load_params['merge'] = 'inner'

    if na_values != 'default':
        self.default_load_params['na_values'] = na_values
    elif 'na_values' not in self.default_load_params:
        self.default_load_params['na_values'] = ['777', '999']

    if drop_na != 'default':
        self.default_load_params['drop_na'] = drop_na
    elif 'drop_na' not in self.default_load_params:
        self.default_load_params['drop_na'] = True

    if drop_or_na != 'default':
        self.default_load_params['drop_or_na'] = drop_or_na
    elif 'drop_or_na' not in self.default_load_params:
        self.default_load_params['drop_or_na'] = 'drop'

    self._print('Default load params set within self.default_load_params.')
    self._print('----------------------')
    for param in self.default_load_params:
        self._print(param + ':', self.default_load_params[param])

    self._print()


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


def _get_data_file_cnt(self):

    if len(self.file_mapping) > 0:
        return max(self.file_mapping.keys())
    else:
        return 0


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
              merge='default', na_values='default', drop_na='default',
              drop_or_na='default',
              filter_outlier_percent=None, filter_outlier_std=None,
              unique_val_drop=None, unique_val_warn=.05,
              drop_col_duplicates=None,
              clear_existing=False, ext=None):
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

        (default = None)

    df : pandas DataFrame or None, optional
        This parameter represents the option for the user to pass in
        a raw custom dataframe. A loc and/or a df must be passed.

        When pasing a raw DataFrame, the loc and dataset_type
        param will be ignored, as those are for loading data from a file.
        Otherwise, it will be treated the same as
        if loading from a file, which means, there should be a column within
        the passed dataframe with subject_id, and e.g. if eventname params are
        passed, they will be applied along with any other proc. specified.

        (default = None)

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
    merge :
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

        The threshold is determined by the passed value as either a float for
        a percentage of the data,
        e.g., computed as unique_val_drop * len(data),
        or if passed a number greater then 1, then that number, where a
        ny column with less unique values then this threshold will be dropped.

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

    ext : None or str, optional
        Optional fixed extension to append to all loaded col names,
        leave as None to ignore this param. Note: applied after
        name mapping.

        (default = None)
    """

    # Clear existing if requested, otherwise append to
    if clear_existing:
        self.Clear_Data()

    # Get the common load params as a mix of user-passed + default values
    load_params = self._make_load_params(args=locals())

    # Load in the raw dataframe - based on dataset type and/or passed user df
    data = self._load_datasets(loc, df, load_params, ext=ext)
    self._print()

    # Set to only overlap subjects if passed
    data = self._set_overlap(data, load_params['overlap_subjects'])

    # Drop cols by drop keys and inclusion_keys
    data = self._drop_data_cols(data, drop_keys, inclusion_keys)

    # Handle missing data
    data = self._drop_na(data, load_params['drop_na'])

    # Filter based on passed filter_outlier_percent
    data = filter_data_cols(data, filter_outlier_percent,
                            filter_outlier_std,
                            load_params['drop_or_na'],
                            seperate_keys=self.data_file_keys,
                            subject_id=self.subject_id,
                            _print=self._print)

    # Drop/warn about number of unique cols
    data = self._proc_data_unique_cols(data, unique_val_drop, unique_val_warn)

    # Drop column duplicates if param passed
    data = drop_duplicate_cols(data, drop_col_duplicates)

    # Show final na info after all proc
    self._show_na_info(data)

    # Display final shape
    self._print('Loaded Shape:', data.shape)

    # Merge self.data with new loaded data
    self.data = self._merge_existing(self.data, data, load_params['merge'])


def Load_Data_Files(self, loc=None, df=None, files=None,
                    file_to_subject=None, load_func=np.load,
                    dataset_type='default', drop_keys=None,
                    inclusion_keys=None,
                    subject_id='default', eventname='default',
                    eventname_col='default', overlap_subjects='default',
                    merge='default',
                    reduce_func=np.mean, filter_outlier_percent=None,
                    filter_outlier_std=None, clear_existing=False, ext=None):
    """Class method for loading in data as file paths, where file paths correspond
    to some sort of raw data which should only be actually loaded / proc'ed
    within the actual modelling. The further assumption made is
    that these files represent 'Data' in the same sense that :func:`Load_Data`
    represents data, where once loaded / proc'ed
    (See :ref:`Loaders`), the outputted features should be
    continuous / float datatype.

    Parameters
    ----------
    loc :
    df :

    files : dict, optional
        Another alternative for specifying files to load can be done
        by passing a dict to this param.

        Warning: This option right now only works if all files
        to load are the same across each subject, e.g., no missing
        data for one modality. This will hopefully be fixed in the future,
        or atleast provide a better warning!

        Specifically, a python dictionary should be passed where
        each key refers to the name of that feature / column of data files
        to load, and the value is a python list, or array-like of
        str file paths.

        You must also pass a python function to the
        file_to_subject param, which specifies how to convert from passed
        file path, to a subject name.

        E.g., consider the example below, where 2 subjects files are
        loaded for 'feat1' and feat2':

        ::

            files = {'feat1': ['f1/subj_0.npy', 'f1/subj_1.npy'],
                     'feat2': ['f2/subj_0.npy', 'f2/subj_1.npy']}

            def file_to_subject_func(file):
                subject = file.split('/')[1].replace('.npy', '')
                return subject

            file_to_subject = file_to_subject_func
            # or
            file_to_subject = {'feat1': file_to_subject_func,
                               'feat2': file_to_subject_func}

        In this example, subjects are loaded as 'subj_0' and 'subj_1',
        and they have associated loaded data files 'feat1' and 'feat2'.

        ::

            default = None

    file_to_subject : python function, or dict of optional
        If files is passed, then you also need to specify a function
        which takes in a file path, and returns the relevant subject for
        that file path. If just one function is passed, it will be used
        for to load all dictionary entries, alternatively you can pass
        a matching dictionary of funcs, allowing for different funcs
        for each feature to load.

        See the example in param `files`.

        ::

            default = None

    load_func : python function, optional
        Data_Files represent a path to a saved file, which means you must
        also provide some information on how to load the saved file.
        This parameter is where that loading function should be passed.
        The passed `load_func` will be used on each Data_File individually
        and whatever the output of the function is will be passed to
        `loaders` directly in modelling.

        You might need to pass a user defined custom function in some cases,
        e.g., you want to use np.load, but then also np.stack. Just wrap those
        two functions in one, and pass the new function.

        (default = np.load)

    dataset_type :
    drop_keys :
    inclusion_keys :
    subject_id :
    eventname :
    eventname_col :
    overlap_subjects :
    merge :

    reduce_func : python function or list of, optional
        This function is used if either filter_outlier_percent or
        filter_outlier_std is requested.

        The passed python function should reduce the file, once loaded,
        to one number, making it comptabile with the different
        filtering strategies.
        For example, the default function is just to take the
        mean of each loaded file, and to compute outlier detection
        on the mean.

        You may also pass a list to reduce func, where each
        entry of the list is a single reduce func. In this case
        outlier filtering will be computed on each reduce_fun seperately,
        and the union of all subjects marked as outlier will be dropped
        at the end.

        ::

            default = np.mean

    filter_outlier_percent : float, tuple or None
        A percent of values to exclude from either end of the
        data files distribution, provided as either 1 number,
        or a tuple (% from lower, % from higher).
        set `filter_outlier_percent` to None for no filtering.

        For example, if passed (1, 1), then the bottom 1% and top 1%
        of the distribution will be dropped, the same as passing 1.
        Further, if passed (.1, 1), the bottom .1% and top 1% will be
        removed.

        ::

            default = None

    filter_outlier_std : int, float, tuple or None, optional
        Determines outliers as data points within each column where their
        value is less than the mean of the column
        (where column values are determine by the reduce func)
        - `filter_outlier_std[0]` * the standard deviation of the column,
        and greater than the mean of the column + `filter_outlier_std[1]`
        * the standard deviation of the column.

        If a single number is passed, that number is applied to both the lower
        and upper range.  If a tuple with None on one side is passed, e.g.
        (None, 3), then nothing will be taken off that lower or upper bound.

        ::

            default = None

    clear_existing : bool, optional
        If this parameter is set to True, then any existing
        loaded data will first be cleared before loading new data!

        Note: Data_Files and Data are both loaded as data, and will
        both therefore be cleared if this argument is set to True!

        .. WARNING::
            If any subjects have been dropped from a different place,
            e.g. targets, then simply reloading / clearing existing data
            might result in computing a misleading overlap of final
            valid subjects. Reloading should therefore be best used
            right after loading the original data, or if not possible,
            then reloading the notebook or re-running the script.

        (default = False)

    ext : None or str, optional
        Optional fixed extension to append to all loaded col names,
        leave as None to ignore this param. Note: applied after
        name mapping.

        (default = None)
    """

    # Clear existing if requested, otherwise append to
    if clear_existing:
        self.Clear_Data()

    # Get the common load params as a mix of user-passed + default values
    load_params = self._make_load_params(args=locals())

    # If files passed, add files as df to df
    df = proc_file_input(files, file_to_subject, df,
                         load_params['subject_id'])

    # Load in the raw dataframe - based on dataset type and/or passed user df
    data = self._load_datasets(loc, df, load_params, ext=ext)
    self._print()

    # Set to only overlap subjects if passed
    data = self._set_overlap(data, load_params['overlap_subjects'])

    # Drop or include
    data = self._drop_data_cols(data, drop_keys, inclusion_keys)

    # Convert loaded paths into ints within data, as
    # mapped to Data_Files in the file_mapping dicts
    file_mapping = {}
    data_file_keys = list(data)
    cnt = self._get_data_file_cnt()

    # Wrap load_func here if needed.
    if load_func.__module__ == '__main__':
        wrapped_load_func = wrap_non_picklable_objects(load_func)
        self._print('Warning: Passed load_func was defined within the',
                    '__main__ namespace and therefore has been cloud wrapped.',
                    'The function will still work, but it is reccomended to',
                    'define this function in a seperate file, and then import',
                    'it , otherwise loader caching will be limited',
                    'in utility!')
    else:
        wrapped_load_func = load_func

    for col in data:
        for subject in data.index:

            data_file = Data_File(data.at[subject, col], wrapped_load_func)
            file_mapping[cnt] = data_file

            data.at[subject, col] = cnt
            cnt += 1

    # Process for outliers - if requested
    if filter_outlier_percent is not None or filter_outlier_std is not None:
        data, file_mapping =\
            filter_data_file_cols(data, reduce_func,
                                  filter_outlier_percent,
                                  filter_outlier_std,
                                  data_file_keys=data_file_keys,
                                  file_mapping=file_mapping,
                                  subject_id=self.subject_id,
                                  n_jobs=self.n_jobs,
                                  _print=self._print)

    # Merge self.data with new loaded data
    self.data = self._merge_existing(self.data, data,
                                     load_params['merge'])

    # Only once the merge w/ existing has been confirmed,
    # merge with class globals
    self.file_mapping.update(file_mapping)
    self.data_file_keys += data_file_keys


def Load_Targets(self, loc=None, df=None, col_name=None, data_type=None,
                 dataset_type='default', subject_id='default',
                 eventname='default', eventname_col='default',
                 overlap_subjects='default', merge='default',
                 na_values='default', drop_na='default', drop_or_na='default',
                 filter_outlier_percent=None,
                 filter_outlier_std=None, categorical_drop_percent=None,
                 float_bins=10, float_bin_strategy='uniform',
                 clear_existing=False, ext=None):
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

    data_type : {'b', 'c', 'f', 'f2c', 'a'}, optional
        The data types of the different columns to load,
        in the same order as the column names passed in.
        Shorthands for datatypes can be used as well.

        If a list is passed to col_name, then you can either supply
        one data_type to be applied to all passed cols, or a list with
        corresponding data types by index for each col_name passed.

        - 'binary' or 'b'
            Binary input

        - 'categorical' or 'c'
            Categorical input

        - 'float' or 'f'
            Float numerical input

        - 'float_to_cat', 'f2c', 'float_to_bin' or 'f2b'
            This specifies that the data should be loaded
            initially as float, then descritized to be a binned
            categorical feature.

        - 'auto' or 'a'
            This specifies that the type should be automatically inferred.
            Current inference rules are: if 2 unique non-nan categories then
            binary, if pandas datatype category, then categorical, otherwise float.

        Datatypes are explained further in Notes.

        (default = None)

    dataset_type :
    subject_id :
    eventname :
    eventname_col :
    overlap_subjects :
    merge :
    na_values :
    drop_na :
    drop_or_na :

    filter_outlier_percent : float, tuple, list of or None, optional
        For float datatypes only.
        A percent of values to exclude from either end of the
        target distribution, provided as either 1 number,
        or a tuple (% from lower, % from higher).
        set `filter_outlier_percent` to None for no filtering.

        For example, if passed (1, 1), then the bottom 1% and top 1%
        of the distribution will be dropped, the same as passing 1.
        Further, if passed (.1, 1), the bottom .1% and top 1% will be
        removed.

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

    float_bins : int or list of, optional
        If any columns are loaded as 'float_to_bin' or 'f2b' then
        input must be discretized into bins. This param controls
        the number of bins to create. As with other params, if one value
        is passed, it is applied to all columns, but if different values
        per column loaded are desired, a list of ints (with inds correponding)
        should be pased. For columns that are not specifed as 'f2b' type,
        anything can be passed in that list index spot as it will be igored.

        (default = 10)

    float_bin_strategy : {'uniform', 'quantile', 'kmeans'}, optional
        If any columns are loaded as 'float_to_bin' or 'f2b' then
        input must be discretized into bins. This param controls
        the strategy used to define the bins. Options are,

        - 'uniform'
            All bins in each feature have identical widths.

        - 'quantile'
            All bins in each feature have the same number of points.

        - 'kmeans'
            Values in each bin have the same nearest center of a 1D
            k-means cluster.

        As with float_bins, if one value
        is passed, it is applied to all columns, but if different values
        per column loaded are desired, a list of choices
        (with inds correponding) should be pased.

        (default = 'uniform')

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

    ext : None or str, optional
        Optional fixed extension to append to all loaded col names,
        leave as None to ignore this param. Note: applied after
        name mapping.

        (default = None)

    Notes
    ----------
    Targets can be either 'binary', 'categorical',
    or 'float',

    - binary
        Targets are read in and label encoded to be 0 or 1,
        Will also work if passed column of unique string also,
        e.g. 'M' and 'F'.

    - categorical
        Targets are treated as taking on one fixed value from a
        limited set of possible values.

    - float
        Targets are read in as a floating point number,
        and optionally then filtered.
    '''

    if clear_existing:
        self.Clear_Targets()

    # Get the common load params as a mix of user-passed + default values
    load_params = self._make_load_params(args=locals())

    if not load_params['drop_na']:
        self._print('Warning: you are choosing to keep NaNs loaded',
                    'in the target variable.')

    # Load in the targets w/ basic pre-processing
    targets, col_names = self._common_load(loc, df, dataset_type,
                                           load_params,
                                           col_names=col_name,
                                           ext=ext)

    # Proccess the passed in data_types - get right number and in list
    data_types, col_names = proc_datatypes(data_type, col_names)

    # Process pass in other args to be list of len(datatypes)
    fops = proc_args(filter_outlier_percent, data_types)
    foss = proc_args(filter_outlier_std, data_types)
    cdps = proc_args(categorical_drop_percent, data_types)
    fbs = proc_args(float_bins, data_types)
    fbss = proc_args(float_bin_strategy, data_types)

    # Set the drop_val
    if load_params['drop_or_na'] == 'na':
        drop_val = np.nan
    else:
        drop_val = get_unused_drop_val(targets)

    self._print()

    # Process each target to load
    for key, d_type, fop, fos, cdp, fb, fbs, in zip(col_names, data_types,
                                                    fops, foss, cdps,
                                                    fbs, fbss):
        targets =\
            self._proc_target(targets, key, d_type,
                              fop, fos, cdp, fb, fbs,
                              drop_val, load_params['drop_or_na'])
    self._print()

    # Drop rows set to drop
    targets = drop_from_filter(targets, drop_val, _print=self._print)

    self._print('Loaded Shape:', targets.shape)

    # Merge with existing and set self.targets
    self.targets = self._merge_existing(self.targets, targets,
                                        load_params['merge'])

    # Print out info on all loaded targets, w/ index names / keys
    self._print_loaded_targets()


def _proc_target(self, targets, key, d_type, fop, fos, cdp, fb,
                 fbs, drop_val, don, add_key=True):

    # If float to bin, recursively call this func with d_type float first
    if is_f2b(d_type):
        targets = self._proc_target(targets, key,
                                    d_type='float',
                                    fop=fop,
                                    fos=fos,
                                    cdp=None,
                                    fb=None,
                                    fbs=None,
                                    drop_val=drop_val,
                                    don=don,
                                    add_key=False)

    else:
        self._print('loading:', key)

    # Set to only the non-Nan subjects for this column
    non_nan_subjects = targets[(~targets[key].isna()) &
                               (targets[key] != drop_val)].index
    non_nan_targets = targets.loc[non_nan_subjects]

    # Check for auto data type
    if d_type == 'a' or d_type == 'auto':
        d_type = auto_data_type(non_nan_targets[key])

    # Processing for binary, with some tolerance to funky input
    if d_type == 'b' or d_type == 'binary':

        non_nan_targets, self.targets_encoders[key] =\
            process_binary_input(non_nan_targets, key, drop_val, self._print)

    # Proc. Categoirical as ordinal
    elif d_type == 'c' or d_type == 'categorical':

        non_nan_targets, self.targets_encoders[key] =\
            process_ordinal_input(non_nan_targets, key, cdp, drop_val,
                                  _print=self._print)

    # For float, just optionally apply outlier percent filter
    elif d_type == 'f' or d_type == 'float':

        # Encoder set to None for non_nan_targets
        self.targets_encoders[key] = None

        if (fop is not None and fos is not None) and (don == 'na'):
            raise RuntimeError('You may only pass one of filter outlier',
                               ' percent or std with drop_or_na == na')

        if fop is not None:
            non_nan_targets =\
                filter_float_by_outlier(non_nan_targets, key,
                                        fop, drop_val=drop_val,
                                        _print=self._print)

        if fos is not None:
            non_nan_targets =\
                filter_float_by_std(non_nan_targets, key,
                                    fos, drop_val=drop_val,
                                    _print=self._print)

    # If float to binary
    elif is_f2b(d_type):

        # K-bins encode
        non_nan_targets, self.targets_encoders[key] =\
            process_float_input(data=non_nan_targets, key=key,
                                bins=fb, strategy=fbs, drop_percent=cdp,
                                drop_val=drop_val, nac=False,
                                _print=self._print)

    else:
        raise RuntimeError('Invalid data type passed:', d_type)

    if targets.shape == non_nan_targets.shape:
        targets = non_nan_targets
    else:
        targets = fill_df_with(targets, non_nan_subjects, non_nan_targets)

    # Keep track of each loaded target in targets_keys
    if key not in self.targets_keys and add_key:
        self.targets_keys.append(key)

    return targets


def _print_loaded_targets(self):

    self._print('All loaded targets')
    for i in range(len(self.targets_keys)):
        self._print(i, ':', self.targets_keys[i])
    self._print()


def Load_Covars(self, loc=None, df=None, col_name=None, data_type=None,
                dataset_type='default', subject_id='default',
                eventname='default', eventname_col='default',
                overlap_subjects='default', merge='default',
                na_values='default', drop_na='default',
                drop_or_na='default',
                nan_as_class=False,
                code_categorical_as='depreciated',
                filter_outlier_percent=None,
                filter_outlier_std=None,
                categorical_drop_percent=None,
                float_bins=10,
                float_bin_strategy='uniform',
                clear_existing=False, ext=None):
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

    data_type : {'b', 'c', 'f', 'm', 'f2c'} or None, optional
        The data types of the different columns to load,
        in the same order as the column names passed in.
        Shorthands for datatypes can be used as well.

        If a list is passed to col_name, then you can either supply
        one data_type to be applied to all passed cols, or a list with
        corresponding data types by index for each col_name passed.

        - 'binary' or 'b'
            Binary input

        - 'categorical' or 'c'
            Categorical input

        - 'float' or 'f'
            Float numerical input

        - 'float_to_cat', 'f2c', 'float_to_bin' or 'f2b'
            This specifies that the data should be loaded
            initially as float, then descritized to be a binned
            categorical feature.

        - 'multilabel' or 'm'
            Multilabel categorical input

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
    merge :
    na_values :
    drop_na :
    drop_or_na :

    nan_as_class : bool, or list of, optional
        If True, then when data_type is categorical, instead of keeping
        rows with NaN (explicitly this parameter does not override drop_na,
        so to use this, drop_na must be set to not True).
        the NaN values will be treated as a unique category.

        A list of values can also be passed in the case that
        multiple col_names / covars are being loaded. In this
        case, the index should correspond. If a list is not passed
        here, then the same value is used when loading all covars.

        ::

            default = False

    code_categorical_as: 'depreciated', optional
        This parameter has been removed, please use transformers
        within the actual modelling to accomplish something simillar.

        ::

            default = 'depreciated'

    filter_outlier_percent : int, float, tuple, None or list of, optional
        For float datatypes only.
        A percent of values to exclude from either end of the
        covars distribution, provided as either 1 number,
        or a tuple (% from lower, % from higher).
        set `filter_outlier_percent` to None for no filtering.

        For example, if passed (1, 1), then the bottom 1% and top 1%
        of the distribution will be dropped, the same as passing 1.
        Further, if passed (.1, 1), the bottom .1% and top 1% will be
        removed.

        A list of values can also be passed in the case that
        multiple col_names / covars are being loaded. In this
        case, the index should correspond. If a list is not passed
        here, then the same value is used when loading all covars.

        Note: If loading a variable with type 'float_to_cat' / 'float_to_bin',
        the outlier filtering will be performed before kbin encoding.

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

        Note: If loading a variable with type 'float_to_cat' / 'float_to_bin',
        the outlier filtering will be performed before kbin encoding.

        (default = None)

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

        Note: percent in the name might be a bit misleading.
        For 1%, you should pass .01, for 10%, you should pass .1.

        If loading a categorical variable, this filtering will be applied
        before ordinally encoding that variable. If instead loading a variable
        with type 'float_to_cat' / 'float_to_bin', the outlier filtering will
        be performed after kbin encoding
        (as before then it is not categorical).
        This can yield gaps in the oridinal outputted values.

        (default = None)

    float_bins : int or list of, optional
        If any columns are loaded as 'float_to_bin' or 'f2b' then
        input must be discretized into bins. This param controls
        the number of bins to create. As with other params, if one value
        is passed, it is applied to all columns, but if different values
        per column loaded are desired, a list of ints (with inds correponding)
        should be pased. For columns that are not specifed as 'f2b' type,
        anything can be passed in that list index spot as it will be igored.

        (default = 10)

    float_bin_strategy : {'uniform', 'quantile', 'kmeans'}, optional
        If any columns are loaded as 'float_to_bin' or 'f2b' then
        input must be discretized into bins. This param controls
        the strategy used to define the bins. Options are,

        - 'uniform'
            All bins in each feature have identical widths.

        - 'quantile'
            All bins in each feature have the same number of points.

        - 'kmeans'
            Values in each bin have the same nearest center of a 1D
            k-means cluster.

        As with float_bins, if one value
        is passed, it is applied to all columns, but if different values
        per column loaded are desired, a list of choices
        (with inds correponding) should be pased.

        (default = 'uniform')

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

    ext : None or str, optional
        Optional fixed extension to append to all loaded col names,
        leave as None to ignore this param. Note: applied after
        name mapping.

        (default = None)
    '''

    if code_categorical_as != 'depreciated':
        print('Warning: code_categorical_as has been depreciated. ' +
              'Please move performing categorical encoding to within the ' +
              'Cross-validated loop via Transformers.')

    if clear_existing:
        self.Clear_Covars()

    # Get the common load params as a mix of user-passed + default values
    load_params = self._make_load_params(args=locals())

    # Load in covars w/ basic pre-proc
    covars, col_names = self._common_load(loc, df, dataset_type,
                                          load_params,
                                          col_names=col_name, ext=ext)

    # Proccess the passed in data_types / get right number and in list
    data_types, col_names = proc_datatypes(data_type, col_names)

    # Process pass in other args to be list of len(datatypes)
    nacs = proc_args(nan_as_class, data_types)
    cdps = proc_args(categorical_drop_percent, data_types)
    fops = proc_args(filter_outlier_percent, data_types)
    foss = proc_args(filter_outlier_std, data_types)
    fbs = proc_args(float_bins, data_types)
    fbss = proc_args(float_bin_strategy, data_types)

    # Set the drop_val
    if load_params['drop_or_na'] == 'na':
        drop_val = np.nan
    else:
        drop_val = get_unused_drop_val(covars)

    # Load in each covar
    for key, d_type, nac, cdp, fop, fos, fb, fbs in zip(col_names, data_types,
                                                        nacs, cdps, fops, foss,
                                                        fbs, fbss):
        covars =\
            self._proc_covar(covars, key, d_type, nac, cdp, fop, fos,
                             fb, fbs, drop_val, load_params['drop_or_na'])

    # Have to remove rows with drop_val if drop_val not NaN
    if drop_val is not np.nan:
        covars = drop_from_filter(covars, drop_val, _print=self._print)

    self._print('Loaded Shape:', covars.shape)

    # If other data is already loaded,
    # merge this data with existing loaded data.
    self.covars = self._merge_existing(self.covars, covars,
                                       load_params['merge'])


def _proc_covar(self, covars, key, d_type, nac, cdp,
                fop, fos, fb, fbs, drop_val, don):

    # If float to binary, recursively call this func with d_type float first
    if is_f2b(d_type):

        covars = self._proc_covar(covars, key, d_type='float',
                                  nac=nac, cdp=None,
                                  fop=fop,
                                  fos=fos,
                                  fb=None,
                                  fbs=None,
                                  drop_val=drop_val,
                                  don=don)

    else:
        self._print('loading:', key)

    # Special behavior if categorical data_type + nan_as_class
    if nac and (d_type == 'c' or d_type == 'categorical' or is_f2b(d_type)):

        # Set to all
        non_nan_subjects = covars.index
        non_nan_covars = covars.loc[non_nan_subjects]

        # Set column to load as str - makes NaNs unique
        non_nan_covars[key] = non_nan_covars[key].astype('str')

    # Set to only the non-Nan subjects for this column
    else:
        non_nan_subjects = covars[(~covars[key].isna()) &
                                  (covars[key] != drop_val)].index
        non_nan_covars = covars.loc[non_nan_subjects]

    # Binary
    if d_type == 'b' or d_type == 'binary':

        non_nan_covars, self.covars_encoders[key] =\
            process_binary_input(non_nan_covars, key, drop_val=drop_val,
                                 _print=self._print)

    # Float
    elif d_type == 'f' or d_type == 'float':

        # Float type needs an encoder set to None
        self.covars_encoders[key] = None

        if (fop is not None and fos is not None) and (don == 'na'):
            raise RuntimeError('You may only pass one of filter outlier',
                               ' percent or std with drop_or_na == na')

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

    # Categorical
    elif d_type == 'c' or d_type == 'categorical':

        # Always encode as ordinal
        non_nan_covars, self.covars_encoders[key] =\
                process_ordinal_input(non_nan_covars, key, drop_percent=cdp,
                                      drop_val=drop_val, nac=nac,
                                      _print=self._print)

    # Float to Categorical case
    elif is_f2b(d_type):

        # K-bins encode
        non_nan_covars, self.covars_encoders[key] =\
            process_float_input(data=non_nan_covars, key=key,
                                bins=fb, strategy=fbs, drop_percent=cdp,
                                drop_val=drop_val, nac=nac, _print=self._print)

    # Multilabel
    elif d_type == 'm' or d_type == 'multilabel':

        common_name = process_multilabel_input(key)

        self.covars_encoders[common_name] = key
        self._print('Base str indicator/name for loaded multilabel =',
                    common_name)

    # Otherwise raise error
    else:
        raise RuntimeError('Unknown data_type: ' + d_type)

    if covars.shape == non_nan_covars.shape:
        covars = non_nan_covars
    else:
        fill_df_with(covars, non_nan_subjects, non_nan_covars)

    # Check for special code nan as categorical case
    if nac and hasattr(self.covars_encoders[key], 'nan_val'):

        # Make sure any NaNs are replaced with the nan val of the
        # categorical encoder
        nan_subjects = covars[covars[key].isna()].index

        if len(nan_subjects) > 0:

            # Make sure to add the category if new
            n_v = self.covars_encoders[key].nan_val
            if n_v not in covars[key].dtype.categories:
                covars[key].cat.add_categories(n_v, inplace=True)

            covars.loc[nan_subjects, key] = n_v

    return covars


def Load_Strat(self, loc=None, df=None, col_name=None, dataset_type='default',
               subject_id='default', eventname='default',
               eventname_col='default', overlap_subjects='default',
               binary_col=False, float_to_binary=False, float_col=False,
               float_bins=10, float_bin_strategy='uniform',
               filter_outlier_percent=None, filter_outlier_std=None,
               categorical_drop_percent=None,
               na_values='default', clear_existing=False, ext=None):
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

    float_to_binary : False, int, (int, int), or list of
        Strat values are loaded as ordinal categorical, but one
        could also want to load a float value, and force it to be
        binary via thresholding.

        If False is passed, or False within a list of values,
        this will be ignored. Otherwise, a single int can be
        passed in the case of one threshold when values lower than
        or equal should be converted to 0, and values > to 1. If
        a tuple of ints passed, that corresponds to the case of
        passing a lower and upper binary threshold.

        (default = False)

    float_col : bool, or list or None, optional
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

    filter_outlier_percent : int, float, tuple, None or list of, optional
        If any float_col are set to True, then you may perform float based
        outlier removal.

        A percent of values to exclude from either end of the
        covars distribution, provided as either 1 number,
        or a tuple (% from lower, % from higher).
        set `filter_outlier_percent` to None for no filtering.

        For example, if passed (1, 1), then the bottom 1% and top 1%
        of the distribution will be dropped, the same as passing 1.
        Further, if passed (.1, 1), the bottom .1% and top 1% will be
        removed.

        As with float_col and float_bins, if one value
        is passed, it is applied to all columns, but if different values
        per column loaded are desired, a list of choices
        (with inds correponding) should be pased.

        Note: this filtering will be applied before binning.

        (default = None)

    filter_outlier_std : int, float, tuple, None or list of, optional
        If any float_col are set to True, then you may perform float based
        outlier removal.

        Determines outliers as data points within each column where their
        value is less than the mean of the column - `filter_outlier_std[0]`
        * the standard deviation of the column,
        and greater than the mean of the column + `filter_outlier_std[1]`
        * the standard deviation of the column.

        If a single number is passed, that number is applied to both the lower
        and upper range. If a tuple with None on one side is passed, e.g.
        (None, 3), then nothing will be taken off that lower or upper bound.

        As with float_col and float_bins, if one value
        is passed, it is applied to all columns, but if different values
        per column loaded are desired, a list of choices
        (with inds correponding) should be pased.

        Note: this filtering will be applied before binning.

        (default = None)

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

        Note: if this is used with float col, then the outlier
        removal will be performed after the k-binning. If also provided
        filter_outlier_percent or std, that will be applied before binning.

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

    ext : None or str, optional
        Optional fixed extension to append to all loaded col names,
        leave as None to ignore this param. Note: applied after
        name mapping.

        (default = None)

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
                                         col_names=col_name, ext=ext)

    # Add strat unique name to end of each col name
    col_mapping = {col: col + self.strat_u_name for col in strat}
    strat = strat.rename(col_mapping, axis=1)

    # Proc list optional args to right length
    bcs = proc_args(binary_col, col_names)
    ftbs = proc_args(float_to_binary, col_names)
    fcs = proc_args(float_col, col_names)
    fbs = proc_args(float_bins, col_names)
    fbss = proc_args(float_bin_strategy, col_names)
    cdps = proc_args(categorical_drop_percent, col_names)
    fops = proc_args(filter_outlier_percent, col_names)
    foss = proc_args(filter_outlier_std, col_names)

    # Get drop val, no option for keeping NaN for strat
    drop_val = get_unused_drop_val(strat)

    # Load in each strat w/ passed args
    for key, bc, ftb, fc, fb, fbs, cdp, fop, fos in zip(col_names, bcs,
                                                        ftbs, fcs,
                                                        fbs, fbss, cdps,
                                                        fops, foss):
        strat = self._proc_strat(strat, key, bc, ftb, fc, fb, fbs, cdp,
                                 fop, fos, drop_val)

    # Drop rows set to drop
    strat = drop_from_filter(strat, drop_val, _print=print)

    self._print('Loaded Shape:', strat.shape)

    # Merge with existing if any, and process new overlap of global subjects
    self.strat = self._merge_existing(self.strat, strat, 'inner')


def _proc_strat(self, strat, key, bc, ftb, fc, fb, fbs,
                cdp, fop, fos, drop_val):

    key = key + self.strat_u_name

    # Binary
    if bc:
        strat, self.strat_encoders[key] =\
            process_binary_input(strat, key, drop_val, self._print)

    elif ftb is not False:

        try:
            lower, upper = ftb
            threshold = None
        except TypeError:
            threshold = ftb
            lower, upper = None, None

        key, strat =\
            self._proc_threshold(threshold, lower, upper, key,
                                 strat, replace=True, merge='inner')

        if threshold is None:
            self.strat_encoders[key] =\
                {0: '<' + str(lower), 1: '>' + str(upper)}
        else:
            self.strat_encoders[key] =\
                {0: '<' + str(threshold), 1: '>=' + str(threshold)}

    # Float
    elif fc:

        if fop is not None:
            strat = filter_float_by_outlier(strat, key,
                                            fop, drop_val=drop_val,
                                            _print=self._print)
            strat = drop_from_filter(strat, drop_val, _print=print)

        if fos is not None:
            strat = filter_float_by_std(strat, key, fos,
                                        drop_val=drop_val,
                                        _print=self._print)
            strat = drop_from_filter(strat, drop_val, _print=print)

        strat, self.strat_encoders[key] =\
            process_float_input(data=strat, key=key,
                                bins=fb, strategy=fbs,
                                drop_percent=cdp,
                                drop_val=drop_val, nac=False,
                                _print=self._print)

    # Categorical
    else:
        strat, self.strat_encoders[key] =\
            process_ordinal_input(strat, key, drop_percent=cdp,
                                  drop_val=drop_val, nac=False,
                                  _print=self._print)

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

    If multiple inclusions are loaded, the final set of inclusions
    is computed as the union of all passed inclusions, not the intersection!
    In this way, inclusions acts more as an iterative whitelist.

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

    self.data = filter_data_cols(data, filter_outlier_percent,
                                 filter_outlier_std,
                                 load_params['drop_or_na'],
                                 seperate_keys=self.data_file_keys,
                                 subject_id=self.subject_id,
                                 _print=self._print)


def Filter_Data_Files_Cols(self, reduce_func=np.mean,
                           filter_outlier_percent=None,
                           filter_outlier_std=None,
                           overlap_subjects='default'):

    '''Perform filtering on all loaded data-files based on an outlier percent,
    or filtering by std.

    Note, if overlap_subject is set to True here, only the overlap will
    be saved after proc within self.data.

    Further note: you can only drop data with this function right now,
    keeping files as NaN is not implemented.

    Parameters
    ----------
    reduce_func : python function or list of, optional
        This function should reduce the loaded file to one number,
        making it comptabile with the different filtering strategies.
        For example, the default function is just to take the
        mean of each loaded file, and to compute outlier detection
        on the mean.

        You may also pass a list to reduce func, where each
        entry of the list is a single reduce func. In this case
        outlier filtering will be computed on each reduce_fun seperately,
        and the union of all subjects marked as outlier will be dropped
        at the end.

        ::

            default = np.mean

    filter_outlier_percent : int, float, tuple or None
        A percent of values to exclude from either end of the
        data files distribution, provided as either 1 number,
        or a tuple (% from lower, % from higher).
        set `filter_outlier_percent` to None for no filtering.
        If over 1 then treated as a percent, if under 1, then
        used directly.

        ::

            default = None

    filter_outlier_std : int, float, tuple or None, optional
        Determines outliers as data points within each column where their
        value is less than the mean of the column
        (where column values are determine by the reduce func)
        - `filter_outlier_std[0]` * the standard deviation of the column,
        and greater than the mean of the column + `filter_outlier_std[1]`
        * the standard deviation of the column.

        If a single number is passed, that number is applied to both the lower
        and upper range.  If a tuple with None on one side is passed, e.g.
        (None, 3), then nothing will be taken off that lower or upper bound.

        ::

            default = None

    overlap_subjects :

    '''

    load_params = self._make_load_params(args=locals())
    data = self._set_overlap(self.data, load_params['overlap_subjects'])

    self.data, self.file_mapping =\
        filter_data_file_cols(data, reduce_func, filter_outlier_percent,
                              filter_outlier_std,
                              data_file_keys=self.data_file_keys,
                              file_mapping=self.file_mapping,
                              subject_id=self.subject_id,
                              n_jobs=self.n_jobs,
                              _print=self._print)


def Consolidate_Data_Files(self, save_dr, cast_to=None):

    # Make sure save_dr exists
    os.makedirs(save_dr, exist_ok=True)

    # Make new df
    new = pd.DataFrame(index=self.data.index)
    new['consolidated'] = ''

    for index in self.data.index:

        subj_data = []
        for key in self.data_file_keys:
            int_key = self.data.loc[index, key]
            subj_data.append(self.file_mapping[int_key].load())

        # Stack the subj data with extra columns at last axis
        subj_data = np.stack(subj_data, axis=-1)

        # Optional cast to dtype
        if cast_to is not None:
            subj_data = subj_data.astype(cast_to)

        # Save as name of index in save loc
        save_loc = os.path.join(save_dr, str(index) + '.npy')
        np.save(save_loc, subj_data)

        # Add to new df
        new.loc[index, 'consolidated'] = save_loc

    # Drop the old data_file_keys
    self.data.drop(self.data_file_keys, axis=1, inplace=True)

    # Reset data file keys and file mapping
    self.data_file_keys = []
    self.file_mapping = {}

    # Re-load the consolidated data
    self.Load_Data_Files(df=new)


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

        The threshold is determined by the passed value as either a float for
        a percentage of the data, e.g.,
        computed as unique_val_drop * len(data), 
        or if passed a number greater then 1, then that number, where a
        ny column with less unique values then this threshold will be dropped.

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

    if unique_val_drop is None and unique_val_warn is None:
        return data

    # Seperate data from data files if applicable
    file_keys = [key for key in list(data) if key in self.data_file_keys]
    data_files = data[file_keys]
    data = data.drop(file_keys, axis=1)

    if unique_val_drop is None:
        unique_val_drop = 0
    if unique_val_warn is None:
        unique_val_warn = 0

    if unique_val_drop >= 1:
        drop_thresh = unique_val_drop
    else:
        drop_thresh = unique_val_drop * len(data)

    if unique_val_warn >= 1:
        warn_thresh = unique_val_warn
    else:
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

    # Re-merge
    data = pd.merge(data, data_files, on=self.subject_id)

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

    # Seperate data from data files if applicable
    file_keys = [key for key in list(data) if key in self.data_file_keys]
    data_files = data[file_keys]
    data = data.drop(file_keys, axis=1)

    # Drop the duplicates
    data = drop_duplicate_cols(data, corr_thresh)

    # Re-merge
    self.data = pd.merge(data, data_files, on=self.subject_id)


def Binarize_Target(self, threshold=None, lower=None, upper=None, target=0,
                    replace=True, merge='outer'):
    '''This function binarizes a loaded target variable,
    assuming that a float type target is loaded,
    otherwise this function will break!

    Parameters
    ----------
    threshold : float or None, optional
        Single binary threshold, where any value less than the threshold
        will be set to 0 and any value greater than or equal to the
        threshold will be set to 1. Leave threshold as None, and use
        lower and upper instead to 'cut' out a chunk of values in the middle.

        (default = None)

    lower : float or None, optional
        Any value that is greater than lower will be set to 1,
        and any value <= upper and >= lower will be dropped.

        If a value is set for lower, one cannot be set for threshold,
        and one must bet set for upper.

        (default = None)

    upper : float or None, optional
        Any value that is less than upper will be set to 0,
        and any value <= upper and >= lower will be dropped.

        If a value is set for upper, one cannot be set for threshold,
        and one must bet set for lower.

        (default = None)

    target : int or str, optional
        The loaded target in which to Binarize. This can be
        the int index, or the name of the target column.
        If only one target is loaded, just leave as default.

        (default = 0)

    replace : bool, optional
        If True, then replace the target to be binarized in
        place, otherwise if False, add the binarized version as a
        new target.

        (default = True)

    merge : {'inner' or 'outer'}
        This argument is used only when replace is False,
        and is further relevant only when upper and lower
        arguments are passed. If 'inner', then drop from
        the loaded target dataframe any subjects which do not
        overlap, if 'outer', then set any non-overlapping subjects
        data to NaN's.

        (default = 'outer')

    '''

    targets_key = self._get_targets_key(target)

    targets_key, self.targets =\
        self._proc_threshold(threshold, lower, upper, targets_key,
                             self.targets, replace, merge)

    if not replace:
        self.targets_keys.append(targets_key)

    # Save new encoder, either replacing or adding new
    if threshold is None:
        self.targets_encoders[targets_key] =\
            {0: '<' + str(lower), 1: '>' + str(upper)}
    else:
        self.targets_encoders[targets_key] =\
            {0: '<' + str(threshold), 1: '>=' + str(threshold)}

    if not replace:
        self._print_loaded_targets()


def _proc_threshold(self, threshold, lower, upper, key, df, replace, merge):

    if threshold is None and lower is None and upper is None:
        raise RuntimeError('Some value must be set.')
    if lower is not None and upper is None:
        raise RuntimeError('Upper must be set.')
    if upper is not None and lower is None:
        raise RuntimeError('Lower must be set.')

    values = df[key]
    original_key = key
    original_df = df.copy()

    self._print('Binarizing', key)

    if not replace:
        key = 'binary_' + key

        if key in df:

            cnt = 1
            while key + str(cnt) in df:
                cnt += 1
            key = key + str(cnt)

    if threshold is None:
        one_sum = (values > upper).sum()
        zero_sum = (values < lower).sum()
        drop_sum = len(values) - one_sum + zero_sum

        self._print('Setting:', zero_sum, 'as 0.')
        self._print('Setting:', one_sum, 'as 1.')
        self._print('Dropping:', drop_sum)

        # Drop out the middle
        to_drop = values[(values <= upper) &
                         (values >= lower)].index
        df = df.drop(to_drop)

        # If inner merge, drop subjects from original too
        if merge == 'inner':
            original_df = original_df.drop(to_drop)

        drop_middle = df[original_key]

        # Binarize remaining
        binarize = drop_middle.where(drop_middle > lower, 0)
        binarize = binarize.where(binarize < upper, 1)

    else:
        one_sum = (values >= threshold).sum()
        zero_sum = (values < threshold).sum()

        self._print('Setting:', zero_sum, 'as 0.')
        self._print('Setting:', one_sum, 'as 1.')

        # Grab targets, and binarize
        binarize = values.where(values >= threshold, 0)
        binarize = binarize.where(binarize < threshold, 1)

    # Fill back into df, either replacing or adding new
    original_df[key] = binarize

    # Conv to categorical data type
    original_df[key] = original_df[key].astype('category')

    return key, original_df


def Binarize_Covar(self, threshold=None, lower=None, upper=None, covar=0,
                   replace=True, merge='outer'):
    '''This function binarizes a loaded covar variable,
    assuming that originally float type covar is loaded,
    otherwise this function will break!

    Parameters
    ----------
    threshold : float or None, optional
        Single binary threshold, where any value less than the threshold
        will be set to 0 and any value greater than or equal to the
        threshold will be set to 1. Leave threshold as None, and use
        lower and upper instead to 'cut' out a chunk of values in the middle.

        (default = None)

    lower : float or None, optional
        Any value that is greater than lower will be set to 1,
        and any value <= upper and >= lower will be dropped.

        If a value is set for lower, one cannot be set for threshold,
        and one must bet set for upper.

        (default = None)

    upper : float or None, optional
        Any value that is less than upper will be set to 0,
        and any value <= upper and >= lower will be dropped.

        If a value is set for upper, one cannot be set for threshold,
        and one must bet set for lower.

        (default = None)

    covar : str, optional
        The loaded covar in which to Binarize. This should be
        the loaded name of the covar column.

        (default = 0)

    replace : bool, optional
        If True, then replace the covar to be binarized in
        place, otherwise if False, add the binarized version as a
        new covar.

        (default = True)

    merge : {'inner' or 'outer'}
        This argument is used only when replace is False,
        and is further relevant only when upper and lower
        arguments are passed. If 'inner', then drop from
        the loaded covar dataframe any subjects which do not
        overlap, if 'outer', then set any non-overlapping subjects
        data to NaN's.

        (default = 'outer')
    '''

    covars_key = covar

    if covars_key in self.name_map:
        covars_key = self.name_map[covars_key]

    covars_key, self.covars =\
        self._proc_threshold(threshold, lower, upper, covars_key,
                             self.covars, replace, merge)

    # Save new encoder, either replacing or adding new
    if threshold is None:
        self.covars_encoders[covars_key] =\
            {0: '<' + str(lower), 1: '>' + str(upper)}
    else:
        self.covars_encoders[covars_key] =\
            {0: '<' + str(threshold), 1: '>=' + str(threshold)}


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
    self.file_mapping = {}
    self.data_file_keys = []
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
        self._print('Calling Prepare_All_Data()',
                    'to change the default merge behavior',
                    'call it again!')
        self.Prepare_All_Data()

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


def _load_datasets(self, locs, df, load_params, ext=None):
    '''Helper function to load in multiple datasets with default
    load and drop behavior based on type. And calls proc_df on each
    before merging.

    Parameters
    ----------
    locs : list of str, Path
        The location of the  files to load data load from.

    df : pandas df, or list of
        User passed

    load_params : dict
        load params

    Returns
    ----------
    pandas DataFrame
        BPt formatted pd DataFrame, with the loaded
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

        if not isinstance(df, list):
            df = [df]

        for single_df in df:
            single_df = self._load_user_passed(single_df,
                                               load_params['na_values'])
            single_df = self._proc_df(single_df, load_params)
            dfs.append(single_df)

    # Add ext to col names if any ext
    if ext is not None:
        for d in range(len(dfs)):
            col_mapping = {col: col + ext for col in dfs[d]}
            dfs[d] = dfs[d].rename(col_mapping, axis=1)

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

        data = pd.merge(data, more_data, on=self.subject_id,
                        how=load_params['merge'])

    return data


def _load_user_passed(self, df, na_values):

    self._print('Loading from df or files')

    if df.index.name is not None:
        df = df.reset_index()

    df = df.replace(to_replace=na_values, value=np.nan)
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
        BPt formatted pd DataFrame, with the loaded
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
    return data


def _common_load(self, loc, df, dataset_type, load_params,
                 col_names=None, ext=None):

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

    # Add ext to col names if any ext
    if ext is not None:
        col_mapping = {col: col + ext for col in data}
        data = data.rename(col_mapping, axis=1)

    return data, list(data)


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


def _merge_existing(self, class_data, local_data, merge='inner'):
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

        class_data = pd.merge(class_data, local_data, on=self.subject_id,
                              how=merge)
        self._print('Merged with existing (merge=' + str(merge) + ')')
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
        self._print('Note: BPt does not currently support',
                    'duplicate subjects loaded as seperate rows!',
                    before - after, 'subjects have been dropped',
                    'accordingly.')

    # Set the subject_id column as the index column
    data = data.set_index(self.subject_id)

    # Drop excluded and/or included subjects, if any
    data = self._drop_excluded(data)
    data = self._drop_included(data)

    return data


def _load_set_of_subjects(self, loc=None, subjects=None, auto=None):
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

    if isinstance(auto, str):
        loc = auto
    elif auto is not None:
        subjects = auto

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

    if self.use_abcd_subject_ids:
        subject = subject.strip().upper()

        if 'NDAR_' not in subject:
            subject = 'NDAR_' + subject
        return subject

    else:
        return subject


def _drop_na(self, data, drop_na=True):

    # Seperate data from data files if applicable
    file_keys = [key for key in list(data) if key in self.data_file_keys]
    data_files = data[file_keys]
    data = data.drop(file_keys, axis=1)

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

    # Re-merge
    data = pd.merge(data, data_files, on=self.subject_id)

    return data


def _filter_by_eventname(self, data, eventname, eventname_col):
    '''Internal helper function, to filter a dataframe by eventname,
    and then return the dataframe.

    Parameters
    ----------
    data : pandas DataFrame
        BPt formatted.

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
                self._print(before - after, 'data points have been dropped',
                            'based on the passed eventname params.')

        else:
            self._print('Warning: filter by eventname_col:',
                        eventname_col, 'specfied, with value:',
                        eventname, 'but the column was not found!')

    # Remove eventname column
    if eventname_col in list(data):
        data = data.drop(eventname_col, axis=1)

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
        BPt formatted.

    Returns
    ----------
    pandas DataFrame
        Input df, with dropped excluded subjects
    '''

    in_num = len(data)
    overlap = set(data.index).intersection(self.exclusions)
    data = data.drop(overlap)
    if in_num != len(data):
        self._print('Dropped', in_num - len(data), 'excluded subjects')

    return data


def _drop_included(self, data):
    '''Wrapper to drop subjects from a df,
    if any do not overlap with inclusions.

    Parameters
    ----------
    data : pandas DataFrame
        BPt formatted.

    Returns
    ----------
    pandas DataFrame
        Input df, with dropped non included subjects
    '''

    if len(self.inclusions) > 0:

        in_num = len(data)

        to_drop = set(data.index) - self.inclusions
        data = data.drop(to_drop)

        if in_num != len(data):
            self._print('Dropped', in_num - len(data), 'outside of included '
                        'subjects')

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


def Prepare_All_Data(self, merge='default', low_memory_mode='default'):
    '''Helper function to merge all loaded data from different sources.
    self.data, self.covars, self.targets ect...
    into self.all_data for use directly in ML.
    This function must be called before Modelling.

    Parameters
    ----------
    merge : {'inner' or 'outer'}, optional
        To generate all data, the different loaded dataframes must
        be merged. This parameter controls the behavior of that merge.
        If 'inner' is passed, then only the overlapping subjects will be
        considered. If 'outer' is passed, then the union of all subjects
        will be taken, with missing spots filled with NaN!

        If left as 'default' will use the default value for
        merge set in Set_Default_Load_Params,
        which is initially 'inner', unless changed

        ::

            default = 'default'

    low_memory_mode : bool or 'default', optional
        If True, then all of the individual dataframes, e.g.,
        self.data, self.covars, etc... will be removed from memory
        once self.all_data is created. If False then they will
        not.

        If Prepare_All_Data is called with low_memory_mode True, then
        calling it again will not work, as the data to create self.all_data
        will have been deleted. In this case, just re-run your script.

        If left as default, will use the saved class value, set upon init.

        ::

            default = 'default'

    '''

    load_params = self._make_load_params({'merge': merge})

    self._print('Preparing final data, in self.all_data')
    self._print('Any changes to loaded data, covars or strat will not be',
                'included, from now on.')
    self._print()

    dfs = []

    if len(self.data) == 0 and len(self.covars) == 0:
        raise RuntimeError('Some data must be loaded!')

    if len(self.data) > 0:
        dfs.append(self.data)

    if len(self.covars) > 0:
        dfs.append(self.covars)

    if len(self.strat) > 0:
        dfs.append(self.strat)

    if len(self.targets) == 0:
        raise RuntimeError('Targets must be loaded!')

    dfs.append(self.targets)

    self.all_data = dfs[0]
    for i in range(1, len(dfs)):

        overlap_c_names = np.array([c_name in list(self.all_data) for
                                    c_name in list(dfs[i])])

        if overlap_c_names.any():

            self._print('Col names from data, covars, targets and strat',
                        'must be unique!')

            overlap = np.array(list(dfs[i]))[overlap_c_names]
            raise RuntimeWarning(str(overlap) + ' col(s) overlap!')

        self.all_data = pd.merge(self.all_data, dfs[i],
                                 on=self.subject_id, how=load_params['merge'])

    # Set data keys, covars, strat, ect...
    self._set_data_scopes()

    self._print('Final data (w/ target) for modeling loaded shape:',
                self.all_data.shape)

    if low_memory_mode == 'default':
        low_memory_mode = self.low_memory_mode

    if low_memory_mode:
        self._print('Low memory mode is on!')
        self._print('Clearing self.data, self.covars, self.targets',
                    'and self.strat from memory!')
        self._print('Note: Final data, self.all_data, the',
                    'merged dataframe is still in memory')

        self.Clear_Data()
        self.Clear_Targets()
        self.Clear_Covars()
        self.Clear_Strat()

    # IMPORTANT, @ end sort all data by index, so that calculated splits
    # are reporoducible
    self.all_data.sort_index(inplace=True)


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


def _set_data_scopes(self):

    self.Data_Scopes = Data_Scopes(data_keys=list(self.data),
                                   data_file_keys=self.data_file_keys,
                                   cat_keys=self._get_cat_keys(),
                                   strat_keys=list(self.strat),
                                   covars_keys=list(self.covars),
                                   file_mapping=self.file_mapping)


def _get_base_targets_names(self):

    targets_base_keys = []

    for targets_key in self.targets_keys:

        if isinstance(targets_key, list):
            base_targets_key = get_common_name(targets_key)
        else:
            base_targets_key = targets_key

        targets_base_keys.append(base_targets_key)

    return targets_base_keys
