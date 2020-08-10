


class Dataset():

    def __init__(self, subject_id='src_subject_id',
                 eventname=None, eventname_col=None,
                 merge='inner', drop_na=True,
                 drop_or_na='drop', ext=None):
        ''' This class is used in order to load and prepare a dataset for modelling.
        The values set upon initialization here reflect the setting of default
        values for Dataset methods, e.g., for Load_Variable and Load_Variables.

        Parameters
        ----------
        subject_id : str, optional
            The default name of the column in a provided location or dataframe
            in which the unique subject identifier or subject id is stored.

            In the case of longitudinal data, it may be neccisary to further
            specify an eventname and eventname_col arguments.

            ::

                default = 'src_subject_id'

        eventname : str or None, optional
            Optional default eventname value,
            if set to None as is by default, this parameter is ignored.
            Eventname is used mostly in the context of loading data
            from longitudinal studies.
            This value provided here must be provided along with the following
            eventname_col
            (which specified the name of the column where this value).

            The value passed here specifies which value in the eventname column
            should be kept.

            For example, lets say the eventname column contained entries for
            'year 1' and 'year 2'. By passing 'year 1' here, only data points
            with a value of 'year 1' will be kept.

            ::

                default = None

        eventname_col : str or None, optional
            Default eventname_col value.

            If an eventname is provided, this param refers to
            the name of the column within the dataframe. This could
            also be used along with eventname to be set to any
            arbitrary value, in order to perform selection by specific
            column value.

            If `eventname` is None, this parameter will be skipped.

            ::

                default = None

        merge : {'inner' or 'outer'}, optional
            This default parameter controls the merge
            behavior when loading a new variable or set of variables.
            Specifically, the options are:

            - 'inner'
                Only the overlapping subjects between those already loaded
                and those about to be loaded will be kept.

            - 'outer'
                All subjects are kept, and in any columns where data
                is missing, NaN's will be inserted.

            ::

                default = 'inner'

        drop_na : bool, optional
            This setting sets the default value for drop_na.

            If set to True, then any row containing 1 or more missing
            values will be dropped. If False, the missing value will remain.

            If missing data is kept, missing data imputation
            will likely be required later on.


            ::

                default = True

        drop_or_na : {'drop', 'na'}, optional

            This setting sets the default value for drop_na,
            which is relevant when performing outlier filtering.
            In this case the default behavior is for outliers,
            however defined by the user, to be dropped ('drop'),
            dropping the whole row of subjects data.
            Alternatively, by setting this parameter to 'na',
            you can specify that the value prompting the drop
            to instead be replaced with NaN (to be imputed later on).

            ::

                default = 'drop'

        ext : None, or str, optional
            Optional fixed extension in which to append to
            all loaded column names. If left as default value
            of None, ignore this parameter.
            leave as None to ignore this param. Note: applied after
            name mapping.

            This parameter can be useful, especially when loading
            data from multiple timepoints in a longitudinal context.

            ::

                default = None
        '''

        self.subject_id = subject_id
        self.eventname = eventname
        self.eventname_col = eventname_col
        self.merge = merge
        self.drop_na = drop_na
        self.drop_or_na = drop_or_na
        self.ext = ext

    def Load_Variable(self,
                      df,
                      col_name,
                      type,
                      data_type,
                      scopes='default',
                      filter_outlier_percent=None,
                      filter_outlier_std=None,
                      categorical_drop_percent=None,
                      nan_as_class=False,
                      float_bins=10,
                      float_bin_strategy='uniform',
                      subject_id='default',
                      eventname='default',
                      eventname_col='default',
                      merge='default',
                      drop_na='default',
                      drop_or_na='default',
                      ext='default',
                      clear_existing=False):

    '''
    Load a single variable from a pandas dataframe into the Dataset.

    Parameters
    ----------
    df : pandas DataFrame
        The DataFrame in which the variable to load is included in.
        This can be prepared ahead of time with the pandas library.

    col_name : str
        The name of the column to load.

    type : {'data', 'target', 'strat'}
        This determines the role of the data to
        be loaded in. Valid values are:

        - 'data'
            Data represents variables which will ultimately be
            used as input features for an ML expiriment.

        - 'target'
            Target's are the variables to be predicted using
            the loaded data.

        - 'strat'
            Strat represent non-input variables. These
            are converted internally to ordinal variables
            and are used to inform things like cross-validation
            strategies.

     data_type : {'b', 'c', 'f', 'f2c', 'm'}
        The data type of the column to load in.

        The extensive list of values accepted here is listed below:

        - 'binary' or 'b'
            Binary variable. If more than two unique values
            are found, all but the most freuqent two will be
            dropped, ensuring that this variable is loaded
            in as binary.

        - 'categorical' or 'c'
            Categorical input. This variable will be
            ordinally encoded, and if specified as non-null, the
            `categorical_drop_percent` parameter will be applied.

        - 'float', 'f' or 'continuous'
            A floating point numerical / continuous variable.
            If either `filter_outlier_percent` or `filter_outlier_std`
            is specified as non-null, then the respective outlier
            removal method will be applied.

        - 'float_to_cat', 'f2c', 'float_to_bin' or 'f2b'
            This specifies that the data should be loaded
            initially as float, then descritized via k-bin
            encoding to be a categorical feature.
            The `float_bins` and `float_bin_strategy` will
            be applied if this data_type is selected.

        - 'multilabel' or 'm'
            Multilabel categorical input.
            If 'multilabel' datatype is specified, then the associated col_name
            should be a list of columns.

    scopes : str, list of str or 'default', optional
        When loading a variable you have the option to
        associate either a single scope (as a str) or multiple
        scopes (list of strs). By default, both the type and
        data_type arguments will be added as scopes.
        If any additional scopes are passed here, they will be added
        in addition.

        Scopes are used in a number of places in order to select single
        or subsets of variables. For example, you may want to only perform
        a certain type of processing on categorical variables.

        ::

            default = 'default'

    filter_outlier_percent : float, tuple or None, optional
        For float datatypes only.
        A percent of values to exclude from either end of the
        variables distribution, provided as either 1 number,
        or a tuple (% from lower, % from higher).
        set `filter_outlier_percent` to None for no filtering.

        For example, if passed (1, 1), then the bottom 1% and top 1%
        of the distribution will be dropped, the same as passing 1.
        Further, if passed (.1, 1), the bottom .1% and top 1% will be
        removed.

        Note: If loading a variable with data_type 'float_to_cat'
         / 'float_to_bin',
        the outlier filtering will be performed before kbin encoding.

        ::

            default = None

    filter_outlier_std : float, tuple, or None, optional
        For float datatypes only.
        Determines outliers as data points within each column where their
        value is less than the mean of the column - `filter_outlier_std[0]`
        * the standard deviation of the column,
        and greater than the mean of the column + `filter_outlier_std[1]`
        * the standard deviation of the column.

        If a single number is passed, that number is applied to both the lower
        and upper range. If a tuple with None on one side is passed, e.g.
        (None, 3), then nothing will be taken off that lower or upper bound.

        Note: If loading a variable with type 'float_to_cat' / 'float_to_bin',
        the outlier filtering will be performed before kbin encoding.

        ::

            default = None


    
    
    
    # See :class:`Dataset`
    '''

                  
 
                  
                  
                  f
                  clear_existing=False, ext=None):
    '''Load a single variable, or 

    Parameters
    ----------
   

   

   

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
                             fb, fbs, drop_val)

    # Have to remove rows with drop_val if drop_val not NaN
    if drop_val is not np.nan:
        covars = drop_from_filter(covars, drop_val, _print=self._print)

    self._print('Loaded Shape:', covars.shape)

    # If other data is already loaded,
    # merge this data with existing loaded data.
    self.covars = self._merge_existing(self.covars, covars,
                                       load_params['merge'])


def _proc_covar(self, covars, key, d_type, nac, cdp,
                fop, fos, fb, fbs, drop_val):

    # If float to binary, recursively call this func with d_type float first
    if is_f2b(d_type):

        covars = self._proc_covar(covars, key, d_type='float',
                                  nac=nac, cdp=None,
                                  fop=fop,
                                  fos=fos,
                                  fb=None,
                                  fbs=None,
                                  drop_val=drop_val)

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
        non_nan_subjects = covars[~covars[key].isna()].index
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

        if fop is not None and fos is not None:
            raise RuntimeError('You may only pass one of filter outlier',
                               ' percent or std')

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

    # Now update the changed values within covars
    covars.loc[non_nan_subjects] = non_nan_covars

    # Check for special code nan as categorical case
    if nac and hasattr(self.covars_encoders[key], 'nan_val'):

        # Make sure any NaNs are replaced with the nan val of the
        # categorical encoder
        nan_subjects = covars[covars[key].isna()].index
        covars.loc[nan_subjects, key] = self.covars_encoders[key].nan_val

    # Update col datatype
    for dtype, k in zip(non_nan_covars.dtypes, list(covars)):
        covars[k] = covars[k].astype(dtype.name)

    return covars