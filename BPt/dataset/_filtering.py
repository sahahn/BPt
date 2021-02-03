import numpy as np
from itertools import combinations
from .helpers import proc_fop


def _drop_subjects(self, subjects):

    if len(subjects) > 0:
        self.drop(subjects, axis=0, inplace=True)
        self._print('Dropped', len(subjects), 'Rows', level=1)
        self._print('Dropped Rows:', subjects, level=2)

    return self


def drop_nan_subjects(self, scope, inplace=True):
    '''This method is used for
    dropping all of the subjects which have NaN
    values for a given scope / column.

    Parameters
    ----------
    scope : :ref:`Scope`
        The BPt style :ref:`Scope` input that will be
        used to determine which column names to drop
        subjects with missing values by.

    inplace : bool, optional
        If this operation should take place on
        the original object (inplace = True),
        or if it should be done on a copy of the Dataset
        object (inplace = False).

        If done inplace, then None will be returned.
        If done with inplace = False, then a copy
        of the Dataset with the operation applied
        will be returned.

        ::

            default = True
    '''

    if not inplace:
        df_copy = self.copy(deep=False)
        df_copy.drop_nan_subjects(scope=scope, inplace=True)
        return df_copy

    cols = self.get_cols(scope)
    for col in cols:
        nan_subjects = self._get_nan_subjects(col)
        self._drop_subjects(nan_subjects)


def apply_inclusions(self, subjects, inplace=True):
    '''This method will drop all subjects
    that do not overlap with the passed subjects to
    this function. In this sense, this method acts
    like a whitelist, where you could pass for example
    only valid subjects that passed some QC, and all
    other's loaded will be dropped.

    Parameters
    -----------
    subjects : :ref:`Subjects`
        This argument can be any of the BPt accepted
        subject style inputs. E.g., None, 'nan' for subjects
        with any nan data, the str location of a file
        formatted with one subject per line, or directly an
        array-like of subjects, to name some options.

        See :ref:`Subjects` for all options.

    inplace : bool, optional
        If this operation should take place on
        the original object (inplace = True),
        or if it should be done on a copy of the Dataset
        object (inplace = False).

        If done inplace, then None will be returned.
        If done with inplace = False, then a copy
        of the Dataset with the operation applied
        will be returned.

        ::

            default = True
    '''

    if not inplace:
        df_copy = self.copy(deep=False)
        df_copy.apply_inclusions(subjects=subjects, inplace=True)

        return df_copy

    # Load inclusions
    inclusions = self.get_subjects(subjects, return_as='set')

    if len(inclusions) > 0:
        to_drop = set(self.index) - inclusions
        self.drop(list(to_drop), axis=0, inplace=True)


def apply_exclusions(self, subjects, inplace=True):
    '''This method will drop all subjects
    that overlap with the passed subjects to
    this function.

    Parameters
    -----------
    subjects : :ref:`Subjects`
        This argument can be any of the BPt accepted
        subject style inputs. E.g., None, 'nan' for subjects
        with any nan data, the str location of a file
        formatted with one subject per line, or directly an
        array-like of subjects, to name some options.

        See :ref:`Subjects` for all options.

    inplace : bool, optional
        If this operation should take place on
        the original object (inplace = True),
        or if it should be done on a copy of the Dataset
        object (inplace = False).

        If done inplace, then None will be returned.
        If done with inplace = False, then a copy
        of the Dataset with the operation applied
        will be returned.

        ::

            default = True
    '''

    if not inplace:
        df_copy = self.copy(deep=False)
        df_copy.apply_exclusions(subjects=subjects, inplace=True)
        return df_copy

    # Load exclusions
    exclusions = self.get_subjects(subjects, return_as='set')

    if len(exclusions) > 0:
        to_drop = set(self.index).intersection(exclusions)
        self.drop(list(to_drop), axis=0, inplace=True)


def drop_subjects_by_nan(self, threshold=.5, scope='all', inplace=True):
    '''This method is used for dropping subjects based on
    the amount of missing values found across a subset of
    columns as selected by scope. Each subject is dropped
    if it has greater than or equal to the passed threshold
    of NaN values.

    Parameters
    -----------
    threshold : float or int, optional
        Passed as a float between 0 and 1, or
        as an int. If greater than 0 or less than 1,
        this threshold represents the percent of columns
        a subject needs to have missing for it to be dropped.
        If 1 or over, then it represents an absolute number of
        columns that a subject needs to have greater than or equal
        to that value in order to drop.

        ::

            default = .5

    scope : :ref:`Scope`, optional
        A BPt style :ref:`Scope` used to select a subset of
        columns in which to check for if a subject should be dropped
        or not.

        By default this is set to 'all' and will check across
        all loaded subjects.

        ::

            default = 'all'

    inplace : bool, optional
        If this operation should take place on
        the original object (inplace = True),
        or if it should be done on a copy of the Dataset
        object (inplace = False).

        If done inplace, then None will be returned.
        If done with inplace = False, then a copy
        of the Dataset with the operation applied
        will be returned.

        ::

            default = True
    '''

    if not inplace:
        df_copy = self.copy(deep=False)
        df_copy.drop_subjects_by_nan(threshold=threshold,
                                     scope=scope, inplace=True)
        return df_copy

    # Get cols from scope
    cols = self.get_cols(scope)

    # Get nan counts by column
    col_nan_cnts = self[cols].isnull().sum(axis=1)

    # If between 0 and 1
    if threshold > 0 and threshold < 1:

        # Change threshold from percent to abs
        threshold = threshold * len(cols)
        self._print('Setting NaN threshold to:', threshold, level=1)

    # Calculate subjects to drop if greater than or equal to threshold
    to_drop = self.loc[col_nan_cnts >= threshold].index

    self._drop_subjects(to_drop)


def _drop_or_nan(self, col, to_drop_index, all_to_drop, drop):
    '''Internal helper function for commonly re-used drop or
    nan function.'''

    # If drop, add to drop list at end
    if drop:
        all_to_drop.update(set(to_drop_index))

    # Otherwise, set to NaN in place
    else:
        self.loc[to_drop_index, col] = np.nan


def _data_file_fail_check(self, cols):

    for col in cols:
        if 'data file' in self.scopes[col]:
            raise RuntimeError('Loaded column: ' + col + ' cannot be a'
                               ' data file.')


def _drop_cols(self, to_drop):

    if len(to_drop) > 0:

        self.drop(to_drop, axis=1, inplace=True)
        self._print('Dropped', len(to_drop), 'Columns', level=1)
        self._print('Dropped Columns:', to_drop, level=2)

    return self


def filter_outliers_by_percent(self, fop=1, scope='float', drop=True,
                               reduce_func=np.mean, n_jobs=1, inplace=True):
    '''This method is designed to allow dropping a fixed percent of outliers
    from the requested columns. This method is designed to work
    on float type / cont. variables.

    Note: This method operates on each of the columns specified by scope
    independently. In the case that multiple columns are passed, then the
    overlap of all outliers from each column will dropped after all
    have been calculated (therefore the order won't matter).

    This method can be used with data file's as well, the
    reduce_func and n_jobs parameters are specific to this case.

    Parameters
    -----------
    fop : float, tuple, optional
        This parameter represents the percent of outliers to drop.
        It should be passed as a percent, e.g., therefore 1 for
        one percent, or 5 for five percent.

        This can also be passed as a tuple with two elements, where
        the first entry represents the percent to filter from the lower
        part of the distribution and the second element the percent from
        the upper half of the distribution. For example,

        ::

            filter_outlier_percent = (5, 1)

        This set of parameters with drop 5 percent from the lower part
        of the distribution and only 1 percent from the top portion.
        Likewise, you can use None on one side to skip dropping from
        one half, for example:

        ::

            filter_outlier_percent = (5, None)

        Would drop only five percent from the bottom half, and not drop
        any from the top half.

        ::

            default = 1

    scope : :ref:`Scope`, optional
        A BPt style :ref:`Scope` used to select a subset of
        columns in which to percent this outlier filtering by.

        By default this is set to only the 'float' style data.

        ::

            default = 'float'

    drop : bool, optional
        By default this function will drop any subject's that are
        determined as outliers. On the otherhand, you
        may instead set specific outlier values as NaN values instead.
        To do this, you can pass drop=False here, and now those specific
        values identified as outliers will be replaced with NaN.

        ::

            default = True

    reduce_func : python function, optional
        The passed python function will be applied only if
        the requested col/column is a 'data file'. In the case
        that it is, the function should accept as input
        the data from one data file, and should return a single
        scalar value. For example, the default value is
        numpy's mean function, which returns one value.

        ::

            default = np.mean

    n_jobs : int, optional
        As with reduce_func, this parameter is only
        valid when the passed col/column is a 'data file'.
        In that case, this specifies the number of cores
        to use in loading and applying the reduce_func to each
        data file. This can provide a signifigant speed up when
        passed the number of avaliable cores, but can sometimes
        be memory intensive depending on the underlying size of the file.

        ::

            default = 1

    inplace : bool, optional
        If this operation should take place on
        the original object (inplace = True),
        or if it should be done on a copy of the Dataset
        object (inplace = False).

        If done inplace, then None will be returned.
        If done with inplace = False, then a copy
        of the Dataset with the operation applied
        will be returned.

        ::

            default = True
    '''

    # Check scope and role
    self._check_sr()

    if not inplace:
        df_copy = self.copy(deep=False)
        df_copy.filter_outliers_by_percent(fop=fop, scope=scope, drop=drop,
                                           reduce_func=reduce_func,
                                           n_jobs=n_jobs, inplace=True)
        return df_copy

    # Get cols from scope
    cols = self._get_cols(scope)

    # For if to drop
    all_to_drop = set()

    # Proc fop
    fop = proc_fop(fop)
    lower, upper = fop

    for col in cols:

        # Extract non-nan values
        values = self._get_values(col, dropna=True,
                                  reduce_func=reduce_func,
                                  n_jobs=n_jobs)

        if lower is not None:

            # Compute index under lower bound
            mask = values < values.quantile(lower)
            to_drop_index = values[mask].index

            # Drop or NaN
            self._drop_or_nan(col=col, to_drop_index=to_drop_index,
                              all_to_drop=all_to_drop, drop=drop)

        if upper is not None:

            # Compute index above upper bound
            mask = values > values.quantile(upper)
            to_drop_index = values[mask].index

            # Drop or NaN
            self._drop_or_nan(col=col, to_drop_index=to_drop_index,
                              all_to_drop=all_to_drop, drop=drop)

    # Drop now, if to drop
    if drop:
        self.drop(list(all_to_drop), axis=0, inplace=True)

    # Update file mapping if needed
    self._check_file_mapping()


def filter_outliers_by_std(self, n_std=10, scope='float', drop=True,
                           reduce_func=np.mean, n_jobs=1, inplace=True):
    '''This method is designed to allow dropping outliers
    from the requested columns based on comparisons with that columns
    standard deviation.

    Note: This method operates on each of the columns specified by scope
    independently. In the case that multiple columns are passed, then the
    overlap of all outliers from each column will dropped after all
    have been calculated (therefore the order won't matter).

    This method can be used with data file's as well, the
    reduce_func and n_jobs parameters are specific to this case.

    Parameters
    -----------
    n_std : float, tuple, optional
        This value is used to set an outlier threshold by
        standrad deviation. For example if passed n_std = 10,
        then it will be converted internally to (10, 10).
        This parameter determines outliers as
        data points within each
        relevant column (as determined by the scope argument) where their
        value is less than the mean of the
        column - n_std[0] * the standard deviation of the column,
            and greater than the mean of the column + `n_std[1]`
        * the standard deviation of the column.

        If a single number is passed, that number is applied to
        both the lower
        and upper range.
        If a tuple with None on one side is passed, e.g.
        (None, 3), then nothing will be taken off
        that lower or upper bound.

        ::

            default = 10

    scope : :ref:`Scope`, optional
        A BPt style :ref:`Scope` used to select a subset of
        columns in which to percent this outlier filtering by.

        By default this is set to only the 'float' style data.

        ::

            default = 'float'

    drop : bool, optional
        By default this function will drop any subject's that are
        determined as outliers. On the otherhand, you
        may instead set specific outlier values as NaN values instead.
        To do this, you can pass drop=False here, and now those specific
        values identified as outliers will be replaced with NaN.

        ::

            default = True

    reduce_func : python function, optional
        The passed python function will be applied only if
        the requested col/column is a 'data file'. In the case
        that it is, the function should accept as input
        the data from one data file, and should return a single
        scalar value. For example, the default value is
        numpy's mean function, which returns one value.

        ::

            default = np.mean

    n_jobs : int, optional
        As with reduce_func, this parameter is only
        valid when the passed col/column is a 'data file'.
        In that case, this specifies the number of cores
        to use in loading and applying the reduce_func to each
        data file. This can provide a signifigant speed up when
        passed the number of avaliable cores, but can sometimes
        be memory intensive depending on the underlying size of the file.

        ::

            default = 1

    inplace : bool, optional
        If this operation should take place on
        the original object (inplace = True),
        or if it should be done on a copy of the Dataset
        object (inplace = False).

        If done inplace, then None will be returned.
        If done with inplace = False, then a copy
        of the Dataset with the operation applied
        will be returned.

        ::

            default = True
    '''

    # Check scope and role
    self._check_sr()

    if not inplace:
        df_copy = self.copy(deep=False)
        df_copy.filter_outliers_by_std(n_std=n_std, scope=scope, drop=drop,
                                       reduce_func=reduce_func, n_jobs=n_jobs,
                                       inplace=True)
        return df_copy

    # Get cols from scope
    cols = self._get_cols(scope)

    # For if to drop
    all_to_drop = set()

    # Wrap correctly passed n_std
    if not isinstance(n_std, tuple):
        n_std = (n_std, n_std)

    for col in cols:

        # Extract non-nan values
        values = self._get_values(col, dropna=True,
                                  reduce_func=reduce_func,
                                  n_jobs=n_jobs)

        mean = values.mean()
        std = values.std()

        # If lower
        if n_std[0] is not None:
            l_scale = mean - (n_std[0] * std)
            to_drop_index = values[values < l_scale].index

            # Drop or NaN
            self._drop_or_nan(col=col, to_drop_index=to_drop_index,
                              all_to_drop=all_to_drop, drop=drop)

        # If upper
        if n_std[1] is not None:
            u_scale = mean + (n_std[1] * std)
            to_drop_index = values[values > u_scale].index

            # Drop or NaN
            self._drop_or_nan(col=col, to_drop_index=to_drop_index,
                              all_to_drop=all_to_drop, drop=drop)

    # Drop now, if to drop
    if drop:
        self._drop_subjects(all_to_drop)

    # Update file mapping if needed
    self._check_file_mapping()


def filter_categorical_by_percent(self, drop_percent=1, scope='category',
                                  drop=True, inplace=True):
    '''This method is designed to allow performing outlier filting
    on categorical type variables. Note that this method assume
    all columns passed are of type 'category', and they if not already
    will be cast first to pandas data type 'category'.

    Note: NaN values will be skipped. If desired to treat them as a class,
    use the method nan_to_class to first. It is worth noting further that
    this method will not work on data files.

    This method operates on each of the columns specified by scope
    independently. In the case that multiple columns are passed, then the
    overlap of all outliers from each column will dropped after all
    have been calculated (therefore the order won't matter).

    Parameters
    -----------
    drop_percent : float, optional
        This parameter acts as a percentage threshold for dropping
        categories when loading categorical data. This parameter
        should be passed as a percent, such that a category
        will be dropped if it makes up less than that % of the data points.
        For example:

        ::

            drop_percent = 1

        In this case any data points within the
        relevant categories as specified by scope
        with a category constituing less than 1% of total
        vali data points will be dropped (or set to NaN if drop=False).

    scope : :ref:`Scope`, optional
        A BPt style :ref:`Scope` used to select a subset of
        columns in which to percent this outlier filtering by.

        By default this is set to only the 'category' style data.

        ::

            default = 'category'

    drop : bool, optional
        By default this function will drop any subject's that are
        determined as outliers. On the otherhand, you
        may instead set specific outlier values as NaN values instead.
        To do this, you can pass drop=False here, and now those specific
        values identified as outliers will be replaced with NaN.

        ::

            default = True

    inplace : bool, optional
        If this operation should take place on
        the original object (inplace = True),
        or if it should be done on a copy of the Dataset
        object (inplace = False).

        If done inplace, then None will be returned.
        If done with inplace = False, then a copy
        of the Dataset with the operation applied
        will be returned.

        ::

            default = True

    '''

    # Check scope and role
    self._check_sr()

    if not inplace:
        df_copy = self.copy(deep=False)
        df_copy.filter_categorical_by_percent(drop_percent=drop_percent,
                                              scope=scope, drop=drop,
                                              inplace=True)
        return df_copy

    # Get cols from scope
    cols = self._get_cols(scope)

    # Check for data files
    self._data_file_fail_check(cols)

    # For if to drop
    all_to_drop = set()

    # Divide drop percent by 100
    dp = drop_percent / 100

    for col in cols:

        # Make sure categorical
        self._add_scope(col, 'category')

        # Extract non-nan values
        values = self._get_values(col, dropna=True)

        # Get to drop subjects
        unique_vals, counts = np.unique(values, return_counts=True)
        drop_inds = np.where((counts / len(values)) < dp)

        drop_vals = unique_vals[drop_inds]
        to_drop_index = values.index[values.isin(drop_vals)]

        # Drop or NaN
        self._drop_or_nan(col=col, to_drop_index=to_drop_index,
                          all_to_drop=all_to_drop, drop=drop)

    # Drop now, if to drop
    if drop:
        self._drop_subjects(all_to_drop)


def drop_id_cols(self, scope='all', inplace=True):
    '''This method will drop any str-type / object type columns
    where the number of unique columns is equal
    to the length of the dataframe.

    Parameters
    ----------
    scope : :ref:`Scope`, optional
        A BPt style :ref:`Scope` used to select a subset of
        columns in which to check.

        By default this is set to 'all' and will check all
        loaded columns

        ::

            default = 'all'

    inplace : bool, optional
        If this operation should take place on
        the original object (inplace = True),
        or if it should be done on a copy of the Dataset
        object (inplace = False).

        If done inplace, then None will be returned.
        If done with inplace = False, then a copy
        of the Dataset with the operation applied
        will be returned.

        ::

            default = True
    '''

    if not inplace:
        df_copy = self.copy(deep=False)
        df_copy.drop_id_cols(scope=scope, inplace=True)
        return df_copy

    # Get cols from scope
    cols = self.get_cols(scope)

    to_drop = []

    for col in cols:
        if self[col].dtype.name == 'object':
            if len(self[col].unique()) == len(self):
                to_drop.append(col)

    self._drop_cols(to_drop)


def drop_duplicate_cols(self, scope='all', inplace=True):
    '''This method is used for checking to see if there are
    any columns loaded with duplicate values. If there is, then
    one of the duplicates will be dropped.

    scope : :ref:`Scope`, optional
        A BPt style :ref:`Scope` used to select a subset of
        columns in which to check.

        By default this is set to 'all' and will check all
        loaded columns

        ::

            default = 'all'

    inplace : bool, optional
        If this operation should take place on
        the original object (inplace = True),
        or if it should be done on a copy of the Dataset
        object (inplace = False).

        If done inplace, then None will be returned.
        If done with inplace = False, then a copy
        of the Dataset with the operation applied
        will be returned.

        ::

            default = True
    '''

    if not inplace:
        df_copy = self.copy(deep=False)
        df_copy.drop_duplicate_cols(scope=scope, inplace=True)
        return df_copy

    # Get cols from scope
    cols = self.get_cols(scope)

    to_drop = []
    for col1, col2 in combinations(cols, 2):
        if np.array_equal(self[col1], self[col2]):
            to_drop.append(col2)

    self._drop_cols(to_drop)


def drop_cols(self, exclusions=None,
              inclusions=None,
              scope='all', inplace=True):
    '''This method is designed to allow
    dropping columns based on some flexible arguments.
    Essentially, exclusions, inclusions and scope are
    all scope style arguments for selecting subsets of columns,
    and the way they are composed to specify dropping a column
    is as follows:

    For any given column, if within the columns selected
    by scope and EITHER in the passed exclusions columns or
    not in the passed inclusions columns, drop that column.

    exclusions : :ref:`Scope`, optional
        A BPt style :ref:`Scope` used to select a subset of
        columns in which if a column is in the selected
        subset of columns, it will be dropped (though if scope is
        set, then only those columns within scope will be checked to
        see if inside of the passed inclusions here.)


    inclusions : :ref:`Scope`, optional
        A BPt style :ref:`Scope` used to select a subset of
        columns in which if a column is not in the selected
        subset of columns, it will be dropped (though if scope is
        set, then only those columns within scope will be checked to
        see if outside of the passed inclusions here.)


    scope : :ref:`Scope`, optional
        A BPt style :ref:`Scope` used to select a subset of
        columns in which to apply the combination of exclusions
        and inclusions to. This is allows for performing inclusions
        or exclusions on a subset of columns.

        By default this is set to 'all' and will consider all
        loaded columns.

        ::

            default = 'all'

    inplace : bool, optional
        If this operation should take place on
        the original object (inplace = True),
        or if it should be done on a copy of the Dataset
        object (inplace = False).

        If done inplace, then None will be returned.
        If done with inplace = False, then a copy
        of the Dataset with the operation applied
        will be returned.

        ::

            default = True
    '''

    if not inplace:
        df_copy = self.copy(deep=False)
        df_copy.drop_cols(exclusions=exclusions,
                          inclusions=inclusions,
                          scope=scope, inplace=True)
        return df_copy

    # Get cols from scope
    cols = self.get_cols(scope)

    if exclusions is not None:
        exclusions = self.get_cols(exclusions)
    else:
        exclusions = []

    if inclusions is not None:
        inclusions = self.get_cols(inclusions)
    else:
        inclusions = cols

    to_drop = []
    for col in cols:
        if col in exclusions or col not in inclusions:
            to_drop.append(col)

    self._drop_cols(to_drop)


def drop_cols_by_unique_val(self, threshold=1, scope='all',
                            dropna=True, inplace=True):
    '''This method will drop any columns with less than or equal to
    the number of unique values.
    This is a coarse filtering method for removing
    uninformative variables which may have been
    loaded or included with a dataset.

    Parameters
    -----------
    threshold : int, optional
        The threshold in which a column should be dropped
        if it has less than or equal to this number of
        unique values.

        For example, the default value of 1, will
        only drop column with only 1 unique value.

        ::

            default = 1

    scope : :ref:`Scope`, optional
        A BPt style :ref:`Scope` used to select a subset of
        columns in which to check for under the unique value threshold.

        By default this is set to 'all' and will check all
        loaded columns

        ::

            default = 'all'

    dropna : bool, optional
        This parameter controls if NaN values are
        ignored when determining how many unique values there
        are, when set to True, and if set to False will
        count NaN as a unique value.

        ::

            default = True

    inplace : bool, optional
        If this operation should take place on
        the original object (inplace = True),
        or if it should be done on a copy of the Dataset
        object (inplace = False).

        If done inplace, then None will be returned.
        If done with inplace = False, then a copy
        of the Dataset with the operation applied
        will be returned.

        ::

            default = True
    '''

    # Check scope and role
    self._check_sr()

    if not inplace:
        df_copy = self.copy(deep=False)
        df_copy.drop_cols_by_unique_val(threshold=threshold, scope=scope,
                                        dropna=dropna, inplace=True)
        return df_copy

    # Get cols from scope
    cols = self._get_cols(scope)

    to_drop = []
    for col in cols:
        values = self._get_values(col, dropna=dropna)
        if len(values.unique()) <= threshold:
            to_drop.append(col)

    self._drop_cols(to_drop)


def drop_cols_by_nan(self, threshold=.5, scope='all', inplace=True):
    '''This method is used for dropping columns based on
    the amount of missing values found across all subjects.
    Each column is dropped if it has greater than or equal
    to the passed threshold of NaN values.

    Parameters
    -----------
    threshold : float or int, optional
        Passed as a float between 0 and 1, or
        as an int. If greater than 0 or less than 1,
        this parameter represents the threshold in which
        a column should be dropped if it has greater than or
        equal to this percent of its columns as NaN values.

        If greater than 1, then this threshold represents the
        absolute value in which if a column has that number
        of subjects or greater with NaN, it will be dropped.

        For example, if a column has 3 values and 7 NaN values,
        passing .7 or lower here would drop this column, but passing
        anything above .7, e.g., .8 would not.

        ::

            default = .5

    scope : :ref:`Scope`, optional
        A BPt style :ref:`Scope` used to select a subset of
        columns in which to check for under the NaN value threshold.

        By default this is set to 'all' and will check all
        loaded columns

        ::

            default = 'all'

    inplace : bool, optional
        If this operation should take place on
        the original object (inplace = True),
        or if it should be done on a copy of the Dataset
        object (inplace = False).

        If done inplace, then None will be returned.
        If done with inplace = False, then a copy
        of the Dataset with the operation applied
        will be returned.

        ::

            default = True
    '''

    # Check scope and role
    self._check_sr()

    if not inplace:
        df_copy = self.copy(deep=False)
        df_copy.drop_cols_by_nan(threshold=threshold,
                                 scope=scope, inplace=True)
        return df_copy

    # Get cols from scope
    cols = self._get_cols(scope)

    # Change threshold from percent to abs
    if threshold > 0 and threshold < 1:
        threshold = threshold * self.shape[0]
        self._print('Setting NaN threshold to:', threshold, level=1)

    to_drop = []
    for col in cols:
        values = self._get_values(col, dropna=False)
        nan_percent = values.isnull().sum()

        if nan_percent >= threshold:
            to_drop.append(col)

    self._drop_cols(to_drop)
