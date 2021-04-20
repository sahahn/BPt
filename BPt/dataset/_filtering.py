import numpy as np
from itertools import combinations
from .helpers import proc_fop
from pandas.util._decorators import doc
from .Dataset import _shared_docs, _sip_docs, _file_docs


def _drop_subjects(self, subjects):

    if len(subjects) > 0:
        self.drop(subjects, axis=0, inplace=True)
        self._print('Dropped', len(subjects), 'Rows', level=1)
        self._print('Dropped Rows:', subjects, level=2)


@doc(**_sip_docs)
def drop_nan_subjects(self, scope, inplace=False):
    '''This method is used for
    dropping all of the subjects which have NaN
    values for a given scope / column.

    Parameters
    ----------
    {scope}

    {inplace}

    Examples
    ---------

    .. ipython:: python

        data = bp.read_csv('data/example1.csv')
        data
        data.drop_nan_subjects(scope='all')
        data.drop_nan_subjects(scope='numbers')

    '''

    if not inplace:
        return self._inplace('drop_nan_subjects', locals())

    cols = self.get_cols(scope)

    if len(cols) == 0:
        raise RuntimeError(f'No columns found for scope: {scope}.')

    nan_subjects = self._get_nan_subjects(cols[0])
    for col in cols[1:]:
        nan_subjects = nan_subjects.union(self._get_nan_subjects(col))

    # Drop all nan subjects
    self._drop_subjects(nan_subjects)


_si_docs = {'subjects': _shared_docs['subjects'],
            'inplace': _shared_docs['inplace']}


@doc(**_si_docs)
def apply_inclusions(self, subjects, inplace=False):
    '''This method will drop all subjects
    that do not overlap with the passed subjects to
    this function. In this sense, this method acts
    like a whitelist, where you could pass for example
    only valid subjects that passed some QC, and all
    other's loaded will be dropped.

    Parameters
    -----------
    {subjects}

    {inplace}

    '''

    if not inplace:
        return self._inplace('apply_inclusions', locals())

    # Load inclusions
    inclusions = self.get_subjects(subjects, return_as='set')

    if len(inclusions) > 0:
        to_drop = set(self.index) - inclusions
        self.drop(list(to_drop), axis=0, inplace=True)


@doc(**_si_docs)
def apply_exclusions(self, subjects, inplace=False):
    '''This method will drop all subjects
    that overlap with the passed subjects to
    this function.

    Parameters
    -----------
    {subjects}

    {inplace}

    '''

    if not inplace:
        return self._inplace('apply_exclusions', locals())

    # Load exclusions
    exclusions = self.get_subjects(subjects, return_as='set')

    if len(exclusions) > 0:
        to_drop = set(self.index).intersection(exclusions)
        self.drop(list(to_drop), axis=0, inplace=True)


@doc(**_sip_docs)
def drop_subjects_by_nan(self, scope='all', threshold=.5, inplace=False):
    '''This method is used for dropping subjects based on
    the amount of missing values found across a subset of
    columns as selected by scope. Each subject is dropped
    if it has greater than or equal to the passed threshold
    of NaN values.

    Parameters
    -----------
    {scope}
        ::

            default = 'all'

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

    {inplace}

    '''

    if not inplace:
        return self._inplace('drop_subjects_by_nan', locals())

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


_outlier_docs = {**_sip_docs, **_file_docs, 'drop': _shared_docs['drop']}


@doc(**_outlier_docs)
def filter_outliers_by_percent(self, scope='float', fop=1, drop=True,
                               reduce_func=np.mean, n_jobs=-1, inplace=False):
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
    {scope}
        ::

            default = 'float'

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

    {drop}

    {reduce_func}

    {n_jobs}

    {inplace}

    '''

    if not inplace:
        return self._inplace('filter_outliers_by_percent', locals())

    # Check scope and role
    self._check_sr()

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


@doc(**_outlier_docs)
def filter_outliers_by_std(self, scope='float', n_std=10, drop=True,
                           reduce_func=np.mean, n_jobs=-1, inplace=False):
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
    {scope}
        ::

            default = 'float'

    n_std : float, tuple, optional
        This value is used to set an outlier threshold by
        standrad deviation. For example if passed n_std = 10,
        then it will be converted internally to (10, 10).
        This parameter determines outliers as
        data points within each
        relevant column (as determined by the scope argument) where their
        value is less than the mean of the
        column - n_std[0] * the standard deviation of the column,
        and greater than the mean of the column + n_std[1] * the standard
        deviation of the column.

        If a single number is passed, that number is applied to
        both the lower
        and upper range.
        If a tuple with None on one side is passed, e.g.
        (None, 3), then nothing will be taken off
        that lower or upper bound.

        ::

            default = 10

    {drop}

    {reduce_func}

    {n_jobs}

    {inplace}

    Examples
    ----------

    If we define a dataset, we can check the std.

    .. ipython:: python

        import BPt as bp
        import numpy as np
        data = bp.Dataset()
        data.verbose = 1
        data['1'] = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
        np.std(data['1']), np.mean(data['1'])

    We can now see how different thresholds work.

    .. ipython:: python

        # This won't drop anything
        data.filter_outliers_by_std(n_std=2)

        # This will
        data.filter_outliers_by_std(n_std=1)

    What if there was more than one column?

    .. ipython:: python

        data['2'] = [1, 1, 1, 1, 10, 1, 1, 1, 1, 1]

        # Now a subject will be dropped
        data.filter_outliers_by_std(n_std=2)
        data.filter_outliers_by_std(n_std=1)

    We can also apply it only to one column, and instead of dropping
    subjects, replace outliers with NaN's

    .. ipython:: python

        data.filter_outliers_by_std(n_std=1, scope='1', drop=False)

    '''

    if not inplace:
        return self._inplace('filter_outliers_by_std', locals())

    # Check scope and role
    self._check_sr()

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


@doc(**_sip_docs, drop=_shared_docs['drop'])
def filter_categorical_by_percent(self, scope='category', drop_percent=1,
                                  drop=True, inplace=False):
    '''This method is designed to allow performing outlier filtering
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
    {scope}
        ::

            default = 'category'

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
        with a category constituting less than 1% of total
        valid data points will be dropped (or set to NaN if drop=False).

    {drop}

    {inplace}

    '''

    if not inplace:
        return self._inplace('filter_categorical_by_percent', locals())

    # Check scope and role
    self._check_sr()

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


@doc(**_sip_docs)
def drop_id_cols(self, scope='all', inplace=False):
    '''This method will drop any str-type / object type columns
    where the number of unique columns is equal
    to the length of the dataframe.

    Parameters
    ----------
    {scope}
        ::

            default = 'all'

    {inplace}

    '''

    if not inplace:
        return self._inplace('drop_id_cols', locals())

    # Get cols from scope
    cols = self.get_cols(scope)

    to_drop = []

    for col in cols:
        if self[col].dtype.name == 'object':
            if len(self[col].unique()) == len(self):
                to_drop.append(col)

    self._drop_cols(to_drop)


@doc(**_sip_docs)
def drop_duplicate_cols(self, scope='all', inplace=False):
    '''This method is used for checking to see if there are
    any columns loaded with duplicate values. If there is, then
    one of the duplicates will be dropped.

    {scope}
        ::

            default = 'all'

    {inplace}

    '''

    if not inplace:
        return self._inplace('drop_duplicate_cols', locals())

    # Get cols from scope
    cols = self.get_cols(scope)

    to_drop = []
    for col1, col2 in combinations(cols, 2):
        if np.array_equal(self[col1], self[col2]):
            to_drop.append(col2)

    self._drop_cols(to_drop)


@doc(**_sip_docs)
def drop_cols(self,
              scope='all',
              exclusions=None,
              inclusions=None,
              inplace=False):
    '''This method is designed to allow
    dropping columns based on some flexible arguments.
    Essentially, exclusions, inclusions and scope are
    all scope style arguments for selecting subsets of columns,
    and the way they are composed to specify dropping a column
    is as follows:

    For any given column, if within the columns selected
    by scope and EITHER in the passed exclusions columns or
    not in the passed inclusions columns, drop that column.

    Parameters
    ------------
    {scope}
        ::

            default = 'all'

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

    {inplace}

    Examples
    ---------

    Given the below dataset, we could try dropping some different columns.
    Each drop will be done on a copy of the Dataset.

    .. ipython:: python

        import BPt as bp
        data = bp.Dataset(columns=['cat1', 'cat2', 'cat3', 'dog1', 'dog2'])
        list(data)

        list(data.drop_cols(exclusions='cat'))
        list(data.drop_cols(exclusions='2', scope='cat'))
        list(data.drop_cols(inclusions=['1', '2'], scope='cat'))
        list(data.drop_cols(inclusions='dog'))

    '''

    if not inplace:
        return self._inplace('drop_cols', locals())

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


@doc(**_sip_docs)
def drop_cols_by_unique_val(self, scope='all', threshold=1,
                            dropna=True, inplace=False):
    '''This method will drop any columns with less than or equal to
    the number of unique values.
    This is a coarse filtering method for removing
    uninformative variables which may have been
    loaded or included with a dataset.

    Parameters
    -----------
    {scope}
        ::

            default = 'all'

    threshold : int, optional
        The threshold in which a column should be dropped
        if it has less than or equal to this number of
        unique values.

        For example, the default value of 1, will
        only drop column with only 1 unique value.

        ::

            default = 1

    dropna : bool, optional
        This parameter controls if NaN values are
        ignored when determining how many unique values there
        are, when set to True, and if set to False will
        count NaN as a unique value.

        ::

            default = True

    {inplace}
    '''

    if not inplace:
        return self._inplace('drop_cols_by_unique_val', locals())

    # Check scope and role
    self._check_sr()

    # Get cols from scope
    cols = self._get_cols(scope)

    to_drop = []
    for col in cols:
        values = self._get_values(col, dropna=dropna)
        if len(values.unique()) <= threshold:
            to_drop.append(col)

    self._drop_cols(to_drop)


@doc(**_sip_docs)
def drop_cols_by_nan(self, scope='all', threshold=.5, inplace=False):
    '''This method is used for dropping columns based on
    the amount of missing values found across all subjects.
    Each column is dropped if it has greater than or equal
    to the passed threshold of NaN values.

    Parameters
    -----------
    {scope}
        ::

            default = 'all'

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

    {inplace}

    '''

    if not inplace:
        return self._inplace('drop_cols_by_nan', locals())

    # Check scope and role
    self._check_sr()

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
