import numpy as np
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from .helpers import (add_new_categories, get_str_round)
from operator import add
from functools import reduce


def _add_new_copy(self, old, new):

    # Copy as new key
    self[new] = self[old].copy()

    # Try to copy scope and role too
    try:
        self.scopes[new] = self.scopes[old].copy()
    except KeyError:
        pass

    try:
        self.roles[new] = self.roles[old]
    except KeyError:
        pass


def to_binary(self, scope, drop=True, inplace=False):
    '''This method works by setting all
    columns within scope to just two binary
    categories. This works by setting the two values
    as the top two categories, any others will have
    their subjects either dropped or replaced with NaN.

    Note: This function will not work on data files.

    See: :func:`binarize <Dataset.binarize>` for converting
    float style columns to binary.

    Parameters
    ----------
    scope : :ref:`Scope`
        A BPt style :ref:`Scope` used to select a subset of
        columns in which to apply to_binary on.

    drop : bool, optional
        If set to True, default, then if more
        than two categories are found when converting
        a column to binary, then the subjects / rows
        with these extra values will be dropped from
        the Dataset. If False, then these values will
        be set to NaN and no rows dropped.

        ::

            default = True

    inplace : bool, optional
        If True, do operation inplace and return None.

        ::

            default = False
    '''

    if not inplace:
        return self._inplace('to_binary', locals())

    # Make sure encoders init'ed
    self._check_encoders()

    # Check scope and role
    self._check_sr()

    # Get cols from scope
    cols = self._get_cols(scope)

    # Check for data files
    self._data_file_fail_check(cols)

    # Binarize each
    for col in cols:
        self._base_binarize(col=col, drop=drop)

    # Update scopes
    self._check_scopes()


def _base_binarize(self, col, drop):

    # Extract non-nan values / series
    values = self._get_values(col, dropna=True)

    # Get value counts
    value_counts = values.value_counts()

    # If only 1 values
    if len(value_counts) == 1:
        self._print('Warning: to_binary ' + repr(col) + ' was '
                    'passed with only 1 unique value.')

    # Assuming should be binary, so 2 unique values
    if len(value_counts) > 2:

        # Select all but top two by count to drop
        drop_vals = value_counts.sort_values(ascending=False).index[2:]

        # Get to drop
        to_drop = values.index[values.isin(drop_vals)]

        # Drop or NaN
        if drop:
            self.drop(list(to_drop), inplace=True)
        else:
            self.loc[to_drop, col] = np.nan

    # Ordinalize
    self._ordinalize(col)


def binarize(self, scope, threshold, replace=True, drop=True, inplace=False):
    '''This method contains a utilities for binarizing a variable.
    These are dichotomizing an existing variable with parameter
    threshold, and applying binarization via two thresholds
    (essentially chopping out the middle of the distribution).

    Note: This function will not work on data files.

    See Related: :func:`to_binary <Dataset.to_binary>`

    Parameters
    ----------
    scope : :ref:`Scope`
        A BPt style :ref:`Scope` used to select a subset of
        columns in which to apply binarize on.

    threshold : float or (float, float)
        This parameter can be used to either set a single threshold
        where any values less than or equal (>=) to the threshold
        will be set to 0, and any values greater (<) than the threshold
        will be set to 1.

        Alternatively, in the case that a tuple with two
        values is passed, e.g., (5, 10), then this requests
        that a lower and upper threshold be set, with any
        values in the middle either dropped or set to NaN
        (as dependent on the drop parameter). The first
        value of the tuple represents the lower threshold,
        where any values less than this threshold will be set to
        0. The second element of the tuple represents the upper
        threshold where any values greater than this threshold will
        be set to 1. Note these equalities are strictly less than
        or greater than, e.g., not less than or equal to.

    replace : bool, optional
        This parameter controls if the original columns
        should be replaced with their binarized version, when
        set to True, and if set to False will add a new
        binarized column as well as leave the original column.
        The new columns will share the name of the original
        columns but with '_binary' appended.

        ::

            default = True

    drop : bool, optional
        If set to True, then any values between lower and upper
        will be dropped. If False, they will be set to NaN.

        Note: This parameter is only relevant if using
        the configuration with parameter's upper and lower.

        ::

            default = True

    inplace : bool, optional
        If True, do operation inplace and return None.

        ::

            default = False
    '''

    if not inplace:
        return self._inplace('binarize', locals())

    # Proc if tuple or not
    if isinstance(threshold, tuple):
        if len(threshold) != 2:
            raise RuntimeError('If a tuple passed for threshold it '
                               'must be of length 2!')

        lower, upper = threshold
        threshold = None

    # Not tuple case
    else:
        lower, upper = None, None

    # Make sure encoders init'ed
    self._check_encoders()

    # Check scope and role
    self._check_sr()

    # Get cols from scope
    cols = self._get_cols(scope)

    # Check for data files
    self._data_file_fail_check(cols)

    # Binarize each column individually
    for col in cols:
        self._binarize(col=col, threshold=threshold, lower=lower,
                       upper=upper, replace=replace, drop=drop)

    # Finalize scope
    self._check_scopes()


def _binarize(self, col, threshold, lower, upper, replace, drop):

    # Keep track of initial NaN subjects
    nan_subjects = self._get_nan_subjects(col)

    # Extract non-nan values / series
    values = self._get_values(col, dropna=True)

    # Add new column
    if not replace:
        new_key = col + '_binary'

        if new_key in self.columns:
            cnt = 1
            while new_key + str(cnt) in self.columns:
                cnt += 1

            new_key += str(cnt)

        # Make a copy under the new key
        self._add_new_copy(old=col, new=new_key)

        # Change col to new_key
        col = new_key

    # If upper and lower passed instead of threshold
    if threshold is None:

        # Determine subjects that are between the values
        to_drop = values[(values <= upper) &
                         (values >= lower)].index

        # If drop
        if drop:
            self.drop(to_drop, axis=0, inplace=True)

        # Binarize
        self[col] = self[col].where(self[col] > lower, 0)
        self[col] = self[col].where(self[col] < upper, 1)

        # If not dropping, replace as NaN here
        if not drop:
            self.loc[to_drop, col] = np.nan

        # Add to name map
        self.encoders[col] =\
            {0: '<' + str(lower), 1: '>' + str(upper)}

    # If a single threshold
    else:

        # Binarize
        self[col] = self[col].where(self[col] >= threshold, 0)
        self[col] = self[col].where(self[col] < threshold, 1)

        # Add to name map
        self.encoders[col] =\
            {0: '<' + str(threshold), 1: '>=' + str(threshold)}

    # Make sure NaN's are filled in
    self.loc[nan_subjects, col] = np.nan

    # Make sure category scope
    self._add_scope(col, 'category')


def k_bin(self, scope, n_bins=5, strategy='uniform', inplace=False):
    '''This method is used to apply k binning to
    a column, or columns. On the backend
    this function used the scikit-learn
    KBinsDiscretizer.

    Parameters
    ----------
    scope : :ref:`Scope`
        A BPt style :ref:`Scope` used to select a subset of
        columns in which to apply k bins to.

    n_bins : int, optional
        The number of bins to discretize the passed
        columns to. This same value is applied for
        all columns within scope.

        ::

            default = 5

    strategy : {'uniform', 'quantile', 'kmeans'}, optional
        The strategy in which the binning should be adhere to.
        Options are:

        - 'uniform'
            All bins in each feature have identical widths.

        - 'quantile'
            All bins in each feature have the same number of points.

        - 'kmeans'
            Values in each bin have the same nearest center of a 1D
            k-means cluster.

        ::

            default = 'uniform'

    inplace : bool, optional
        If True, do operation inplace and return None.

        ::

            default = False
    '''

    if not inplace:
        return self._inplace('k_bin', locals())

    # Check scope and role
    self._check_sr()

    # Get cols from scope
    cols = self._get_cols(scope)

    # K bin each requested column individually
    for col in cols:
        self._k_bin(col, n_bins, strategy)

    # Make sure scope updated
    self._check_scopes()


def _k_bin(self, col, n_bins, strategy):

    # Check encoders init'ed
    self._check_encoders()

    # Get non-nan values
    all_values = self[col]
    non_nan_subjects = all_values[~all_values.isna()].index
    values = all_values.loc[non_nan_subjects]

    # Prepare values for encoding
    values = np.array(values.astype('float')).reshape(-1, 1)

    # Perform encoding
    kbin_encoder = KBinsDiscretizer(n_bins=n_bins,
                                    encode='ordinal',
                                    strategy=strategy)
    new_values = np.squeeze(kbin_encoder.fit_transform(values))

    self._print('k_bin encoded col:', col, 'with:',
                kbin_encoder.n_bins, 'bins and strategy:',
                strategy, level=1)

    # Add any needed new categories to the column
    add_new_categories(self[col], new_values)

    # Replace with new values in place
    self.loc[non_nan_subjects, col] = new_values

    # Create encoder information
    encoder = {}
    bin_edges = kbin_encoder.bin_edges_[0]

    for i in range(len(bin_edges)-1):

        # Get rounded start and end values
        start = get_str_round(bin_edges[i], places=3)
        end = get_str_round(bin_edges[i+1], places=3)

        # Set ordinal index to str name of range values
        encoder[i] = start + ' - ' + end

    # Save in encoders
    self.encoders[col] = encoder

    # Make sure col is category type
    self._add_scope(col, 'category')


def ordinalize(self, scope, nan_to_class=False, inplace=False):
    '''This method is used to ordinalize
    a group of columns. Ordinalization is
    performed by setting all n unique
    categories present in each column to
    values 0 to n-1.

    The LabelEncoder from sklearn is used
    on the backend for this operation.

    Parameters
    -----------
    scope : :ref:`Scope`
        A BPt style :ref:`Scope` used to select a subset of
        columns in which to apply ordinalize to.

    nan_to_class : bool, optional
        If set to True, then treat NaN values as
        as a unique class, otherwise if False then
        ordinalization will be applied on just non-NaN
        values, and any NaN values will remain NaN.

        See: :func:`nan_to_class <Dataset.nan_to_class>`
        for more generally adding NaN values as a new
        category to any arbitrary categorical column.
        All this parameter does is if True calls
        self.nan_to_class after normal ordinalization.

        ::

            default = False

    inplace : bool, optional
        If True, do operation inplace and return None.

        ::

            default = False
    '''

    if not inplace:
        return self._inplace('ordinalize', locals())

    # Check scope and role
    self._check_sr()

    # Get cols from scope
    cols = self._get_cols(scope)

    # Ordinalize each column individually
    for col in cols:
        self._ordinalize(col)

    # Optionally add NaN as class
    if nan_to_class:
        self.nan_to_class(scope=scope, inplace=True)

    # Make sure scope updated
    self._check_scopes()


def _ordinalize(self, col):

    # Check encoders init'ed
    self._check_encoders()

    # Get non-nan values
    all_values = self[col]
    non_nan_subjects = all_values[~all_values.isna()].index
    values = all_values.loc[non_nan_subjects]

    # Perform actual encoding
    label_encoder = LabelEncoder()
    new_values = label_encoder.fit_transform(np.array(values))

    # Add any needed new categories to the column
    add_new_categories(self[col], new_values)

    # Replace values in place
    self.loc[non_nan_subjects, col] = new_values

    # Convert label encoder to encoder
    encoder = {}
    for i, c in enumerate(label_encoder.classes_):
        encoder[i] = c

    # Save in encoders
    self.encoders[col] = encoder

    # Make sure col is category type
    self._add_scope(col, 'category')


def nan_to_class(self, scope='category', inplace=False):
    '''This method will cast any columns that were not categorical that are
    passed here to categorical. Will also ordinally encode them if
    they have not already been encoded
    (e.g., by either via ordinalize, binarize, or a similar function...).

    scope : :ref:`Scope`
        A BPt style :ref:`Scope` used to select a subset of
        columns in which to apply nan_to_class.

        ::

            default = 'category'

    inplace : bool, optional
        If True, do operation inplace and return None.

        ::

            default = False
    '''

    if not inplace:
        return self._inplace('nan_to_class', locals())

    # Check scope and roles
    self._check_sr()

    # Make sure encoders are up init'ed
    self._check_encoders()

    # For each col in scope
    cols = self._get_cols(scope)
    for col in cols:

        # If this column hasn't been encoded yet, ordinalize
        if col not in self.encoders:
            self._ordinalize(col)

        # Get nan subjects
        all_values = self[col]
        nan_subjects = all_values[all_values.isna()].index

        # If no nan subjects, skip
        if len(nan_subjects) == 0:
            continue

        # Get non-nan values
        non_nan_subjects = all_values[~all_values.isna()].index
        values = all_values.loc[non_nan_subjects]

        # Add nan class as next avaliable
        nan_class = np.max(values.astype(int)) + 1
        add_new_categories(self[col], [nan_class])
        self.loc[nan_subjects, col] = nan_class

        # Update encoder entry with NaN
        self.encoders[col][nan_class] = np.nan

    # Make sure scopes updated
    self._check_scopes()


def copy_as_non_input(self, col, new_col, copy_scopes=True, inplace=False):
    '''This method is a used for making a copy of an
    existing column, ordinalizing it and then setting it
    to have role = non input.

    Parameters
    ----------
    col : str
        The name of the loaded column to make a copy of.

    new_col : str
        The new name of the non input and ordinalized column.

    copy_scopes : bool, optional
        If the associated scopes with the original column should be copied
        over as well. If False, then the new col will
        only have scope 'category'.

        Note: The scopes will be copied before ordinalizing,
        s.t., the new copy will have the scope 'category'
        regardless of if that was a scope of the original variable.

        ::

            default = True

    inplace : bool, optional
        If True, do operation inplace and return None.

        ::

            default = False
    '''

    if not inplace:
        return self._inplace('copy_as_non_input', locals())

    # Copy as new col
    self[new_col] = self[col].copy()

    # Update scope and role
    self._check_sr()

    # Only copy scopes if requested, and do before
    # ordinalize call
    if copy_scopes:
        self.scopes[new_col] = self.scopes[col].copy()

    # Ordinalize
    self._ordinalize(new_col)

    # Set new role
    self.set_role(new_col, 'non input', inplace=True)

    # Make sure scopes updated
    self._check_scopes()


def add_unique_overlap(self, cols, new_col, decode_values=True, inplace=False):
    '''This function is designed to add a new column
    with the overlapped unique values from passed two or more columns.
    For example, say you had two binary columns, A and B. This function
    would compute a new column with 4 possible values, where:

    ::

        # If these are the values in the dataset
        A == 0 and B == 0, A == 0 and B == 1,
        A == 1 and B == 0 and A == 1 and B == 1

        # After calling,
        self.add_unique_overlap(['A', 'B'], new_col='new')

        # This new column 'new' will look like below, before encoding.

        0    A=0 B=0
        1    A=0 B=1
        2    A=1 B=1
        3    A=0 B=0
        ...

        # After encoding, i.e., the end of the function, it will be

        0    0
        1    1
        2    2
        3    0
        ...

    The new added column will be default be added with role data,
    except if all of the passed cols have a different role. In the
    case that all of the passed cols have the same role, the new
    col will share that role.

    Simmilar to role, the scope of the new column will be the overlap
    of shared scopes from all of the passed new_col. If no overlap,
    then no scope.

    Parameters
    -----------
    cols : list of str
        The names of the columns to compute the overlap with.
        E.g., in the example above, cols = ['A', 'B'].

        Note: You must pass atleast two columns here.

    new_col : str
        The name of the new column where these values will be stored.

    decode_values : bool, optional
        This is an optional parameter, which is set to True
        will when creating the overlapping values will try to
        replace values with the encoded value (if any). For example
        if a variable being added had an originally encoded values of
        'cat' and 'dog', then the replace value before ordinalization
        would be col_name=cat and col_name=dog, vs. if set to False
        would have values of col_name=0 and col_name=1.

        ::

            default = True

    inplace : bool, optional
        If True, do operation inplace and return None.

        ::

            default = False

    '''

    if not inplace:
        return self._inplace('add_unique_overlap', locals())

    # Make sure up to date
    self._check_sr()

    # Some input validation, make sure str
    if isinstance(cols, str):
        raise RuntimeError('You must pass cols as a list or array-like.')

    # Make sure 2 or more cols
    if len(cols) < 2:
        raise RuntimeError('cols must be of length 2 or more.')

    # Make sure passed cols exist
    for col in cols:
        if col not in list(self):
            raise KeyError('Passed col: ' + col + ' is not a valid '
                           'loaded column.')

    # Make sure new col doesn't exist
    if new_col in list(self):
        raise KeyError('Passed new col: ' + new_col + ' already exists!')

    # Generate a list of modified series
    combo = []
    for col in cols:

        vals = self._get_values(col, dropna=False,
                                decode_values=decode_values)
        combo.append(col + '=' + vals.astype(str) + ' ')

    # Combine
    combo = reduce(add, combo)

    # Add as new column
    self[new_col] = combo

    # Update scope and role
    self._check_sr()

    # If all roles agree, set new col as
    roles = set([self.roles[col] for col in cols])
    if len(roles) == 1:
        self._set_role(new_col, roles.pop())

    # Add scope if in all passed
    scopes = [self.scopes[col] for col in cols]
    for scope in scopes[0]:
        in_rest = [scope in other for other in scopes[1:]]
        if all(in_rest):
            self._add_scope(new_col, scope)

    # Lastly, ordinalize
    self._ordinalize(new_col)

    # Make sure scopes updated
    self._check_scopes()