import pandas as pd
import numpy as np
from .data_file import load_data_file_proxy
from copy import deepcopy
from .helpers import verbose_print
from pandas.util._decorators import doc
import pandas.core.common as com
from pandas.core.dtypes.generic import ABCMultiIndex


_shared_docs = {}
_shared_docs['scope'] = """scope : :ref:`Scope`
            A BPt style :ref:`Scope` used to select a subset of
            column(s) in which to apply the current function to.
            See :ref:`Scope` for more information on how this
            can be applied.
"""

_shared_docs['inplace'] = """inplace : bool, optional
            If True, perform the current function inplace and return None.

            ::

                default = False

"""
_sip_docs = _shared_docs.copy()

_shared_docs['load_func'] = """load_func : python function, optional
            | Fundamentally columns of type 'data file' represent
              a path to a saved file, which means you must
              also provide some information on how to load the saved file.
              This parameter is where that loading function should be passed.
              The passed `load_func` will be called on each file individually
              and whatever the output of the function is will be passed to
              the different loading functions.

            | You might need to pass a user defined custom function
              in some cases, e.g., you want to use :func:`numpy.load`,
              but then also :func:`numpy.stack`.
              Just wrap those two functions in one,
              and pass the new function.

            ::

                def my_wrapper(x):
                    return np.stack(np.load(x))

            | Note that in this case where a custom function is defined
              it is reccomended that you define this function in
              a separate file from where the main script will be run,
              and then import the function.

            | By default this function assumes data files are passed
              as numpy arrays, and uses the default function
              :func:`numpy.load`, when nothing else is specified.

            ::

                default = np.load

"""

_file_docs = {}
_file_docs['reduce_func'] = '''reduce_func : python function, optional
            The passed python function will be applied only if
            the requested col/column is a 'data file'. In the case
            that it is, the function should accept as input
            the data from one data file, and should return a single
            scalar value. For example, the default value is
            numpy's mean function, which returns one value.

            ::

                default = np.mean

'''

_file_docs['n_jobs'] = '''n_jobs : int, optional
            As with reduce_func, this parameter is only
            valid when the passed col/column is a 'data file'.
            In that case, this specifies the number of cores
            to use in loading and applying the reduce_func to each
            data file. This can provide a significant speed up when
            passed the number of avaliable cores, but can sometimes
            be memory intensive depending on the underlying size of the file.

            If set to -1, will try to automatically use all avaliable cores.

            ::

                default = -1

'''

_shared_docs['drop'] = '''drop : bool, optional
        By default this function will drop any subjects / index that are
        determined to be outliers. On the otherhand, you
        may instead set specific outlier values as NaN values instead.
        To do this, set drop=False. Now those specific
        values identified as outliers will be replaced with NaN.

        ::

            default = True

'''

_shared_docs['subjects'] = '''subjects : :ref:`Subjects`
        This argument can be any of the BPt accepted
        subject style inputs. E.g., None, 'nan' for subjects
        with any nan data, the str location of a file
        formatted with one subject per line, or directly an
        array-like of subjects, to name some options.

        See :ref:`Subjects` for all options.
'''


class Dataset(pd.DataFrame):
    '''| The BPt Dataset class is the main class used for preparing data
      into a compatible format to work with machine learning. This class is new
      as of BPt version 2 (replacing the building in loading functions of the
      old BPt_ML).

    | See :ref:`loading_data` for more a comprehensive guide on this object.

    | This class can be initialized like a pandas.DataFrame, or
      typically from a pandas.DataFrame. This class has some constraints
      relative to using DataFrames. Some of these are that columns must be
      strings (if passed as int-like will be cast to strings), and that
      there cannot be duplicate column names.

    | This class can be initialized in most of the same
      ways that a pandas DataFrame can be initialized, for example

    .. ipython:: python

        data = bp.Dataset(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                          columns=['a', 'b', 'c'])
        data

    Or from a pandas DataFrame.

    .. ipython:: python

        import pandas as pd
        df = pd.DataFrame([1, 2, 3], columns=['a'])
        df

        data = bp.Dataset(df)
        data

    The Dataset also has some extra optional constructor parameters:
    roles, scopes, targets and non_inputs, which are just helpers
    for setting parameters at the time of construction. For example:

    .. ipython:: python

        data = bp.Dataset([1, 2, 3], columns=['1'], targets=['1'])
        data.get_roles()

    .. versionadded:: 2.0.0

    '''

    _metadata = ['roles', 'scopes', 'encoders', 'file_mapping',
                 'verbose_', 'test_subjects', 'train_subjects']

    _print = verbose_print

    def __init__(self, data=None, index=None,
                 columns=None, dtype=None,
                 copy=None, roles=None,
                 scopes=None, targets=None,
                 non_inputs=None):

        super().__init__(data=data, index=index,
                         columns=columns, dtype=dtype,
                         copy=copy)

        if roles is not None:
            self.roles = roles
            self._check_roles()

        if scopes is not None:
            self.scopes = scopes
            self._check_scopes()

        if targets is not None:
            self.set_target(targets, inplace=True)
            self._check_roles()

        if non_inputs is not None:
            self.set_non_input(non_inputs, inplace=True)
            self._check_roles()

    def __getitem__(self, key):

        try:
            item = super().__getitem__(key)

        # If the original passed index isn't a key,
        # then try again treating it as a scope.
        except KeyError:
            item = super().__getitem__(self.get_cols(key))

        # If index returns Dataset, copy over metadata
        if isinstance(item, Dataset):
            self._copy_meta_data(item)

        return item

    @property
    def _constructor(self):
        return Dataset

    @property
    def reserved_roles(self):
        '''The dataset class has three fixed and therefore
        reserved (i.e., cannot be set as a scope) roles,
        these are ::

            ['data', 'target', 'non input']

        '''
        return set(['data', 'target', 'non input'])

    @property
    def reservered_scopes(self):
        '''There are a number of reserved fixed scopes
        in the Dataset class, these are ::

            ['all', 'float', 'category', 'data float',
             'data', 'data file', 'data category',
             'non input', 'target', 'target category',
             'target float']
        '''
        return set(['all', 'float', 'category', 'data float',
                    'data', 'data file', 'data category',
                    'non input', 'target', 'target category',
                    'target float'])

    @property
    def verbose(self):
        '''This parameter takes a verbosity level
        as an integer, where level 0 is just warnings,
        a value lower than 0 can be set to mute warnings
        and then higher values for more verbosity.
        E.g., level 1 will print basic information,
        where level 2 or higher will print more extensive information.'''

        if not hasattr(self, 'verbose_'):
            self.verbose_ = 0
        if self.verbose_ is None:
            self.verbose_ = 0

        return self.verbose_

    @verbose.setter
    def verbose(self, verbose):
        self.verbose_ = verbose

    def _inplace(self, func_name, args):
        '''Assumes that inplace is True for this func to be called.'''

        # Create shallow copy
        data_copy = self.copy(deep=False)

        # Get args to pass to copy
        copy_args = {key: args[key] for key in args
                     if key != 'self' and key != 'inplace'}
        copy_args['inplace'] = True

        # Apply func in place
        getattr(data_copy, func_name)(**copy_args)

        # Return copy
        return data_copy

    def _check_test_subjects(self):

        if not hasattr(self, 'test_subjects'):
            self.test_subjects = None

    def _get_test_subjects(self):

        self._check_test_subjects()
        return self.test_subjects

    def _check_train_subjects(self):

        if not hasattr(self, 'train_subjects'):
            self.train_subjects = None

    def _get_train_subjects(self):

        self._check_train_subjects()
        return self.train_subjects

    def _check_file_mapping(self):

        if not hasattr(self, 'file_mapping'):
            self.file_mapping = {}
        elif getattr(self, 'file_mapping') is None:
            self.file_mapping = {}

        # Check to remove any non relevant file mappings
        # e.g., ones that have been dropped
        data_file_cols = self.get_cols(scope='data file')

        # Get all unique files, including if any NaNs
        unique_files = pd.unique(self[data_file_cols].values.ravel())

        # If any NaN's, make sure np.nan is in the file_mapping
        if pd.isnull(unique_files).any():
            self.file_mapping[np.nan] = np.nan

        # Compute the difference
        file_mapping_keys = np.array(list(self.file_mapping))

        # Make sure types are lined up
        if file_mapping_keys.dtype.name != unique_files.dtype.name:
            unique_files = unique_files.astype(file_mapping_keys.dtype.name)

        # Compute difference
        to_remove = np.setdiff1d(file_mapping_keys, unique_files,
                                 assume_unique=True)

        # Remove any difference
        for key in to_remove:
            try:
                self.file_mapping.pop(key)
            except KeyError:
                pass

    def _check_encoders(self):

        if not hasattr(self, 'encoders'):
            self.encoders = {}
        elif getattr(self, 'encoders') is None:
            self.encoders = {}

    def _get_encoders(self):

        self._check_encoders()
        return self.encoders

    def _check_cols(self):
        '''BPt / scopes assume that all columns are of
        type str. This function should be called before
        check roles or check scope. Also checks to make sure
        no columns have one of the reserved role names'''

        col_names = list(self)

        # Check if any col names not str's
        non_str = []
        for col in col_names:
            if not isinstance(col, str):
                non_str.append(col)

        if len(non_str) > 0:

            rename = {c: str(c) for c in non_str}
            super().rename(rename, axis=1, inplace=True)

            self._print('Warning: the columns:', repr(non_str),
                        'were cast to str', level=0)

        # Check if any columns have the same name as a reserved scope
        reserved_overlap = set(self.columns).intersection(
            self.reservered_scopes)
        if len(reserved_overlap) > 0:
            raise RuntimeError('The columns: ' + repr(reserved_overlap) + ' '
                               'overlap with reserved saved names! '
                               'They must be changed.')

        # Check to make sure there are no columns with the same name,
        # i.e., duplicates
        self._check_duplicate_cols()

    def _check_duplicate_cols(self):

        if len(set(self.columns)) != len(self.columns):
            raise RuntimeError('Duplicate columns with the same name ',
                               'are not allowed!')

    def _check_roles(self, check_type=True):

        # Make sure cols type str
        if check_type:
            self._check_cols()

        if not hasattr(self, 'roles'):
            self.roles = {}
        elif getattr(self, 'roles') is None:
            self.roles = {}

        if not isinstance(self.roles, dict):
            raise RuntimeError('roles must be a dict.')

        # Fill in any column without a role.
        for col in list(self.columns):
            if col not in self.roles:
                self._set_role(col, 'data')

        # Remove any saved roles that are not
        # in the data any more.
        for col in list(self.roles):
            if col not in self.columns:
                del self.roles[col]

    def _check_sr(self):
        '''Helper to check scopes and roles'''

        # Only check once when both being checked
        self._check_cols()

        # Check scopes and roles
        self._check_scopes(check_type=False)
        self._check_roles(check_type=False)

    def _check_scopes(self, check_type=True):

        # Make sure cols type str
        if check_type:
            self._check_cols()

        # If doesn't exist, create scopes
        if not hasattr(self, 'scopes'):
            self.scopes = {}

        # Or is set to None
        elif getattr(self, 'scopes') is None:
            self.scopes = {}

        if not isinstance(self.scopes, dict):
            raise RuntimeError('scopes must be a dict.')

        # Make sure each col is init'ed with a scope
        for col in self.columns:
            if col not in self.scopes:
                self.scopes[col] = set()

        # Make sure to get rid of any removed cols
        for col in list(self.scopes):
            if col not in self.columns:
                del self.scopes[col]

        # Compute columns which are categorical + columns with scope category
        dtype_category = set(self.dtypes[self.dtypes == 'category'].index)
        scope_category = set([col for col in self.columns if
                             'category' in self.scopes[col]])

        # End if equal
        if dtype_category == scope_category:
            return

        # Add scope to columns which have dtype category but not scope category
        needs_scope = dtype_category - scope_category
        for col in needs_scope:
            self._add_scope(col, 'category')

        # For any columns which have scope category, but the dtype isn't
        # assume that the scope should be removed
        remove_scope = scope_category - dtype_category
        for col in remove_scope:
            self._remove_scope(col, 'category')

    def get_roles(self):
        ''' This function can be
        used to get a dictionary with the currently
        loaded roles, See :ref:`Role` for more information
        on how roles are defined and used within BPt.

        Returns
        --------
        roles : dict
            Returns the up to date dictionary
            of roles stored in self.roles, where
            each key is a column name and each value
            is the corresponding role.
        '''

        self._check_roles()
        return self.roles

    @doc(**_sip_docs)
    def set_role(self, scope, role, inplace=False):
        '''This method is used to set a role for
        either a single column or multiple, as set
        through the scope parameter. See :ref:`Role`
        for more information about how roles are
        used within BPt.

        Parameters
        ----------
        {scope}

        role : :ref:`Role`
            A valid role in which to set all columns
            in the passed scope to. Input must be
            either 'data', 'target' or 'non input'.
            Note: by default all columns start with role 'data'.

            See :ref:`Role` for more information on how
            each role differs.

        {inplace}

        See Also
        ----------
        set_target : Specifically for setting target role.
        set_non_input : Specifically for setting non input role.
        get_roles : Returns a dictionary with saved roles.

        Examples
        ---------
        Setting columns role's within the :class:`Dataset` is an
        essential part of using the object.

        .. ipython:: python

            data = bp.read_csv('data/example1.csv')
            data = data.set_role('animals', 'target')
            data
            data.get_roles()

        We can also use the method to set columns to role non input,
        which has the additional constraint that no NaN values
        can be present in that column. So we can see below
        that one row is dropped.

        .. ipython:: python

            data = data.set_role('numbers', 'non input')
            data
            data.get_roles()

        '''

        if not inplace:
            return self._inplace('set_role', locals())

        self._check_sr()

        cols = self._get_cols(scope)
        for col in cols:
            self._set_role(col, role)

    def set_roles(self, scopes_to_roles, inplace=False):
        '''This method is used to set multiple roles
        across multiple scopes as specified by a passed
        dictionary with keys as scopes and values as
        the role to set for all columns corresponding to that
        scope. See :ref:`Role`
        for more information about how roles are
        used within BPt.

        Parameters
        -----------
        scope_to_roles : dict of :ref:`Scope` to :ref:`Role`
            A python dictionary with keys as :ref:`Scope`
            and their corresponding value's as the :ref:`Role`
            in which those columns should take.

            For example ::

                scope_to_roles = dict()
                scope_to_roles['col1'] = 'target'
                scope_to_roles['col2'] = 'non input'

        inplace : bool, optional
            If True, do operation inplace and return None.

            ::

                default = False

        '''

        if not inplace:
            return self._inplace('set_roles', locals())

        # Init checks just once
        self._check_sr()

        # Set for each passed scope - all in place
        for scope, role in zip(scopes_to_roles, scopes_to_roles.values()):
            [self._set_role(col, role) for col in self._get_cols(scope)]

    def set_target(self, scope, inplace=False):
        '''This method is used to set
        either a single column, or multiple, specifically
        with role `target`. This function is simply
        a helper wrapper around :func:`Dataset.set_role`.

        See :ref:`Role` for more information
        about how roles are used within BPt.

        Parameters
        ----------
        scope : :ref:`Scope`
            A BPt style :ref:`Scope` used to select a subset of
            columns in which to set with role `target`.

        inplace : bool, optional
            If True, do operation inplace and return None.

            ::

                default = False
        '''

        return self.set_role(scope=scope, role='target', inplace=inplace)

    def set_non_input(self, scope, inplace=False):
        '''This method is used to set
        either a single column, or multiple, specifically
        with role `non input`. This function is simply
        a helper wrapper around :func:`Dataset.set_role`.

        See :ref:`Role` for more information
        about how roles are used within BPt.

        Parameters
        ----------
        scope : :ref:`Scope`
            A BPt style :ref:`Scope` used to select a subset of
            columns in which to set with role `non input`.

        inplace : bool, optional
            If True, do operation inplace and return None.

            ::

                default = False
        '''

        return self.set_role(scope=scope, role='non input', inplace=inplace)

    def _set_role(self, col, role):
        '''Internal function for setting a single columns role.'''

        if role not in self.reserved_roles:
            raise AttributeError(
                'Passed role "' + str(role) + '" must be one of ' +
                str(self.reserved_roles))

        # Set as role
        self.roles[col] = role

        # If role is non input, can't have any NaN
        if role == 'non input':
            self.drop_nan_subjects(scope=col, inplace=True)

    def get_scopes(self):
        '''This returns the up to date scopes for the
        Dataset. Note that scopes can be accessed through
        self.scopes, but this function is reccomended instead,
        as self.scope may be missing the latest change applied.

        Returns
        --------
        scopes : dict of sets
            Returns a dictionary with column name as the
            key and a set containing all scopes associated with
            each column.
        '''

        self._check_scopes()
        return self.scopes

    def add_scope(self, scope, scope_val, inplace=False):
        '''This method is designed as helper for adding a new scope val
        to a number of columns at once, using the existing scope system.
        Don't be confused about the arguments, the scope parameter is used
        to select the columns in which the scope_val should be added as a
        scope to those columns.

        See :ref:`Scope` for more information
        on how scope's are used in BPt. This
        function specifically is for adding
        custom, tag-like, scopes to certain columns.

        Parameters
        -----------
        scope : :ref:`Scope`
            A BPt style :ref:`Scope` used to select a subset of
            column in which to add new scopes (scope_val) to.

        scope_val : str or array-like of str
            A single string scope value to add
            to the columns select by scope, or an array-like of
            scope values to all add to the selected
            col. E.g.,

            ::

                scope_val = '1'

            Would add '1' as a scope

            ::

                scope_val = ['1', '2']

            Would add '1' and '2' as scopes.

        inplace : bool, optional
            If True, do operation inplace and return None.

            ::

                default = False
        '''

        if not inplace:
            return self._inplace('add_scope', locals())

        self._check_sr()

        cols = self._get_cols(scope)
        for col in cols:
            self._add_scope(col, scope_val)

    def _add_scope(self, col, scope_val):

        # Check if category
        if scope_val == 'category':
            self.scopes[col].add(scope_val)

            # If already categorical, skip
            if self[col].dtype.name == 'category':
                return

            # Special case, if converting from bool to category
            if self[col].dtype.name == 'bool':

                # Need to cast to int first as intermediate
                self[col] = self[col].astype('int')

            new_col = self[col].copy().astype('category')
            new_col.cat.as_ordered(inplace=True)
            self[col] = new_col

            return

        # If a int or float
        if isinstance(scope_val, int) or isinstance(scope_val, float):
            scope_val = str(scope_val)

            # @TODO Add a verbose here saying cast to str?

        # If str
        if isinstance(scope_val, str):

            # Make sure doesn't overlap with loaded column
            if scope_val in list(self):
                raise RuntimeError('Warning scope of: ' + scope_val + ' '
                                   'overlaps with a loaded column. This '
                                   'can cause index errors, as if this '
                                   'scope is '
                                   'requested, then the column will be '
                                   'returned '
                                   'instead of this scope!')

            # Add
            self.scopes[col].add(scope_val)

        # Or assume array-like
        else:
            for s in set(list(scope_val)):
                self._add_scope(col, s)

    def remove_scope(self, scope, scope_val, inplace=False):
        '''This method is used for removing scopes
        from an existing column or subset of columns,
        as selected by
        the scope parameter.

        See :ref:`Scope` for more information
        on how scope's are used in BPt. This
        function specifically is for removing
        custom, tag-like, scopes to certain columns.

        See related method :func:`add_scope <Dataset.add_scope>`

        Parameters
        -----------
        scope : :ref:`Scope`
            A BPt style :ref:`Scope` used to select a subset of
            column in which to remove scopes (scope_val).

        scope_val : str or array-like of str
            A single string scope value to remove
            from the column(s), or an array-like of
            scope values to all remove from the selected
            column(s).

        inplace : bool, optional
            If True, do operation inplace and return None.

            ::

                default = False
        '''

        if not inplace:
            return self._inplace('remove_scope', locals())

        self._check_sr()

        cols = self._get_cols(scope)
        for col in cols:
            self._remove_scope(col, scope_val)

    def _remove_scope(self, col, scope_val):

        try:
            self.scopes[col].remove(scope_val)

            # If removing category and currently pandas dtype is category,
            # change to float32.
            if scope_val == 'category' and \
               self[col].dtype.name == 'category':
                self[col] = self[col].astype('float32')

        except KeyError:
            pass

    def _get_reserved_cols(self, scope, columns):
        '''If passed here then we know the
        passed scope is one the options.'''

        if scope == 'all':
            return list(columns)

        elif scope == 'data float':
            return [col for col in columns if
                    'category' not in self.scopes[col] and
                    self.roles[col] == 'data']

        elif scope == 'target float':
            return [col for col in columns if
                    'category' not in self.scopes[col] and
                    self.roles[col] == 'target']

        elif scope == 'float':
            return [col for col in columns if
                    'category' not in self.scopes[col]]

        elif scope == 'category':
            return [col for col in columns if
                    'category' in self.scopes[col]]

        elif scope == 'data category':
            return [col for col in columns if
                    'category' in self.scopes[col] and
                    self.roles[col] == 'data']

        elif scope == 'target category':
            return [col for col in columns if
                    'category' in self.scopes[col] and
                    self.roles[col] == 'target']

        elif scope == 'data':
            return [col for col in columns if
                    self.roles[col] == 'data']

        elif scope == 'target':
            return [col for col in columns if
                    self.roles[col] == 'target']

        elif scope == 'non input':
            return [col for col in columns if
                    self.roles[col] == 'non input']

        elif scope == 'data file':
            return [col for col in columns if
                    'data file' in self.scopes[col]]

        else:
            raise RuntimeError('Should be impossible to reach here.')

    def _g_cols(self, scope, columns, saved_scopes):
        '''This method is used internally to avoid
        some repeated actions.

        It will return columns as a list.'''

        # If int or float, cast to str.
        if isinstance(scope, int) or isinstance(scope, float):
            scope = str(scope)

        # Check is passed scope is reserved
        if isinstance(scope, str):

            # First case, check if reserved scope
            if scope in self.reservered_scopes:
                return self._get_reserved_cols(scope, columns)

            # Next case is if the passed scope
            # is a valid column.
            elif scope in columns:
                return [scope]

            # Check if a saved scope
            if scope in saved_scopes:
                return [col for col in columns if
                        scope in self.scopes[col]]

            # Last string case is a stub search
            return [col for col in columns if scope in col]

        # If scope not str, assume iterable, and call recursively
        # on each iterable. If not iterable, will fail
        cols = []
        for scp in scope:
            cols += self._g_cols(scp, columns, saved_scopes)

        # In the case that scope was an iterable, their exists
        # the possibility for overlapping columns.
        # Cast to set and back to list to remove these
        return list(set(cols))

    def _get_cols(self, scope, limit_to=None):

        # Set columns to check within based on if
        # limit_to is passed
        if limit_to is None:
            columns = self.columns
        else:
            columns = self._get_cols(limit_to, limit_to=None)

        # Get the set of saved valid scopes for these columns
        saved_scopes = set()
        for col in columns:
            saved_scopes.update(self.scopes[col])

        # Get columns from internal g_cols func
        cols = self._g_cols(scope, columns, saved_scopes)

        # For maximum reproducibility and reliable behavior
        # always return the sorted columns.
        return sorted(cols)

    def get_cols(self, scope, limit_to=None):
        '''This method is the main internal and external
        facing way of getting the names of columns which match a
        passed scope from the Dataset. For example this
        method may be useful for user's when they want to ensure
        that a scope returns exactly the subset of columns they expect.

        Parameters
        -----------
        scope : :ref:`Scope`
            The BPt style :ref:`Scope` input that will be
            used to determine which column names from the Dataset
            to return.

        limit_to : :ref:`Scope` or None, optional
            Can optionally limit the columns
            to search over with another scope.
            If None, then will use 'all'.

            ::

                default = None

        Returns
        ----------
        cols : list
            This method returns the columns specified by
            the input `scope` argument as a sorted python
            list

        '''

        # Perform checks first
        self._check_sr()

        return self._get_cols(scope=scope, limit_to=limit_to)

    def _get_data_cols(self, ps_scope):
        return self._get_cols('data', limit_to=ps_scope)

    def _get_data_inds(self, ps_scope, scope):
        '''This function always limits first by the data cols,
        then ps_scope refers to the problem_spec scope, and
        lastly scope can be used to specify of subset of those columns'''

        # Get data cols, these are the ordered columns
        # actually passed to an evaluate function
        data_cols = self._get_data_cols(ps_scope)

        # If scope is 'all', data inds
        if scope == 'all':
            return list(range(len(data_cols)))

        # Otherwise, get subset of inds, then return sorted
        inds = [data_cols.index(k) for k in
                self._get_cols(scope, limit_to=data_cols)]

        return sorted(inds)

    def _is_data_cat(self, ps_scope, scope):

        # Get data cols
        data_cols = self._get_data_cols(ps_scope)

        # Check if all are categorical cat
        # In the case where the underlying scope
        # is nothing, return False
        try:
            all_cat = self._is_category(scope, limit_to=data_cols,
                                        check_scopes=False)
        except KeyError:
            all_cat = False

        return all_cat

    @doc(**_file_docs)
    def get_values(self, col, dropna=True, decode_values=False,
                   reduce_func=np.mean, n_jobs=-1):
        '''This method is used to obtain the either normally loaded and
        stored values from a passed column, or in the case of a data file
        column, the data file proxy values will be loaded. There is likewise
        an option to return these values with and without missing values
        included.

        Parameters
        -----------
        col : str
            The name of the column in which to load/extract values for.

        dropna : bool, optional
            Boolean argument, if True, return only non-nan values.
            If False, return everything regardless of if NaN.

            ::

                default = True

        decode_values : bool, optional
            Boolean argument, if True, then try to
            return the original values before any encoding,
            otherwise default of False will return the current
            loaded values.

            ::

                default = False

        {reduce_func}

        {n_jobs}

        Returns
        ---------
        values : pandas Series
            This method returns a single Series with the extracted
            values for the requested column, which either include or
            exclude missing values and may be data file proxy values
            depending on the nature of the requested column.

        '''

        # Check scopes and roles
        self._check_sr()

        return self._get_values(col=col, dropna=dropna,
                                decode_values=decode_values,
                                reduce_func=reduce_func, n_jobs=n_jobs)

    def _get_values(self, col, dropna=True, decode_values=False,
                    reduce_func=np.mean, n_jobs=-1):

        # Extract base series depending on if dropna
        if dropna:
            values = self[col].loc[self._get_non_nan_subjects(col)]
        else:
            values = self[col]

        # If data file, load and return proxy
        if 'data file' in self.scopes[col]:

            # If NaN, make sure NaN is in file mapping
            # if passing any NaN's here
            if pd.isnull(values).any():
                self.file_mapping[np.nan] = np.nan

            return load_data_file_proxy(values,
                                        reduce_func=reduce_func,
                                        file_mapping=self.file_mapping,
                                        n_jobs=n_jobs)

        if not decode_values:
            return values

        # Check for if to try and de-code
        # Note that we will assume data files can't be de-coded
        # Check encoders init'ed
        self._check_encoders()

        # Make copy of values
        values = values.copy()

        try:
            encoder = self.encoders[col]

        # If no encoder, return values as is
        except KeyError:
            return values

        # If dict style encoder - any other cases?
        if isinstance(encoder, dict):
            return values.replace(encoder)

        return values

    def _get_nan_subjects(self, col):
        return self[col][self[col].isna()].index

    def _get_non_nan_subjects(self, col):
        return self[col][~self[col].isna()].index

    def auto_detect_categorical(self, scope='all', obj_thresh=30,
                                all_thresh=None, inplace=False):
        '''This function will attempt to automatically add scope "category" to
        any loaded categorical variables. Note that any columns with pandas
        data type category should already be detected without
        calling this function.

        Default heuristic threshold settings are used by default, by
        they can be changed.

        Note: if any of the conditions are
        met the column will be set to categorical, it is not the case
        that if a single condition is not met, then it won't be categorical.

        Any column with only two unique non-nan values
        is considered binary- and therefore categorical. This is a
        fixed behavior of this function.

        If any data file's within scope, will always treat as not categorical,
        and further will make sure it is not categorical if for some reason
        set as that.

        Parameters
        -----------
        scope : :ref:`Scope`, optional
            Any valid BPt style scope used to select which columns this
            function should operate on. E.g., if known that only
            a subset of columns might be categorical, you could
            specify only this subset to work on.

            By default this is set to 'all', which will check all columns.

            ::

                default = 'all'

        obj_thresh : int or None, optional
            This threshold is used for any columns of pandas
            datatype object. If the number of unique non-nan values in
            this object datatype column is less than this threshold,
            this column will be set to categorical.

            To ignore this condition, you may pass None.

            ::

                default = 30

        all_thresh : int or None, optional
            Simmilar to obj_thresh, except that this condition is
            for any column regardless of datatype, this threshold
            is set such that if the number of unique non-nan values
            in this column is less than the passed value, this column
            will be set to categorical.

            To ignore this condition, you may pass None.

            ::

                default = None

        inplace : bool, optional
            If True, do operation inplace and return None.

            ::

                default = False

        '''

        if not inplace:
            return self._inplace('auto_detect_categorical', locals())

        # Check scope and role
        self._check_sr()

        # Get cols from scope
        cols = self._get_cols(scope)

        for col in cols:

            # Skip if data file
            if 'data file' in self.scopes[col]:

                # Make sure not categorical then skip rest of
                # checks.
                self._remove_scope(col, 'category')
                continue

            # Get non-nan values to check
            values = self._get_values(col, dropna=True)

            # First check for binary by unique number of columns
            n_unique = len(values.unique())
            if n_unique == 2:
                self._add_scope(col, 'category')

            # Check object threshold
            if obj_thresh is not None and values.dtype.name == 'object':
                if n_unique < obj_thresh:
                    self._add_scope(col, 'category')

            # Check all threshold
            if all_thresh is not None and n_unique < all_thresh:
                self._add_scope(col, 'category')

        self._print('Num. categorical variables in dataset:',
                    len(self._get_cols(scope='category')), level=1)
        self._print('Categorical variables in dataset:',
                    self._get_cols(scope='category'), level=2)

    def _get_problem_type(self, col):
        '''Return the default problem type for a given column.'''

        self._check_scopes()

        # Doesn't matter if column is categorical
        # if just two unique values, assume binary
        if self[col].nunique(dropna=True) == 2:

            # Further, if not categorical, make sure values
            # are 0 and 1.
            if 'category' not in self.scopes[col]:
                u_values = self.get_values(col, dropna=True).unique()
                if 0 not in u_values or 1 not in u_values:
                    raise RuntimeError(
                        'Error determining default problem type: Requested'
                        ' target column: ' + str(col) + ' has only two unique'
                        ' values: ' + repr(u_values) + ', but they are not'
                        ' categorical and not 0 and 1. Either change the'
                        ' values / type of target column: ' + str(col) + ', '
                        'or pass an explicit option to problem_type'
                        ', e.g., problem_type="binary"')

            self._print('Problem type auto-detected as binary', level=2)
            return 'binary'

        if 'category' in self.scopes[col]:
            self._print('Problem type auto-detected as categorical', level=2)
            return 'categorical'

        self._print('Problem type auto-detected as regression', level=2)
        return 'regression'

    def rename(self, mapper=None, index=None,
               columns=None, axis=None, copy=True,
               inplace=False, level=None, errors='ignore'):
        '''Calls method according to:
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rename.html

        This version is updated to reflect changes
        to meta-data after renaming, e.g.,
        scopes will be updated to their new name accordingly.

        Parameters
        -------------
        mapper : dict-like or function, optional
            Dict-like or function transformations to apply to that
            axis’ values. Use either mapper and axis to specify
            the axis to target with mapper, or index and columns.

            ::

                default = None

        index : dict-like or function, optional
            Alternative to specifying axis
            (mapper, axis=0 is equivalent to index=mapper).

            ::

                default = None

        columns : dict-like or function, optional
            Alternative to specifying axis
            (mapper, axis=1 is equivalent to columns=mapper).

            ::

                default = None

        axis : {0 or 'index', 1 or 'columns'}, optional
            Axis to target with mapper. Can be either the axis name
            ('index', 'columns') or number (0, 1). The default is ‘index’.

            ::

                default = 0

        copy : bool, optional
            Also copy underlying data.

            ::

                default = True

        inplace : bool, optional
            Whether to return a new :class:`Dataset`.
            If True then value of copy is ignored.

            ::

                default = False

        level : int or level name, optional
            In case of a MultiIndex, only rename labels in the specified level.

            ::

                default = None

        errors : {'ignore', 'raise'}, optional
            If 'raise', raise a KeyError when a dict-like mapper,
            index, or columns contains labels that are not present
            in the Index being transformed. If 'ignore',
            existing keys will be renamed and extra keys will be ignored.

            ::

                default = 'ignore'
        '''

        # Make sure everything up to data
        # and that the attributes exist
        self._check_sr()
        self._check_encoders()
        self._check_train_subjects()
        self._check_test_subjects()

        # First apply parent function
        data = super().rename(mapper=mapper, index=index, columns=columns,
                              axis=axis, copy=copy, inplace=inplace,
                              level=level,
                              errors=errors)

        # Next proc using either result if not inplace
        # or self if inplace
        result = self if inplace else data

        # Set columns and index
        if not (index is not None or columns is not None):
            if axis and self._get_axis_number(axis) == 1:
                columns = mapper
            else:
                index = mapper

        # Process each passed
        for axis_no, replacements in enumerate((index, columns)):
            if replacements is None:
                continue

            # Get rename function
            func = com.get_rename_function(replacements)

            if axis_no == 0:
                result._rename_index(func, level=level)
            elif axis_no == 1:
                result._rename_columns(func)
            else:
                raise RuntimeError(f'axis_no == {axis_no}, should be 0 or 1.')

        return result

    def _rename_columns(self, func):

        # Can't have mapping to duplicate cols
        self._check_duplicate_cols()

        to_change = ['roles', 'scopes', 'encoders']
        for attr in to_change:
            curr_attr = getattr(self, attr)
            new_attr = {func(key): curr_attr[key] for key in curr_attr}
            setattr(self, attr, new_attr)

    def _rename_index(self, func, level=None):

        # Skip if no train or test subjects
        if self.train_subjects is None and self.test_subjects is None:
            return

        if not isinstance(self, ABCMultiIndex):
            to_change = ['test_subjects', 'train_subjects']
            for attr in to_change:
                curr_attr = getattr(self, attr)
                if curr_attr is not None:
                    new_vals = [func(val) for val in curr_attr.values]
                    new_attr = pd.Index(new_vals)
                    setattr(self, attr, new_attr)

        # @TODO Support at some point
        else:
            raise RuntimeError('renaming index in a MultiIndex case '
                               'is not currently supported.')
            '''
            if level is not None:
                level = self.index._get_level_number(level)

            if level is not None:
                items = [
                    tuple(f(y) if i == level else y for i, y in enumerate(x))
                    for x in self
                ]

            else:
                items = [tuple(f(y) for y in x) for x in self]
            '''

    def _is_category(self, scope, limit_to=None, check_scopes=True):
        '''Tests if a set of columns are categorical,
        returns True only if all are categorical.'''

        # Only if requested check scopes
        if check_scopes:
            self._check_scopes()

        # Get cols
        cols = self._get_cols(scope, limit_to=limit_to)

        # If no valid, raise error
        if len(cols) == 0:
            raise KeyError('Passed: ' + repr(scope) +
                           ' not a valid scope.')

        # Check for each col
        is_category = []
        for col in self._get_cols(scope):

            # Only check is there is a scope for this col
            if col in self.scopes:
                is_category.append('category' in self.scopes[col])
            else:
                is_category.append(False)

        return all(is_category)

    def copy(self, deep=True):
        '''Creates and returns a dopy of this dataset, either
        a deep copy or shallow.

        Parameters
        ----------
        deep : bool, optional
            If the returned copy should be
            a deep copy (True), or a shallow copy (False).

            ::

                default == True
        '''

        # Base parent pandas behavior
        dataset = super().copy(deep=deep)

        # Make copy of meta data from self to dataset in place
        # Always makes deep copy for meta-data.
        self._copy_meta_data(dataset)

        return dataset

    def _copy_meta_data(self, dataset):
        '''Add copy of meta data from self to dataset in place'''

        # Make copy of meta data
        for m in self._metadata:

            # If has this attribute, set in the copy a copy.
            if hasattr(self, m):
                setattr(dataset, m, deepcopy(getattr(self, m)))

        return self

    def get_Xy(self, problem_spec='default', **problem_spec_params):
        '''This function is used to get a sklearn-style
        grouping of input data (X) and target data (y)
        from the Dataset as according to a passed problem_spec.

        Note: X and y are returned as pandas DataFrames not Datasets,
        so none of the Dataset meta data is accessible through the
        returned X, y here.

        Parameters
        -----------
        problem_spec : :class:`ProblemSpec` or 'default', optional
            This argument accepts an instance of the
            params class :class:`ProblemSpec`.
            This object is essentially a wrapper around commonly used
            parameters needs to define the context
            the model pipeline should be evaluated in.
            It includes parameters like problem_type, scorer, n_jobs,
            random_state, etc...
            See :class:`ProblemSpec` for more information
            and for how to create an instance of this object.

            If left as 'default', then will initialize a
            ProblemSpec with default params.

            ::

                default = 'default'

        problem_spec_params : :class:`ProblemSpec` params, optional
            You may also pass any valid parameter value pairs here,
            e.g.

            ::

                get_Xy(problem_spec=problem_spec, problem_type 'binary')

            Any parameters passed here will override the original
            value in problem spec. This can be useful when using all default
            values for problem spec except for one, e.g., you just want
            to change random_state.

            ::

                get_Xy(random_state=5)

        Returns
        --------
        X : pandas DataFrame
            DataFrame with the input data and columns as
            specified by the passed problem_spec.

        y : pandas Series
            Series with the the target values as requested
            by the passed problem_spec.
        '''
        from ..main.funcs import problem_spec_check

        # Get proc'ed problem spec
        ps = problem_spec_check(problem_spec, dataset=self,
                                **problem_spec_params)

        # Get sorted subjects from problem spec
        subjects = sorted(self.get_subjects(ps.subjects, return_as='set'))

        # Get X cols
        X_cols = self._get_data_cols(ps.scope)

        # Get X as pandas df subset, y as Series
        X = pd.DataFrame(self.loc[subjects, X_cols]).astype(ps.base_dtype)
        y = self.loc[subjects, ps.target].astype('float64')

        # Add warning for unique n subjs
        n_unique_index = len(pd.unique(X.index))
        if n_unique_index != len(X):
            self._print('The number of unique index rows (',
                        str(n_unique_index),
                        ') does not match the number of rows (',
                        str(len(X)), ').',
                        'There may be duplicate subjects / index.', level=0)

        return X, y

    def _replace_datafiles_with_repr_(self):

        def is_int(i):
            try:
                int(i)
                return True
            except ValueError:
                return False

        for col in self:
            if 'data file' in self.scopes[col]:
                self[col] = [f'Loc({int(i)})' if is_int(i)
                             else repr(i) for i in self[col]]

    def _repr_html_(self):

        # Checks
        self._check_sr()

        template = """<div style="float: left; padding: 10px;">
        <h3>{0}</h3>{1}</div>"""

        html = ''
        for scope in ['data', 'target', 'non input']:
            cols = self._get_cols(scope)

            # Minor formatting for display
            display_scope = scope[0].upper() + scope[1:]
            if display_scope == 'Target' and len(cols) > 1:
                display_scope = 'Targets'
            elif display_scope == 'Non input':
                display_scope = 'Non Input'

            if len(cols) > 0:

                display_copy = self[cols].copy()
                display_copy._replace_datafiles_with_repr_()

                html += template.format(display_scope,
                                        display_copy._base_repr_html_())
                html += '\n'

        return html

    def _base_repr_html_(self):

        html = super()._repr_html_()

        try:
            train_subjects = self.get_subjects('train', return_as='set')
            test_subjects = self.get_subjects('test', return_as='set')
        except RuntimeError:
            return html

        train_color = 'RGBA(176, 224, 230, .25)'
        test_color = 'RGBA(249, 121, 93, .25)'

        train_subjects = set(['>' + str(s) for s in train_subjects])
        test_subjects = set(['>' + str(s) for s in test_subjects])

        # Generate splits 1
        start = html.index('</thead>')
        s1 = html[start:].split('<th')
        for i in range(len(s1)):
            s2 = s1[i].split('</th>')

            for j in range(len(s2)):
                if s2[j] in train_subjects:
                    s2[j] = ' style="background: ' + train_color + '"' + s2[j]
                elif s2[j] in test_subjects:
                    s2[j] = ' style="background: ' + test_color + '"' + s2[j]

            s1[i] = '</th>'.join(s2)

        # Set as re-joined html str
        s1_post = '<th'.join(s1)
        html = html[:start] + s1_post

        n_cols = str(len(self.columns))

        train_info = '<p style="margin-top: .35em"><span style="background: '
        train_info += train_color + '">'
        train_info += str(len(train_subjects)) + " rows × "
        train_info += n_cols + " columns - Train Set </span></p>"

        test_info = '<p style="margin-top: .35em"><span style="background: '
        test_info += test_color + '">'
        test_info += str(len(test_subjects)) + " rows × "
        test_info += n_cols + " columns - Test Set </span></p>"

        extra_info = train_info + test_info

        # Add before end
        html = html[:-6] + extra_info + html[-6:]

        return html

    from ._subjects import (_apply_only_level,
                            _get_nan_loaded_subjects,
                            _get_value_subset_loaded_subjects,
                            _get_base_loaded_subjects,
                            _return_subjects_as,
                            get_subjects)

    from ._data_files import (get_file_mapping,
                              add_data_files,
                              to_data_file,
                              _series_to_data_file,
                              consolidate_data_files,
                              update_data_file_paths,
                              _get_next_ind)

    from ._plotting import (plot,
                            nan_info,
                            _cont_info,
                            _cat_info,
                            summary,
                            plot_bivar,
                            _plot_cat_cat,
                            _plot_cat_float,
                            _plot_float_float,
                            _get_plot_values,
                            _plot_float,
                            _plot_col,
                            _plot_category,
                            _print_plot_info)

    from ._encoding import (_add_new_copy,
                            to_binary,
                            binarize,
                            _base_binarize,
                            _binarize,
                            k_bin,
                            _k_bin,
                            ordinalize,
                            _ordinalize,
                            nan_to_class,
                            copy_as_non_input,
                            add_unique_overlap,
                            _replace_cat_values)

    from ._validation import (_validate_group_key,
                              _proc_cv_strategy,
                              _validate_split,
                              _finish_split,
                              set_test_split,
                              test_split,
                              train_split,
                              set_train_split,
                              save_test_split,
                              save_train_split)

    from ._filtering import (_drop_subjects,
                             drop_nan_subjects,
                             apply_inclusions,
                             apply_exclusions,
                             drop_subjects_by_nan,
                             _drop_or_nan,
                             _data_file_fail_check,
                             _drop_cols,
                             filter_outliers_by_percent,
                             filter_outliers_by_std,
                             filter_categorical_by_percent,
                             drop_id_cols,
                             drop_duplicate_cols,
                             drop_cols,
                             drop_cols_by_unique_val,
                             drop_cols_by_nan)


def read_csv(*args, **kwargs):
    '''Passes along all arguments and kwargs to
    :func:`pandas.read_csv` then casts to :class:`Dataset`.

    This method is just a helper wrapper function.
    '''

    return Dataset(pd.read_csv(*args, **kwargs))
