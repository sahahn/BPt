import pandas as pd
import numpy as np
from itertools import product

from joblib import wrap_non_picklable_objects
from ..helpers.Data_File import Data_File, load_data_file_proxy
from copy import copy, deepcopy
from ..helpers.ML_Helpers import conv_to_list
from .helpers import (base_load_subjects, proc_file_input)
from ..main.Input_Tools import Value_Subset

# @TODO Loook into pandas finalize
# https://github.com/pandas-dev/pandas/blob/ce3e57b44932e7131968b9bcca97c1391cb6b532/pandas/core/generic.py#L5422

# @TODO Customize the appearence of Dataset class, e.g.
# add to repr and to_html, etc...


class Dataset(pd.DataFrame):
    '''The BPt Dataset class is the main class used for preparing data
    into a compatible format to work with machine learning. This class is new
    as of BPt version >= 2 (replacing the building in loading functions of the
    old BPt_ML).'''

    ROLES = set(['data', 'target', 'non input'])
    RESERVED_SCOPES = set(['all', 'float', 'category', 'data float',
                           'data', 'data file', 'data category',
                           'non input', 'target', 'target category',
                           'target float'])
    _metadata = ['roles', 'scopes', 'encoders', 'file_mapping',
                 'verbose', 'test_subjects', 'train_subjects']

    @property
    def _constructor(self):
        return Dataset

    @property
    def _constructor_sliced(self):
        return pd.Series

    def _print(self, *args, **kwargs):
        '''Overriding the print function to allow for
        customizable verbosity.

        According to passed level:

        Warnings are level 0,
        Information on sizes / how many dropped are level 1.
        Set to level -1 to mute warnings too.

        Parameters
        ----------
        args
            Anything that would be passed to default python print
        '''

        self._check_verbose()

        if 'level' in kwargs:
            level = kwargs.pop('level')
        else:
            level = 1

        if self.verbose >= level:
            print(flush=True, *args, **kwargs)

    # Checks #
    def _check_verbose(self):

        if not hasattr(self, 'verbose'):
            self.verbose = 0
        elif getattr(self, 'verbose') is None:
            self.verbose = 0

    def _check_test_subjects(self):

        if not hasattr(self, 'test_subjects'):
            self.test_subjects = None

    def _check_train_subjects(self):

        if not hasattr(self, 'train_subjects'):
            self.train_subjects = None

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

    def _check_cols_type(self):
        '''BPt / scopes assume that all columns are of
        type str. This function should be called before
        check roles or check scope.'''

        col_names = list(self)

        # Check if any col names not strs
        non_str = []
        for col in col_names:
            if not isinstance(col, str):
                non_str.append(col)

        if len(non_str) > 0:

            rename = {c: str(c) for c in non_str}
            super().rename(rename, axis=1, inplace=True)

            self._print('Warning: the columns:', repr(non_str),
                        'were cast to str', level=0)

    def _check_roles(self):

        # Make sure cols type str
        self._check_cols_type()

        if not hasattr(self, 'roles'):
            self.roles = {}
        elif getattr(self, 'roles') is None:
            self.roles = {}

        # Fill in any column without a role.
        for col in list(self.columns):
            if col not in self.roles:
                self._set_role(col, 'data')

        # Remove any saved roles that are not
        # in the data any more.
        for col in list(self.roles):
            if col not in self.columns:
                del self.roles[col]

    def _check_scopes(self):

        # Make sure cols type str
        self._check_cols_type()

        # If doesnt exist, create scopes
        if not hasattr(self, 'scopes'):
            self.scopes = {}

        # Or is set to None
        elif getattr(self, 'scopes') is None:
            self.scopes = {}

        # Make sure each col is init'ed with a scope
        for col in self.columns:
            if col not in self.scopes:
                self.scopes[col] = set()

        # Make sure to get rid of any removed cols
        for col in list(self.scopes):
            if col not in self.columns:
                del self.scopes[col]

        # @TODO maybe optimize with self.dtypes ?

        # Compute columns which are categorical + columns with scope category
        dtype_category = set(self.select_dtypes('category'))
        scope_category = set([col for col in self.columns if
                             'category' in self.scopes[col]])

        # End if equal
        if dtype_category == scope_category:
            return self

        # Add scope to columns which have dtype category but not scope category
        needs_scope = dtype_category - scope_category
        for col in needs_scope:
            self._add_scope(col, 'category')

        # For any columns which have scope category, but the dtype isn't
        # assume that the scope should be removed
        remove_scope = scope_category - dtype_category
        for col in remove_scope:
            self._remove_scope(col, 'category')

        return self

    def _apply_only_level(self, subjects, only_level):

        # If only level not None and MultiIndex - only keep
        if isinstance(self.index, pd.MultiIndex) and only_level is not None:
            drop_names = list(self.index.names)
            drop_names.remove(self.index.names[only_level])
            subjects = subjects.droplevel(drop_names)

        return set(list(subjects))

    def _get_nan_loaded_subjects(self, only_level):

        # Get nan subjects
        nan_subjects = self[pd.isnull(self[:]).any(axis=1)].index

        # Apply only level + convert to sat
        nan_subjects = self._apply_only_level(nan_subjects, only_level)

        return nan_subjects

    def _get_value_subset_loaded_subjects(self, subjects, only_level):

        if subjects.name not in list(self):
            raise KeyError('Passed Value_Subset name: ' +
                           repr(subjects.name) +
                           ' is not loaded')

        # Get the relevant series
        if subjects.encoded_values:
            data = self.get_encoded_values(subjects.name)
        else:
            data = self[subjects.name]

        # Extract the values as list
        values = conv_to_list(subjects.values)

        # Get subjects by values
        loaded_subjects = data[data.isin(values)].index

        # Apply only level + convert to set
        loaded_subjects =\
            self._apply_only_level(loaded_subjects, only_level)

        return loaded_subjects

    def _get_base_loaded_subjects(self, subjects, only_level):

        loaded_subjects = base_load_subjects(subjects)

        # In multi-index case
        if isinstance(self.index, pd.MultiIndex):

            # If want multi-index returned
            if only_level is None:
                loaded_subjects = set(self.loc[loaded_subjects].index)

        # Non multi-index case try to cast to correct type
        else:

            dtype = self.index.dtype.name
            if 'int' in dtype:
                loaded_subjects = {int(s) for s in loaded_subjects}
            elif 'float' in dtype:
                loaded_subjects = {float(s) for s in loaded_subjects}
            elif dtype == 'object':
                loaded_subjects = {str(s) for s in loaded_subjects}
            else:
                raise RuntimeError('Index data type:' + dtype + ' '
                                   'is not currently supported!')

        return loaded_subjects

    def _return_subjects_as(self, subjects, return_as, only_level):

        if return_as == 'set':
            return subjects

        # If not set, treat as pandas index or flat index case
        subjects = sorted(list(subjects))

        # If not multi index, just cast and return
        if not isinstance(self.index, pd.MultiIndex):
            return pd.Index(data=subjects, name=self.index.name)

        # Multi index case, with only level
        if only_level is not None:
            return pd.Index(data=subjects,
                            name=self.index.names[only_level])

        # RConvert to multi index
        subjects = list(map(tuple, subjects))
        multi_index = pd.MultiIndex.from_tuples(
            subjects, names=self.index.names)

        if return_as == 'flat index':
            return multi_index.to_flat_index()
        else:
            return multi_index

    def get_subjects(self, subjects, return_as='set', only_level=None):
        '''Method to get a set of subjects, from
        a set of already loaded ones, or from a saved location.

        Parameters
        -----------
        subjects : :ref:`Subjects`
            This argument can be any of the BPt accepted
            :ref:`Subjects` style inputs.
            E.g., None, 'nan' for subjects
            with any nan data, the str location of a file
            formatted with one subject per line, or directly an
            array-like of subjects, to name some options.

            See :ref:`Subjects` for all options.

        return_as : {'set', 'index', 'flat index'}, optional
            - If 'set', return as set of subjects.
            - If 'index', return as sorted pandas index.
            - If 'flat index', will return as sorted pandas index
                i.e., the same output as index, when not MultiIndex,
                but when MultiIndex, will return the index as a flat
                Index of tuples.

        only_level : int or None, optional
            This parameter is only relevant when the
            underlying index is a MultiIndex.

            Note: this param is not relevant
            when using special tuple style input for subjects.

            ::

                default = None

        Returns
        ----------
        subjects : {set, np.array, pd.Index}
            Based on value of return_as, returns as
            a set of subjects, sorted numpy array or
            sorted pandas Index.
        '''

        # input validation
        if return_as not in ['set', 'index', 'flat index']:
            raise TypeError('Invalid parameter passed to return as!')

        if isinstance(subjects, tuple):

            if len(subjects) != len(self.index.names):
                raise RuntimeError('Passed special tuple must match the '
                                   'number of MultiIndex levels.')

            # Proc each ind seperately
            inds = []
            for i, subject_arg in enumerate(subjects):
                inds.append(self.get_subjects(subject_arg,
                                              return_as='set',
                                              only_level=i))
            # Create set of sets
            loaded_subjects = set(product(*inds))

            # Return as requested, note only_level fixed as None
            return self._return_subjects_as(loaded_subjects,
                                            return_as=return_as,
                                            only_level=None)

        # Check if None
        if subjects is None:
            loaded_subjects = set()

        # Check for special keywords
        elif isinstance(subjects, str) and subjects == 'nan':
            loaded_subjects =\
                self._get_nan_loaded_subjects(only_level=only_level)

        elif isinstance(subjects, str) and subjects == 'all':
            loaded_subjects =\
                self._apply_only_level(self.index, only_level)

        elif isinstance(subjects, str) and subjects == 'train':
            if not hasattr(self, 'train_subjects') or self.train_subjects is None:
                raise RuntimeError('Train subjects undefined')
            loaded_subjects = set(self.train_subjects)

        elif isinstance(subjects, str) and subjects == 'test':
            if not hasattr(self, 'test_subjects') or self.test_subjects is None:
                raise RuntimeError('Test subjects undefined')
            loaded_subjects = set(self.test_subjects)

        # Check for Value Subset or Values Subset
        elif isinstance(subjects, Value_Subset):
            loaded_subjects =\
                self._get_value_subset_loaded_subjects(subjects,
                                                       only_level=only_level)
        else:
            loaded_subjects =\
                self._get_base_loaded_subjects(subjects, only_level=only_level)

        # Return based on return as value
        return self._return_subjects_as(loaded_subjects, return_as,
                                        only_level=only_level)

    def get_roles(self):

        self._check_roles()
        return self.roles

    def set_role(self, col, role):

        self._check_roles()
        self._set_role(col, role)

        return self

    def set_roles(self, scope, role):

        self._check_roles()

        cols = self.get_cols(scope)
        for col in cols:
            self._set_role(col, role)

        return self

    def _set_role(self, col, role):

        if role not in self.ROLES:
            raise AttributeError(
                'Passed role "' + str(role) + '" must be one of ' +
                str(self.ROLES))

        # If col is int or float, cast
        if isinstance(col, int) or isinstance(col, float):
            col = str(col)

        # Set as role
        self.roles[col] = role

        # If role is non input, can't have any NaN
        if role == 'non input':
            self.drop_nan_subjects(scope=col)

        return self

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

    def add_scope(self, col, scope_val):
        '''This method is used for adding scopes
        to an existing column.

        See :ref:`Scope` for more information
        on how scope's are used in BPt. This
        function specifically is for adding
        custom, tag-like, scopes to certain columns.

        Note: All passed scopes if not a str are
        cast to a str! This means that if a scope of
        say 1 is set, and a scope of "1", they will
        be equivelent.

        This method is applied in place. Also,
        see related methods :func:`add_scopes <Dataset.add_scopes>`
        and :func:`remove_scope <Dataset.remove_scope>`

        Parameters
        -----------
        col : str
            The name of the column in which the scope_val
            parameter should be added as a scope.

        scope_val : str or array-like of str
            A single string scope value to add
            to the column, or an array-like of
            scope values to all add to the selected
            col. E.g.,

            ::

                scope_val = '1'

            Would add '1' as a scope

            ::

                scope_val = ['1', '2']

            Would add '1' and '2' as scopes.
        '''

        self._check_scopes()
        self._add_scope(col, scope_val)

        return self

    def add_scopes(self, scope, scope_val):
        '''This method is designed as helper for adding a new scope val
        to a number of columns at once, using the existing scope system.
        Don't be confused about the arguments, the scope parameter is used
        to select the columns in which the scope_val should be added as a
        scope to those columns.

        See :ref:`Scope` for more information
        on how scope's are used in BPt. This
        function specifically is for adding
        custom, tag-like, scopes to certain columns.

        This method is applied in place. Also,
        see related methods :func:`add_scope <Dataset.add_scope>`
        and :func:`remove_scopes <Dataset.remove_scopes>`

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
        '''

        self._check_scopes()

        cols = self.get_cols(scope)
        for col in cols:
            self._add_scope(col, scope_val)

        return self

    def _add_scope(self, col, scope_val):

        # If col is int or float, cast
        if isinstance(col, int) or isinstance(col, float):
            col = str(col)

        # Check if category
        if scope_val == 'category':
            self.scopes[col].add(scope_val)
            self[col] = self[col].astype('category')

            return self

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

        return self

    def remove_scope(self, col, scope_val):
        '''This method is used for removing scopes
        from an existing column.

        See :ref:`Scope` for more information
        on how scope's are used in BPt. This
        function specifically is for removing
        custom, tag-like, scopes to certain columns.

        This method is applied in place.

        See related methods :func:`add_scope <Dataset.add_scope>`
        and :func:`remove_scopes <Dataset.remove_scopes>`

        Parameters
        -----------
        col : str
            The name of the column in which the scope_val
            parameter should be removed as a scope.

        scope_val : str or array-like of str
            A single string scope value to remove
            from the column, or an array-like of
            scope values to all remove from the selected
            column.
        '''

        self._check_scopes()
        self._remove_scope(col, scope_val)

        return self

    def remove_scopes(self, scope, scope_val):
        '''This method is used for removing scopes
        from an existing subset of columns, as selected by
        the scope parameter.

        See :ref:`Scope` for more information
        on how scope's are used in BPt. This
        function specifically is for removing
        custom, tag-like, scopes to certain columns.

        This method is applied in place.

        See related methods :func:`add_scopes <Dataset.add_scopes>`
        and :func:`remove_scope <Dataset.remove_scope>`

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
        '''

        self._check_scopes()

        cols = self.get_cols(scope)
        for col in cols:
            self._remove_scope(col, scope_val)

        return self

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

        return self

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
        self._check_scopes()
        self._check_roles()

        if limit_to is None:
            columns = self.columns
        else:
            columns = self.get_cols(limit_to, limit_to=None)

        saved_scopes = set()
        for col in columns:
            saved_scopes.update(self.scopes[col])

        # If int or flot, cast to str.
        if isinstance(scope, int) or isinstance(scope, float):
            scope = str(scope)

        # Check is passed scope is reserved
        if isinstance(scope, str):
            if scope in self.RESERVED_SCOPES:

                if scope == 'all':
                    return list(columns)

                # Float refers to essentially not category and not data file
                elif scope == 'data float':
                    return [col for col in columns if
                            'category' not in self.scopes[col] and
                            'data file' not in self.scopes[col] and
                            self.roles[col] == 'data']

                elif scope == 'target float':
                    return [col for col in columns if
                            'category' not in self.scopes[col] and
                            'data file' not in self.scopes[col] and
                            self.roles[col] == 'target']

                elif scope == 'float':
                    return [col for col in columns if
                            'category' not in self.scopes[col] and
                            'data file' not in self.scopes[col]]

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

            # Check if passed scope is a loaded column
            elif scope in columns:
                return [scope]

            # Check if a saved scope
            elif scope in saved_scopes:
                return [col for col in columns if
                        scope in self.scopes[col]]

            # Do a search, see if passed scope is a stub of any strings
            else:
                return [col for col in columns if
                        scope in col]

        cols = []
        for scp in scope:
            cols += self.get_cols(scp)

        return sorted(list(set(cols)))

    def _get_data_inds(self, ps_scope, scope='all'):
        '''This function always limits first by the data cols,
        then ps_scope refers to the problem_spec scope, and
        lastly scope can be used to specify of subset of those columns'''

        data_cols = self.get_cols('data', limit_to=ps_scope)

        inds = [data_cols.index(k) for k in
                self.get_cols(scope, limit_to=data_cols)]

        return inds

    def get_values(self, col, dropna=True, reduce_func=np.mean, n_jobs=1):
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

        Returns
        ---------
        values : pandas Series
            This method returns a single Series with the extracted
            values for the requested column, which either include or
            exclude missing values and may be data file proxy values
            depending on the nature of the requested column.

        '''

        # Check scopes
        self._check_scopes()

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

        return values

    def _get_nan_subjects(self, col):
        return self[col][self[col].isna()].index

    def _get_non_nan_subjects(self, col):
        return self[col][~self[col].isna()].index

    def auto_detect_categorical(self, scope='all', obj_thresh=30,
                                all_thresh=None):
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

        '''

        # Get cols from scope
        cols = self.get_cols(scope)

        for col in cols:

            # Skip if data file
            if 'data file' in self.scopes[col]:

                # Make sure not categorical then skip rest of
                # checks.
                self.remove_scope(col, 'category')
                continue

            # Get non-nan values to check
            values = self.get_values(col, dropna=True)

            # First check for binary by unique number of columns
            n_unique = len(values.unique())
            if n_unique == 2:
                self.add_scope(col, 'category')

            # Check object threshold
            if obj_thresh is not None and values.dtype.name == 'object':
                if n_unique < obj_thresh:
                    self.add_scope(col, 'category')

            # Check all threshold
            if all_thresh is not None and n_unique < all_thresh:
                self.add_scope(col, 'category')

        self._print('Num. categorical variables in dataset:',
                    len(self.get_cols(scope='category')), level=1)
        self._print('Categorical variables in dataset:',
                    self.get_cols(scope='category'), level=2)

        return self

    def get_encoded_values(self, col):
        '''Returns a copy of the column with values replaced, if any
        valid encoders. This function is typically used internally, but
        may be useful if you are doing custom plotting.

        Parameters
        -----------
        col : str
            A loaded column name in which to return the
            encoded values as a Series.

        Returns
        -----------
        values : Series
            Returns a Series with values as the encoded values.

        '''

        # Check encoders init'ed
        self._check_encoders()

        # Make copy of values
        values = self[col].copy()

        try:
            encoder = self.encoders[col]

        # If no encoder, return values as is
        except KeyError:
            return values

        # If dict style encoder
        if isinstance(encoder, dict):
            return values.replace(encoder)

        # Any other cases?
        return values

    def get_file_mapping(self):
        '''This function is used to access the
        up to date file mapping.

        Returns
        --------
        file_mapping : dict
            Return a dictionary with keys as
            integer's loaded in the Dataset referring
            to Data Files.
        '''

        self._check_file_mapping()
        return self.file_mapping

    def add_data_files(self, files, file_to_subject, load_func=np.load):
        '''This method is the main way of adding columns of type
        'data file' to the Dataset class.

        Parameters
        ----------
        files : dict
            This argument must be passed as a python dict.
            Specifically, a python dictionary should be passed where
            each key refers to the name of that feature / column of data files
            to load, and the value is a python list, or array-like of
            str file paths.

            In addition to this parameter, you must also pass a
            python function to the file_to_subject param,
            which specifies how to convert from passed
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

        file_to_subject : python function or dict of
            If files is passed, then you also need to specify a function
            which takes in a file path, and returns the relevant subject for
            that file path. If just one function is passed, it will be used
            for to load all dictionary entries, alternatively you can pass
            a matching dictionary of funcs, allowing for different funcs
            for each feature to load.

            See the example in files, e.g.,

            ::

                file_to_subject = file_to_subject_func
                # or
                file_to_subject = {'feat1': file_to_subject_func,
                                   'feat2': file_to_subject_func}

            In the case that the underlying index is a MultiIndex, this
            function should be designed to return the subject in correct
            tuple form. E.g.,

            ::

                # The underlying dataset is indexed by subject and event
                data.set_index(['subject', 'event'], inplace=True)

                # Only one feature
                files = {'feat1': ['f1/s0_e0.npy',
                                   'f1/s0_e1.npy',
                                   'f1/s1_e0.npy',
                                   'f1/s1_e1.npy']}

                def file_to_subject_func(file):

                    # This selects the substring
                    # at the last part seperated by the '/'
                    # so e.g. the stub, 's0_e0.npy', 's0_e1.npy', etc...
                    subj_split = file.split('/')[-1]

                    # This removes the .npy from the end, so
                    # stubs == 's0_e0', 's0_e1', etc...
                    subj_split = subj_split.replace('.npy', '')

                    # Set the subject name as the first part
                    # and the eventname as the second part
                    subj_name = subj_split.split('_')[0]
                    event_name = subj_split.split('_')[1]

                    # Lastly put it into the correct return style
                    # This is tuple style e.g., ('s0', 'e0'), ('s0', 'e1')
                    ind = (subj_name, eventname)

                    return ind

            While this is a bit longer than the previous case, it is flexible.

        load_func : python function, optional
            Fundamentally columns of type 'data file' represent
            a path to a saved file, which means you must
            also provide some information on how to load the saved file.
            This parameter is where that loading function should be passed.
            The passed `load_func` will be called on each file individually
            and whatever the output of the function is will be passed to
            the different loading functions.

            You might need to pass a user defined custom function
            in some cases, e.g., you want to use np.load,
            but then also np.stack. Just wrap those two functions in one,
            and pass the new function.

            ::

                def my_wrapper(x):
                    return np.stack(np.load(x))

            In this case though, it is reccomended that
            you define this function in a seperate file from
            where the main script will be run (for ease of caching)

            By default this function assumes data files are passed
            as numpy arrays.

            ::

                default = np.load
        '''

        # Wrap load_func here if needed.
        if load_func.__module__ == '__main__':
            wrapped_load_func = wrap_non_picklable_objects(load_func)
            self._print('Warning: Passed load_func was defined within the',
                        '__main__ namespace and therefore has been '
                        'cloud wrapped.',
                        'The function will still work, but it is '
                        'reccomended to',
                        'define this function in a seperate file, '
                        'and then import',
                        'it , otherwise loader caching will be limited',
                        'in utility!', level=0)
        else:
            wrapped_load_func = load_func

        # Init if needed
        self._check_file_mapping()

        # Get dict of key to files
        file_series = proc_file_input(files, file_to_subject)

        # Get next file mapping ind
        cnt = self._get_next_ind()

        # For each column
        for file in file_series:

            # For each subject, fill in with Data File
            series = file_series[file]

            for subject in series.index:

                # Create data file and add to file mapping
                data_file = Data_File(series[subject], wrapped_load_func)
                self.file_mapping[cnt] = data_file

                # Replace cnt index in data
                self.at[subject, file] = cnt

                # Increment
                cnt += 1

            # Set scope
            self.add_scope(file, 'data file')

    def _get_next_ind(self):

        if len(self.file_mapping) > 0:
            return np.nanmax(self.file_mapping.keys()) + 1
        else:
            return 0

    def _get_problem_type(self, col):
        '''Return the default problem type for a given column.'''

        self._check_scopes()

        if 'category' in self.scopes[col]:
            if self[col].nunique(dropna=True) == 2:
                return 'binary'
            return 'categorical'
        return 'regression'

    def rename(self, **kwargs):
        print('Warning: rename might cause errors!')
        print('Until this is supported, re-name before casting to a Dataset.')
        return super().rename(**kwargs)

    def _is_category(self, col):
        '''Tests if a column is categorical'''

        self._check_scopes()

        # Make sure valid column
        if col not in self.columns:
            raise KeyError('Passed: ' + repr(col) + ' not a valid loaded col.')

        # Only check is there is a scope for this col
        if col in self.scopes:
            return 'category' in self.scopes[col]

        # If no scope, assume isn't category
        return False

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

        # Deep copy rest of meta data
        for m in self._metadata:
            if hasattr(self, m):
                c = getattr(dataset, m)

                if deep:
                    setattr(dataset, m, deepcopy(c))
                else:
                    setattr(dataset, m, copy(c))

        return dataset

    def get_Xy(self, problem_spec, subjects='all', as_np=True):
        from ..main.funcs import problem_spec_check

        # Get proc'ed problem spec
        ps = problem_spec_check(problem_spec, dataset=self)

        # Get subjects as intersection between problem spec subjects
        # and passed subjects.
        ps_subjects =\
            self.get_subjects(ps.subjects, return_as='set')
        subjects = self.get_subjects(subjects, return_as='set')
        overlap = ps_subjects.intersection(subjects)

        # Sort subjects for reproducibility
        overlap = sorted(list(overlap))

        # Get X cols
        X_cols = self.get_cols('data', limit_to=ps.scope)

        # Get X as pandas df subset, y as Series
        X = pd.DataFrame(self.loc[overlap, X_cols]).astype(ps.base_dtype)
        y = self.loc[overlap, ps.target].astype('float64')

        # Get as np arrays
        if as_np:
            X = np.array(X)
            y = np.array(y)

        return X, y

    def get_train_Xy(self, problem_spec):
        return self.get_Xy(problem_spec, subjects='train')

    def get_test_Xy(self, problem_spec):
        return self.get_Xy(problem_spec, subjects='test')

    def _repr_html_(self):

        template = """<div style="float: left; padding: 10px;"><h3>{0}</h3>{1}</div>"""

        html = ''
        for scope in ['data', 'target', 'non input']:
            cols = self.get_cols(scope)
            if len(cols) > 0:
                html += template.format(scope,
                                        self[cols]._base_repr_html_())
                html += '\n'

        return html

    def _base_repr_html_(self):

        html = super()._repr_html_()

        try:
            train_subjects = self.get_subjects('train', return_as='set')
            test_subjects = self.get_subjects('test', return_as='set')
        except RuntimeError:
            return html

        train_color = 'RGBA(176, 224, 230, .3)'
        test_color = 'RGBA(249, 121, 93, .3)'

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

        train_info = '<p><span style="background: ' + train_color + '">'
        train_info += str(len(train_subjects)) + " rows × "
        train_info += n_cols + " columns - Train Set </span></p>"

        test_info = '<p><span style="background: ' + test_color + '">'
        test_info += str(len(test_subjects)) + " rows × "
        test_info += n_cols + " columns - Test Set </span></p>"

        extra_info = train_info + test_info

        # Add before end
        html = html[:-6] + extra_info + html[-6:]

        return html

    from ._plotting import (plot,
                            show,
                            info,
                            plot_bivar,
                            _plot_cat_cat,
                            _plot_cat_float,
                            _plot_float_float,
                            plot_vars,
                            _get_plot_values,
                            _plot_float,
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
                            add_unique_overlap)

    from ._validation import (_validate_cv_key,
                              _proc_cv_strategy,
                              _validate_split,
                              _finish_split,
                              set_test_split,
                              set_train_split,
                              save_test_subjects,
                              save_train_subjects)

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
