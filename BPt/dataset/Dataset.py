import pandas as pd
import numpy as np
from itertools import combinations, product

from joblib import wrap_non_picklable_objects
from ..helpers.Data_File import Data_File, load_data_file_proxy
from copy import copy as shallow_copy
from copy import deepcopy
from ..helpers.ML_Helpers import conv_to_list
from .helpers import (base_load_subjects, proc_file_input,
                      proc_fop)
from ..main.Input_Tools import Value_Subset


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
            if not hasattr(self, 'train_subjects'):
                raise RuntimeError('Train subjects undefined')
            loaded_subjects = self.train_subjects

        elif isinstance(subjects, str) and subjects == 'test':
            if not hasattr(self, 'test_subjects'):
                raise RuntimeError('Test subjects undefined')
            loaded_subjects = self.test_subjects

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

    def apply_inclusions(self, subjects, only_level=None):
        '''This method will drop all subjects
        that do not overlap with the passed subjects to
        this function. In this sense, this method acts
        like a whitelist, where you could pass for example
        only valid subjects that passed some QC, and all
        other's loaded will be dropped.

        This method operates in place.

        Parameters
        -----------
        subjects : :ref:`Subjects`
            This argument can be any of the BPt accepted
            subject style inputs. E.g., None, 'nan' for subjects
            with any nan data, the str location of a file
            formatted with one subject per line, or directly an
            array-like of subjects, to name some options.

            See :ref:`Subjects` for all options.
        '''

        # Load inclusions
        inclusions = self.get_subjects(subjects, return_as='set')

        if len(inclusions) > 0:
            to_drop = set(self.index) - inclusions
            self.drop(list(to_drop), axis=0, inplace=True)

        return self

    def apply_exclusions(self, subjects):
        '''This method will drop all subjects
        that overlap with the passed subjects to
        this function.

        This method operates in place.

        Parameters
        -----------
        subjects : :ref:`Subjects`
            This argument can be any of the BPt accepted
            subject style inputs. E.g., None, 'nan' for subjects
            with any nan data, the str location of a file
            formatted with one subject per line, or directly an
            array-like of subjects, to name some options.

            See :ref:`Subjects` for all options.
        '''

        # Load exclusions
        exclusions = self.get_subjects(subjects, return_as='set')

        if len(exclusions) > 0:
            to_drop = set(self.index).intersection(exclusions)
            self.drop(list(to_drop), axis=0, inplace=True)

        return self

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

    def get_roles(self):

        self._check_roles()
        return self.roles

    def set_role(self, col, role):

        self._check_roles()
        self._set_role(col, role)

        return self

    def set_roles(self, col_to_roles):

        self._check_roles()
        for col, role in zip(col_to_roles, col_to_roles.values()):
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

    def _drop_subjects(self, subjects):

        if len(subjects) > 0:

            self.drop(subjects, axis=0, inplace=True)
            self._print('Dropped:', len(subjects), level=1)

        return self

    def drop_nan_subjects(self, scope):
        '''This method is used for
        dropping all of the subjects which have NaN
        values for a given scope / column.

        Parameters
        ----------
        scope : :ref:`Scope`
            The BPt style :ref:`Scope` input that will be
            used to determine which column names to drop
            subjects with missing values by.
        '''

        cols = self.get_cols(scope)
        for col in cols:
            nan_subjects = self._get_nan_subjects(col)
            self._drop_subjects(nan_subjects)

        return self

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

        # Make sure scopes includes if categorical or not
        for col in self.columns:

            try:
                if self[col].dtype.name == 'category':
                    self._add_scope(col, 'category')
                else:
                    self._remove_scope(col, 'category')
            except AttributeError:
                pass

        return self

    def get_scopes(self):

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
               self.scopes[col].dtype.name == 'category':
                self[col] = self[col].astype('float32')

        except KeyError:
            pass

        return self

    def get_cols(self, scope, columns='all'):
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

        columns : :ref:`Scope` or None, optional
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

        if columns is None:
            columns = self.columns
        else:
            columns = self.get_cols(columns, None)

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
                            self.roles['col'] == 'data']

                elif scope == 'target float':
                    return [col for col in columns if
                            'category' not in self.scopes[col] and
                            'data file' not in self.scopes[col] and
                            self.roles['col'] == 'target']

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
                            self.roles['col'] == 'data']

                elif scope == 'target category':
                    return [col for col in columns if
                            'category' in self.scopes[col] and
                            self.roles['col'] == 'target']

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

        data_cols = self.get_cols('data', columns=ps_scope)

        inds = [data_cols.index(k) for k in
                self.get_cols(scope, columns=data_cols)]

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

    def _drop_or_nan(self, col, to_drop_index, all_to_drop, drop):
        '''Internal helper function for commonly re-used drop or
        nan function.'''

        # If drop, add to drop list at end
        if drop:
            all_to_drop.update(set(to_drop_index))

        # Otherwise, set to NaN in place
        else:
            self.loc[to_drop_index, col] = np.nan

    def filter_outliers_by_percent(self, fop=1, scope='float', drop=True,
                                   reduce_func=np.mean, n_jobs=1):
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
        '''

        # Get cols from scope
        cols = self.get_cols(scope)

        # For if to drop
        all_to_drop = set()

        # Proc fop
        fop = proc_fop(fop)
        lower, upper = fop

        for col in cols:

            # Extract non-nan values
            values = self.get_values(col, dropna=True,
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

        return self

    def _get_file_mapping(self):

        self._check_file_mapping()
        return self.file_mapping

    def filter_outliers_by_std(self, n_std=10, scope='float', drop=True,
                               reduce_func=np.mean, n_jobs=1):
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
        '''

        # Get cols from scope
        cols = self.get_cols(scope)

        # For if to drop
        all_to_drop = set()

        # Wrap correctly passed n_std
        if not isinstance(n_std, tuple):
            n_std = (n_std, n_std)

        for col in cols:

            # Extract non-nan values
            values = self.get_values(col, dropna=True,
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

        return self

    def _data_file_fail_check(self, cols):

        for col in cols:
            if 'data file' in self.scopes[col]:
                raise RuntimeError('Loaded column: ' + col + ' cannot be a'
                                   ' data file.')

    def filter_categorical_by_percent(self, drop_percent=1, scope='category',
                                      drop=True):
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

        '''

        # Get cols from scope
        cols = self.get_cols(scope)

        # Check for data files
        self._data_file_fail_check(cols)

        # For if to drop
        all_to_drop = set()

        # Divide drop percent by 100
        dp = drop_percent / 100

        for col in cols:

            # Make sure categorical
            self.add_scope(col, 'category')

            # Extract non-nan values
            values = self.get_values(col, dropna=True)

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

        return self

    def _drop_cols(self, to_drop):

        if len(to_drop) > 0:

            self.drop(to_drop, axis=1, inplace=True)
            self._print('Dropped:', len(to_drop), 'columns', level=1)
            self._print('Dropped:', to_drop, level=2)

        return self

    def drop_id_cols(self, scope='all'):
        '''This method will drop any str-type / object type columns
        where the number of unique columns is equal
        to the length of the dataframe.

         scope : :ref:`Scope`, optional
            A BPt style :ref:`Scope` used to select a subset of
            columns in which to check.

            By default this is set to 'all' and will check all
            loaded columns

            ::

                default = 'all'

        '''

        # Get cols from scope
        cols = self.get_cols(scope)

        to_drop = []

        for col in cols:
            if self[col].dtype.name == 'object':
                if len(self[col].unique()) == len(self):
                    to_drop.append(col)

        self._drop_cols(to_drop)

        return self

    def drop_duplicate_cols(self, scope='all'):
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
        '''

        # Get cols from scope
        cols = self.get_cols(scope)

        to_drop = []
        for col1, col2 in combinations(cols, 2):
            if np.array_equal(self[col1], self[col2]):
                to_drop.append(col2)

        self._drop_cols(to_drop)
        return self

    def drop_cols(self,
                  exclusions=None,
                  inclusions=None,
                  scope='all'):
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

        '''

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

        return self

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

    def drop_cols_by_unique_val(self, threshold=1, scope='all', dropna=True):
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

        '''

        # Get cols from scope
        cols = self.get_cols(scope)

        to_drop = []
        for col in cols:
            values = self.get_values(col, dropna=dropna)
            if len(values.unique()) <= threshold:
                to_drop.append(col)

        self._drop_cols(to_drop)

        return self

    def drop_cols_by_nan(self, threshold=.5, scope='all'):
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
        '''

        # Get cols from scope
        cols = self.get_cols(scope)

        # Change threshold from percent to abs
        if threshold > 0 and threshold < 1:
            threshold = threshold * self.shape[0]
            self._print('Setting NaN threshold to:', threshold)

        to_drop = []
        for col in cols:
            values = self.get_values(col, dropna=False)
            nan_percent = values.isnull().sum()

            if nan_percent >= threshold:
                to_drop.append(col)

        self._drop_cols(to_drop)

        return self

    def drop_subjects_by_nan(self, threshold=.5, scope='all'):
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
        '''

        # Get cols from scope
        cols = self.get_cols(scope)

        # Get nan counts by column
        col_nan_cnts = self[cols].isnull().sum(axis=1)

        # If between 0 and 1
        if threshold > 0 and threshold < 1:

            # Change threshold from percent to abs
            threshold = threshold * len(cols)
            self._print('Setting NaN threshold to:', threshold)

        # Calculate subjects to drop if greater than or equal to threshold
        to_drop = self.loc[col_nan_cnts >= threshold].index

        self._drop_subjects(to_drop)

    def print_nan_info(self, scope):
        pass

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
            print('Warning: Passed load_func was defined within the',
                  '__main__ namespace and therefore has been cloud wrapped.',
                  'The function will still work, but it is reccomended to',
                  'define this function in a seperate file, and then import',
                  'it , otherwise loader caching will be limited',
                  'in utility!')
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
                    setattr(dataset, m, shallow_copy(c))

        return dataset

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

    from ._encoding import (to_binary,
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
