from operator import add
from functools import reduce
import pandas as pd
import numpy as np
from itertools import combinations
import warnings
from sklearn.preprocessing import LabelEncoder
from joblib import wrap_non_picklable_objects
from ..helpers.Data_File import Data_File, load_data_file_proxy
from copy import copy as shallow_copy
from copy import deepcopy
from ..helpers.ML_Helpers import conv_to_list
from .Dataset_helpers import (base_load_subjects, proc_file_input,
                              proc_fop, add_new_categories)
from .Input_Tools import Value_Subset


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

    def _check_test_subjects(self):

        if not hasattr(self, 'test_subjects'):
            self.test_subjects = None

    def _check_train_subjects(self):

        if not hasattr(self, 'train_subjects'):
            self.train_subjects = None

    def _check_file_mapping(self):

        if not hasattr(self, 'file_mapping'):
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

    def get_subjects(self, subjects, return_as='set'):
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

        return_as : {'set', 'array', 'index'}, optional
            - If 'set', return as set of subjects.
            - If 'array', return as sorted numpy array.
            - If 'index', return as sorted pandas index.
        '''

        # Check if None
        if subjects is None:
            loaded_subjects = set()

        # Check for special nan keyword
        elif isinstance(subjects, str) and subjects == 'nan':
            nan_subjects = self[pd.isnull(self[:]).any(axis=1)].index
            loaded_subjects = set(list(nan_subjects))

        # Check for Value Subset or Values Subset
        elif isinstance(subjects, Value_Subset):

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

            # Make sure subjects is set-like
            loaded_subjects = set(list(loaded_subjects))

        else:
            loaded_subjects = base_load_subjects(subjects)

        # Cast to correct type
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

        if return_as == 'set':
            return loaded_subjects

        # Array or index
        else:

            # Sort and cast to array
            loaded_subjects = np.array(sorted(list(loaded_subjects)))

            if return_as == 'array':
                return loaded_subjects
            elif return_as == 'index':
                return pd.Index(data=loaded_subjects, name=self.index.name)

        raise TypeError('Invalid parameter passed to as!')

    def apply_inclusions(self, subjects):
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

        '''

        # Load inclusions
        inclusions = self.get_subjects(subjects, return_as='set')

        if len(inclusions) > 0:
            to_drop = set(self.index) - inclusions
            self.drop(list(to_drop), axis=0, inplace=True)

        return self

    def apply_exclusions(self, subjects):

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
            super().rename(mapper=rename, index='columns')

            self._print('Warning: the columns: ', repr(non_str),
                        'were cast to str', level=0)

    def _check_roles(self):

        # Make sure cols type str
        self._check_cols_type()

        if not hasattr(self, 'roles'):
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

        # Set as role
        self.roles[col] = role

        # If role is non input, can't have any NaN
        if role == 'non input':
            self.drop_nan_subjects(scope=col)

        return self

    def drop_nan_subjects(self, scope):
        '''This method is a quick helper for
        dropping all of the subject's which have NaN
        values for just a given scope.

        '''

        cols = self.get_cols(scope)
        for col in cols:
            nan_subjects = self._get_nan_subjects(col)
            self.drop(nan_subjects, inplace=True)

    def _check_scopes(self):

        # Make sure cols type str
        self._check_cols_type()

        # If doesnt exist, create scopes
        if not hasattr(self, 'scopes'):
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
            if self[col].dtype.name == 'category':
                self._add_scope(col, 'category')
            else:
                self._remove_scope(col, 'category')

        return self

    def get_scopes(self):

        self._check_scopes()
        return self.scopes

    def add_scope(self, col, scope):
        '''This method is used for adding scopes
        to an existing column.

        Note: All passed scopes if not a str are
        cast to a str! This means that if a scope of
        say 1 is set, and a scope of "1", they will
        be equivilent.
        '''

        self._check_scopes()
        self._add_scope(col, scope)

        return self

    def add_scopes(self, col_to_scopes):

        self._check_scopes()
        for col, scope in zip(col_to_scopes, col_to_scopes.values()):
            self._add_scope(col, scope)

        return self

    def _add_scope(self, col, scope):

        # Check if category
        if scope == 'category':
            self.scopes[col].add(scope)
            self[col] = self[col].astype('category')

            return self

        # If a int or float
        if isinstance(scope, int) or isinstance(scope, float):
            scope = str(scope)

            # @TODO Add a verbose here saying cast to str?

        # If str
        if isinstance(scope, str):

            # Make sure doesn't overlap with loaded column
            if scope in list(self):
                raise RuntimeError('Warning scope of: ' + scope + ' '
                                   'overlaps with a loaded column. This '
                                   'can cause index errors, as if this '
                                   'scope is '
                                   'requested, then the column will be '
                                   'returned '
                                   'instead of this scope!')

            # Add
            self.scopes[col].add(scope)

        # Or assume array-like
        else:
            for s in set(list(scope)):
                self._add_scope(col, s)

        return self

    def remove_scope(self, col, scope):

        self._check_scopes()
        self._remove_scope(col, scope)

        return self

    def remove_scopes(self, col_to_scopes):

        self._check_scopes()
        for col, scope in zip(col_to_scopes, col_to_scopes.values()):
            self._remove_scope(col, scope)

        return self

    def _remove_scope(self, col, scope):

        try:
            self.scopes[col].remove(scope)

            # If removing category and currently pandas dtype is category,
            # change to float32.
            if scope == 'category' and \
               self.scopes[col].dtype.name == 'category':
                self[col] = self[col].astype('float32')

        except KeyError:
            pass

        return self

    def get_cols(self, scope):
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

        saved_scopes = set()
        for col in self.columns:
            saved_scopes.update(self.scopes[col])

        # If int or flot, cast to str.
        if isinstance(scope, int) or isinstance(scope, float):
            scope = str(scope)

        # Check is passed scope is reserved
        if isinstance(scope, str):
            if scope in self.RESERVED_SCOPES:

                if scope == 'all':
                    return list(self.columns)

                # Float refers to essentially not category and not data file
                elif scope == 'data float':
                    return [col for col in self.columns if
                            'category' not in self.scopes[col] and
                            'data file' not in self.scopes[col] and
                            self.roles['col'] == 'data']

                elif scope == 'target float':
                    return [col for col in self.columns if
                            'category' not in self.scopes[col] and
                            'data file' not in self.scopes[col] and
                            self.roles['col'] == 'target']

                elif scope == 'float':
                    return [col for col in self.columns if
                            'category' not in self.scopes[col] and
                            'data file' not in self.scopes[col]]

                elif scope == 'category':
                    return [col for col in self.columns if
                            'category' in self.scopes[col]]

                elif scope == 'data category':
                    return [col for col in self.columns if
                            'category' in self.scopes[col] and
                            self.roles['col'] == 'data']

                elif scope == 'target category':
                    return [col for col in self.columns if
                            'category' in self.scopes[col] and
                            self.roles['col'] == 'target']

                elif scope == 'data':
                    return [col for col in self.columns if
                            self.roles[col] == 'data']

                elif scope == 'target':
                    return [col for col in self.columns if
                            self.roles[col] == 'target']

                elif scope == 'non input':
                    return [col for col in self.columns if
                            self.roles[col] == 'non input']

                elif scope == 'data file':
                    return [col for col in self.columns if
                            'data file' in self.scopes[col]]

            # Check if passed scope is a loaded column
            elif scope in self.columns:
                return [scope]

            # Check if a saved scope
            elif scope in saved_scopes:
                return [col for col in self.columns if
                        scope in self.scopes[col]]

            # Do a search, see if passed scope is a stub of any strings
            else:
                return [col for col in self.columns if
                        scope in col]

        cols = []
        for scp in scope:
            cols += self.get_cols(scp)

        return sorted(list(set(cols)))

    def get_values(self, col, dropna=True, reduce_func=np.mean, n_jobs=1):
        '''Returns either normal values or data file proxy values, e.g.,
        (pandas Series) for the passed column, with or without NaN values
        included.

        Parameters
        -----------
        col : str
            The name of the column in which to load values for.

        dropna : bool, optional
            Boolean, if True, return only non-nan values.
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

        '''

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

        Fixed behavior is that any column with only two unique non-nan values
        is considered binary and therefore categorical.

        If any data file's within scope, will always treat as not categorical,
        and further will make sure it is not categorical if for some reason
        set as that.

        Parameters
        -----------
        scope : :ref:`Scopes`
            A valid BPt style scope used to select which columns this
            function should operate on.

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

    def _drop_or_nan(self, col, to_drop_index, all_to_drop, drop):

        # If drop, add to drop list at end
        if drop:
            all_to_drop.update(set(to_drop_index))

        # Otherwise, set to NaN in place
        else:
            self.loc[to_drop_index, col] = np.nan

    def filter_outliers_by_percent(self, fop=1, scope='float', drop=True,
                                   reduce_func=np.mean, n_jobs=1):
        '''This method is designed to allow dropping a fixed percent of outliers
        from the requested columns.

        Parameters
        -----------
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
            self.drop(list(all_to_drop), inplace=True)

        # Update file mapping if needed
        self._check_file_mapping()

        return self

    def filter_outliers_by_std(self, n_std=10, scope='float', drop=True,
                               reduce_func=np.mean, n_jobs=1):
        '''This method is designed to allow dropping outliers
        from the requested columns based on comparisons with that columns
        standard deviation.

        Parameters
        -----------
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
            self.drop(list(all_to_drop), inplace=True)

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
        '''This function is designed to allow performing outlier filting
        on categorical type variables. Note that this function assume
        all columns passed are of type 'category', and they if not already
        will be cast first to pandas data type 'category'.

        Note: NaN values will be skipped. If desired to treat them as a class,
        use the method nan_to_class to first.

        It is worth noting that this function will not work on data files.

        Parameters
        -----------
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
            self.drop(list(all_to_drop), inplace=True)

        return self

    def drop_non_unique(self, scope='all'):
        '''This method will drop any columns with only one unique value.'''

        # Get cols from scope
        cols = self.get_cols(scope)

        to_drop = []
        for col in cols:
            if len(self[col].unique()) == 1:
                to_drop.append(col)

        self.drop(to_drop, axis=1, inplace=True)
        return self

    def drop_id_cols(self, scope='all'):
        '''This method will drop any str-type / object type columns
        where the number of unique columns is equal
        to the length of the dataframe.'''

        # Get cols from scope
        cols = self.get_cols(scope)

        to_drop = []

        for col in cols:
            if self[col].dtype.name == 'object':
                if len(self[col].unique()) == len(self):
                    to_drop.append(col)

        self.drop(to_drop, axis=1, inplace=True)
        return self

    def drop_duplicate_cols(self, scope='all'):

        # Get cols from scope
        cols = self.get_cols(scope)

        to_drop = []
        for col1, col2 in combinations(cols, 2):
            if np.array_equal(self[col1], self[col2]):
                to_drop.append(col2)

        self.drop(to_drop, axis=1, inplace=True)
        return self

    def drop_cols(self,
                  exclusions=None,
                  inclusions=None,
                  scope='all'):

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

        self.drop(to_drop, axis=1, inplace=True)
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

    def binarize(self, scope, base=False, threshold=None, lower=None,
                 upper=None, replace=True, drop=True):
        '''This method contains a few different common utilities
        for binarizing a variable. These range from simply cleaning
        up an existing binary variable, to dichatomizing an existing one,
        to dichatomizing an existing one with two thresholds
        (essentially chopping out the middle of the distribution).

        Note: This function will not work on data files.

        Parameters
        -----------

        replace : bool, optional

            Note: Ignored if base == True.

        drop : bool, optional
            Note: This parameter is only relevant if using
            the configuration with parameter's upper and lower.

        '''

        # Input checks
        if base is False:
            if threshold is None and lower is None and upper is None:
                raise RuntimeError('Some value must be set.')
            if lower is not None and upper is None:
                raise RuntimeError('Upper must be set.')
            if upper is not None and lower is None:
                raise RuntimeError('Lower must be set.')

        # Make sure encoders init'ed
        self._check_encoders()

        # Get cols from scope
        cols = self.get_cols(scope)

        # Check for data files
        self._data_file_fail_check(cols)

        # Binarize each column individually
        for col in cols:

            # Choose which binarize behavior
            if base:
                self._base_binarize(col=col, drop=drop)
            else:
                self._binarize(col=col, threshold=threshold, lower=lower,
                               upper=upper, replace=replace, drop=drop)

        return self

    def _base_binarize(self, col, drop):

        # Extract non-nan values / series
        values = self.get_values(col, dropna=True)

        # Get non-nan counts
        unique_vals, counts = np.unique(values, return_counts=True)

        # If only 1 values
        if len(unique_vals) == 1:
            warnings.warn('binarize base=True ' + repr(col) + ' was '
                          'passed with only 1 unique value.')

        # Assuming should be binary, so 2 unique values
        if len(unique_vals) > 2:

            # Select top two scores by count
            keep_inds = np.argpartition(counts, -2)[-2:]
            keep_vals = unique_vals[keep_inds]
            keep_vals.sort()

            # Get to drop
            to_drop = values.index[~values.isin(keep_vals)]

            # Drop or NaN
            if drop:
                self.drop(list(to_drop), inplace=True)
            else:
                self.loc[to_drop, col] = np.nan

        # Ordinalize
        self._ordinalize(col)

    def _binarize(self, col, threshold, lower, upper, replace, drop):

        # Keep track of initial NaN subjects
        nan_subjects = self._get_nan_subjects(col)

        # Extract non-nan values / series
        values = self.get_values(col, dropna=True)

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
        self.add_scope(col, 'category')

        return self

    def k_bin(self, scope):
        pass

    def ordinalize(self, scope):

        # Get cols from scope
        cols = self.get_cols(scope)

        # Ordinalize each column individually
        for col in cols:
            self._ordinalize(col)

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
        self.add_scope(col, 'category')

    def get_encoded_values(self, col):
        '''Returns a copy of the column with values replaced, if any
        valid encoders.'''

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

    def nan_to_class(self, scope='category'):
        '''This method will cast any columns that were not categorical that are
        passed here to categorical. Will also ordinally encode them if
        they have not already been encoded
        (i.e., by either via ordinalize, binarize, or a simmilar function...).
        '''

        cols = self.get_cols(scope)

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

            # Update encoder entry
            self.encoders[col][nan_class] = np.nan

        return self

    def drop_by_unique_val(self, scope):
        pass

    def print_nan_info(self, scope):
        pass

    def copy_as_non_input(self, col, new_col, copy_scopes=False):
        '''This method is a used for making a copy of an
        existing column, ordinalizing it and then setting it
        to have role == non input.

        Parameters
        ----------
        col : str
            The name of the loaded column to make a copy of.

        new_col : str
            The new name of the non input column

        copy_scopes : bool, optional
            If the associated scopes with the original column should be copied
            over as well.

            ::

                default = False
        '''

        # Copy as new col
        self[new_col] = self[col].copy()

        # Only copy scopes if requested, and do before
        # ordinalize call
        if copy_scopes:
            self.scopes[new_col] = self.scopes[col].copy()

        # Ordinalize
        self._ordinalize(new_col)

        # Set new role
        self.set_role(new_col, 'non input')

    def add_data_files(self, files, file_to_subject, load_func):

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
                data_file = Data_File(series.loc[subject], wrapped_load_func)
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

    def rename(self, **kwargs):
        print('Warning: rename might cause errors!')
        print('Until this is supported, re-name before casting to a Dataset.')
        return super().rename(**kwargs)

    def copy(self, deep=True):

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

    def add_unique_overlap(self, cols, new_col, encoded_values=True):
        '''This function is designed to add a new column
        with the overlapped unique values from passed two or more columns.
        For example, say you had two binary columns, A and B. This function
        would compute a new column with 4 possible values, where:

        ::

            A == 0 and B == 0, A == 0 and B == 1,
            A == 1 and B == 0 and A == 1 and B == 1

            0    A=0 B=0
            1    A=0 B=1
            2    A=1 B=1
            3    A=0 B=0
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

        encoded_values : bool, optional
            This is an optional parameter, which is set to True
            will when creating the overlapping values will try to
            replace values with the encoded value (if any). For example
            if a variable being added had an originally encoded values of
            'cat' and 'dog', then the replace value before ordinalization
            would be col_name=cat and col_name=dog, vs. if set to False
            would have values of col_name=0 and col_name=1.

            ::

                default = True

        '''

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

            if encoded_values:
                vals = self.get_encoded_values(col)
            else:
                vals = self[col]

            combo.append(col + '=' + vals.astype(str) + ' ')

        # Combine
        combo = reduce(add, combo)

        # Add as new column
        self[new_col] = combo

        # If all roles agree, set new col as
        roles = set([self.roles[col] for col in cols])
        if len(roles) == 1:
            self.set_role(new_col, roles.pop())

        # Add scope if in all passed
        scopes = [self.scopes[col] for col in cols]
        for scope in scopes[0]:
            in_rest = [scope in other for other in scopes[1:]]
            if all(in_rest):
                self.add_scope(new_col, scope)

        # Lastly, ordinalize
        self._ordinalize(new_col)

    from .Dataset_Plotting import (plot,
                                   show,
                                   info,
                                   plot_vars,
                                   _plot_category)

    from .Dataset_Validation import (_validate_cv_key,
                                     _proc_cv,
                                     _validate_split,
                                     set_test_split,
                                     set_train_split,
                                     save_test_subjects,
                                     save_train_subjects)
