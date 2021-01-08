import pandas as pd
import numpy as np
from itertools import combinations
import warnings
from sklearn.preprocessing import LabelEncoder


def proc_fop(fop):

    # If provided as just % number, divide by 100
    if not isinstance(fop, tuple):
        fop /= 100
        return (fop, 1-fop)

    elif fop[0] is None:
        return tuple([None, 1-(fop[1] / 100)])

    elif fop[1] is None:
        return tuple([fop[0] / 100, None])

    return tuple([fop[0]/100, 1-(fop[1] / 100)])


def load_subjects(subjects):

    # Include checks here for Value_Subset and Value Subsets!!!
    # @TODO

    loaded_subjects = set()

    if isinstance(subjects, str):
        with open(subjects, 'r') as f:
            lines = f.readlines()

            for line in lines:
                subject = line.rstrip()
                loaded_subjects.add(subject)

    else:
        loaded_subjects = set([s for s in subjects])

    return loaded_subjects


class Dataset(pd.DataFrame):

    ROLES = set(['data', 'target', 'non input'])
    RESERVED_SCOPES = set(['all', 'float', 'category',
                           'data', 'data file',
                           'non input', 'target'])
    _metadata = ['roles', 'scopes', 'encoders']

    @property
    def _constructor(self):
        return Dataset

    @property
    def _constructor_sliced(self):
        return pd.Series

    def _check_encoders(self):

        if not hasattr(self, 'encoders'):
            self.encoders = {}

    def apply_inclusions(self, subjects):

        # Load inclusions
        inclusions = load_subjects(subjects)

        if len(inclusions) > 0:
            to_drop = set(self.index) - inclusions
            self.drop(to_drop, axis=0, inplace=True)

        return self

    def apply_exclusions(self, subjects):

        # Load exclusions
        exclusions = load_subjects(subjects)

        if len(exclusions) > 0:
            to_drop = set(self.index).intersection(exclusions)
            self.drop(to_drop, axis=0, inplace=True)

        return self

    def _check_roles(self):

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

        return self

    def _check_scopes(self):

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

        self._check_scopes()
        self._add_scope(col, scope)

        return self

    def add_scopes(self, col_to_scopes):

        self._check_scopes()
        for col, scope in zip(col_to_scopes, col_to_scopes.values()):
            self._add_scope(col, scope)

        return self

    def _add_scope(self, col, scope):

        if scope == 'category':
            self.scopes[col].add(scope)
            self[col] = self[col].astype('category')

        elif isinstance(scope, str):
            self.scopes[col].add(scope)

        elif isinstance(scope, int) or isinstance(scope, float):
            self.scopes[col].add(str(scope))

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

    def _get_cols_from_scope(self, scope):

        # Perform checks first
        self._check_scopes()
        self._check_roles()

        saved_scopes = set()
        for col in self.columns:
            saved_scopes.update(self.scopes[col])

        # Check is passed scope is reserved
        if isinstance(scope, str):
            if scope in self.RESERVED_SCOPES:

                if scope == 'all':
                    return list(self.columns)

                elif scope == 'float':
                    return [col for col in self.columns if
                            'category' not in self.scopes[col] and
                            'data file' not in self.scopes[col]]

                elif scope == 'category':
                    return [col for col in self.columns if
                            'category' in self.scopes[col]]

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
            cols += self._get_cols_from_scope(scp)

        return sorted(list(set(cols)))

    def _get_non_nan_values(self, col):

        # Extract values / series
        values = self[col]

        # Determine nan and non nan subjects
        nan_subjects = values[values.isna()].index
        non_nan_subjects = values[~values.isna()].index

        return values.loc[non_nan_subjects], nan_subjects

    def get_category_cols(self):
        '''Return all categorical columns (sorted for reproducible behavior)'''

        scopes = self.get_scopes()

        category_cols = []
        for col in list(self.columns):
            if 'category' in scopes[col]:
                category_cols.append(col)

        return sorted(category_cols)

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
        cols = self._get_cols_from_scope(scope)

        for col in cols:

            # Get non-nan values to check
            values, _ = self._get_non_nan_values(col)

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

    def filter_outliers_by_percent(self, fop=1, scope='float', drop=True):

        # Proc fop
        fop = proc_fop(fop)
        lower, upper = fop

        # Get cols from scope
        cols = self._get_cols_from_scope(scope)

        if drop:
            to_drop = set()

            if lower is not None:
                mask = self[cols] < self[cols].quantile(lower)
                to_drop.update(set(self[cols][mask.any(axis=1)].index))

            if upper is not None:
                mask = self[cols] > self[cols].quantile(upper)
                to_drop.update(set(self[cols][mask.any(axis=1)].index))

            self.drop(list(to_drop), inplace=True)

        else:

            for c in cols:
                if lower is not None:
                    l_thresh = self[c] < self[c].quantile(lower)
                else:
                    l_thresh = None

                if upper is not None:
                    u_thresh = self[c] > self[c].quantile(upper)
                else:
                    u_thresh = None

                if lower is not None:
                    self.loc[l_thresh, c] = np.nan
                if upper is not None:
                    self.loc[u_thresh, c] = np.nan

        return self

    def filter_outliers_by_std(self, n_std=10, scope='float', drop=True):

        if not isinstance(n_std, tuple):
            n_std = (n_std, n_std)

        # Get cols from scope
        cols = self._get_cols_from_scope(scope)

        if drop:

            mean = self[cols].mean()
            std = self[cols].std()

            to_drop = set()

            if n_std[0] is not None:
                l_scale = mean - (n_std[0] * std)
                mask = self[cols] < l_scale
                to_drop.update(set(self[cols][mask.any(axis=1)].index))

            if n_std[1] is not None:
                u_scale = mean + (n_std[1] * std)
                mask = self[cols] > u_scale
                to_drop.update(set(self[cols][mask.any(axis=1)].index))

            self.drop(list(to_drop), inplace=True)

        else:

            for c in cols:

                if n_std[0] is not None:
                    l_scale = self[c].mean() - (n_std[0] * self[c].std())
                else:
                    l_scale = None

                if n_std[1] is not None:
                    u_scale = self[c].mean() + (n_std[1] * self[c].std())
                else:
                    u_scale = None

                if n_std[0] is not None:
                    self.loc[self[c] < l_scale, c] = np.nan
                if n_std[1] is not None:
                    self.loc[self[c] > u_scale, c] = np.nan

        return self

    def filter_categorical_by_percent(self, drop_percent=1, scope='category',
                                      drop=True):
        '''This function is designed to allow performing outlier filting
        on categorical type variables. Note that this function assume
        all columns passed are of type 'category', and they if not already
        will be cast first to pandas data type 'category'.

        Note: NaN values will be skipped. If desired to treat them as a class,
        use nan_to_class to first.
        '''

        # Get cols from scope
        cols = self._get_cols_from_scope(scope)

        # For if to drop
        all_to_drop = set()

        # Divide drop percent by 100
        dp = drop_percent / 100

        for col in cols:

            # Make sure categorical
            self.add_scope(col, 'category')

            # Extract non-nan values and subjects
            values, nan_subjects = self._get_non_nan_values(col)

            # Get to drop subjects
            unique_vals, counts = np.unique(values, return_counts=True)
            drop_inds = np.where((counts / len(values)) < dp)

            drop_vals = unique_vals[drop_inds]
            to_drop = values.index[values.isin(drop_vals)]

            # If drop, add to drop list at end
            if drop:
                all_to_drop.update(set(to_drop))

            # Otherwise, set to NaN in place
            else:
                self.loc[to_drop, col] = np.nan

            # Make sure nan subjects nan
            self.loc[nan_subjects, col] = np.nan

        # Drop now, if to drop
        if drop:
            self.drop(list(all_to_drop), inplace=True)

        return self

    def drop_non_unique(self, scope='all'):
        '''This method will drop any columns with only one unique value.'''

        # Get cols from scope
        cols = self._get_cols_from_scope(scope)

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
        cols = self._get_cols_from_scope(scope)

        to_drop = []

        for col in cols:
            if self[col].dtype.name == 'object':
                if len(self[col].unique()) == len(self):
                    to_drop.append(col)

        self.drop(to_drop, axis=1, inplace=True)
        return self

    def drop_duplicate_cols(self, scope='all'):

        # Get cols from scope
        cols = self._get_cols_from_scope(scope)

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
        cols = self._get_cols_from_scope(scope)

        if exclusions is not None:
            exclusions = self._get_cols_from_scope(exclusions)
        else:
            exclusions = []

        if inclusions is not None:
            inclusions = self._get_cols_from_scope(inclusions)
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
        '''
        Note: drop is only relevant if using upper and lower.

        Parameters:

        replace : bool, optional

            Note: Ignored if base == True.

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
        cols = self._get_cols_from_scope(scope)

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
        values, _ = self._get_non_nan_values(col)

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

        # Extract non-nan values / series
        values, nan_subjects = self._get_non_nan_values(col)

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

        # Make sure encoders init'ed
        self._check_encoders()

        # Get cols from scope
        cols = self._get_cols_from_scope(scope)

        # Ordinalize each column individually
        for col in cols:
            self._ordinalize(col)

    def _ordinalize(self, col):

        # Get non-nan values
        all_values = self[col]
        non_nan_subjects = all_values[~all_values.isna()].index
        values = all_values.loc[non_nan_subjects]

        # Perform actual encoding
        label_encoder = LabelEncoder()
        self.loc[non_nan_subjects, col] =\
            label_encoder.fit_transform(np.array(values))

        # Convert label encoder to encoder
        encoder = {}
        for i, c in enumerate(label_encoder.classes_):
            encoder[i] = c

        # Save in encoders
        self.encoders[col] = encoder

        # Make sure col is category type
        self.add_scope(col, 'category')

    def _replace_values(self, col, values):

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

        cols = self._get_cols_from_scope(scope)

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
            if nan_class not in self[col].dtype.categories:
                self[col].cat.add_categories(nan_class, inplace=True)
            self.loc[nan_subjects, col] = nan_class

            # Update encoder entry
            self.encoders[col][nan_class] = np.nan

        return self

    def drop_by_unique_val(self, scope):
        pass

    def print_nan_info(self, scope):
        pass

    # need to handle data files still***
    def add_data_files(self):
        pass

    def rename(self, **kwargs):
        print('Warning: rename might cause errors!')
        print('Until this is supported, re-name before casting to a Dataset.')
        return super().rename(**kwargs)

    from .Dataset_Plotting import (plot,
                                   _plot_category)
