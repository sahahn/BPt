import pandas as pd
import numpy as np


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


class Dataset(pd.DataFrame):

    ROLES = set(['data', 'target', 'non input'])
    RESERVED_SCOPES = set(['all', 'float', 'category',
                           'data', 'data file',
                           'non input', 'target'])
    _metadata = ['roles', 'scopes', 'name_mapping']

    @property
    def _constructor(self):
        return Dataset

    @property
    def _constructor_sliced(self):
        return pd.Series

    def _check_roles(self):

        if not hasattr(self, 'roles'):
            self.roles = {}

        # Fill in any column without a role.
        for col in self.columns:
            if col not in self.roles:
                self._set_role(col, 'data')

        # Remove any saved roles that are not
        # in the data any more.
        for col in self.roles:
            if col not in self.columns:
                del self.roles[col]

    def get_roles(self):

        self._check_roles()
        return self.roles

    def set_role(self, col, role):

        self._check_roles()
        self._set_role(col, role)

    def set_roles(self, col_to_roles):

        self._check_roles()
        for col, role in zip(col_to_roles, col_to_roles.values()):
            self._set_role(col, role)

    def _set_role(self, col, role):

        if role not in self.ROLES:
            raise AttributeError(
                'Passed role "' + str(role) + '" must be one of ' +
                str(self.ROLES))

        # Set as role
        self.roles[col] = role

    def _check_scopes(self):

        # If doesnt exist, create scopes
        if not hasattr(self, 'scopes'):
            self.scopes = {}

        # Make sure each col is init'ed with a scope
        for col in self.columns:
            if col not in self.scopes:
                self.scopes[col] = set()

        # Make sure to get rid of any removed cols
        for col in self.scopes:
            if col not in self.columns:
                del self.scopes[col]

        # Make sure scopes includes if categorical or not
        for col in self.columns:
            if self[col].dtype.name == 'category':
                self._add_scope(col, 'category')
            else:
                self._remove_scope(col, 'category')

    def get_scopes(self):

        self._check_scopes()
        return self.scopes

    def add_scope(self, col, scope):

        self._check_scopes()
        self._add_scope(col, scope)

    def add_scopes(self, col_to_scopes):

        self._check_scopes()
        for col, scope in zip(col_to_scopes, col_to_scopes.values()):
            self._add_scope(col, scope)

    def _add_scope(self, col, scope):

        # Add checks to make sure not adding reserved key word scope
        # i.e., 'all'.

        if isinstance(scope, str):
            self.scopes[col].add(scope)

        elif isinstance(scope, int) or isinstance(scope, float):
            self.scopes[col].add(str(scope))

        else:
            for s in set(list(scope)):
                self._add_scope(col, s)

    def remove_scope(self, col, scope):

        self._check_scopes()
        self._remove_scope(col, scope)

    def remove_scopes(self, col_to_scopes):

        self._check_scopes()
        for col, scope in zip(col_to_scopes, col_to_scopes.values()):
            self._remove_scope(col, scope)

    def _remove_scope(self, col, scope):

        try:
            self.scopes[col].remove(scope)
        except KeyError:
            pass

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

        return list(set(cols))

    def auto_detect_categorical(self):
        pass

    def filter_outliers_by_percent(self, fop=1, scope='float', drop=True):
        '''Right now is fixed to work as in place'''

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
                if upper is not None:
                    u_thresh = self[c] > self[c].quantile(upper)
                if lower is not None:
                    self.loc[l_thresh, c] = np.nan
                if upper is not None:
                    self.loc[u_thresh, c] = np.nan

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
                if n_std[1] is not None:
                    u_scale = self[c].mean() + (n_std[1] * self[c].std())
                if n_std[0] is not None:
                    self.loc[self[c] < l_scale, c] = np.nan
                if n_std[1] is not None:
                    self.loc[self[c] > u_scale, c] = np.nan

    def filter_categorical_by_percent(self, scope):

        # Add cast to categorical
        pass

    def binarize(self, scope):
        pass

    def k_bin(self, scope):
        pass

    def ordinalize(self, scope):
        pass

    def nan_to_class(self, scope):

        # Have checks s.t., only works if all columns are categorical
        pass

    def drop_duplicate_cols(self, scope):
        pass

    def drop_by_unique_val(self, scope):
        pass

    def print_nan_info(self, scope):
        pass

    # need to handle data files still***
    def add_data_files(self):
        pass

    # Plotting
    def plot_var(self, scope):
        pass

    def plot_vars(self, scope):
        pass


def get_fake_dataset():

    fake = Dataset()
    fake['1'] = [1, 2, 3]
    fake['2'] = ['6', '7', '8']
    fake['2'] = fake['2'].astype('category')
    fake['3'] = [np.nan, 2, 3]

    return fake


def test_add_scope():

    df = get_fake_dataset()

    df.add_scope(col='1', scope='a')
    assert df.scopes['1'] == set(['a'])

    df.add_scope(col='1', scope='b')
    assert df.scopes['1'] != set(['a'])
    assert df.scopes['1'] == set(['a', 'b'])

    # Test some get cols from scope
    assert(set(df._get_cols_from_scope('a')) == set(['1']))
    assert(set(df._get_cols_from_scope('b')) == set(['1']))
    assert(set(df._get_cols_from_scope(['a', 'b'])) == set(['1']))


def test_set_roles():

    df = get_fake_dataset()
    df.set_role('1', 'target')
    df.set_role('2', 'non input')

    assert(set(df._get_cols_from_scope('target')) == set(['1']))
    assert(set(df._get_cols_from_scope('non input')) == set(['2']))


def test_get_cols_from_scope():

    df = get_fake_dataset()
    assert(set(df._get_cols_from_scope('all')) == set(['1', '2', '3']))
    assert(set(df._get_cols_from_scope('data')) == set(['1', '2', '3']))
    assert(set(df._get_cols_from_scope('1')) == set(['1']))
    assert(set(df._get_cols_from_scope('category')) == set(['2']))


def test_filter_outliers():

    df = get_fake_dataset()
    df.filter_outliers_by_percent(20, scope='3', drop=False)
    assert pd.isnull(df['3']).all()


test_add_scope()
test_set_roles()
test_get_cols_from_scope()
test_filter_outliers()





# If the plotting is moved within this dataset class,
# could likewise move the feature importance plotting within its own class
# as well!