import pandas as pd
import numpy as np
from itertools import combinations


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
    _metadata = ['roles', 'scopes', 'name_map']

    @property
    def _constructor(self):
        return Dataset

    @property
    def _constructor_sliced(self):
        return pd.Series

    def _check_name_map(self):

        if not hasattr(self, 'name_map'):
            self.name_map = {}

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

    def auto_detect_categorical(self):
        pass

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

    def filter_categorical_by_percent(self, scope):

        return self

        # Add cast to categorical
        pass

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

    def binarize(self, scope, threshold=None, lower=None,
                 upper=None, replace=True):

        if threshold is None and lower is None and upper is None:
            raise RuntimeError('Some value must be set.')
        if lower is not None and upper is None:
            raise RuntimeError('Upper must be set.')
        if upper is not None and lower is None:
            raise RuntimeError('Lower must be set.')

        # Make sure name_map init'ed
        self._check_name_map()

        # Get cols from scope
        cols = self._get_cols_from_scope(scope)

        # Binarize each column
        for col in cols:
            self._binarize(col, threshold, lower, upper, replace)

        return self

    def _binarize(self, col, threshold, lower, upper, replace):

        # Extract values / series
        values = self[col]

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

            # Drop between the values
            to_drop = values[(values <= upper) &
                             (values >= lower)].index
            self.drop(to_drop, axis=0, inplace=True)

            # Binarize
            self[col] = self[col].where(self[col] > lower, 0)
            self[col] = self[col].where(self[col] < upper, 1)

            # Add to name map
            self.name_map[col] =\
                {0: '<' + str(lower), 1: '>' + str(upper)}

        # If a single threshold
        else:

            # Binarize
            self[col] = self[col].where(self[col] >= threshold, 0)
            self[col] = self[col].where(self[col] < threshold, 1)

            # Add to name map
            self.name_map[col] =\
                {0: '<' + str(threshold), 1: '>=' + str(threshold)}

        # Make sure category scope
        self.add_scope(col, 'category')

    def k_bin(self, scope):
        pass

    def ordinalize(self, scope):
        pass

    def nan_to_class(self, scope):

        # Have checks s.t., only works if all columns are categorical
        pass

    def drop_by_unique_val(self, scope):
        pass

    def print_nan_info(self, scope):
        pass

    # need to handle data files still***
    def add_data_files(self):
        pass

    def rename(self, **kwargs):
        print('Warning: Renaming might cause errors!')
        super().rename(**kwargs)

    from .Dataset_Plotting import (plot,
                                   _plot_category)


def get_fake_dataset():

    fake = Dataset()
    fake['1'] = [1, 2, 3]
    fake['2'] = ['6', '7', '8']
    fake['2'] = fake['2'].astype('category')
    fake['3'] = [np.nan, 2, 3]

    return fake


def get_fake_dataset2():

    fake = Dataset()
    fake['1'] = [1, 1, 1]
    fake['2'] = [1, 2, 3]
    fake['3'] = ['1', '2', '3']

    return fake


def get_fake_dataset3():

    fake = Dataset()
    fake['1'] = [1, 1, 1]
    fake['2'] = [1, 1, 1]
    fake['3'] = ['2', '2', '2']
    fake['4'] = ['2', '2', '2']
    fake['5'] = ['2', 1, '2']

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

    df = get_fake_dataset()
    df.add_scope(col='1', scope='category')
    assert(df['1'].dtype.name == 'category')


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


def test_drop_non_unique():

    df = get_fake_dataset2()
    df.drop_non_unique()

    assert '1' not in df
    assert '2' in df
    assert '3' in df


def test_drop_id_cols():

    df = get_fake_dataset2()
    df.drop_id_cols()

    assert '1' in df
    assert '2' in df
    assert '3' not in df


def test_drop_duplicate_cols():

    df = get_fake_dataset3()
    df.drop_duplicate_cols()

    assert '5' in df
    assert df.shape == (3, 3)


def test_apply_inclusions():

    df = get_fake_dataset3()
    df.apply_inclusions([0])
    assert len(df) == 1


def test_apply_exclusions():

    df = get_fake_dataset()
    df.apply_exclusions([0, 1])
    assert len(df) == 1

    df = get_fake_dataset()
    df.apply_exclusions([0])
    assert len(df) == 2


def test_drop_cols_inclusions():

    df = get_fake_dataset()
    df.drop_cols(inclusions='1')
    assert '1' in df
    assert df.shape[1] == 1

    df = get_fake_dataset()
    df.drop_cols(inclusions='category')
    assert '2' in df

    df = get_fake_dataset()
    df.drop_cols(inclusions=['1', '2'])
    assert df.shape[1] == 2

    df = Dataset(columns=['xxx1', 'xxx2', 'xxx3', '4'])
    df.drop_cols(inclusions=['xxx'])
    assert '4' not in df
    assert df.shape[1] == 3


def test_drop_cols_exclusions():

    df = get_fake_dataset()
    df.drop_cols(exclusions='1')
    assert '1' not in df
    assert df.shape[1] == 2

    df = get_fake_dataset()
    df.drop_cols(exclusions=['1', '2'])
    assert '3' in df
    assert df.shape[1] == 1

    df = Dataset(columns=['xxx1', 'xxx2', 'xxx3', '4'])
    df.drop_cols(exclusions=['xxx'])
    assert '4' in df
    assert df.shape[1] == 1


def test_binarize_threshold():

    df = get_fake_dataset()
    df.binarize('1', threshold=1.5)

    assert df.loc[0, '1'] == 0
    assert df.loc[1, '1'] == 1
    assert 'category' in df.scopes['1']
    assert df.name_map['1'] == {0: '<1.5', 1: '>=1.5'}


def test_binarize_upper_lower():

    df = get_fake_dataset()
    df.binarize('1', lower=2, upper=2)

    assert len(df) == 2
    assert df.loc[0, '1'] == 0
    assert df.loc[2, '1'] == 1
    assert 'category' in df.scopes['1']
    assert df.name_map['1'] == {0: '<2', 1: '>2'}


test_add_scope()
test_set_roles()
test_get_cols_from_scope()
test_filter_outliers()
test_drop_non_unique()
test_drop_id_cols()
test_drop_duplicate_cols()
test_apply_inclusions()
test_apply_exclusions()
test_drop_cols_inclusions()
test_drop_cols_exclusions()
test_binarize_threshold()
test_binarize_upper_lower()
