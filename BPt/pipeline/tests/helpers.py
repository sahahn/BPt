import os
import tempfile
import numpy as np
from ...dataset.data_file import DataFile
from ...main.input import ProblemSpec
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection._base import SelectorMixin
from ...main.input import ParamSearch
from ...main.CV import BPtCV, CVStrategy
from ...dataset.Dataset import Dataset


class ToFixedTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, to, n_jobs=1):
        self.to = to
        self.n_jobs = n_jobs

    def fit(self, X, y):
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X_trans = np.zeros(X.shape)
        X_trans[:] = self.to
        return X_trans


class IdentityListLoader(BaseEstimator, TransformerMixin):
    needs_mapping = True

    def __init__(self):
        pass

    def fit(self, X, y=None, mapping=None):
        assert mapping is not None
        return self

    def transform(self, X):

        assert isinstance(X, list)

        X_trans = []
        for x in X:
            X_trans.append(x.flatten())

        return np.array(X_trans)


class FakeSelector(SelectorMixin, BaseEstimator):

    def __init__(self, mask):
        self.mask = mask

    def fit(self, X, y):
        return self

    def _get_support_mask(self):
        return self.mask


def get_param_search():

    param_search = ParamSearch(search_type='RandomSearch',
                               cv='default',
                               n_iter=10,
                               scorer='default',
                               weight_scorer=False,
                               mp_context='loky',
                               n_jobs='default',
                               dask_ip=None,
                               memmap_X=False,
                               search_only_params=None,
                               progress_loc=None)

    ps = ProblemSpec(random_state=1,
                     n_jobs=2,
                     problem_type='regression')

    ps_dict = param_search._as_dict(ps)
    ps_dict['cv'] = BPtCV(splits=3, n_repeats=1,
                          cv_strategy=CVStrategy(), splits_vals=None)

    return ps_dict


def get_temp_files(n):

    temp_dr = tempfile.gettempdir()
    return [os.path.join(temp_dr, 'test_' + str(i) + '.npy') for i in range(n)]


def get_fake_mapping(n):

    locs = get_temp_files(n)

    mapping = {}
    for i, loc in enumerate(locs):
        data = np.zeros((2, 2))
        data[:] = i

        np.save(loc, data)

        mapping[i] = DataFile(loc=loc, load_func=np.load)

    return mapping


def clean_fake_mapping(n):

    locs = get_temp_files(n)
    for loc in locs:
        os.unlink(loc)


def get_fake_data_dataset(data_keys=None,
                          cat_keys=None):

    if data_keys is None:
        data_keys = []

    if cat_keys is None:
        cat_keys = []

    dataset = Dataset()

    for key in data_keys:
        dataset[key] = []
        dataset.set_role(key, 'data', inplace=True)

    for key in cat_keys:
        dataset[key] = []
        dataset.set_role(key, 'data', inplace=True)
        dataset.add_scope(key, 'category', inplace=True)

    dataset._check_scopes()

    return dataset
