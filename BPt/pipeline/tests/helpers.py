import os
import tempfile
import numpy as np
from ...helpers.Data_File import Data_File
from ...helpers.Data_Scopes import Data_Scopes
from ...main.Params_Classes import Problem_Spec
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection._base import SelectorMixin
from ...main.Params_Classes import Param_Search
from ...helpers.CV import CV


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


class FakeSelector(SelectorMixin, BaseEstimator):

    def __init__(self, mask):
        self.mask = mask

    def fit(self, X, y):
        return self

    def _get_support_mask(self):
        return self.mask


def get_param_search():

    param_search = Param_Search(search_type='RandomSearch',
                                splits=3,
                                n_repeats=1,
                                cv='default',
                                n_iter=10,
                                scorer='default',
                                weight_scorer=False,
                                mp_context='loky',
                                n_jobs='default',
                                dask_ip=None,
                                memmap_X=False,
                                search_only_params=None)

    param_search.set_random_state(1)
    param_search.set_n_jobs(2)
    param_search.set_scorer('regression')
    param_search.set_cv(CV())
    param_search.set_split_vals(None)

    return param_search


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

        mapping[i] = Data_File(loc=loc, load_func=np.load)

    return mapping


def clean_fake_mapping(n):

    locs = get_temp_files(n)
    for loc in locs:
        os.unlink(loc)


def get_fake_data_scopes(data_keys=None,
                         cat_keys=None):

    if data_keys is None:
        data_keys = []

    if cat_keys is None:
        cat_keys = []

    data_scopes =\
        Data_Scopes(data_keys=data_keys,
                    data_file_keys=[],
                    cat_keys=cat_keys,
                    strat_keys=[],
                    covars_keys=[],
                    file_mapping=None)

    fake_ps = Problem_Spec(target='target', scope='all')
    data_scopes.set_all_keys(fake_ps)

    return data_scopes
