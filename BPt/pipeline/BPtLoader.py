from .helpers import (update_mapping, proc_mapping, get_reverse_mapping)
import numpy as np
from joblib import Parallel, delayed
from sklearn.utils.validation import check_memory
from sklearn.base import clone
from .ScopeObjs import ScopeTransformer
from operator import itemgetter
from .base import _get_est_trans_params
from ..util import get_top_substrs
import pandas as pd


def load_and_trans(transformer, load_func, loc):
    '''This function is designed to be able to be wrapped in a cache
    check_memory.'''

    data = load_func(loc)
    trans_data = np.squeeze(transformer.fit_transform(data))
    return trans_data


def get_trans_chunk(transformer, data_files, func):
    '''This function is designed to be used for multi-processing'''

    X_trans_chunk = []
    for DataFile in data_files:
        loc = DataFile.loc
        load_func = DataFile.load_func
        trans_data = func(clone(transformer), load_func, loc)
        X_trans_chunk.append(trans_data)

    return X_trans_chunk


class BPtLoader(ScopeTransformer):

    # Override
    _required_parameters = ["estimator", "inds", "file_mapping"]

    def __init__(self, estimator, inds, file_mapping,
                 n_jobs=1, fix_n_jobs=False,
                 cache_loc=None):
        '''The inds for loaders are special, they should not be
        set with Ellipsis. Instead in the case of all, should be
        passed inds as usual.'''

        # Set Super params
        super().__init__(estimator=estimator, inds=inds, cache_loc=cache_loc)

        # Set rest of params
        self.file_mapping = file_mapping

        # Make sure to set fix n jobs before n_jobs
        self.fix_n_jobs = fix_n_jobs
        self.n_jobs = n_jobs

    # Override inherited n_jobs propegate behavior
    @property
    def n_jobs(self):
        return self.n_jobs_proxy

    @n_jobs.setter
    def n_jobs(self, n_jobs):
        self.n_jobs_proxy = n_jobs

    @property
    def _n_jobs(self):

        if self.fix_n_jobs is False:
            return self.n_jobs

        return self.fix_n_jobs

    def fit(self, X, y=None, mapping=None,
            fit_index=None, **fit_params):

        # Need the output from a transform to full fit,
        # so when fit is called, call fit_transform instead
        self.fit_transform(X=X, y=y, mapping=mapping,
                           fit_index=fit_index,
                           **fit_params)

        return self

    def _fit(self, X, y=None, **fit_params):
        '''Special fit for loader, only concerned with
        fitting first data point for e.g., use of reverse
        transform
        '''

        # Get the first data point
        first_feat = self.inds_[0]
        fit_fm_key = X[0, first_feat]
        fit_X = self.file_mapping[int(fit_fm_key)].load()

        # Fit the first data point
        self.estimator_.fit(fit_X, y=y, **fit_params)

    def _update_loader_mappings(self, mapping):

        # Need to set out_mapping_ and update mapping
        self.out_mapping_ = {}

        # Add changed X_trans by col
        for c in range(len(self.inds_)):
            ind = self.inds_[c]
            self.out_mapping_[ind] = self.X_trans_inds_[c]

        # Fill the remaining spots sequentially,
        # for each of the rest inds.
        for c in range(len(self.rest_inds_)):
            ind = self.rest_inds_[c]
            self.out_mapping_[ind] = self.n_trans_feats_ + c

        # Update the original mapping, this is the mapping which
        # will be passed to the next piece of the pipeline
        update_mapping(mapping, self.out_mapping_)

    def fit_transform(self, X, y=None, mapping=None,
                      fit_index=None, **fit_params):

        if mapping is None:
            mapping = {}

        # Call parent fit but passing only the first data point
        super().fit(X, y=y, mapping=mapping,
                    fit_index=fit_index,
                    **fit_params)

        # If skipped, skip
        if self.estimator_ is None:
            return X

        # The parent fit takes care of, in addition to
        # fitting the loader on one
        # data point, sets base_dtype, processes the mapping,
        # sets rest inds, etc...

        # Now transform X - this sets self.X_trans_inds_
        X_trans = self.transform(X, transform_index=fit_index)

        # Update the mapping + out_mapping_
        self._update_loader_mappings(mapping)

        # Now return X_trans
        return X_trans

    def transform(self, X, transform_index=None):

        # Skip if skipped
        if self.estimator_ is None:
            return X

        # @ TODO transform index just exists for compat
        # with loader right now, won't actually propegate.

        # Init lists + mappings
        X_trans, self.X_trans_inds_ = [], []
        cnt = 0

        # For each column to fit_transform
        for col in self.inds_:

            # Get transformer column
            fm_keys = [key for key in X[:, col]]
            X_trans_cols = self._get_trans_col(fm_keys)

            # Stack + append new features
            X_trans_cols = np.stack(X_trans_cols)
            X_trans.append(X_trans_cols)

            # Add + append inds
            X_trans_cols_inds =\
                [i for i in range(cnt, X_trans_cols.shape[1] + cnt)]
            self.X_trans_inds_.append(X_trans_cols_inds)

            # Update cnt
            cnt = X_trans_cols.shape[1] + cnt

        # Stack final
        X_trans = np.hstack(X_trans)

        # Save number of output features after X_trans
        self.n_trans_feats_ = X_trans.shape[1]

        # Return stacked X_trans with rest inds
        return np.hstack([X_trans, X[:, self.rest_inds_]])

    def get_chunks(self, data_files):

        per_chunk = len(data_files) // self._n_jobs
        chunks = [list(range(i * per_chunk, (i+1) * per_chunk))
                  for i in range(self._n_jobs)]

        last = chunks[-1][-1]
        chunks[-1] += list(range(last+1, len(data_files)))
        return [[data_files[i] for i in c] for c in chunks]

    def _get_trans_col(self, fm_keys):

        # Grab the right data files from the file mapping (casting to int!)
        data_files = [self.file_mapping[int(fm_key)] for fm_key in fm_keys]

        # Clone the base loader
        cloned_estimator = clone(self.estimator)

        # If a caching location is passed, create new load_and_trans_c func
        if self.cache_loc is not None:
            memory = check_memory(self.cache_loc)
            load_and_trans_c = memory.cache(load_and_trans)
        else:
            load_and_trans_c = load_and_trans

        if self._n_jobs == 1:
            X_trans_cols = get_trans_chunk(cloned_estimator,
                                           data_files, load_and_trans_c)
        else:
            chunks = self.get_chunks(data_files)

            X_trans_chunks =\
                Parallel(n_jobs=self._n_jobs)(
                    delayed(get_trans_chunk)(
                        transformer=cloned_estimator,
                        data_files=chunk,
                        func=load_and_trans_c)
                    for chunk in chunks)

            X_trans_cols = []
            for chunk in X_trans_chunks:
                X_trans_cols += chunk

        return X_trans_cols

    def transform_df(self, df, base_name='loader', encoders=None):

        return super().transform_df(df, base_name=base_name)

    def _proc_new_names(self, feat_names, base_name=None, encoders=None):

        # If skip, return passed names as is
        if self.estimator_ is None:
            return feat_names

        # Store original passed feat names here
        self.feat_names_in_ = feat_names

        # If skip, return passed names as is
        if self.estimator_ is None:
            return feat_names

        # If base loader has stored feat names, use those.
        if hasattr(self.estimator_, 'feat_names_'):
            return getattr(self.estimator_, 'feat_names_')

        # Get new names
        new_names = []
        for c in range(len(self.inds_)):

            ind = self.inds_[c]
            col_name = feat_names[ind]

            new_inds = self.X_trans_inds_[c]
            new_names += [str(col_name) + '_' + str(i)
                          for i in range(len(new_inds))]

        # Remove old names - using parent method
        feat_names = self._remove_old_names(feat_names)

        # New names come first, then rest of names
        all_names = new_names + feat_names

        return all_names

    def inverse_transform_FIs(self, fis):

        # Skip if skipped
        if self.estimator_ is None:
            return fis

        # If doesn't have inverse_transform, return as is.
        if not hasattr(self.estimator_, 'inverse_transform'):
            return fis

        # Get feature importances also as array
        fis_data = np.array(fis)
        fis_names = np.array(fis.index)

        # Compute reverse mapping
        reverse_mapping = get_reverse_mapping(self.mapping_)

        # Prep return fis
        return_fis_data = np.zeros(len(reverse_mapping), dtype='object')
        return_fis_names = ['' for _ in range(len(reverse_mapping))]

        # Process each feature
        for col_ind in self.inds_:

            # Get reverse inds and data for just this col
            reverse_inds = proc_mapping([col_ind], self.out_mapping_)
            col_fis = fis_data[reverse_inds]
            col_names = fis_names[reverse_inds]

            # Run inverse transform
            inv_fis = self.estimator_.inverse_transform(col_fis)

            # Place into return fis_data
            original_ind = reverse_mapping[col_ind]
            return_fis_data[original_ind] = inv_fis

            # Add return name
            return_fis_names[original_ind] =\
                self._get_reverse_feat_name(col_names, original_ind)

        # Fill in with original
        for col_ind in self.rest_inds_:
            reverse_ind = proc_mapping([col_ind], self.out_mapping_)[0]
            original_ind = reverse_mapping[col_ind]

            # Just pass along data and name, but in original spot
            return_fis_data[original_ind] = fis_data[reverse_ind]
            return_fis_names[original_ind] = fis_names[reverse_ind]

        # Return new series
        return pd.Series(return_fis_data, index=return_fis_names)

    def _get_reverse_feat_name(self, col_names, original_ind):

        # Get + add return name
        if hasattr(self, 'feat_names_in_'):
            return self.feat_names_in_[original_ind]

        # If reverse_feat_names hasn't been called
        substrs = get_top_substrs(col_names)

        # If no common sub string
        if len(substrs) == 0:
            return 'loader_ind_' + str(original_ind)

        # Just take first substring
        new_name = substrs[0]

        # Remove ending _ if any
        if new_name.endswith('_'):
            new_name = new_name[:-1]

        return new_name


class CompatArray(list):

    def __init__(self, arr_2d):

        self.dtype = arr_2d.dtype
        super().__init__(np.swapaxes(arr_2d, 0, 1))

    @property
    def shape(self):
        return (len(self[0]), len(self))

    def conv_rest_back(self, rest_inds):

        if len(rest_inds) == 0:
            empty = np.array([], dtype=self.dtype)
            return empty.reshape((self.shape[0], 0))

        # Create an array from the requested rest inds
        base = np.array(itemgetter(*rest_inds)(self),
                        dtype=self.dtype)

        # If only one return axis, conv to correct shape
        if len(base.shape) == 1:
            return base[:, np.newaxis]

        # Reverse initial swap
        return np.swapaxes(base, 0, 1)


class BPtListLoader(BPtLoader):

    def fit_transform(self, X, y=None, mapping=None,
                      fit_index=None, **fit_params):

        # Process the mapping
        if mapping is None:
            mapping = {}
        self._proc_mapping(mapping)

        if len(self.inds_) != 1:
            raise RuntimeWarning('BPtListLoader can only work on one column.')

        # Calls super fit_transform, but passing
        # X with the data columns replaced by CompatArray
        return super().fit_transform(self._get_X_compat(X), y=y,
                                     mapping=mapping,
                                     fit_index=fit_index,
                                     **fit_params)

    def _get_X_compat(self, X):

        ind = self.inds[0]
        X_loaded = CompatArray(X)
        X_loaded[ind] = self._load_col(X[:, ind])

        return X_loaded

    def _load_col(self, X_col):
        '''Load X col as list of subjects'''

        X_col_loaded = []
        for key in X_col:
            DataFile = self.file_mapping[int(key)]
            data = DataFile.load()
            X_col_loaded.append(data)

        return X_col_loaded

    def _fit(self, X, y=None, **fit_params):
        '''Override the internal fit function to fit only
        the single requested column.'''
        self.estimator_.fit(X[self.inds_[0]], y=y, **fit_params)

        return self

    def transform(self, X, transform_index=None):

        # If None, pass along as is
        if self.estimator_ is None:
            return X

        # Load if not loaded
        if not isinstance(X, CompatArray):
            X = self._get_X_compat(X)

        # Get transform params
        trans_params = _get_est_trans_params(self.estimator_,
                                             transform_index=transform_index)

        # Get X_trans
        X_trans = self.estimator_.transform(X=X[self.inds_[0]], **trans_params)

        # Save number of output features after X_trans
        self.n_trans_feats_ = X_trans.shape[1]

        # For compat
        self.X_trans_inds_ = [list(range(self.n_trans_feats_))]

        # Return stacked X_trans with rest inds
        return np.hstack([X_trans, X.conv_rest_back(self.rest_inds_)])