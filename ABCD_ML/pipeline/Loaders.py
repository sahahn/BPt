from ..helpers.ML_Helpers import (get_obj_and_params, update_mapping,
                                  proc_mapping)
import numpy as np
from .Transformers import Transformer_Wrapper
from ..extensions.Loaders import Identity, SurfLabels
from joblib import Parallel, delayed
import warnings


def get_trans_chunk(transformer, data_files):

    X_trans_chunk = []

    for data_file in data_files:

        data = data_file.load()
        trans_data = np.squeeze(transformer.transform(data))
        X_trans_chunk.append(trans_data)

    return X_trans_chunk


class Loader_Wrapper(Transformer_Wrapper):

    def __init__(self, wrapper_transformer,
                 wrapper_inds, file_mapping,
                 wrapper_n_jobs=1, **params):

        super().__init__(wrapper_transformer,
                         wrapper_inds, **params)

        self.file_mapping = file_mapping
        self.wrapper_n_jobs = wrapper_n_jobs

    def _fit(self, X, y=None):

        fit_fm_key = X[0, self.wrapper_inds[0]]
        fit_data = self.file_mapping[int(fit_fm_key)].load()

        self.wrapper_transformer.fit(fit_data, y)

        return self

    def fit_transform(self, X, y=None, mapping=None, **kwargs):

        if mapping is None:
            mapping = {}

        # If any changes to mapping, update
        self._proc_mapping(mapping)

        # Fit on the first data-point only
        self._fit(X, y)

        # Transform X
        X_trans, self._X_trans_inds = self._get_X_trans(X)

        # Then create new mapping
        new_mapping = {}

        # Add changed X_trans by col
        for c in range(len(self.wrapper_inds)):
            ind = self.wrapper_inds[c]
            new_mapping[ind] = self._X_trans_inds[c]

        # Update rest of inds, as just shifted over
        self.rest_inds_ = [i for i in range(X.shape[1])
                           if i not in self.wrapper_inds]

        for c in range(len(self.rest_inds_)):
            ind = self.rest_inds_[c]
            new_mapping[ind] = X_trans.shape[1] + c

        self._out_mapping = new_mapping.copy()

        # Update mapping
        update_mapping(mapping, new_mapping)
        return np.hstack([X_trans, X[:, self.rest_inds_]])

    def get_chunks(self, data_files):

        per_chunk = len(data_files) // self.wrapper_n_jobs
        chunks = [list(range(i * per_chunk, (i+1) * per_chunk))
                  for i in range(self.wrapper_n_jobs)]

        last = chunks[-1][-1]
        chunks[-1] += list(range(last+1, len(data_files)))
        return [[data_files[i] for i in c] for c in chunks]

    def _get_trans_col(self, fm_keys):

        data_files = [self.file_mapping[int(fm_key)] for fm_key in fm_keys]

        if self.wrapper_n_jobs == 1:
            X_trans_cols = get_trans_chunk(self.wrapper_transformer,
                                           data_files)

        else:
            chunks = self.get_chunks(data_files)

            X_trans_chunks =\
                Parallel(n_jobs=self.wrapper_n_jobs)(
                    delayed(get_trans_chunk)(
                        transformer=self.wrapper_transformer, data_files=chunk)
                    for chunk in chunks)

            X_trans_cols = []
            for chunk in X_trans_chunks:
                X_trans_cols += chunk

        return X_trans_cols

    def _get_X_trans(self, X):

        # Init lists + mappings
        X_trans, X_trans_inds = [], []
        cnt = 0

        # For each column to fit_transform
        for col in self.wrapper_inds:

            # Get transformer column
            fm_keys = [key for key in X[:, col]]
            X_trans_cols = self._get_trans_col(fm_keys)

            # Stack + append new features
            X_trans_cols = np.stack(X_trans_cols)
            X_trans.append(X_trans_cols)

            # Add + append inds
            X_trans_cols_inds =\
                [i for i in range(cnt, X_trans_cols.shape[1] + cnt)]
            X_trans_inds.append(X_trans_cols_inds)

            # Update cnt
            cnt = X_trans_cols.shape[1] + cnt

        # Stack final
        X_trans = np.hstack(X_trans)

        return X_trans, X_trans_inds

    def transform(self, X):

        # Transform X
        X_trans, _ = self._get_X_trans(X)
        return np.hstack([X_trans, X[:, self.rest_inds_]])

    def _get_new_df_names(self, feat_names):
        '''Create new feature names for the transformed features,
        in loaders this is done per feature/column'''

        new_names = []
        for c in range(len(self.wrapper_inds)):

            ind = self.wrapper_inds[c]
            base_name = feat_names[ind]

            new_inds = self._X_trans_inds[c]
            new_names += [base_name + '_' + str(i)
                          for i in range(len(new_inds))]

        return new_names

    def inverse_transform(self, X, name='base loader'):

        # For each column, compute the inverse transform of what's loaded
        inverse_X = {}

        for col_ind in self.wrapper_inds:
            reverse_inds = proc_mapping([col_ind], self._out_mapping)

            # for each subject
            X_trans = []
            for subject_X in X[:, reverse_inds]:
                try:
                    X_trans.append(
                        self.wrapper_transformer.inverse_transform(subject_X))
                except AttributeError:
                    X_trans.append('No inverse_transform')
                    warnings.warn('Passed loader: "' + name + '" has no '
                                  'inverse_transform! '
                                  'Setting relevant inverse '
                                  'feat importances to "No inverse_transform".')

            # Store the list of inverse_transformed X's by subject
            # In a dictionary with the original col_ind as the key
            inverse_X[col_ind] = X_trans

        # Now need to do two things, it is assumed the output from loader
        # cannot be put in a standard X array, but also
        # in the case with multiple loaders, we still need to return
        # An otherwise inversed X, we will just set values to 0 in this version
        reverse_rest_inds = proc_mapping(self.rest_inds_, self._out_mapping)

        all_inds_len = len(self.wrapper_inds) + len(self.rest_inds_)
        Xt = np.zeros((X.shape[0], all_inds_len), dtype=X.dtype)

        Xt[:, self.wrapper_inds] = 0
        Xt[:, self.rest_inds_] = X[:, reverse_rest_inds]

        return Xt, inverse_X

    def set_params(self, **params):

        if 'file_mapping' in params:
            self.file_mapping = params.pop('file_mapping')
        if 'wrapper_n_jobs' in params:
            self.wrapper_n_jobs = params.pop('wrapper_n_jobs')

        return super().set_params(**params)

    def get_params(self, deep=False):

        params = super().get_params()
        params['file_mapping'] = self.file_mapping
        params['wrapper_n_jobs'] = self.wrapper_n_jobs

        return params


LOADERS = {
    'identity': (Identity, ['default']),
    'surface rois': (SurfLabels, ['default']),
}

# If nilearn dependencies
try:
    from nilearn.input_data import NiftiLabelsMasker
    from ..extensions.Loaders import Connectivity
    LOADERS['volume rois'] = (NiftiLabelsMasker, ['default'])
    LOADERS['connectivity'] = (Connectivity, ['default'])

except ImportError:
    pass


def get_loader_and_params(loader_str, extra_params, params, search_type,
                          random_state=None, num_feat_keys=None):

    loader, extra_loader_params, loader_params =\
        get_obj_and_params(loader_str, LOADERS, extra_params, params,
                           search_type)

    return loader(**extra_loader_params), loader_params
