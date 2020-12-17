from ..helpers.ML_Helpers import (get_obj_and_params, update_mapping,
                                  proc_mapping, get_reverse_mapping)
import numpy as np
from .Transformers import Transformer_Wrapper
from ..extensions.Loaders import Identity, SurfLabels
from joblib import Parallel, delayed
import warnings
from sklearn.utils.validation import check_memory
from sklearn.base import clone


def load_and_trans(transformer, load_func, loc):
    '''This function is designed to be able to be wrapped in a cache
    check_memory.'''

    data = load_func(loc)
    trans_data = np.squeeze(transformer.fit_transform(data))
    return trans_data


def get_trans_chunk(transformer, data_files, func):
    '''This function is designed to be used for multi-processing'''

    X_trans_chunk = []
    for data_file in data_files:
        loc = data_file.loc
        load_func = data_file.load_func
        trans_data = func(clone(transformer), load_func, loc)
        X_trans_chunk.append(trans_data)

    return X_trans_chunk


class Loader_Wrapper(Transformer_Wrapper):

    def __init__(self, wrapper_transformer,
                 wrapper_inds, file_mapping,
                 wrapper_n_jobs=1, cache_loc=None,
                 fix_n_wrapper_jobs='default',
                 **params):

        super().__init__(wrapper_transformer=wrapper_transformer,
                         wrapper_inds=wrapper_inds, cache_loc=cache_loc,
                         fix_n_wrapper_jobs=fix_n_wrapper_jobs,
                         **params)

        self.file_mapping = file_mapping
        self.wrapper_n_jobs = wrapper_n_jobs

    @property
    def _n_jobs(self):

        if self.fix_n_wrapper_jobs == 'default':
            return self.wrapper_n_jobs

        return self.fix_n_wrapper_jobs

    def _fit(self, X, y=None):

        fit_fm_key = X[0, self.wrapper_inds_[0]]
        fit_data = self.file_mapping[int(fit_fm_key)].load()

        self.wrapper_transformer_ = clone(self.wrapper_transformer)
        self.wrapper_transformer_.fit(fit_data, y)

        return self

    def fit_transform(self, X, y=None, mapping=None, **kwargs):

        # Save base dtype of input
        self._base_dtype = X.dtype

        if mapping is None:
            mapping = {}

        # If any changes to mapping, update
        self._proc_mapping(mapping)

        # Fit a copy on the first data-point only
        # this is used for say reverse transformations
        self._fit(X, y)

        # Transform X
        X_trans, self._X_trans_inds = self._get_X_trans(X)

        # Then create new mapping
        new_mapping = {}

        # Add changed X_trans by col
        for c in range(len(self.wrapper_inds_)):
            ind = self.wrapper_inds_[c]
            new_mapping[ind] = self._X_trans_inds[c]

        # Set rest inds as any inds not in wrapper inds
        self.rest_inds_ = list(np.setdiff1d(list(range(X.shape[1])),
                                            self.wrapper_inds_,
                                            assume_unique=True))

        for c in range(len(self.rest_inds_)):
            ind = self.rest_inds_[c]
            new_mapping[ind] = X_trans.shape[1] + c

        self._out_mapping = new_mapping.copy()

        # Update mapping
        update_mapping(mapping, new_mapping)

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

        # Clone the base loader transformer
        cloned_transformer = clone(self.wrapper_transformer)

        # If a caching location is passed, create new load_cand_trans_c func
        if self.cache_loc is not None:
            memory = check_memory(self.cache_loc)
            load_and_trans_c = memory.cache(load_and_trans)
        else:
            load_and_trans_c = load_and_trans

        if self._n_jobs == 1:
            X_trans_cols = get_trans_chunk(cloned_transformer,
                                           data_files, load_and_trans_c)
        else:
            chunks = self.get_chunks(data_files)

            X_trans_chunks =\
                Parallel(n_jobs=self._n_jobs)(
                    delayed(get_trans_chunk)(
                        transformer=cloned_transformer,
                        data_files=chunk,
                        func=load_and_trans_c)
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
        for col in self.wrapper_inds_:

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

    def _get_new_df_names(self, base_name=None, feat_names=None):
        '''Create new feature names for the transformed features,
        in loaders this is done per feature/column'''

        new_names = []
        for c in range(len(self.wrapper_inds_)):

            ind = self.wrapper_inds_[c]
            base_name = feat_names[ind]

            new_inds = self._X_trans_inds[c]
            new_names += [base_name + '_' + str(i)
                          for i in range(len(new_inds))]

        return new_names

    def inverse_transform(self, X, name='base loader'):

        # For each column, compute the inverse transform of what's loaded
        inverse_X = {}
        reverse_mapping = get_reverse_mapping(self.mapping_)

        no_it_warns = set()
        other_warns = set()

        for col_ind in self.wrapper_inds_:
            reverse_inds = proc_mapping([col_ind], self._out_mapping)

            # for each subject
            X_trans = []
            for subject_X in X[:, reverse_inds]:

                # If pipeline
                if hasattr(self.wrapper_transformer_, 'steps'):
                    for step in self.wrapper_transformer_.steps[::-1]:
                        s_name = step[0]
                        pipe_piece = self.wrapper_transformer_[s_name]

                        try:
                            subject_X = pipe_piece.inverse_transform(subject_X)
                        except AttributeError:
                            no_it_warns.add(name + '__' + s_name)
                        except:
                            other_warns.add(name + '__' + s_name)

                else:
                    try:
                        subject_X =\
                            self.wrapper_transformer_.inverse_transform(
                                subject_X)
                    except AttributeError:
                        no_it_warns.add(name)
                    except:
                        other_warns.add(name)

                # Regardless of outcome, add to X_trans
                X_trans.append(subject_X)

            # If X_trans only has len 1, get rid of internal list
            if len(X_trans) == 1:
                X_trans = X_trans[0]

            # Store the list of inverse_transformed X's by subject
            # In a dictionary with the original col_ind as the key
            inverse_X[reverse_mapping[col_ind]] = X_trans

        # Send out warn messages
        if len(no_it_warns) > 0:
            warnings.warn(repr(list(no_it_warns)) + ' skipped '
                          'in calculating inverse FIs due to no '
                          'inverse_transform')
        if len(other_warns) > 0:
            warnings.warn(repr(list(other_warns)) + ' skipped '
                          'in calculating inverse FIs due to '
                          'an error in inverse_transform')

        # Now need to do two things, it is assumed the output from loader
        # cannot be put in a standard X array, but also
        # in the case with multiple loaders, we still need to return
        # An otherwise inversed X, we will just set values to 0 in this version
        reverse_rest_inds = proc_mapping(self.rest_inds_, self._out_mapping)

        all_inds_len = len(self.wrapper_inds_) + len(self.rest_inds_)
        Xt = np.zeros((X.shape[0], all_inds_len), dtype=X.dtype)

        Xt[:, self.wrapper_inds_] = 0
        Xt[:, self.rest_inds_] = X[:, reverse_rest_inds]

        return Xt, inverse_X

    def set_params(self, **params):

        if 'file_mapping' in params:
            self.file_mapping = params.pop('file_mapping')
        if 'wrapper_n_jobs' in params:
            self.wrapper_n_jobs = params.pop('wrapper_n_jobs')

        return super().set_params(**params)

    def get_params(self, deep=False):

        params = super().get_params(deep=deep)

        # Passing file_mapping as just a reference *should* be okay
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
        get_obj_and_params(loader_str, LOADERS, extra_params, params)

    return loader(**extra_loader_params), loader_params
