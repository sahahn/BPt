from ..helpers.ML_Helpers import (get_obj_and_params, update_mapping,
                                  proc_mapping, get_reverse_mapping)
import numpy as np
from ..extensions.Loaders import Identity, SurfLabels
from joblib import Parallel, delayed
import warnings
from sklearn.utils.validation import check_memory
from sklearn.base import clone
from .ScopeObjs import ScopeTransformer


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


class BPtLoader(ScopeTransformer):

    # Override
    _required_parameters = ["estimator", "inds", "file_mapping"]

    def __init__(self, estimator, inds, file_mapping,
                 n_jobs=1, fix_n_jobs=False,
                 cache_loc=None):

        # Set Super params
        super().__init__(estimator=estimator, inds=inds, cache_loc=cache_loc)

        # Set rest of params
        self.file_mapping = file_mapping

        # Make sure to set fix n jobs before n_jobs
        self.fix_n_jobs = fix_n_jobs
        self.n_jobs = n_jobs

    @property
    def _n_jobs(self):

        if self.fix_n_jobs is False:
            return self.n_jobs

        return self.fix_n_jobs

    def fit(self, X, y=None, mapping=None,
            train_data_index=None, **fit_params):

        # Need the output from a transform to full fit,
        # so when fit is called, call fit_transform instead
        self.fit_transform(X=X, y=y, mapping=mapping,
                           train_data_index=train_data_index,
                           **fit_params)

        return self

    def _fit(self, X, y=None, **fit_params):
        '''Special fit for loader, only concerned with
        fitting first data point for e.g., use of reverse
        transform
        '''

        # Get the first data point
        fit_fm_key = X[0, self.inds_[0]]
        fit_X = self.file_mapping[int(fit_fm_key)].load()

        # Fit the first data point
        self.estimator_.fit(fit_X, y=y, **fit_params)

    def fit_transform(self, X, y=None, mapping=None,
                      train_data_index=None, **fit_params):

        if mapping is None:
            mapping = {}

        # Call parent fit but passing only the first data point
        super().fit(X, y=y, mapping=mapping,
                    train_data_index=train_data_index,
                    **fit_params)

        # The parent fit takes care of, in addition to
        # fitting the loader on one
        # data point, sets base_dtype, processes the mapping,
        # sets rest inds, etc...

        # Now transform X - this sets self.X_trans_inds_
        X_trans = self.transform(X)

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

        # Now return X_trans
        return X_trans

    def transform(self, X):

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

    def transform_df(self, df, base_name='loader'):

        return super().transform_df(df, base_name=base_name)

    def _proc_new_names(self, feat_names, base_name):

        # Get new names
        new_names = []
        for c in range(len(self.inds_)):

            ind = self.inds_[c]
            base_name = feat_names[ind]

            new_inds = self.X_trans_inds_[c]
            new_names += [base_name + '_' + str(i)
                          for i in range(len(new_inds))]

        # Remove old names - using parent method
        feat_names = self._remove_old_names(feat_names)

        # New names come first, then rest of names
        all_names = new_names + feat_names

        return all_names

    def inverse_transform(self, X, name='base loader'):

        # For each column, compute the inverse transform of what's loaded
        inverse_X = {}
        reverse_mapping = get_reverse_mapping(self.mapping_)

        no_it_warns = set()
        other_warns = set()

        for col_ind in self.inds_:
            reverse_inds = proc_mapping([col_ind], self.out_mapping_)

            # for each subject
            X_trans = []
            for subject_X in X[:, reverse_inds]:

                # If pipeline
                if hasattr(self.estimator_, 'steps'):
                    for step in self.estimator_.steps[::-1]:
                        s_name = step[0]
                        pipe_piece = self.estimator_[s_name]

                        try:
                            subject_X = pipe_piece.inverse_transform(subject_X)
                        except AttributeError:
                            no_it_warns.add(name + '__' + s_name)
                        except:
                            other_warns.add(name + '__' + s_name)

                else:
                    try:
                        subject_X =\
                            self.estimator_.inverse_transform(
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
        reverse_rest_inds = proc_mapping(self.rest_inds_, self.out_mapping_)

        all_inds_len = len(self.inds_) + len(self.rest_inds_)
        Xt = np.zeros((X.shape[0], all_inds_len), dtype=X.dtype)

        Xt[:, self.inds_] = 0
        Xt[:, self.rest_inds_] = X[:, reverse_rest_inds]

        return Xt, inverse_X


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


def get_loader_and_params(loader_str, extra_params, params,
                          random_state=None, num_feat_keys=None):

    loader, extra_loader_params, loader_params =\
        get_obj_and_params(loader_str, LOADERS, extra_params, params)

    return loader(**extra_loader_params), loader_params
