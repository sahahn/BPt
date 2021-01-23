from sklearn.pipeline import Pipeline
import numpy as np
from ..helpers.VARS import ORDERED_NAMES
from ..helpers.ML_Helpers import hash
from joblib import load, dump
from joblib import hash as joblib_hash
import os


class BPtPipeline(Pipeline):

    _needs_mapping = True
    _needs_train_data_index = True

    def __init__(self, steps, memory=None,
                 verbose=False, names=None,
                 cache_fit_dr=None):

        self.names = names
        self.cache_fit_dr = cache_fit_dr

        super().__init__(steps=steps, memory=memory, verbose=verbose)

    @property
    def n_jobs(self):
        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, n_jobs):

        # Store ... in self._n_jobs
        self._n_jobs = n_jobs

        # If set here, try to propegate to all steps
        for step in self.steps:
            if hasattr(step[1], 'n_jobs'):
                setattr(step[1], 'n_jobs', n_jobs)

    @property
    def feature_importances_(self):
        if hasattr(self.__getitem__(-1), 'feature_importances_'):
            return getattr(self.__getitem__(-1), 'feature_importances_')
        return None

    @property
    def coef_(self):
        if hasattr(self[-1], 'coef_'):
            return getattr(self[-1], 'coef_')
        return None

    def get_params(self, deep=True):
        params = super()._get_params('steps', deep=deep)
        return params

    def set_params(self, **kwargs):
        super()._set_params('steps', **kwargs)
        return self

    def fit(self, X, y=None, mapping=None,
            train_data_index=None, **fit_params):

        if self.cache_fit_dr is not None:

            # Compute the hash for this fit
            # Store as an attribute
            self.hash_ = hash([X, y, mapping,
                               train_data_index, fit_params],
                              self.steps)

            # Check if hash exists - if it does load
            if os.path.exists(self._get_hash_loc()):
                self._load_from_hash()

                # end / return!
                return self

            # Otherwise, continue to fit as normal

        # Add mapping to fit params if already passed, e.g., in nested context
        # Or init new
        if mapping is not None:
            self.mapping_ = mapping.copy()
        else:
            self.mapping_ = {i: i for i in range(X.shape[1])}

        # Add to the fit parameters according to estimator tags
        # adding mapping + needs_train_index info
        for step in self.steps:
            name, estimator = step[0], step[1]
            if hasattr(estimator, '_needs_mapping'):
                fit_params[name + '__mapping'] = self.mapping_
            if hasattr(estimator, '_needs_train_data_index'):
                fit_params[name + '__train_data_index'] = train_data_index

        # Call parent fit
        super().fit(X, y, **fit_params)

        # If cache fit enabled, hash fitted pipe here
        if self.cache_fit_dr is not None:
            self._hash_fit()

        return self

    def _get_hash_loc(self):

        # Make sure directory exists
        os.makedirs(self.cache_fit_dr, exist_ok=True)

        # Set hash loc as directory + hash of fit args
        hash_loc = os.path.join(self.cache_fit_dr, self.hash_)

        return hash_loc

    def _hash_fit(self):

        # Just save full fitted pipeline
        dump(self, self._get_hash_loc())
        return self

    def _load_from_hash(self):

        # Load from saved hash, by
        # loading the fitted object
        # and then copying over
        # each relevant saved fitted parameter
        fitted_pipe = load(self._get_hash_loc())

        # Copy mapping
        self.mapping_ = fitted_pipe.mapping_

        # Copy each step with the fitted version
        for (step_idx,
             name,
             fitted_piece) in fitted_pipe._iter(with_final=True,
                                                filter_passthrough=False):
            self.steps[step_idx] = (name, fitted_piece)

        # Set flag for testing
        self.loaded_ = True

        return self

    def _get_objs_by_name(self):

        if self.names is None:
            self.names = []

        fitted_objs = [[self.__getitem__(name) for name in obj]
                       for obj in self.names]
        return fitted_objs

    def _get_ordered_objs_and_names(self, fs=True, model=False):

        fitted_objs = self._get_objs_by_name()
        ordered_objs = []
        ordered_base_names = []

        for name in ORDERED_NAMES:

            # Check if should add or not based on passed params
            add = True
            if name == 'feat_selectors' and not fs:
                add = False
            if name == 'model' and not model:
                add = False

            if add:
                ind = ORDERED_NAMES.index(name)
                ordered_objs += fitted_objs[ind]
                ordered_base_names += self.names[ind]

        return ordered_objs, ordered_base_names

    def transform_df(self, X_df, fs=True):
        '''Transform an input dataframe, keeping track of feature names'''

        # Get ordered objects as list, with or without feat selectors
        # and also the corr. base names
        ordered_objs, ordered_base_names =\
            self._get_ordered_objs_and_names(fs=fs, model=False)

        # Run all of the transformations
        for obj, base_name in zip(ordered_objs, ordered_base_names):
            X_df = obj.transform_df(X_df, base_name=base_name)

        return X_df

    def inverse_transform_FIs(self, fis, feat_names):

        # Make compat w/ subjects x feats
        if len(fis.shape) == 1:
            fis = np.expand_dims(fis, axis=0)

        # To inverse transform FIs, we are only concerned with feat_selectors
        # transformers, and loaders
        fitted_objs = self._get_objs_by_name()

        # Feat selectors
        fs_ind = ORDERED_NAMES.index('feat_selectors')
        for feat_selector in fitted_objs[fs_ind][::-1]:
            fis = feat_selector.inverse_transform(fis)

        # Transformers
        trans_ind = ORDERED_NAMES.index('transformers')
        for transformer, name in zip(fitted_objs[trans_ind][::-1],
                                     self.names[trans_ind][::-1]):
            fis = transformer.inverse_transform(fis, name=name)

        # Loaders - special case
        inversed_loaders = {}
        l_ind = ORDERED_NAMES.index('loaders')
        for loader, name in zip(fitted_objs[l_ind][::-1],
                                self.names[l_ind][::-1]):
            fis, inverse_X = loader.inverse_transform(fis, name=name)
            inversed_loaders.update(inverse_X)

        # Make the final feat_importances dict
        feat_imp_dict = {}
        for i in range(len(feat_names)):
            if i in inversed_loaders:
                feat_imp_dict[feat_names[i]] = inversed_loaders[i]
            else:
                feat_imp_dict[feat_names[i]] = fis[:, i]

        return feat_imp_dict
