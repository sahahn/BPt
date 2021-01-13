from sklearn.pipeline import Pipeline
import numpy as np
from ..helpers.VARS import ORDERED_NAMES
from .helpers import f_array


class BPtPipeline(Pipeline):

    _needs_mapping = True
    _needs_train_data_index = True

    def __init__(self, steps, memory=None, verbose=False, names=None):
        self.names = names

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

            # Also check for wrapper n jobs
            if hasattr(step[1], 'wrapper_n_jobs'):
                setattr(step[1], 'wrapper_n_jobs', n_jobs)

    def get_params(self, deep=True):
        params = super()._get_params('steps', deep=deep)
        return params

    def set_params(self, **kwargs):
        super()._set_params('steps', **kwargs)
        return self

    def fit(self, X, y=None, mapping=None,
            train_data_index=None, **fit_params):

        # Add mapping to fit params if already passed, e.g., in nested context
        # Or init new
        if mapping is not None:
            self._mapping = mapping.copy()
        else:
            self._mapping = {i: i for i in range(X.shape[1])}

        # Add to the fit parameters according to estimator tags
        # adding mapping + needs_train_index info
        for step in self.steps:
            name, estimator = step[0], step[1]
            if hasattr(estimator, '_needs_mapping'):
                fit_params[name + '__mapping'] = self._mapping
            if hasattr(estimator, '_needs_train_data_index'):
                fit_params[name + '__train_data_index'] = train_data_index

        # Call parent fit
        super().fit(X, y, **fit_params)
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
