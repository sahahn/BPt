from sklearn.pipeline import Pipeline
import numpy as np
from ..helpers.VARS import ORDERED_NAMES


def f_array(in_array, tp='float32'):
    return np.array(in_array).astype(tp)


class BPt_Pipeline(Pipeline):

    _needs_mapping = True
    _needs_train_data_index = True

    def __init__(self, steps, memory=None, verbose=False,
                 add_mapping=False, to_map=None, needs_index=None,
                 names=None):

        self.add_mapping = add_mapping
        self.to_map = to_map
        self.needs_index = needs_index
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

        if self.to_map is None:
            self.to_map = []
        if self.needs_index is None:
            self.needs_index = []
        if self.names is None:
            self.names = []

        # Add mapping to fit params, as either passed or new
        if mapping is not None:
            self._mapping = mapping.copy()
        elif self.add_mapping:
            self._mapping = {i: i for i in range(X.shape[1])}
        else:
            self._mapping = {}

        for name in self.to_map:
            fit_params[name + '__mapping'] = self._mapping
        for name in self.needs_index:
            fit_params[name + '__train_data_index'] = train_data_index

        super().fit(X, y, **fit_params)
        return self

    def _get_objs_by_name(self):

        if self.names is None:
            self.names = []

        fitted_objs = [[self.__getitem__(name) for name in obj]
                       for obj in self.names]
        return fitted_objs

    def has_transforms(self):
        fitted_objs = self._get_objs_by_name()

        if len(fitted_objs[0]) > 0 or len(fitted_objs[3]) > 0:
            return True
        return False

    def proc_X_test(self, X_test, y_test, fs=True, tp='float32'):

        # Load all base objects and corresponding fitted objs
        fitted_objs = self._get_objs_by_name()

        feat_names = list(X_test)

        # Process the loaders, while keeping track of feature names
        for loader in fitted_objs[ORDERED_NAMES.index('loaders')]:

            # Use special transform in place df func
            X_test = loader.transform_df(X_test, base_name=feat_names)
            feat_names = list(X_test)

        # Imputer and Scaler are Wrapped in ScopeTransformer
        # so use transform df.
        for obj in fitted_objs[ORDERED_NAMES.index('imputers')]:
            X_test = obj.transform_df(X_test)
            feat_names = list(X_test)
        for obj in fitted_objs[ORDERED_NAMES.index('scalers')]:
            X_test = obj.transform_df(X_test)
            feat_names = list(X_test)

        # Handle transformers, w/ simmilar func to loaders
        trans_ind = ORDERED_NAMES.index('transformers')
        for i in range(len(fitted_objs[trans_ind])):

            # Grab transformer and base name
            transformer = fitted_objs[trans_ind][i]
            base_name = self.names[trans_ind][i]

            # Use special transform in place df func
            X_test = transformer.transform_df(X_test, base_name=base_name)
            feat_names = list(X_test)

        # Drop features according to feat_selectors, keeping track of changes
        # only if passed param fs is True
        if fs:
            fs_ind = ORDERED_NAMES.index('feat_selectors')
            for feat_selector in fitted_objs[fs_ind]:

                # This feat mask corresponds to the already transformed feats
                feat_mask = feat_selector.get_support()

                # So we first need to compute the right order of new
                # names that X gets transformed into, as concat wrapper_inds
                # + rest inds
                out_feat_names =\
                    [feat_names[i] for i in feat_selector.wrapper_inds_] +\
                    [feat_names[i] for i in feat_selector.rest_inds_]

                # Then we can apply the computed mask, and get the actually
                # selected features
                feat_names = np.array(out_feat_names)[feat_mask]

                # Now set within X_test, the results of the transformation
                X_test[feat_names] =\
                    feat_selector.transform(f_array(X_test, tp))
                X_test = X_test[feat_names].copy()

        return X_test, y_test

    def proc_X_train(self, X_train, tp='float32'):

        # Load all base objects
        fitted_objs = self._get_objs_by_name()

        # No need to proc in place, so the transformations are pretty easy
        # Note: Loader and Transformer take care of conv to correct
        # np array type.
        for loader in fitted_objs[ORDERED_NAMES.index('loaders')]:
            X_train = loader.transform(f_array(X_train, tp))
        for imputer in fitted_objs[ORDERED_NAMES.index('imputers')]:
            X_train = imputer.transform(f_array(X_train, tp))
        for scaler in fitted_objs[ORDERED_NAMES.index('scalers')]:
            X_train = scaler.transform(f_array(X_train, tp))
        for transformer in fitted_objs[ORDERED_NAMES.index('transformers')]:
            X_train = transformer.transform(f_array(X_train, tp))
        fs_ind = ORDERED_NAMES.index('feat_selectors')
        for feat_selector in fitted_objs[fs_ind]:
            X_train = feat_selector.transform(f_array(X_train, tp))

        return X_train

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
