from imblearn.pipeline import Pipeline
import numpy as np


def f_array(in_array):
    return np.array(in_array).astype(float)


class ABCD_Pipeline(Pipeline):

    def __init__(self, steps, memory=None, verbose=False,
                 mapping=False, to_map=None, names=None):

        self.mapping = mapping

        if to_map is None:
            to_map = []
        self.to_map = to_map

        if names is None:
            names = []
        self.names = names

        super().__init__(steps, memory, verbose)

    def get_params(self, deep=True):
        params = super()._get_params('steps', deep=deep)
        return params

    def set_params(self, **kwargs):
        super()._set_params('steps', **kwargs)
        return self

    def fit(self, X, y=None, mapping=None,
            train_data_index=None, **fit_params):

        # Add mapping to fit params, as either passed or new
        if mapping is not None:
            self._mapping = mapping
        elif self.mapping:
            self._mapping = {i: i for i in range(X.shape[1])}
        else:
            self._mapping = {}

        for name in self.to_map:
            fit_params[name + '__mapping'] = self._mapping

        super().fit(X, y, **fit_params)

        return self

    def _get_objs_by_name(self):

        fitted_objs = [[self.__getitem__(name) for name in obj]
                       for obj in self.names]
        return fitted_objs

    def has_transforms(self):
        fitted_objs = self._get_objs_by_name()

        if len(fitted_objs[0]) > 0 or len(fitted_objs[3]) > 0:
            return True
        return False

    def proc_X_test(self, X_test, y_test, fs=True):

        # Order of names is:
        #
        #  0 - 'loaders'
        #  1 - 'imputers'
        #  2 - 'scalers'
        #  3 - 'transformers'
        #  4 - 'samplers'
        #  5 - 'drop_strat'
        #  6 - 'feat_selectors',
        #  7 - 'models'
        #  8 - 'ensembles'

        # Load all base objects and corresponding fitted objs
        fitted_objs = self._get_objs_by_name()

        feat_names = list(X_test)

        # Process the loaders, while keeping track of feature names
        for loader in fitted_objs[0]:

            # Use special transform in place df func
            X_test = loader.transform_df(X_test, base_name=feat_names)
            feat_names = list(X_test)

        # Apply pipeline operations in place
        for imputer in fitted_objs[1]:
            X_test[feat_names] = imputer.transform(f_array(X_test))
        for scaler in fitted_objs[2]:
            X_test[feat_names] = scaler.transform(f_array(X_test))

        # Handle transformers, w/ simmilar func to loaders
        for i in range(len(fitted_objs[3])):

            # Grab transformer and base name
            transformer = fitted_objs[3][i]
            base_name = self.names[3][i]

            # Use special transform in place df func
            X_test = transformer.transform_df(X_test, base_name=base_name)
            feat_names = list(X_test)

        # Skip fitted_objs[4] here, as it is samplers

        # Make sure to keep track of col changes w/ drop + feat_selector
        for drop in fitted_objs[5]:

            valid_inds = np.array(drop.transformers[0][2])
            feat_names = np.array(feat_names)[valid_inds]
            X_test = X_test[feat_names]

        # Drop features according to feat_selectors, keeping track of changes
        # only if passed param fs is True
        if fs:
            for feat_selector in fitted_objs[6]:

                feat_mask = feat_selector.get_support()
                feat_names = np.array(feat_names)[feat_mask]

                X_test[feat_names] = feat_selector.transform(X_test)
                X_test = X_test[feat_names]

        return X_test, y_test

    def proc_X_train(self, X_train, y_train):

        # Load all base objects
        fitted_objs = self._get_objs_by_name()

        # No need to proc in place, so the transformations are pretty easy
        for loader in fitted_objs[0]:
            X_train = loader.transform(f_array(X_train))
        for imputer in fitted_objs[1]:
            X_train = imputer.transform(f_array(X_train))
        for scaler in fitted_objs[2]:
            X_train = scaler.transform(X_train)
        for transformer in fitted_objs[3]:
            X_train = transformer.transform(X_train)
        for sampler in fitted_objs[4]:
            X_train, y_train = sampler.fit_resample(X_train, y_train)
        for drop in fitted_objs[5]:
            X_train = drop.transform(X_train)
        for feat_selector in fitted_objs[6]:
            X_train = feat_selector.transform(X_train)

        return X_train

    def inverse_transform_FIs(self, fis, feat_names):

        # Make compat w/ subjects x feats
        if len(fis.shape) == 1:
            fis = np.expand_dims(fis, axis=0)

        # To inverse transform FIs, we are only concerned with feat_selectors
        # transformers, and loaders
        fitted_objs = self._get_objs_by_name()

        # Feat selectors
        for feat_selector in fitted_objs[6][::-1]:
            fis = feat_selector.inverse_transform(fis)

        # Reverse drop strat
        for drop in fitted_objs[5]:
            fis = drop.inverse_transform(fis)

        # Transformers
        for transformer, name in zip(fitted_objs[3][::-1],
                                     self.names[3][::-1]):
            fis = transformer.inverse_transform(fis, name=name)

        # Loaders - special case
        inversed_loaders = {}
        for loader, name in zip(fitted_objs[0][::-1],
                                self.names[0][::-1]):
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
