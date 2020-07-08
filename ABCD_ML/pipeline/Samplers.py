from imblearn.over_sampling import (RandomOverSampler, SMOTE, ADASYN,
                                    BorderlineSMOTE, SVMSMOTE, KMeansSMOTE,
                                    SMOTENC)
from imblearn.under_sampling import (ClusterCentroids, RandomUnderSampler,
                                     NearMiss, TomekLinks,
                                     EditedNearestNeighbours,
                                     RepeatedEditedNearestNeighbours,
                                     AllKNN, CondensedNearestNeighbour,
                                     OneSidedSelection,
                                     NeighbourhoodCleaningRule)
from imblearn.combine import SMOTEENN, SMOTETomek

import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import OneHotEncoder

from ..helpers.ML_Helpers import get_obj_and_params, find_ind, proc_mapping
from ..helpers.Data_Helpers import get_unique_combo, reverse_unique_combo


class Sampler_Wrapper():

    def __init__(self, sampler_obj, sampler_type='change', strat_inds=[],
                 sample_target=True, sample_strat=[], categorical=True,
                 recover_strat=False, covars_inds=[],
                 regression_bins=5, regression_bin_strategy='uniform',
                 copy=True, **kwargs):

        # Sampler + sampler info
        self.sampler_obj = sampler_obj
        self.sampler_type = sampler_type

        # Which inds are strat (to be ignored / re-filled)
        self.strat_inds = strat_inds

        # Binary, sample target or not
        self.sample_target = sample_target

        # As actual inds, therefore could also refer to a covar, if right type
        # make sure passed as list even if only 1
        self.sample_strat = sample_strat

        self.categorical = categorical

        self.recover_strat = recover_strat

        # Covar info
        self.covars_inds = covars_inds
        self.all_covars_inds = []
        for c_inds in self.covars_inds:
            self.all_covars_inds += c_inds

        # For handling regression
        self.regression_bins = regression_bins
        self.regression_bin_strategy = regression_bin_strategy

        self.copy = copy

        # Init sampler, w/ rest of passed args if any
        self.init_sampler(kwargs)

    def init_sampler(self, kwargs):

        try:
            self.sampler = self.sampler_obj(**kwargs)

        except TypeError:

            kwargs['categorical_features'] = 'temp'
            self.sampler = self.sampler_obj(**kwargs)

    def get_return_X(self, X, X_resamp, strat_resamp):
        ''' Processes X_resamp based on a few different things.

        If base behavior was just re-sample based on y, then nothing
        needs to be done.

        Otherwise, the point of this function is to fill back in
        the strat columns, that were left out of base_X.
        There is slightly different behavior for over-sampler vs. undersampler.

        In general, strat is recoverable if sampler_type is 'no change',
        but there is a high overhead.
        Otherwise, only any strat col explicitly re-sampled
        will be preserved, with the rest set to NaN.
        '''

        # If no strat then X_resamp is ready as is
        if (self.base_X_mask).all():
            return X_resamp

        o_size, n_size = X.shape[0], X_resamp.shape[0]
        return_X = np.zeros((n_size, X.shape[1]))

        # If nothing changed, return original
        if o_size == n_size:
            return X

        # If over-sampler
        if o_size < n_size:

            return_X[:o_size] = X
            return_X[o_size:, self.base_X_mask] = X_resamp[o_size:]

            # If sampler type is 'no change' and want to recover
            if self.sampler_type == 'no change' and self.recover_strat is True:

                for r_ind in range(o_size, n_size):
                    o_ind = find_ind(X, self.base_X_mask, X_resamp, r_ind)
                    return_X[r_ind] = X[o_ind]

            else:

                # One behavior if some sample_strat vals are known
                if len(self.sample_strat) > 0:

                    # Fill with known vals
                    return_X[o_size:, self.sample_strat] =\
                        strat_resamp[o_size:]

                    # Rest as NaN
                    rest = ~self.base_X_mask.copy()
                    rest[self.sample_strat] = False
                    return_X[o_size:, rest] = np.nan

                # otherwise all rest as nan
                else:
                    return_X[o_size:, ~self.base_X_mask] = np.nan

        # Under-sampler
        else:

            # If under-sampler that doesn't change / insert new vals
            if self.sampler_type == 'no change' and self.recover_strat is True:

                for r_ind in range(len(X_resamp)):
                    o_ind = find_ind(X, self.base_X_mask, X_resamp, r_ind)
                    return_X[r_ind] = X[o_ind]

            # If special under-sampler, return just filled with NaN
            else:
                return_X[:, self.base_X_mask] = X_resamp

                if len(self.sample_strat) > 0:

                    # Atleast fill with known strat inds
                    return_X[:, self.sample_strat] = strat_resamp

                    # Set rest to NaN
                    rest = ~self.base_X_mask.copy()
                    rest[self.sample_strat] = False
                    return_X[:, rest] = np.nan

                else:
                    return_X[:, ~self.base_X_mask] = np.nan

        return return_X

    def fix_cat(self, X_copy, X_resamp, c_inds, y=None):

        try:
            original_col_vals = X_copy[:, c_inds]
        except IndexError:
            original_col_vals = y

        resamp_col_vals = X_resamp[:, c_inds].astype(float)

        # Binary or ordinal
        if original_col_vals.shape[1] == 1:

            if np.max(original_col_vals) > 1:
                X_resamp[:, c_inds] = np.rint(resamp_col_vals)
            else:
                X_resamp[:, c_inds] =\
                    np.where(resamp_col_vals > 0.5, 1, 0)

        # One-hot or dummy
        else:

            dummy = False
            if (np.sum(original_col_vals, axis=1) == 0).any():
                dummy = True

            clean = np.zeros(resamp_col_vals.shape)

            for i in range(len(clean)):
                if not dummy and (resamp_col_vals[i] > .5).any():
                    mx = np.argmax(resamp_col_vals[i])
                    clean[i, mx] = 1

            X_resamp[:, c_inds] = clean

        return X_resamp

    def fix_X_resamp(self, X_copy, X_resamp):
        ''' Tries to fix X_resamp, if covars and type change'''

        if len(self.covars_inds) > 0 and self.sampler_type == 'change':
            for c_inds in self.covars_inds:
                X_resamp = self.fix_cat(X_copy, X_resamp, c_inds)

        return X_resamp

    def categorical_fit_resample(self, X_copy, base_X, y):

        if self.sample_target is True:

            y_multiclass = y.reshape(-1, 1)

            # Just sample on y
            if len(self.sample_strat) == 0:
                X_resamp, y_multiclass_resamp =\
                    self.sampler.fit_resample(base_X, y_multiclass.ravel())

                if len(y_multiclass_resamp.shape) == 1:
                    y_multiclass_resamp =\
                        np.expand_dims(y_multiclass_resamp, -1)

                strat_resamp = None

            # Sample on a combo of y and passed strat
            else:

                # Make strat + y combo
                merged_y = np.append(X_copy[:, self.sample_strat],
                                     y_multiclass, 1)
                unique_combo_y, le = get_unique_combo(merged_y)

                # Re-sample
                X_resamp, y_unique_combo_resamp =\
                    self.sampler.fit_resample(base_X, unique_combo_y)

                # Reverse the combo to y and strat
                merged_y_resamp =\
                    reverse_unique_combo(y_unique_combo_resamp, le)

                y_multiclass_resamp = merged_y_resamp[:, [-1]]
                strat_resamp = merged_y_resamp[:, :-1]

            # Fix covar cols + add back in self.strat cols to X
            X_resamp = self.fix_X_resamp(X_copy, X_resamp)
            X_resamp = self.get_return_X(X_copy, X_resamp, strat_resamp)

            # Convert multiclass y back to encoded y
            y_resamp = np.squeeze(y_multiclass_resamp)

            return X_resamp, y_resamp

        else:

            y_encoder = OneHotEncoder(categories='auto', sparse=False)

            if len(y.shape) == 1:
                flat_input = True
                y = y.reshape(-1, 1)
            else:
                flat_input = False

            # Multiclass
            if np.max(y) > 1:
                y_one_hot = y_encoder.fit_transform(y)
            else:
                y_one_hot = y

            # Add y to base_X
            base_X_shape, y_shape = base_X.shape, y_one_hot.shape
            base_X = np.append(base_X, y_one_hot, axis=1)

            # Get y_inds within base_X
            y_inds = np.arange(base_X_shape[1],
                               base_X_shape[1] + y_shape[1])

            # Add y as cat inds if special
            if self.sampler_type == 'special':
                self.sampler.categorical_features +=\
                    list(y_inds)

            # Create merged strat
            merged_strat, le = get_unique_combo(X_copy[:, self.sample_strat])

            # Resample
            X_resamp, merged_strat_resamp =\
                self.sampler.fit_resample(base_X, merged_strat)

            # Reverse strat merge
            strat_resamp =\
                reverse_unique_combo(merged_strat_resamp, le)

            # Fix y within X_resamp inplace
            X_resamp = self.fix_cat(X_copy, X_resamp, y_inds, y=y_one_hot)

            # Seperate y from X
            y_resamp = X_resamp[:, y_inds]
            X_resamp = np.delete(X_resamp, y_inds, 1)

            # Fix covar cols if needed + add back strat
            X_resamp = self.fix_X_resamp(X_copy, X_resamp)
            X_resamp = self.get_return_X(X_copy, X_resamp, strat_resamp)

            # Multiclass
            if y_resamp.shape[1] > 1:
                y_resamp = y_encoder.transform(y_resamp)

            if flat_input:
                y_resamp = np.squeeze(y_resamp)

            return X_resamp, y_resamp

    def regression_fit_resample(self, X_copy, base_X, y):

        if self.sample_target is True:

            # If sample on y, must bin first
            kbin = KBinsDiscretizer(n_bins=self.regression_bins,
                                    encode='ordinal',
                                    strategy=self.regression_bin_strategy)
            binned_y = kbin.fit_transform(y.reshape(-1, 1))
            binned_y = np.squeeze(binned_y)

            # Then re-sample as if categorical, multiclass
            X_resamp, y_resamp =\
                self.categorical_fit_resample(X_copy, base_X, binned_y)

            # Recover y values
            o_size, n_size = len(y), len(y_resamp)
            return_y = np.zeros(n_size)
            to_fill = np.arange(0, n_size)

            # If nothing changed, return original y
            if o_size == n_size:
                return X_resamp, y

            # If over-sampler, fill_known + change to_fill
            if o_size < n_size:
                return_y[:o_size] = y
                to_fill = np.arange(o_size, n_size)

            # If sampler_type == 'no change', then explicitly recover new
            if self.sampler_type == 'no change':
                for r_ind in to_fill:

                    o_ind = find_ind(X_copy, self.base_X_mask, X_resamp, r_ind,
                                     mask=False)
                    return_y[r_ind] = y[o_ind]

            # Otherwise, inverse_transform to get new_vals
            else:
                return_y[to_fill] =\
                    np.squeeze(kbin.inverse_transform(
                        y_resamp[to_fill].reshape(-1, 1)))

            return X_resamp, return_y

        # If not sampling target, then don't need to bin!
        else:

            y = y.reshape(-1, 1)
            base_X_shape, y_shape = base_X.shape, y.shape
            base_X = np.append(base_X, y, axis=1)

            # Get y_inds within base_X
            y_inds = np.arange(base_X_shape[1],
                               base_X_shape[1] + y_shape[1])

            # Create merged strat
            merged_strat, le =\
                get_unique_combo(X_copy[:, self.sample_strat])

            # Resample
            X_resamp, merged_strat_resamp =\
                self.sampler.fit_resample(base_X, merged_strat)

            # Reverse strat merge
            strat_resamp =\
                reverse_unique_combo(merged_strat_resamp, le)

            # Seperate y from X
            y_resamp = X_resamp[:, y_inds]
            X_resamp = np.delete(X_resamp, y_inds, 1)

            # Fix covar cols if needed + add back strat
            X_resamp = self.fix_X_resamp(X_copy, X_resamp)
            X_resamp = self.get_return_X(X_copy, X_resamp, strat_resamp)

            # Set y_resamp back to right dims
            y_resamp = np.squeeze(y_resamp)

            return X_resamp, y_resamp

    def _proc_mapping(self, mapping):

        try:
            self._mapping
            return

        except AttributeError:
            self._mapping = mapping

        if len(mapping) > 0:

            self.strat_inds = proc_mapping(self.strat_inds, mapping)
            self.sample_strat = proc_mapping(self.sample_strat, mapping)
            self.covars_inds = proc_mapping(self.covars_inds, mapping)
            self.all_covars_inds = proc_mapping(self.all_covars_inds, mapping)

        return

    def fit_resample(self, X, y, mapping=None):

        if mapping is None:
            mapping = {}

        self._proc_mapping(mapping)

        # Set cat inds if special
        if self.sampler_type == 'special':
            self.sampler.categorical_features =\
                self.all_covars_inds

        self.base_X_mask = np.full(np.shape(X)[1], True)

        if len(self.strat_inds) > 0:
            self.base_X_mask[self.strat_inds] = False

        if self.copy:
            X_copy = X.copy()
            y_copy = y.copy()
        else:
            X_copy = X
            y_copy = y

        base_X = X_copy[:, self.base_X_mask]

        if self.categorical:
            X_resamp, y_resamp =\
                self.categorical_fit_resample(X_copy, base_X, y_copy)

        # Regression type problem
        else:
            X_resamp, y_resamp =\
                self.regression_fit_resample(X_copy, base_X, y_copy)

        return X_resamp, y_resamp

    def set_params(self, **params):

        if 'sampler_obj' in params:
            self.sampler_obj = params.pop('sampler_obj')
        if 'sampler_type' in params:
            self.sampler_type = params.pop('sampler_type')
        if 'strat_inds' in params:
            self.strat_inds = params.pop('strat_inds')
        if 'sample_target' in params:
            self.sample_target = params.pop('sample_target')
        if 'sample_strat' in params:
            self.sample_strat = params.pop('sample_strat')
        if 'sample_strat' in params:
            self.sample_strat = params.pop('sample_strat')
        if 'recover_strat' in params:
            self.recover_strat = params.pop('recover_strat')
        if 'covars_inds' in params:
            self.covars_inds = params.pop('covars_inds')
        if 'regression_bins' in params:
            self.regression_bins = params.pop('regression_bins')
        if 'regression_bin_strategy' in params:
            self.regression_bin_strategy =\
                params.pop('regression_bin_strategy')
        if 'copy' in params:
            self.copy = params.pop('copy')

        # Set rest of passed params to the sampler object
        self.sampler.set_params(**params)

    def get_params(self, deep=False):

        params = {'sampler_obj': self.sampler_obj,
                  'sampler_type': self.sampler_type,
                  'strat_inds': self.strat_inds,
                  'sample_target': self.sample_target,
                  'sample_strat': self.sample_strat,
                  'categorical': self.categorical,
                  'recover_strat': self.recover_strat,
                  'covars_inds': self.covars_inds,
                  'regression_bins': self.regression_bins,
                  'regression_bin_strategy': self.regression_bin_strategy,
                  'copy': self.copy
                  }

        # Add the sampler params
        params.update(self.sampler.get_params())

        return params


SAMPLERS = {
    'random over sampler': (RandomOverSampler, ['base no change sampler']),
    'smote': (SMOTE, ['base change sampler']),
    'adasyn': (ADASYN, ['base change sampler']),
    'borderline smote': (BorderlineSMOTE, ['base change sampler']),
    'svm smote': (SVMSMOTE, ['base change sampler']),
    'kmeans smote': (KMeansSMOTE, ['base change sampler']),
    'smote nc': (SMOTENC, ['base special sampler']),
    'cluster centroids': (ClusterCentroids, ['base change sampler']),
    'random under sampler': (RandomUnderSampler, ['base no change sampler',
                                                  'rus binary ratio']),
    'near miss': (NearMiss, ['base no change sampler']),
    'tomek links': (TomekLinks, ['base no change sampler']),
    'enn': (EditedNearestNeighbours, ['base no change sampler']),
    'renn': (RepeatedEditedNearestNeighbours, ['base no change sampler']),
    'all knn': (AllKNN, ['base no change sampler']),
    'condensed nn': (CondensedNearestNeighbour, ['base no change sampler']),
    'one sided selection': (OneSidedSelection, ['base no change sampler']),
    'neighbourhood cleaning rule': (NeighbourhoodCleaningRule,
                                    ['base no change sampler']),
    'smote enn': (SMOTEENN, ['base change sampler']),
    'smote tomek': (SMOTETomek, ['base change sampler']),
}


def get_sampler_and_params(param, search_type,
                           strat_inds=[], sample_target=False, sample_strat=[],
                           categorical=True, recover_strat=False,
                           covars_inds=[]):

    sampler_str = param.obj,
    extra_params = param.extra_params
    params = param.params

    # Grab base object, params, and param distributions
    base_sampler_obj, sampler_wrapper_params, sampler_params =\
            get_obj_and_params(sampler_str, SAMPLERS, extra_params, params,
                               search_type)

    # Add rest of passed params to sampler wrapper params
    sampler_wrapper_params['sampler_obj'] = base_sampler_obj
    sampler_wrapper_params['strat_inds'] = strat_inds
    sampler_wrapper_params['sample_target'] = sample_target
    sampler_wrapper_params['sample_strat'] = sample_strat
    sampler_wrapper_params['categorical'] = categorical
    sampler_wrapper_params['recover_strat'] = recover_strat
    sampler_wrapper_params['covars_inds'] = covars_inds

    sampler_wrapper_params['regression_bins'] = param.target_bins
    sampler_wrapper_params['regression_bin_strategy'] = param.target_bin_strategy

    # Make object
    sampler = Sampler_Wrapper(**sampler_wrapper_params)

    return sampler, sampler_params


# From Params_Classes
class Sampler(Piece):

    def __init__(self, obj, params=0, sample_on='targets', target_bins=3,
                 target_bin_strategy='uniform', extra_params=None):
        '''Sampler is a special base :class:`Model_Pipeline` pieces designed
        to perform resampling on the data points / subjects themselves. This
        is typically used in cases where there are extreme class imbalances, but can
        also be used to resample based on a loaded Strat value.

        Note: As of right now, the use of samplers, along with nested
        parameter searches may cause problems! Samplers as a whole might
        likely get fully re-designed at some point in the future, for now
        if you use this Object, just be careful.

        Parameters
        ----------
        obj : str
            `obj` refers to one of the pre-defining sampling strategies,
            and is selected by indicated one of the pre-set str indicators
            found at :ref:`Samplers`.

        params : int, str or dict of :ref:`params<Params>`, optional
            `params` determines optionally if the distribution of hyper-parameters to
            potentially search over for this sampler. Preset param distributions are
            listed for each choice of obj at :ref:`Samplers`, and you can read more on
            how params work more generally at :ref:`Params`.

            ::

                default = 0

        sample_on : str, or list of, optional
            This parameter dictates what the underlying sampler should use as its
            variable to re-sample on. While the most typical case is to re-sample
            based on the target variable, i.e., to potentially help correct a class imbalance,
            you may also choose to re-sample based on a loaded Strat value
            (see :func:`Load_Strat<ABCD_ML.Load_Strat>`).

            If a list of values is passed to sample_on, then that will be interpretted as
            sampling based on the unique overlap of values from all of those columns.
            E.g., if passed `sex` and a k_binned `age` variable, then it would internally
            create a new categorical variable as the unique intersection of all elements on the
            passed list.

            Note: if any single value, or value within the list of values passed is
            not a valid loaded column within Strat values, then it will be converted
            to the name of the currently selected target variable. In this way, to just
            by default sample based on the target variable, you may keep sample_on at its
            default value of 'target'

            ::

                default = 'target'

        target_bins : int, optional
            If the target variable is set to or included in `sample_on`, and further if you
            are in the context of a regression problem, where the target variable is of float type,
            then the variable will be binned before use as in sample_on, this parameter dictates
            the number of bins to create.

            ::

                default = 3

        target_bin_strategy : {'uniform', 'quantile', 'kmeans'}, optional
            If the target variable is set to or included in `sample_on`, and further if you
            are in the context of a regression problem, where the target variable is of float type,
            then the variable will be binned according the the number of `target_bins` and also
            this param, which controls what strategy is used to define the bins. Options are:

            - 'uniform'
                All bins in each feature have identical widths.

            - 'quantile'
                All bins in each feature have the same number of points.

            - 'kmeans'
                Values in each bin have the same nearest center of a 1D
                k-means cluster.

            You likely do not want to use `quantile` here...

            ::

                default = 'uniform'

        extra_params : :ref`extra params dict<Extra Params>`, optional

            See :ref:`Extra Params`

            ::

                default = None

        '''

        self.obj = obj
        self.params = params
        self.sample_on = sample_on
        self.target_bins = target_bins
        self.target_bin_strategy = target_bin_strategy
        self.extra_params = extra_params

        self.check_args()

    def add_strat_u_name(self, func):
        self.sample_on = func(self.sample_on)


# From Pipeline_Pieces
class Samplers(Scope_Pieces):

    name = 'samplers'

    def _process(self, params):

        sample_on = [p.sample_on for p in params]
        recover_strats = self._get_recover_strats(sample_on)

        # Get the scalers and params
        scalers_and_params =\
            [self._get_sampler(param, on, recover_strat)
             for param, on, recover_strat in zip(params, sample_on,
                                                 recover_strats)]

        samplers, sampler_params =\
            self._proc_objs_and_params(scalers_and_params)

        # Change Random State + n_jobs
        samplers = self._check_and_replace_samplers(samplers,
                                                    'random_state',
                                                    self.spec['random_state'])

        samplers = self._check_and_replace_samplers(samplers,
                                                    'n_jobs',
                                                    self.spec['n_jobs'])

        return samplers, sampler_params

    def _get_recover_strats(self, sample_on):

        # Creates binary mask of, True if any strat inds used
        uses_strat = [len(self._proc_sample_on(s)[1]) > 0 for s in sample_on]

        # If never use strat, set all to False
        if True not in uses_strat:
            return [False for i in range(len(sample_on))]

        # If strat is used, then just need Trues up to but not including
        # the last true.
        last_true = len(uses_strat) - 1 - uses_strat[::-1].index(True)

        trues = [True for i in range(last_true)]
        falses = [False for i in range(len(sample_on) - last_true)]

        return trues + falses

    def _proc_sample_on(self, on):

        # On should be either a list, or a single str
        if isinstance(on, str):
            on = [on]
        else:
            on = list(on)

        sample_strat_keys = [o for o in on if o in self.Data_Scopes.strat_keys]

        # Set sample_target
        sample_target = False
        if len(sample_strat_keys) != len(on):
            sample_target = True

        return sample_target, sample_strat_keys

    def _get_sampler(self, param, on, recover_strat):

        from .Samplers import get_sampler_and_params

        # Grab sample_target and sample_strat_keys, from on
        sample_target, sample_strat_keys = self._proc_sample_on(on)

        # Set strat inds and sample_strat
        strat_inds = self.Data_Scopes.get_strat_inds()
        sample_strat =\
            self.Data_Scopes.get_train_inds_from_keys(sample_strat_keys)

        # Set categorical flag
        categorical = True
        if self.spec['problem_type'] == 'regression':
            categorical = False

        cat_inds, ordinal_inds =\
            self.Data_Scopes.get_cat_ordinal_inds()
        covars_inds = cat_inds + [[o] for o in ordinal_inds]

        # Get the sampler
        sampler, sampler_params =\
            get_sampler_and_params(param, self.spec['search_type'],
                                   strat_inds=strat_inds,
                                   sample_target=sample_target,
                                   sample_strat=sample_strat,
                                   categorical=categorical,
                                   recover_strat=recover_strat,
                                   covars_inds=covars_inds)

        return param.obj, (sampler, sampler_params)

    def _check_and_replace_samplers(self, samplers, param_name, replace_value):

        for i in range(len(samplers)):

            try:
                getattr(samplers[i][1].sampler, param_name)
                setattr(samplers[i][1].sampler, param_name, replace_value)

            except AttributeError:
                pass

        return samplers


# From Model_Pipeline doc string
'''
samplers : :class:`Sampler`, list of or None, optional
            Each :class:`Sampler` refers to an optional type
            of data point resampling in which to preform, i.e., 
            in attempt to correct for a class imbalance. See the
            base :class:`Sampler` object for more information on
            what different sampler options and restrictions are.

            If passed a list, the sampling will be applied sequentially.

            ::

            default = None
'''


## From Data_Scopes
 def get_cat_ordinal_inds(self):

        cat_keys = self.covar_scopes['categorical']
        ordinal_keys = self.covar_scopes['ordinal categorical']

        cat_inds = [self.get_inds_from_scope(k) for k in cat_keys]
        ordinal_inds = self.get_inds_from_scope(ordinal_keys)

        return cat_inds, ordinal_inds


## From _Data

def _get_covar_scopes(self):

    # categorical also includes multilabel

    covar_scopes = {'float': [],
                    'categorical': [],
                    'ordinal categorical': []}
    cat_encoders = []

    for base_covar in list(self.covars_encoders):

        cov_encoder = self.covars_encoders[base_covar]

        # One-hot or dummy
        if isinstance(cov_encoder, tuple):

            one_hot_encoder = cov_encoder[1]
            cat_encoders.append(cov_encoder)

            categories = one_hot_encoder.categories_[0]
            covar_df_names = [base_covar + '_' + str(c) for
                              c in categories]
            valid_df_names = [c for c in covar_df_names if
                              c in self.all_data]

            covar_scopes['categorical'].append(valid_df_names)

        # Multilabel
        elif isinstance(cov_encoder, list):
            cat_encoders.append(None)
            covar_scopes['categorical'].append(cov_encoder)

        # Float
        elif cov_encoder is None:
            covar_scopes['float'].append(base_covar)

        # Binary/ordinal
        else:
            covar_scopes['ordinal categorical'].append(base_covar)

    return covar_scopes, cat_encoders