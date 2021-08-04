from sklearn.pipeline import Pipeline
import numpy as np
from .helpers import pipe_hash
from joblib import load, dump
import pandas as pd
from sklearn.utils.metaestimators import if_delegate_has_method
import os
from sklearn.utils import _print_elapsed_time
from sklearn.base import clone
from .base import (_get_est_fit_params, _get_est_trans_params,
                   _needs)

# TODO - Try to make caching pipelines ignore the number of jobs
# of the base pipeline components

# @TODO add docstrings here - and add to docs?
class BPtPipeline(Pipeline):

    _needs_mapping = True
    _needs_fit_index = True

    def __init__(self, steps, memory=None,
                 verbose=False,
                 cache_loc=None):

        self.cache_loc = cache_loc

        # The verbose of the base super pipeline class is binary
        super().__init__(steps=steps, memory=memory, verbose=verbose)

    @property
    def _needs_transform_index(self):

        # If any steps need it
        for step in self.steps:
            if _needs(step[1], '_needs_transform_index',
                      'transform_index', 'transform'):
                return True

        # Otherwise False
        return False

    @property
    def n_jobs(self):
        '''Return the first n jobs found in steps.
        This function is just meant to be a check to
        see if any of the pipeline steps have n_jobs to set.'''

        for step in self.steps:
            if hasattr(step[1], 'n_jobs'):
                return getattr(step[1], 'n_jobs')

        # Otherwise, return a step we know
        # doesn't have n_jobs
        return self.steps[0][1].n_jobs

    @n_jobs.setter
    def n_jobs(self, n_jobs):

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

    def _fit(self, X, y=None, fit_index=None, **fit_params_steps):

        # shallow copy of steps - this should really be steps_
        self.steps = list(self.steps)
        self._validate_steps()

        # For each transformer
        for (step_idx,
             name,
             transformer) in self._iter(with_final=False,
                                        filter_passthrough=False):

            with _print_elapsed_time('Pipeline',
                                     self._log_message(step_idx)):

                # Skip if passthrough
                if (transformer is None or transformer == 'passthrough'):
                    continue

                # Clone transformer
                cloned_transformer = clone(transformer)

                # Get the correct fit_transform params
                fit_trans_params =\
                    _get_est_fit_params(
                        estimator=cloned_transformer,
                        mapping=self.mapping_,
                        fit_index=fit_index,
                        other_params=fit_params_steps[name],
                        copy_mapping=False)

                # Fit transform the current transformer
                X = cloned_transformer.fit_transform(X=X, y=y,
                                                     **fit_trans_params)

                # Print if an estimator is skipped, if verbose
                if cloned_transformer.estimator_ is None:
                    if self.verbose:
                        print('Skipping Step:', name, 'due to empty scope.',
                              flush=True)

                # Replace the transformer of the step with the
                # cloned and now fitted transformer
                self.steps[step_idx] = (name, cloned_transformer)

        return X

    def fit(self, X, y=None, mapping=None,
            fit_index=None, **fit_params):

        if isinstance(X, pd.DataFrame):

            # Set train data index
            fit_index = X.index

            # Cast to np array
            X = np.array(X)

        if isinstance(y, (pd.DataFrame, pd.Series)):

            # Cast to np array
            y = np.array(y)

        if self.cache_loc is not None:

            # Compute the hash for this fit
            # Store as an attribute
            self.hash_ = pipe_hash([X, y, mapping,
                                   fit_index, fit_params],
                                   self.steps)

            # Check if hash exists - if it does load
            if os.path.exists(self._get_hash_loc()):
                self._load_from_hash()

                # end / return!
                return self

            # Otherwise, continue to fit as normal

        # Set internal mapping as either passed mapping or
        # initialize a new 1:1 mapping.
        if mapping is not None:
            self.mapping_ = mapping.copy()
        else:
            self.mapping_ = {i: i for i in range(X.shape[1])}

        # The base parent fit
        # -------------------

        # Get fit params as indexed by each step
        fit_params_steps = self._check_fit_params(**fit_params)

        # Fit and transform X for all but the last step.
        Xt = self._fit(X, y, fit_index=fit_index,
                       **fit_params_steps)

        # Fit the final step
        with _print_elapsed_time('Pipeline',
                                 self._log_message(len(self.steps) - 1)):
            if self._final_estimator != 'passthrough':

                # Get last params fit params
                fit_params_last_step = fit_params_steps[self.steps[-1][0]]

                # Add mapping and train data index if valid
                fit_params_last_step =\
                    _get_est_fit_params(self._final_estimator,
                                        mapping=self.mapping_,
                                        fit_index=fit_index,
                                        other_params=fit_params_last_step,
                                        copy_mapping=False)

                # Fit the final estimator
                self._final_estimator.fit(Xt, y, **fit_params_last_step)

        # If cache fit enabled, hash fitted pipe here
        if self.cache_loc is not None:
            self._hash_fit()

        return self

    def fit_transform(X, y=None, **fit_params):
        # @ TODO is there a case where we might need this?
        raise RuntimeError('Not currently implemented')

    def _get_hash_loc(self):

        # Make sure directory exists
        os.makedirs(self.cache_loc, exist_ok=True)

        # Set hash loc as directory + hash of fit args
        hash_loc = os.path.join(self.cache_loc, self.hash_)

        return hash_loc

    def _hash_fit(self):

        # Just save full fitted pipeline
        dump(self, self._get_hash_loc())
        return self

    def _load_from_hash(self):

        if self.verbose:
            print('Loading fitted pipeline from saved cache directory',
                  'cache_loc:', str(self.cache_loc), flush=True)

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

    def _get_ordered_objs_and_names(self):

        # Get all objects except final model
        ordered_names = [step[0] for step in self.steps[:-1]]
        ordered_objs = [self.__getitem__(name) for name in ordered_names]

        return ordered_objs, ordered_names

    def transform(self, X, transform_index=None):

        Xt = X

        # If DataFrame input
        if isinstance(Xt, pd.DataFrame):

            # Set transform index, then cast to array
            transform_index = Xt.index
            Xt = np.array(Xt)

        # For each transformer, but the last
        for step in self.steps[:-1]:
            transformer = step[1]

            # Get any needed transform params
            trans_params =\
                _get_est_trans_params(transformer,
                                      transform_index=transform_index)

            # Transform X - think in place is okay
            Xt = transformer.transform(Xt, **trans_params)

        return Xt

    def transform_df(self, X_df, encoders=None):
        '''Transform an input dataframe, keeping track of feature names'''

        # Get as two lists - all steps but last
        ordered_objs, ordered_base_names =\
            self._get_ordered_objs_and_names()

        # Run all of the transformations
        for obj, base_name in zip(ordered_objs, ordered_base_names):
            X_df = obj.transform_df(X_df, base_name=base_name,
                                    encoders=encoders)

        return X_df

    def transform_feat_names(self, X_df, encoders=None):
        '''Like transform df, but just transform feat names.'''

        # Get as two lists - all steps but last
        ordered_objs, ordered_base_names =\
            self._get_ordered_objs_and_names()

        feat_names = list(X_df)
        for obj, base_name in zip(ordered_objs, ordered_base_names):
            feat_names = obj._proc_new_names(feat_names, base_name=base_name,
                                             encoders=encoders)

        return feat_names

    def inverse_transform_FIs(self, fis):
        '''fis should be a pandas Series as indexed by feature name'''

        # Get as two lists - all steps but last
        ordered_objs, _ =\
            self._get_ordered_objs_and_names()

        # Process each object in reverse order
        for obj in ordered_objs[::-1]:

            # Process only if object has method
            if hasattr(obj, 'inverse_transform_FIs'):
                fis = obj.inverse_transform_FIs(fis)

        return fis

    @if_delegate_has_method(delegate='_final_estimator')
    def predict(self, X, **predict_params):

        # Transform X
        Xt = self.transform(X)

        # Then return final pipeline piece predicting
        # on the transformed data
        return self.steps[-1][-1].predict(Xt, **predict_params)

    @if_delegate_has_method(delegate='_final_estimator')
    def fit_predict(self, X, y=None, **fit_params):
        raise RuntimeError('Not Implemented')

    @if_delegate_has_method(delegate='_final_estimator')
    def predict_proba(self, X):

        # Transform X and predict
        Xt = self.transform(X)
        return self.steps[-1][-1].predict_proba(Xt)

    @if_delegate_has_method(delegate='_final_estimator')
    def decision_function(self, X):

        # Transform X and predict
        Xt = self.transform(X)
        return self.steps[-1][-1].decision_function(Xt)

    @if_delegate_has_method(delegate='_final_estimator')
    def score_samples(self, X):

        # Transform X and score samples
        Xt = self.transform(X)
        return self.steps[-1][-1].score_samples(Xt)

    @if_delegate_has_method(delegate='_final_estimator')
    def predict_log_proba(self, X):

        # Transform X and predict
        Xt = self.transform(X)
        return self.steps[-1][-1].predict_log_proba(Xt)

    @if_delegate_has_method(delegate='_final_estimator')
    def score(self, X, y=None, sample_weight=None):

        # Transform X
        Xt = self.transform(X)

        # Cast y from dataframe or series if needed
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = np.array(y)

        # Rest of function
        score_params = {}
        if sample_weight is not None:
            score_params['sample_weight'] = sample_weight
        return self.steps[-1][-1].score(Xt, y, **score_params)
