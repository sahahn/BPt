from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from operator import add
from functools import reduce


class OverSampler(BaseEstimator, TransformerMixin):

    _needs_fit_index = True
    _needs_transform_index = True

    def __init__(self, groups=None, random_state=None):
        
        self.groups = groups
        self.random_state = random_state

    def _process_groups_to_series(self):

        if isinstance(self.groups, pd.Series):
            return self.groups

        elif isinstance(self.groups, pd.DataFrame):
            
            # If only 1 col
            if len(self.groups) == 1:
                return self.groups[list(self.groups)[0]]

            # Otherwise need to combine into unique overlap
            combo = []
            for col in self.groups:
                combo.append(col + '=' + self.groups[col].astype(str) + ' ')
            combo = reduce(add, combo)

            return combo

        raise RuntimeError('Passed groups must be DataFrame or series')

    def fit(self, X, y=None, fit_index=None):

        # Process the stored groups
        groups_series = self._process_groups_to_series()

        if isinstance(X, pd.DataFrame):
            fit_index = X.index
            X = np.array(X)

        self.groups_vals_ = groups_series.loc[fit_index]
        self.rng_ = np.random.default_rng(self.random_state)

        # Get unique + counts
        unique, cnts = np.unique(self.groups_vals_, return_counts=True)
        highest = np.max(cnts)

        # Go through each group and select extra samples
        extra_inds = []
        for u, c in zip(unique, cnts):
            
            # Skip largest group(s)
            if c == highest:
                continue
            
            # Select minority inds to sample
            minority_inds = np.where(self.groups_vals_ == u)[0]
            
            # Random sample
            to_add = self.rng_.choice(minority_inds, size=highest - c)
            extra_inds.append(to_add)
            
        # Combine
        extra_inds = np.concatenate(extra_inds)

        # Add to base for completed sampled numeric inds
        self.sampled_inds_ = np.concatenate([np.arange(len(self.groups_vals_)), extra_inds])

        return self

    def transform(self, X, transform_index=None):

        if isinstance(X, pd.DataFrame):
            transform_index = X.index
            X = np.array(X)

        # Only transform if the passed transform index overlaps
        # with the new fit index, i.e. in case of fit_transform
        new_fit_index = self.groups_vals_.iloc[self.sampled_inds_].index
        if len(new_fit_index.intersection(transform_index)) == 0:
            return X

        # Return resampled X and new_fit_inds
        return X[self.sampled_inds_], self.sampled_inds_, new_fit_index
                

    def fit_transform(self, X, y=None, fit_index=None):

        if isinstance(X, pd.DataFrame):
            fit_index = X.index
            X = np.array(X)

        # Pass fit index to both fit and transform index
        return self.fit(X, y=y, fit_index=fit_index).transform(X, transform_index=fit_index)

    def __repr__(self):

        with pd.option_context('display.max_rows', 5,
                               'display.max_columns', 1,
                               'display.max_colwidth', 1):
            rep = super().__repr__()
        return rep


class CounterBalance(BaseEstimator, TransformerMixin):

    _needs_fit_index = True
    _needs_transform_index = True

    def __init__(self, groups=None, cb_by=None, random_state=None):
        
        self.groups = groups
        self.cb_by = cb_by
        self.random_state = random_state

    def _process_groups_to_series(self, groups):

        if isinstance(groups, pd.Series):
            return groups

        elif isinstance(groups, pd.DataFrame):
            
            # If only 1 col
            if len(groups) == 1:
                return groups[list(groups)[0]]

            # Otherwise need to combine into unique overlap
            combo = []
            for col in groups:
                combo.append(col + '=' + groups[col].astype(str) + ' ')
            combo = reduce(add, combo)

            return combo

        raise RuntimeError('Passed groups must be DataFrame or series')

    def fit(self, X, y=None, fit_index=None):

        # Process the stored groups
        groups_series = self._process_groups_to_series(self.groups)
        cb_by_series = self._process_groups_to_series(self.cb_by)

        # Check for X is passed as dataframe
        if isinstance(X, pd.DataFrame):
            fit_index = X.index
            X = np.array(X)

        # Subset to current fit index
        self.groups_vals_ = groups_series.loc[fit_index]
        self.cb_by_vals_ = cb_by_series.loc[fit_index]

        # Set rng
        self.rng_ = np.random.default_rng(self.random_state)

        # Okay- idea here is to, go through each of the passed
        # groups, and within just the subset of subjects in that group
        # oversample according to the cb_by_vals_ groups.

        extra_inds = []

        # Start by going through each group, e.g., sex
        for group in np.unique(self.groups_vals_):

            # Get subset inds of just this group
            group_inds = np.where(self.groups_vals_ == group)[0]

            # Use those inds to select the correct subset of counter balance by series
            cb_group_vals = self.cb_by_vals_.iloc[group_inds]

            # Get unique + counts
            unique, cnts = np.unique(cb_group_vals, return_counts=True)
            highest = np.max(cnts)

            # Go through each and select extra samples
            for u, c in zip(unique, cnts):
                
                # Skip largest group(s)
                if c == highest:
                    continue

                # Select minority inds, according to this group by cb_by sample
                # then make sure index reflect numeric index relative to current
                # fit index, not numeric index relative to cb_group_vals directly
                minority_inds = group_inds[np.where(cb_group_vals == u)[0]]
                
                # Random sample highest count  - count of current group from minority
                to_add = self.rng_.choice(minority_inds, size=highest - c, replace=True)

                extra_inds.append(to_add)
            
        # Combine all extra inds
        extra_inds = np.concatenate(extra_inds)

        # Add to base for completed sampled numeric inds
        self.sampled_inds_ = np.concatenate([np.arange(len(self.groups_vals_)), extra_inds])

        return self

    def transform(self, X, transform_index=None):

        if isinstance(X, pd.DataFrame):
            transform_index = X.index
            X = np.array(X)

        # Only transform if the passed transform index overlaps
        # with the new fit index, i.e. in case of fit_transform
        new_fit_index = self.groups_vals_.iloc[self.sampled_inds_].index
        if len(new_fit_index.intersection(transform_index)) == 0:
            return X

        # Return resampled X and new_fit_inds
        return X[self.sampled_inds_], self.sampled_inds_, new_fit_index
                
    def fit_transform(self, X, y=None, fit_index=None):

        if isinstance(X, pd.DataFrame):
            fit_index = X.index
            X = np.array(X)

        # Pass fit index to both fit and transform index
        return self.fit(X, y=y, fit_index=fit_index).transform(X, transform_index=fit_index)

    def __repr__(self):

        with pd.option_context('display.max_rows', 5,
                               'display.max_columns', 1,
                               'display.max_colwidth', 1):
            rep = super().__repr__()
        return rep
