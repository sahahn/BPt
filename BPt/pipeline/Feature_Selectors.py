"""
Feature_Selectors.py
====================================
File with different Feature Selectors
"""
from ..helpers.ML_Helpers import (get_possible_init_params,
                                  get_obj_and_params)
from sklearn.feature_selection import SelectPercentile, VarianceThreshold
from ..extensions.Feat_Selectors import RFE_Wrapper, FeatureSelector
import numpy as np
from numpy.random import RandomState
import nevergrad as ng

from sklearn.base import BaseEstimator, clone
from sklearn.feature_selection._base import SelectorMixin
from ..helpers.ML_Helpers import proc_mapping, update_mapping
from .ScopeObjs import ScopeTransformer


class BPtFeatureSelector(ScopeTransformer, SelectorMixin):

    def fit(self, X, y=None, mapping=None,
            train_data_index=None, **fit_params):

        if mapping is None:
            mapping = {}

        # Call parent fit
        super().fit(X, y=y, mapping=mapping,
                    train_data_index=train_data_index,
                    **fit_params)

        # Need to pass along the correct mapping
        # overwrite existing out mapping
        self.out_mapping_ = {}

        # This is the calculated support from the base estimator
        support = self.estimator_.get_support()

        # First half is for updating the index within scope
        cnt = 0
        for i, ind in enumerate(self.inds_):

            # If kept by feat selection, add, otherwise set to None
            if support[i]:
                self.out_mapping_[ind] = cnt
                cnt += 1
            else:
                self.out_mapping_[ind] = None

        # Next, need to update the mapping for the remaining wrapper inds
        # essentially setting them where the cnt left off, then sequentially
        for rem_ind in range(len(self.rest_inds_)):
            self.out_mapping_[self.rest_inds_[rem_ind]] = cnt
            cnt += 1

        # Update the original mapping, this is the mapping which
        # will be passed to the next piece of the pipeline
        update_mapping(mapping, self.out_mapping_)

        return self

    def _proc_new_names(self, feat_names, base_name=None):

        # Get base new names from parent class
        new_names = super()._proc_new_names(feat_names)

        # This feat mask corresponds to the already transformed feats
        feat_mask = self._get_support_mask()

        # Apply the computed mask to get the actually selected features
        return_names = np.array(new_names)[feat_mask]

        return list(return_names)

    def _get_support_mask(self):

        # Create full support as base support + True's for all rest inds
        # i.e., those features originally out of scope
        base_support = self.estimator_.get_support()
        rest_support = np.ones(len(self.rest_inds_), dtype='bool')
        support = np.concatenate([base_support, rest_support])

        return support


AVALIABLE = {
        'binary': {
                'univariate selection c':
                'univariate selection c',
                'univariate selection':
                'univariate selection c',
                'rfe': 'rfe',
                'variance threshold': 'variance threshold',
                'selector': 'selector',
        },
        'regression': {
                'univariate selection r':
                'univariate selection r',
                'univariate selection':
                'univariate selection r',
                'rfe': 'rfe',
                'variance threshold': 'variance threshold',
                'selector': 'selector',
        },
}

AVALIABLE['categorical'] = AVALIABLE['binary'].copy()

SELECTORS = {
    'univariate selection r': (SelectPercentile,
                               ['base univar fs regression',
                                'univar fs regression dist',
                                'univar fs regression dist2']),

    'univariate selection c': (SelectPercentile,
                               ['base univar fs classifier',
                                'univar fs classifier dist',
                                'univar fs classifier dist2']),

    'rfe': (RFE_Wrapper, ['base rfe', 'rfe num feats dist']),

    'variance threshold': (VarianceThreshold, ['default']),

    'selector': (FeatureSelector, ['random', 'searchable']),
}


def get_special_selector(feat_selector, feat_selector_params, random_state,
                         num_feat_keys):

    # Init feat selector with mask of random feats
    if random_state is None:
        r_state = RandomState(np.random.randint(1000))
    elif isinstance(random_state, int):
        r_state = RandomState(random_state)
    else:
        r_state = random_state

    init_mask = r_state.random(num_feat_keys)
    feat_selector = feat_selector(mask=init_mask)

    # Figure out param passed
    if 'selector__mask' in feat_selector_params:
        p_name = 'selector__mask'

        # If set to searchable, set to searchable...
        if feat_selector_params[p_name] == 'sets as hyperparameters':

            feat_array = ng.p.Array(init=[.5 for i in range(num_feat_keys)])
            feat_array.set_mutation(sigma=1/6).set_bounds(lower=0, upper=1)
            feat_selector_params[p_name] = feat_array

        elif feat_selector_params[p_name] == 'sets as random features':
            del feat_selector_params[p_name]

    return feat_selector, feat_selector_params


def get_feat_selector_and_params(feat_selector_str, extra_params, params,
                                 random_state, num_feat_keys):
    '''Returns a scaler based on proced str indicator input,

    Parameters
    ----------
    feat_selector_str : str
        `feat_selector_str` refers to the type of feature selection
        to apply to the saved data during model evaluation.

    extra_params : dict
        Any extra params being passed.
        These can be supplied by creating another dict within extra_params.
        E.g., extra_params[method] = {'method param' : new_value}
        Where method param is a valid argument for that method,
        and method in this case is the str indicator.

    params : int
        The index of the params to use.

    Returns
    ----------
    feature_selector
        A feature selector object with fit and transform methods.

    dict
        The params for this feature selector
    '''

    feat_selector, extra_feat_selector_params, feat_selector_params =\
        get_obj_and_params(feat_selector_str, SELECTORS, extra_params, params)

    # Special behavior for selector...
    if feat_selector_str == 'selector':

        feat_selector, feat_selector_params =\
            get_special_selector(feat_selector, feat_selector_params,
                                 random_state, num_feat_keys)

        return feat_selector, feat_selector_params

    else:
        # Need to check for estimator, as RFE needs a default param for est.
        # Set as placeholder None if passed
        possible_params = get_possible_init_params(feat_selector)
        if 'estimator' in possible_params:
            extra_feat_selector_params['estimator'] = None

        return (feat_selector(**extra_feat_selector_params),
                feat_selector_params)
