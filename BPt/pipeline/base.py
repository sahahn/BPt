from sklearn.base import BaseEstimator
from .helpers import get_possible_params


def _needs(estimator, flag, attr, method):

    # If has class attribute flag
    if hasattr(estimator, flag):
        if getattr(estimator, flag):
            return True

    # Next check if matches name of method param
    elif attr in get_possible_params(estimator, method):
        return True

    # Otherwise False
    return False


def _get_est_fit_params(estimator, mapping=None,
                        fit_index=None,
                        other_params=None,
                        copy_mapping=True):

    if other_params is None:
        fit_params = {}
    else:
        fit_params = other_params.copy()

    if 'mapping' in fit_params and mapping is None:
        mapping = fit_params.pop('mapping')

    if 'fit_index' in fit_params and fit_index is None:
        fit_index = fit_params.pop('fit_index')

    if mapping is not None and _needs(estimator, '_needs_mapping',
                                      'mapping', 'fit'):

        if copy_mapping:
            fit_params['mapping'] = mapping.copy()
        else:
            fit_params['mapping'] = mapping

    if fit_index is not None and _needs(estimator, '_needs_fit_index',
                                        'fit_index', 'fit'):
        fit_params['fit_index'] = fit_index

    return fit_params


def _get_est_trans_params(estimator, transform_index=None):

    trans_params = {}

    if transform_index is not None and _needs(estimator,
                                              '_needs_transform_index',
                                              'transform_index', 'transform'):
        trans_params['transform_index'] = transform_index

    return trans_params


def _get_est_fit_trans_params(estimator, mapping=None,
                              fit_index=None,
                              other_params=None,
                              copy_mapping=True):

    # Get fit params first
    fit_trans_params =\
        _get_est_fit_params(estimator,
                            mapping=mapping,
                            fit_index=fit_index,
                            other_params=other_params,
                            copy_mapping=copy_mapping)

    # Then update transform params,
    # note that since it is fit_transform, the transform index
    # is the same as the fit_index here
    fit_trans_params.update(
        _get_est_trans_params(estimator,
                              transform_index=fit_index))

    return fit_trans_params


def _fit_single_estimator(estimator, X, y, sample_weight=None,
                          mapping=None, fit_index=None,
                          message_clsname=None, message=None):
    """Private function used to fit an estimator within a job."""

    fit_params = _get_est_fit_params(estimator=estimator, mapping=mapping,
                                     fit_index=fit_index)

    if sample_weight is not None:
        try:
            estimator.fit(X, y, sample_weight=sample_weight, **fit_params)
        except TypeError as exc:
            if "unexpected keyword argument 'sample_weight'" in str(exc):
                raise TypeError(
                    "Underlying estimator {} does not support sample weights."
                    .format(estimator.__class__.__name__)
                ) from exc
            raise
    else:
        estimator.fit(X, y, **fit_params)

    return estimator


class BPtBase(BaseEstimator):

    _required_parameters = ["estimator"]

    def __init__(self, estimator):
        self.estimator = estimator

    @property
    def n_jobs(self):
        if hasattr(self.estimator, 'n_jobs'):
            return getattr(self.estimator, 'n_jobs')

    @n_jobs.setter
    def n_jobs(self, n_jobs):
        if hasattr(self.estimator, 'n_jobs'):
            setattr(self.estimator, 'n_jobs', n_jobs)

    def transform_df(self, X_df, base_name=None, encoders=None):
        '''Override this class in child classes'''
        pass
