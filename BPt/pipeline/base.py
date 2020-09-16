def _get_est_fit_params(estimator, mapping=None, train_data_index=None,
                        other_params=None):

    if other_params is None:
        fit_params = {}
    else:
        fit_params = other_params.copy()

    if mapping is not None:
        if hasattr(estimator, 'needs_mapping'):
            if estimator.needs_mapping:
                fit_params['mapping'] = mapping.copy()

    if train_data_index is not None:
        if hasattr(estimator, 'needs_train_data_index'):
            if estimator.needs_train_data_index:
                fit_params['train_data_index'] = train_data_index

    return fit_params


def _fit_single_estimator(estimator, X, y, sample_weight=None,
                          mapping=None, train_data_index=None,
                          message_clsname=None, message=None):
    """Private function used to fit an estimator within a job."""

    fit_params = _get_est_fit_params(estimator=estimator, mapping=mapping,
                                     train_data_index=train_data_index)

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
