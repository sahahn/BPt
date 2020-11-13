from sklearn.utils.metaestimators import _BaseComposition
from sklearn.utils.metaestimators import if_delegate_has_method
import nevergrad as ng


class Selector(_BaseComposition):

    def __init__(self, estimators, to_use=0):
        self.estimators = estimators
        self.to_use = to_use
        self.example_estimator_ = self.estimators[0][1]

    def get_params(self, deep=True):
        return self._get_params('estimators', deep=deep)

    def set_params(self, **kwargs):

        # Pass params as dict with key select
        select = kwargs['select']

        # Get to use from select
        self.to_use = select['to_use']

        # Set rest of select params
        self._set_params('estimators', **select)
        return self

    @if_delegate_has_method(delegate='example_estimator_')
    def fit(self, *args, **kwargs):
        self.is_fitted_ = True
        self.estimator_ = self.estimators[self.to_use][1]
        self.estimator_.fit(*args, **kwargs)
        return self

    @if_delegate_has_method(delegate='example_estimator_')
    def fit_transform(self, *args, **kwargs):
        self.estimator_ = self.estimators[self.to_use][1]
        return self.estimator_.fit_transform(*args, **kwargs)

    @if_delegate_has_method(delegate='example_estimator_')
    def transform(self, *args, **kwargs):
        return self.estimator_.transform(*args, **kwargs)

    @if_delegate_has_method(delegate='example_estimator_')
    def fit_resample(self, *args, **kwargs):
        self.estimator_ = self.estimators[self.to_use][1]
        return self.estimator_.fit_resample(*args, **kwargs)

    @if_delegate_has_method(delegate='example_estimator_')
    def fit_predict(self, *args, **kwargs):
        self.estimator_ = self.estimators[self.to_use][1]
        return self.estimator_.fit_predict(*args, **kwargs)

    @if_delegate_has_method(delegate='estimator_')
    def predict(self, *args, **kwargs):
        return self.estimator_.predict(*args, **kwargs)

    @if_delegate_has_method(delegate='estimator_')
    def predict_proba(self, *args, **kwargs):
        return self.estimator_.predict_proba(*args, **kwargs)

    @if_delegate_has_method(delegate='estimator_')
    def decision_function(self, *args, **kwargs):
        return self.estimator_.decision_function(*args, **kwargs)

    @if_delegate_has_method(delegate='estimator_')
    def predict_log_proba(self, *args, **kwargs):
        return self.estimator_.predict_log_proba(*args, **kwargs)

    @if_delegate_has_method(delegate='estimator_')
    def score(self, *args, **kwargs):
        return self.estimator_.score(*args, **kwargs)

    @if_delegate_has_method(delegate='estimator_')
    def inverse_transform(self, *args, **kwargs):
        return self.estimator_.inverse_transform(*args, **kwargs)

    @property
    def _estimator_type(self):
        return self.example_estimator_._estimator_type


def selector_wrapper(objs, params, name):

    selector = (name, Selector(objs))

    p_dicts = []
    for i in range(len(objs)):
        obj_name = objs[i][0]
        rel_params =\
            {p: params[p] for p in params if p.split('__')[0] == obj_name}
        rel_params['to_use'] = i

        p_dict = ng.p.Dict(**rel_params)
        p_dicts.append(p_dict)

    select = ng.p.Choice(p_dicts, deterministic=True)
    select_params = {name + '__select': select}

    return selector, select_params
