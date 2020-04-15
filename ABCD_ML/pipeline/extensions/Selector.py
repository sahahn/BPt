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
        self._set_params('estimators', **kwargs)
        return self
    
    @if_delegate_has_method(delegate='example_estimator_')
    def fit(self, *args, **kwargs):
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


def selector_wrapper(objs, params, name):
    
    selector = (name, Selector(objs))
    
    new_params = {}
    for key in params:
        new_params[name + '__' + key] = params[key]
            
    params = new_params
    params[name + '__to_use'] = ng.p.Choice([i for i in range(len(objs))])
    
    return selector, params