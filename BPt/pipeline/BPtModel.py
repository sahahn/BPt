from .ScopeObjs import ScopeObj
from sklearn.utils.metaestimators import if_delegate_has_method


class BPtModel(ScopeObj):

    @property
    def _estimator_type(self):
        return self.estimator._estimator_type

    @property
    def feature_importances_(self):
        if hasattr(self.estimator_, 'feature_importances_'):
            return getattr(self.estimator_, 'feature_importances_')
        return None

    @property
    def coef_(self):
        if hasattr(self.estimator_, 'coef_'):
            return getattr(self.estimator_, 'coef_')
        return None

    @property
    def classes_(self):
        if hasattr(self.estimator_, 'classes_'):
            return getattr(self.estimator_, 'classes_')
        return None

    # Every estimator should have at least predict
    def predict(self, X, *args, **kwargs):
        return self.estimator_.predict(X[:, self.inds_], *args, **kwargs)

    @if_delegate_has_method(delegate='estimator_')
    def predict_proba(self, X, *args, **kwargs):
        return self.estimator_.predict_proba(X[:, self.inds_], *args, **kwargs)

    @if_delegate_has_method(delegate='estimator_')
    def decision_function(self, X, *args, **kwargs):
        return self.estimator_.decision_function(X[:, self.inds_],
                                                 *args, **kwargs)

    @if_delegate_has_method(delegate='estimator_')
    def predict_log_proba(self, X, *args, **kwargs):
        return self.estimator_.predict_log_proba(X[:, self.inds_],
                                                 *args, **kwargs)

    @if_delegate_has_method(delegate='estimator_')
    def score(self, X, *args, **kwargs):
        return self.estimator_.score(X[:, self.inds_], *args, **kwargs)
