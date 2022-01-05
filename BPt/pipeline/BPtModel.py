from .ScopeObjs import ScopeObj
from sklearn.utils.metaestimators import if_delegate_has_method
from .base import _get_est_trans_params


class BPtModel(ScopeObj):

    _needs_transform_nested_model = True

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

    @if_delegate_has_method(delegate='estimator_')
    def transform_feat_names(self, X_df, encoders=None, nested_model=False):
        return self.estimator_.transform_feat_names(X_df=X_df,
                                                    encoders=encoders,
                                                    nested_model=nested_model)

    @if_delegate_has_method(delegate='estimator_')
    def inverse_transform_fis(self, fis):
        return self.estimator_.inverse_transform_fis(fis)

    @if_delegate_has_method(delegate='estimator_')
    def transform(self, X, transform_index=None, nested_model=False):

        trans_params = _get_est_trans_params(self.estimator_,
                                             transform_index=transform_index,
                                             nested_model=nested_model)
        return self.estimator_.transform(X, **trans_params)

    @property
    def _final_estimator(self):
        if hasattr(self.estimator_, '_final_estimator') and \
          getattr(self.estimator_, '_final_estimator') is not None:
            return getattr(self.estimator_, '_final_estimator')

        # Otherwise, return None
        return None
        
     
