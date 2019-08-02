from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split


class EarlyStopLGBMRegressor(LGBMRegressor):

    def __init__(self, val_split_percent=.1, early_stop_rounds=50,
                 boosting_type='gbdt', num_leaves=31, max_depth=-1,
                 learning_rate=0.1, n_estimators=100, subsample_for_bin=200000,
                 objective=None, class_weight=None, min_split_gain=0.0,
                 min_child_weight=0.001, min_child_samples=20, subsample=1.0,
                 subsample_freq=0, colsample_bytree=1.0, reg_alpha=0.0,
                 reg_lambda=0.0, random_state=None, n_jobs=-1, silent=True,
                 importance_type='split', **kwargs):

        self.val_split_percent = val_split_percent
        self.early_stop_rounds = early_stop_rounds

        super().__init__(boosting_type, num_leaves, max_depth, learning_rate,
                         n_estimators, subsample_for_bin, objective,
                         class_weight, min_split_gain, min_child_weight,
                         min_child_samples, subsample, subsample_freq,
                         colsample_bytree, reg_alpha, reg_lambda, random_state,
                         n_jobs, silent, importance_type, **kwargs)

    def fit(self, X, y, sample_weight=None, init_score=None,
            eval_set=None, eval_names=None, eval_sample_weight=None,
            eval_init_score=None, eval_metric=None, early_stopping_rounds=None,
            verbose=False, feature_name='auto', categorical_feature='auto',
            callbacks=None):

        X_train, X_test, y_train, y_test =\
            train_test_split(X, y, test_size=self.val_split_percent,
                             random_state=self.random_state)

        super().fit(X_train, y_train, sample_weight=sample_weight,
                    init_score=init_score, eval_set=(X_test, y_test),
                    eval_names=eval_names,
                    eval_sample_weight=eval_sample_weight,
                    eval_metric=eval_metric,
                    early_stopping_rounds=self.early_stop_rounds,
                    verbose=verbose, feature_name=feature_name,
                    categorical_feature=categorical_feature,
                    callbacks=callbacks)
