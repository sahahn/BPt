from ..BPtLGBM import BPtLGBMClassifier, BPtLGBMRegressor
from ...main.CV import BPtCV, CVStrategy
import numpy as np


def test_basic():

    X = np.ones((20, 20))
    y = np.ones((20))
    y[:10] = 0

    # Just shouldn't fail
    regr = BPtLGBMRegressor()
    regr.fit(X, y)
    regr.predict(X)

    clasif = BPtLGBMClassifier()
    clasif.fit(X, y)
    clasif.predict(X)


def test_with_bpt_cv():

    cv = BPtCV(splits=.5, n_repeats=1, cv_strategy=CVStrategy(),
               splits_vals=None, random_state=1)

    X = np.ones((20, 20))
    y = np.ones((20))

    # Make sure fit works w/ custom CV
    regr = BPtLGBMRegressor(eval_split=cv, early_stopping_rounds=10)
    regr.fit(X, y, fit_index=np.arange(20))

    X_eval, y_eval, eval_set =\
        regr._get_eval_set(X, y, fit_index=np.arange(20))

    assert X_eval.shape == (10, 20)
    assert y_eval.shape == (10, )
    assert eval_set[0].shape == (10, 20)
    assert eval_set[1].shape == (10, )

    # Regressor w/o early stop rounds
    regr = BPtLGBMRegressor(eval_split=cv, early_stopping_rounds=None)
    X_eval, y_eval, eval_set =\
        regr._get_eval_set(X, y, fit_index=np.arange(20))

    assert X_eval.shape == (20, 20)
    assert eval_set is None


def test_cv_as_size():

    X = np.ones((20, 20))
    y = np.ones((20))

    # Check works with cv as size
    regr = BPtLGBMRegressor(eval_split=.5, early_stopping_rounds=10)
    regr.fit(X, y, fit_index=np.arange(20))

    X_eval, y_eval, eval_set =\
        regr._get_eval_set(X, y, fit_index=np.arange(20))

    assert X_eval.shape == (10, 20)
    assert y_eval.shape == (10, )
    assert eval_set[0].shape == (10, 20)
    assert eval_set[1].shape == (10, )


def test_with_cat_features():

    X = np.ones((5, 5))
    y = np.ones((5))
    base_mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}

    regr = BPtLGBMRegressor(cat_inds=[0, 1])
    regr.fit(X, y)

    categorical_feature = regr._get_categorical_feature(base_mapping)
    assert categorical_feature == [0, 1]

    mapping = {0: [0, 1, 2], 1: [1, 2, None], 2: None, 3: 3, 4: 4, 5: [1, 2]}
    categorical_feature = regr._get_categorical_feature(mapping)
    assert categorical_feature == [0, 1, 2]
