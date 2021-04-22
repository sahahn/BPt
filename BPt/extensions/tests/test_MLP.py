from ..MLP import MLPClassifier_Wrapper, MLPRegressor_Wrapper
import numpy as np


def test_classifier():

    est = MLPClassifier_Wrapper(hidden_layer_sizes=[3.0, 3.0], tol=1)
    est.fit(np.ones((20, 20)), np.ones(20))


def test_regressor():

    est = MLPRegressor_Wrapper(hidden_layer_sizes=[3.0, 3.0], tol=1)
    est.fit(np.ones((20, 20)), np.ones(20))
