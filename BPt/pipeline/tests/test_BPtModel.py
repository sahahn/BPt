from ..BPtModel import BPtModel
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import numpy as np


def model_test_classes_none():

    model = BPtModel(estimator=DecisionTreeRegressor(), inds=Ellipsis)

    X = np.ones((20, 20))
    y = np.ones((20))

    model.fit(X, y)

    assert model.classes_ is None

    score = model.score(X, y)
    assert score == 1


def model_test_classes_some():

    model = BPtModel(estimator=DecisionTreeClassifier(), inds=Ellipsis)

    X = np.ones((20, 20))
    y = np.ones((20))

    model.fit(X, y)

    assert model.classes_ is not None
