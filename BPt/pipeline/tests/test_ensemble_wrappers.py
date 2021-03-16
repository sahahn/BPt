from ..ensemble_wrappers import (BPtStackingClassifier, BPtStackingRegressor,
                                 BPtVotingRegressor, BPtVotingClassifier)
from ..BPtModel import BPtModel
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import numpy as np


def test_stacking_regressor():

    estimators = [('1', BPtModel(DecisionTreeRegressor(), inds=[1, 2, 3])),
                  ('2', BPtModel(DecisionTreeRegressor(), inds=[0, 1]))]

    model = BPtStackingRegressor(estimators)
    model2 = BPtVotingRegressor(estimators)

    X = np.ones((20, 10))
    y = np.zeros((20))
    mapping = {i: i for i in range(10)}

    model.fit(X, y, mapping=mapping)
    model2.fit(X, y, mapping=mapping)

    m1 = model.named_estimators_['1']
    m2 = model.named_estimators_['2']

    assert isinstance(m1, BPtModel)
    assert isinstance(m2, BPtModel)

    m1_2 = model.named_estimators_['1']
    m2_2 = model.named_estimators_['2']

    assert isinstance(m1_2, BPtModel)
    assert isinstance(m2_2, BPtModel)

    assert m1.out_mapping_[3] is not None
    assert m2.out_mapping_[3] is None

    assert m1_2.out_mapping_[3] is not None
    assert m2_2.out_mapping_[3] is None


def test_stacking_classifier():

    estimators = [('1', BPtModel(DecisionTreeClassifier(), inds=[1, 2, 3])),
                  ('2', BPtModel(DecisionTreeClassifier(), inds=[0, 1]))]

    model = BPtStackingClassifier(estimators)
    model2 = BPtVotingClassifier(estimators)

    X = np.ones((20, 10))
    y = np.zeros((20))
    y[:10] = 1
    mapping = {i: i for i in range(10)}

    model.fit(X, y, mapping=mapping)
    model2.fit(X, y, mapping=mapping)

    m1 = model.named_estimators_['1']
    m2 = model.named_estimators_['2']

    assert isinstance(m1, BPtModel)
    assert isinstance(m2, BPtModel)

    m1_2 = model.named_estimators_['1']
    m2_2 = model.named_estimators_['2']

    assert isinstance(m1_2, BPtModel)
    assert isinstance(m2_2, BPtModel)

    assert m1.out_mapping_[3] is not None
    assert m2.out_mapping_[3] is None

    assert m1_2.out_mapping_[3] is not None
    assert m2_2.out_mapping_[3] is None
