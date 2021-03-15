from ..ensemble_wrappers import BPtStackingRegressor
from ..BPtModel import BPtModel
from sklearn.tree import DecisionTreeRegressor
import numpy as np


def test_stacking_regressor():

    estimators = [('1', BPtModel(DecisionTreeRegressor(), inds=[1, 2, 3])),
                  ('2', BPtModel(DecisionTreeRegressor(), inds=[0, 1]))]

    model = BPtStackingRegressor(estimators)

    X = np.ones((20, 10))
    y = np.zeros((20))
    mapping = {i: i for i in range(10)}

    model.fit(X, y, mapping=mapping)

    m1 = model.named_estimators_['1']
    m2 = model.named_estimators_['2']

    assert isinstance(m1, BPtModel)
    assert isinstance(m2, BPtModel)

    print(m1.mapping_)
    print(m2.out_mapping_)

    assert False
