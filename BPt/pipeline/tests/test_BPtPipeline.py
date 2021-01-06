from ..BPtPipeline import BPtPipeline
from .helpers import ToFixedTransformer
from ..ScopeObjs import ScopeTransformer, ScopeModel
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd


def test_BPtPipeline():

    # 'loaders', 'imputers',
    #  'scalers',
    #  'transformers',
    #  'feat_selectors', 'model']

    steps = []
    to_ones = ToFixedTransformer(to=1)
    st = ScopeTransformer(estimator=to_ones, inds=[1, 2])
    steps.append(('to_ones', st))

    model = ScopeModel(estimator=LinearRegression(), inds=[0, 1])
    steps.append(('model', model))

    names = [[], [], ['to_ones'], [], [], ['model']]
    pipe = BPtPipeline(steps=steps, names=names)

    X = np.zeros((3, 3))
    y = np.ones(3)

    pipe.fit(X, y)
    assert pipe['to_ones'].inds_ == [1, 2]

    # Should update so that next inds are 0, 2
    # as 1 -> 0, 2 -> 1, 0 -> 2, so 0,1 -> 2, 0, sorted = 0, 2
    assert pipe['model'].inds_ == [0, 2]
    assert len(pipe.mapping_) == 3
    assert pipe.mapping_[0] == 2
    assert pipe.mapping_[1] == 0
    assert pipe.mapping_[2] == 1

    # Make sure when re-fit resets mapping each time
    pipe.fit(X, y)
    assert pipe.mapping_[0] == 2
    assert pipe.mapping_[1] == 0
    assert pipe.mapping_[2] == 1

    # Test propegate n_jobs
    pipe.n_jobs = 2
    assert pipe['to_ones'].n_jobs == 2
    assert pipe['to_ones'].estimator.n_jobs == 2

    X_df = pd.DataFrame(X)
    X_trans = pipe.transform_df(X_df)
    assert X_trans[0].sum() == 0
    assert X_trans[1].sum() == 3
    assert X_trans[2].sum() == 3
