import pandas as pd
from ..residualizer import LinearResidualizer
from sklearn.preprocessing import OneHotEncoder
import numpy as np


def test_base():

    covars = pd.DataFrame()

    covars['1'] = [1, 1, 1]
    covars['2'] = [1, 2, 3]
    covars['2'] = covars['2'].astype('category')

    model = LinearResidualizer(to_resid_df=covars,
                               demean=True,
                               dummy_code=True)

    X = np.ones((3, 5))
    model.fit(X, fit_index=[0, 1, 2])

    assert len(model.means_) == 3

    # Mean of all 1's col should be 1
    assert model.means_[0] == 1
    assert isinstance(model.encoder_, OneHotEncoder)
    assert len(model.estimators_) == 5

    Xt = model.transform(X, transform_index=[0, 1, 2])
    assert Xt.shape == (3, 5)