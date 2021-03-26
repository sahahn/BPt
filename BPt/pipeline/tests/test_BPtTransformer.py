from ..BPtTransformer import BPtTransformer
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from BPt import Dataset
import pandas as pd


def test_all_bpt_transformer():

    X = np.array([[1, 1, 1],
                  [2, 2, 2],
                  [3, 3, 3]])
    y = np.zeros(3)

    tr = BPtTransformer(OneHotEncoder(sparse=False), inds=Ellipsis)
    tr.fit(X, y)

    assert tr.inds_ is Ellipsis
    assert len(tr.rest_inds_) == 0

    X_trans = tr.transform(X)
    assert X_trans.shape == (3, 9)
    df = pd.DataFrame(X)

    X_trans_df = tr.transform_df(df)
    assert X_trans_df.shape == (3, 9)
    for i in range(3):
        for j in range(1, 4):
            assert str(i) + '=' + str(j) in X_trans_df


def test_all_bpt_transformer_drop():

    X = np.array([[1, 1, 1],
                  [2, 2, 2],
                  [3, 3, 3]])
    y = np.zeros(3)

    tr = BPtTransformer(OneHotEncoder(sparse=False, drop='first'), 
                        inds=Ellipsis)
    tr.fit(X, y)

    assert tr.inds_ is Ellipsis
    assert len(tr.rest_inds_) == 0

    X_trans = tr.transform(X)
    assert X_trans.shape == (3, 6)
    df = pd.DataFrame(X)

    X_trans_df = tr.transform_df(df)
    assert X_trans_df.shape == (3, 6)
    for i in range(3):
        for j in range(2, 4):
            assert str(i) + '=' + str(j) in X_trans_df


def test_all_bpt_transformer_dataset():

    data = Dataset()
    data['1'] = [1, 2, 3]
    data['2'] = [4, 5, 6]
    data = data.ordinalize('all')

    assert len(data.encoders) == 2

    tr = BPtTransformer(OneHotEncoder(sparse=False), inds=Ellipsis)
    tr.fit(np.array(data), np.ones(3))

    assert tr.inds_ is Ellipsis
    assert len(tr.rest_inds_) == 0

    X_trans = tr.transform(np.array(data))
    assert X_trans.shape == (3, 6)

    # First without encoders
    X_trans_df1 = tr.transform_df(data)
    assert X_trans_df1.shape == (3, 6)
    assert '1=0' in X_trans_df1
    assert '2=2' in X_trans_df1
    assert '1=3' not in X_trans_df1
    assert '2=5' not in X_trans_df1
    assert '2=6' not in X_trans_df1

    X_trans_df2 = tr.transform_df(data, encoders=data.encoders)
    X_trans_df2.shape == (3, 6)
    assert '1=0' not in X_trans_df2
    assert '2=2' not in X_trans_df2
    assert '1=1' in X_trans_df2
    assert '1=3' in X_trans_df2
    assert '2=5' in X_trans_df2
    assert '2=6' in X_trans_df2

    assert list(X_trans_df1) != list(X_trans_df2)
    assert np.array_equal(np.array(X_trans_df1), np.array(X_trans_df2))
