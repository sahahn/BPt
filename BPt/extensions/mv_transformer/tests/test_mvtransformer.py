from ..MVTransformer import MVTransformer
from ....dataset.Dataset import Dataset
import numpy as np


def basic_test():

    # Skip this test if mv learn not installed
    try:
        from mvlearn.embed import CCA
    except ImportError:
        return

    X = np.array([[0., 0., 1., 0.1, -0.2],
                  [1., 0., 0., 0.9, 1.1],
                  [2., 2., 2., 6.2, 5.9],
                  [3., 5., 4., 11.9, 12.3]])

    mv = MVTransformer(estimator=CCA(multiview_output=False),
                       inds=[[0, 1, 2], [3, 4]])

    X_trans = mv.fit_transform(X)

    # Basic Checks
    assert X_trans.shape == (4, 1)
    assert mv.inds_ == [0, 1, 2, 3, 4]
    assert mv.view_inds_ == [[0, 1, 2], [3, 4]]
    assert mv.out_mapping_[0] == [0]
    assert mv.out_mapping_[4] == [0]
    assert len(mv.out_mapping_) == 5
    assert isinstance(mv.estimator_, CCA)
    assert mv.n_trans_feats_ == 1
    assert mv.n_features_in_ == 5


def test_constructor():

    data = Dataset()
    data['1'] = [0, 0, 1, .1, -1, 1]
    data['2'] = [0, 1, 0, .2, -2, 2.2]
    data['3'] = [2, 2, 2, 0, 2, 0]
    data['4'] = [0, 1, 0, 4.3, 4.3, 10]
