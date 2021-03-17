from ..MVTransformer import MVTransformer
from ....dataset.Dataset import Dataset
import numpy as np


def basic_test():

    # Skip this test if mvlearn not installed
    try:
        from mvlearn.embed import CCA
    except ImportError:
        return

    data = Dataset()
    data['1'] = [1, 1, 1, 1, 1, 1]
    data['2'] = [2, 2, 2, 2, 2, 2]
    data['3'] = [3, 3, 3, 3, 3, 3]
    data['4'] = [4, 4, 4, 4, 4, 4]

    X = np.array([[0., 0., 1., 0.1, -0.2],
                  [1., 0., 0., 0.9, 1.1],
                  [2., 2., 2., 6.2, 5.9],
                  [3., 5., 4., 11.9, 12.3]])

    mv = MVTransformer(estimator=CCA(multiview_output=False),
                       inds=[[0, 1, 2], [3, 4]])

    X_trans = mv.fit_transform(X)

    # Since reduces 5 features to 1 for 4 subjects
    assert X_trans.shape == (4, 1)
