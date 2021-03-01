from ..loaders import SurfMaps, SurfLabels
import numpy as np


def test_surf_maps():

    sl = SurfLabels(labels=np.array([1, 1, 2, 2]))
    data = np.array([5, 5, 10, 10])

    Xt = sl.fit_transform(data)
    assert np.array_equal(Xt, np.array([5, 10]))
