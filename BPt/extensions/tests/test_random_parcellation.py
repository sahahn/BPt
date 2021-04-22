from ..random_parcellation import RandomParcellation
import numpy as np
import pytest
from ..loaders import load_surf


def test_base():

    fake_geo = [[1, 2],
                [2, 3],
                [3, 4],
                [4, 0],
                [0, 1]]

    parc = RandomParcellation(geo=fake_geo, n_parcels=2,
                              medial_wall_inds=None,
                              medial_wall_mask=None,
                              random_state=None)

    p = parc.get_parc()

    assert len(p) == 5
    assert len(np.unique(p) == 2)


def test_random_seed():

    fake_geo = [[1, 2],
                [2, 3],
                [3, 4],
                [4, 0],
                [0, 1]]

    p = RandomParcellation(geo=fake_geo, n_parcels=2,
                           random_state=1).get_parc()

    for _ in range(5):
        p2 = RandomParcellation(geo=fake_geo, n_parcels=2,
                                random_state=1).get_parc()

        assert np.array_equal(p, p2)


def test_mask_inds():

    fake_geo = [[1, 2],
                [2, 3],
                [3, 4],
                [4, 0],
                [0, 1]]

    p = RandomParcellation(geo=fake_geo, n_parcels=2,
                           medial_wall_inds=[0],
                           random_state=1).get_parc()

    assert p[0] == 0
    assert len(np.unique(p) == 3)


def test_mask():

    fake_geo = [[1, 2],
                [2, 3],
                [3, 4],
                [4, 0],
                [0, 1]]

    p = RandomParcellation(geo=fake_geo, n_parcels=2,
                           medial_wall_mask=[1, 0, 0, 0, 0],
                           random_state=1).get_parc()

    assert p[0] == 0
    assert len(np.unique(p) == 3)


def test_with_load_surf():

    fake_geo = [[1, 2],
                [2, 3],
                [3, 4],
                [4, 0],
                [0, 1]]

    p = RandomParcellation(geo=fake_geo, n_parcels=2,
                           medial_wall_mask=[1, 0, 0, 0, 0],
                           random_state=1)

    s = load_surf(p)
    assert s.shape == (5,)


def test_mask_fail():

    fake_geo = [[1, 2],
                [2, 3],
                [3, 4],
                [4, 0],
                [0, 1]]
    with pytest.raises(IndexError):
        RandomParcellation(geo=fake_geo, n_parcels=2,
                           medial_wall_inds=[0, 0, 0, 1, 0],
                           random_state=1).get_parc()
