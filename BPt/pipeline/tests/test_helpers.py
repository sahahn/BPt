from ..helpers import (to_memmap, from_memmap, get_grid_params,
                       is_ng, update_mapping)
import numpy as np
import os
from ...default.params.Params import Choice, TransitionChoice, Scalar
import pytest
import nevergrad as ng


def test_to_memmap():

    X = np.ones((10, 10), dtype='float32')
    f, dtype, shape = to_memmap(X)

    assert os.path.exists(f)
    assert dtype == 'float32'
    assert shape == X.shape

    os.unlink(f)
    assert not os.path.exists(f)


def test_from_memmap():

    X = np.ones((10, 10), dtype='float32')
    X_m = to_memmap(X)
    X_c = from_memmap(X_m)

    assert X_c.shape == X.shape
    assert X.dtype == X_c.dtype

    X_c[:] = 0
    X_c = from_memmap(X_m)
    assert np.all(X == 1)


def test_get_grid_params():

    params = {'1': Choice([1, 2, 3]),
              '2': TransitionChoice([1, 2]),
              '3': 3,
              '4': Scalar(lower=1, upper=2).set_integer_casting()}

    grid_params = get_grid_params(params)

    assert grid_params['1'] == [1, 2, 3]
    assert grid_params['2'] == [1, 2]
    assert grid_params['3'] == 3
    assert grid_params['4'] == [1, 2]


def test_get_grid_params_fail():

    with pytest.raises(RuntimeError):

        params = {'1': ng.p.Scalar(lower=1, upper=2)}
        get_grid_params(params)


def test_is_ng():

    assert not is_ng(6)
    assert not is_ng([1, 2, 3])
    assert not is_ng(set([5]))
    assert not is_ng(from_memmap)
    assert not is_ng(np.ones(10))
    assert is_ng(ng.p.Choice([1, 2]))
    assert is_ng(ng.p.TransitionChoice([1, 2]))
    assert is_ng(ng.p.Scalar())
    assert is_ng(ng.p.Log(init=1))


def test_update_mapping_base():

    mapping = {1: 1}
    new_mapping = {}

    update_mapping(mapping, new_mapping)
    assert mapping[1] == 1
    assert len(mapping) == 1


def test_update_mapping_simple():

    mapping = {0: 0, 1: 1}
    new_mapping = {0: 1, 1: 0}
    update_mapping(mapping, new_mapping)

    assert len(mapping) == 2
    assert mapping[0] == 1
    assert mapping[1] == 0


def test_update_mapping_lists():

    mapping = {0: [0, 1], 1: [2, 3]}
    new_mapping = {1: [11], 2: [22]}
    update_mapping(mapping, new_mapping)

    # Sorts
    assert mapping[0] == [0, 11]
    assert mapping[1] == [3, 22]
    assert len(mapping) == 2


def test_update_mapping_none():

    mapping = {0: None}
    new_mapping = {0: 1}
    update_mapping(mapping, new_mapping)

    assert mapping[0] is None
    assert len(mapping) == 1
