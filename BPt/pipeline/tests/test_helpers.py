from ..helpers import to_memmap, from_memmap, get_grid_params, is_ng
import numpy as np
import os
import nevergrad as ng
from nose.tools import raises


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

    params = {'1': ng.p.Choice([1, 2, 3]),
              '2': ng.p.TransitionChoice([1, 2]),
              '3': 3,
              '4': ng.p.Scalar(lower=1, upper=2).set_integer_casting()}

    grid_params = get_grid_params(params)

    assert grid_params['1'] == [1, 2, 3]
    assert grid_params['2'] == [1, 2]
    assert grid_params['3'] == 3
    assert grid_params['4'] == [1, 2]


@raises(RuntimeError)
def test_get_grid_params_fail():

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
