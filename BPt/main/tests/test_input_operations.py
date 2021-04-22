import pytest
from ..input_operations import Select
from ..input import Model, Scaler
from ...util import BPtInputMixIn


def test_select_fail():

    s = Select([Model('dt'), Scaler('robust')])

    with pytest.raises(RuntimeError):
        s._check_args()


def test_select_constructor():

    s = Select([Model('dt'), Model('ridge')])
    assert s._constructor == Model._constructor


def test_select_attributes():

    s = Select([Model('dt'), Model('ridge')])
    assert isinstance(s, list)
    assert isinstance(s, Select)
    assert isinstance(s, BPtInputMixIn)