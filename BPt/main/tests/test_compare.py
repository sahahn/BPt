from ..compare import Option, Compare, Options, CompareDict


def test_option1():

    o1 = Option(value='10', name='10', key='something')
    o2 = Option(value='20', name='10', key='something')
    assert o1 == o2

    assert not o1 > o2
    assert not o1 < o2
