from ..params.Params import TransitionChoice, Choice, Dict, Scalar, Array
import nevergrad as ng


def test_choice_repr_aligns():

    base = ng.p.Choice([None, 'balanced'])
    base.indices.value = [0]
    p = Choice([None, 'balanced'])

    assert repr(base._content) == repr(p._content)


def test_equiv():

    p1 = Choice([None, 'balanced'])
    p2 = Choice([None, 'balanced'])
    p3 = Choice([None, 'balanced'])

    assert repr(p1) == repr(p2)
    assert repr(p2) == repr(p3)


def test_transition_choice():

    cls_weight = TransitionChoice([None, 'balanced'],
                                  transitions=[1, 1])

    assert repr(cls_weight) == \
        "TransitionChoice([None, 'balanced'], transitions=[1, 1])"

    # Test nested case
    g = TransitionChoice([cls_weight, Dict(q=cls_weight)])
    assert hasattr(g.choices[0], 'name')
    assert hasattr(g.choices[1], 'name')
    assert hasattr(g.choices[1]['q'], 'name')

    r = "TransitionChoice([TransitionChoice([None, 'balanced'], " + \
        "transitions=[1, 1]), Dict(q=TransitionChoice([None, 'balanced']," + \
        " transitions=[1, 1]))])"
    assert repr(g) == r


def test_scalar():

    # Confirm base behavior
    base = ng.p.Scalar(lower=5, upper=10).set_integer_casting()
    assert repr(base).startswith('Scalar{Cl(')
    assert base.integer

    # Confirm cast works
    p = Scalar(lower=5, upper=10)
    p = p.set_integer_casting()
    assert p.integer

    # Make sure values work as expected
    assert base.value == p.value


def test_array():

    base = ng.p.Array(init=(100, 100)).set_mutation(sigma=50)
    assert base.sigma._value == 50

    p = Array(init=(100, 100)).set_mutation(sigma=50)
    assert p.sigma._value == 50

    base.set_bounds(lower=1, upper=300)
    assert base.bounds[0][0] == 1
    assert base.bounds[1][0] == 300

    p = Array(init=(100, 100)).set_mutation(sigma=50)
    p.set_bounds(lower=1, upper=300)

    assert p.bounds[0][0] == 1
    assert p.bounds[1][0] == 300

    repr_p = "Array(init=(100, 100)).set_mutation(sigma=50)." + \
        "set_bounds(lower=1, upper=300)"

    assert repr(p) == repr_p


def test_to_grid():

    params = {'1': Choice([1, 2, 3]),
              '2': TransitionChoice([1, 2]),
              '3': 3,
              '4': Scalar(lower=1, upper=2).set_integer_casting()}

    assert params['1'].to_grid() == [1, 2, 3]
    assert params['2'].to_grid() == [1, 2]
    assert params['4'].to_grid() == [1, 2]

    assert hasattr(params['1'], 'to_grid')
    assert hasattr(params['2'], 'to_grid')


def test_dict_to_grid():

    params = {'select': Dict(x=1),
              's2': Dict(x=Choice([1, 2, 3]))}
    assert params['select'].to_grid() == {'x': 1}

    grid = params['s2'].to_grid()
    assert grid['x'] == [1, 2, 3]
