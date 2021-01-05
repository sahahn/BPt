from ..Pipeline_Pieces import add_estimator_to_params


def test_add_estimator_to_params():

    params = {'s__1': 1, 'my_model__2': 2}
    returned = add_estimator_to_params(params)

    assert returned['s__estimator__1'] == 1
    assert returned['my_model__estimator__2'] == 2