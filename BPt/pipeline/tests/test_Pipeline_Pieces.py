from ..Pipeline_Pieces import add_estimator_to_params, Feat_Selectors
from ...main.Params_Classes import Feat_Selector
from .helpers import get_fake_data_scopes


def test_add_estimator_to_params():

    params = {'s__1': 1, 'my_model__2': 2}
    returned = add_estimator_to_params(params)

    assert returned['s__estimator__1'] == 1
    assert returned['my_model__estimator__2'] == 2


def test_feature_selectors():

    data_keys = [0, 1]
    data_scopes = get_fake_data_scopes(data_keys=data_keys)

    spec = {'problem_type': 'binary',
            'random_state': None,
            'n_jobs': 1}

    fs = Feat_Selectors(Data_Scopes=data_scopes, spec=spec,
                        user_passed_objs={})

    in_params = Feat_Selector('univariate selection')
    objs, params = fs.process(in_params)
    name, obj = objs[0]

    assert 'univariate selection' in name
    assert obj.inds == data_keys
    assert len(params) == 0

    in_params = Feat_Selector('univariate selection', params=1)
    objs, params = fs.process(in_params)
    name, obj = objs[0]

    assert 'univariate selection' in name
    assert obj.inds == data_keys
    assert len(params) > 0

    for key in params:
        assert 'univariate selection' in key

    in_params = Feat_Selector('univariate selection', scope='cat')
    objs, params = fs.process(in_params)
    name, obj = objs[0]

    assert 'univariate selection' in name
    assert len(obj.inds) == 0
    assert len(params) == 0

    data_scopes = get_fake_data_scopes(data_keys=data_keys, cat_keys=[1])
    fs = Feat_Selectors(Data_Scopes=data_scopes, spec=spec,
                        user_passed_objs={})

    in_params = Feat_Selector('univariate selection', scope='cat')
    objs, params = fs.process(in_params)
    name, obj = objs[0]

    assert 'univariate selection' in name
    assert obj.inds == [1]
    assert len(params) == 0
