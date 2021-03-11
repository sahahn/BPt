from ..constructors import (add_estimator_to_params, FeatSelectorConstructor,
                            ScalerConstructor)
from ...main.input import FeatSelector, Model, Scaler
from ..BPtFeatureSelector import BPtFeatureSelector
from .helpers import get_fake_data_dataset


def test_add_estimator_to_params():

    params = {'s__1': 1, 'my_model__2': 2}
    returned = add_estimator_to_params(params)

    assert returned['s__estimator__1'] == 1
    assert returned['my_model__estimator__2'] == 2


def test_feature_selectors():

    data_keys = ['0', '1']
    dataset = get_fake_data_dataset(data_keys=data_keys)

    spec = {'problem_type': 'binary',
            'random_state': None,
            'n_jobs': 1,
            'scope': 'all'}

    fs = FeatSelectorConstructor(dataset=dataset, spec=spec,
                                 user_passed_objs={})

    in_params = FeatSelector('univariate selection')
    objs, params = fs.process(in_params)
    name, obj = objs[0]

    assert 'univariate selection' in name
    assert obj.inds is Ellipsis
    assert len(params) == 0
    assert isinstance(obj, BPtFeatureSelector)

    in_params = FeatSelector('univariate selection', params=1)
    objs, params = fs.process(in_params)
    name, obj = objs[0]

    assert 'univariate selection' in name
    assert obj.inds is Ellipsis
    assert len(params) > 0
    assert isinstance(obj, BPtFeatureSelector)

    for key in params:
        assert 'univariate selection' in key

    in_params = FeatSelector('univariate selection', scope='category')
    objs, params = fs.process(in_params)
    name, obj = objs[0]

    assert 'univariate selection' in name
    assert len(obj.inds) == 0
    assert len(params) == 0
    assert isinstance(obj, BPtFeatureSelector)

    dataset = get_fake_data_dataset(data_keys=data_keys, cat_keys=['1'])
    fs = FeatSelectorConstructor(dataset=dataset, spec=spec,
                                 user_passed_objs={})

    in_params = FeatSelector('univariate selection', scope='category')
    objs, params = fs.process(in_params)
    name, obj = objs[0]

    assert 'univariate selection' in name
    assert obj.inds == [1]
    assert len(params) == 0
    assert isinstance(obj, BPtFeatureSelector)


def test_feature_selectors_submodel():

    data_keys = ['0', '1']
    dataset = get_fake_data_dataset(data_keys=data_keys)
    spec = {'problem_type': 'binary', 'random_state': None,
            'n_jobs': 1, 'scope': 'all'}

    fs = FeatSelectorConstructor(dataset=dataset, spec=spec,
                                 user_passed_objs={})

    in_params = FeatSelector('rfe')
    objs, params = fs.process(in_params)
    name, obj = objs[0]

    assert 'rfe' in name
    assert len(params) == 0
    assert obj.inds is Ellipsis
    assert obj.estimator.estimator is None
    assert isinstance(obj, BPtFeatureSelector)

    in_params = FeatSelector(obj='rfe',
                             base_model=Model(obj='ridge', params=1))
    objs, params = fs.process(in_params)
    name, obj = objs[0]

    assert 'rfe' in name
    assert obj.inds is Ellipsis
    assert obj.estimator.estimator is not None
    assert len(params) > 1
    assert isinstance(obj, BPtFeatureSelector)

    # Each param should start with name estimator, estimator
    for key in params:
        assert key.startswith('__'.join([name, 'estimator', 'estimator']))


def test_empty_scope():

    data_keys = ['0', '1']
    dataset = get_fake_data_dataset(data_keys=data_keys)
    spec = {'problem_type': 'binary', 'random_state': None,
            'n_jobs': 1, 'scope': 'all'}

    sc = ScalerConstructor(dataset=dataset, spec=spec,
                           user_passed_objs={})

    in_params = Scaler('robust', scope='category')
    objs, params = sc.process(in_params)
    _, obj = objs[0]

    assert len(obj.inds) == 0
    assert len(params) == 0
