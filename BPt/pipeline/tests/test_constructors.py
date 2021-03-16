from BPt.pipeline.ensemble_wrappers import BPtVotingRegressor
from ..BPtModel import BPtModel
from ..constructors import (add_estimator_to_params, FeatSelectorConstructor,
                            ScalerConstructor, ModelConstructor)
from ...main.input import Ensemble, FeatSelector, Model, Scaler
from ..BPtFeatureSelector import BPtFeatureSelector
from .helpers import get_fake_data_dataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import AdaBoostRegressor


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


def test_model():

    data_keys = ['0', '1']
    dataset = get_fake_data_dataset(data_keys=data_keys)

    spec = {'problem_type': 'regression', 'random_state': None,
            'n_jobs': 1, 'scope': 'all'}

    mc = ModelConstructor(dataset=dataset, spec=spec, user_passed_objs={})
    in_params = Model('random forest')

    objs, params = mc.process(in_params)
    _, obj = objs[0]
    assert isinstance(obj, BPtModel)
    assert isinstance(obj.estimator, RandomForestRegressor)

    assert obj.inds is Ellipsis
    assert len(params) == 0


def test_model_target_scaler():

    data_keys = ['0', '1']
    dataset = get_fake_data_dataset(data_keys=data_keys)

    spec = {'problem_type': 'regression', 'random_state': None,
            'n_jobs': 1, 'scope': 'all'}

    mc = ModelConstructor(dataset=dataset, spec=spec, user_passed_objs={})
    in_params = Model('random forest', scope='float',
                      target_scaler=Scaler('standard'))

    objs, params = mc.process(in_params)
    _, obj = objs[0]

    assert isinstance(obj, BPtModel)
    assert isinstance(obj.estimator, TransformedTargetRegressor)
    assert isinstance(obj.estimator.regressor, RandomForestRegressor)
    assert obj.inds is Ellipsis

    assert len(params) == 0


def test_ensemble_stacking():

    data_keys = ['0', '1']
    dataset = get_fake_data_dataset(data_keys=data_keys)

    spec = {'problem_type': 'regression', 'random_state': None,
            'n_jobs': 2, 'scope': 'all'}

    mc = ModelConstructor(dataset=dataset, spec=spec, user_passed_objs={})
    in_params = Ensemble('voting', [Model('random forest'),
                                    Model('random forest')],
                         n_jobs_type='ensemble')

    objs, params = mc.process(in_params)
    _, obj = objs[0]

    assert isinstance(obj, BPtModel)
    assert obj.inds is Ellipsis
    assert isinstance(obj.estimator, BPtVotingRegressor)
    assert obj.estimator.n_jobs == 2

    ests = obj.estimator.estimators
    for est in ests:
        assert isinstance(est[1], BPtModel)
        assert est[1].estimator.n_jobs == 1

    assert len(params) == 0


def test_ensemble_nested_stacking():

    data_keys = ['0', '1']
    dataset = get_fake_data_dataset(data_keys=data_keys)

    spec = {'problem_type': 'regression', 'random_state': None,
            'n_jobs': 2, 'scope': 'all'}

    mc = ModelConstructor(dataset=dataset, spec=spec, user_passed_objs={})
    in_params = Ensemble('voting', [Model('random forest'),
                                    Model('random forest'),
                                    Ensemble('voting', [Model('random forest'),
                                                        Model('random forest')])
                                    ],
                         n_jobs_type='ensemble')

    objs, params = mc.process(in_params)
    _, obj = objs[0]

    assert isinstance(obj, BPtModel)
    assert obj.inds is Ellipsis
    assert isinstance(obj.estimator, BPtVotingRegressor)
    assert obj.estimator.n_jobs == 2

    estimators = obj.estimator.estimators
    assert isinstance(estimators[1][1], BPtModel)
    assert isinstance(estimators[2][1], BPtModel)

    nested_ensemble = estimators[0][1].estimator
    assert isinstance(nested_ensemble, BPtVotingRegressor)
    assert nested_ensemble.n_jobs == 1

    nested_ests = nested_ensemble.estimators
    assert isinstance(nested_ests[0][1], BPtModel)
    assert isinstance(nested_ests[0][1].estimator, RandomForestRegressor)
    assert isinstance(nested_ests[1][1], BPtModel)
    assert isinstance(nested_ests[1][1].estimator, RandomForestRegressor)

    assert len(params) == 0


def test_ensemble_se():

    data_keys = ['0', '1']
    dataset = get_fake_data_dataset(data_keys=data_keys)

    spec = {'problem_type': 'regression', 'random_state': None,
            'n_jobs': 2, 'scope': 'all'}

    mc = ModelConstructor(dataset=dataset, spec=spec, user_passed_objs={})
    in_params = Ensemble(obj='adaboost',
                         models=Model('random forest'),
                         single_estimator=True)

    objs, params = mc.process(in_params)
    _, obj = objs[0]

    assert isinstance(obj, BPtModel)
    assert isinstance(obj.estimator, AdaBoostRegressor)
    assert len(params) == 0
