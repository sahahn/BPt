from ...pipeline.BPtPipeline import BPtPipeline
from ...pipeline.BPtSearchCV import NevergradSearchCV
from ...pipeline.ScopeObjs import ScopeTransformer
from ...pipeline.BPtModel import BPtModel
from ..input import (Model, ModelPipeline, Pipeline, CV, Scaler,
                     ProblemSpec, ParamSearch, Imputer, Transformer)
from ..funcs import (pipeline_check, problem_spec_check, get_estimator,
                     _preproc_pipeline, _preproc_param_search)
from ..CV import BPtCV
from ...dataset.Dataset import Dataset
import pytest
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import RobustScaler
import numpy as np
from ..compare import CompareDict, Options, Compare, Option


def test_no_overlap_param_names():

    ps_params = set(ProblemSpec._get_param_names())
    pipe_params = set(ModelPipeline._get_param_names())
    assert len(ps_params.intersection(pipe_params)) == 0


def test_pipeline_check():

    mp_params = ModelPipeline(imputers=None,
                              model=Model('ridge'))
    mp = pipeline_check(mp_params)
    assert isinstance(mp, ModelPipeline)

    mp_params = ModelPipeline(imputers=None,
                              model='ridge')
    mp = pipeline_check(mp_params)
    assert isinstance(mp, ModelPipeline)

    mp_params = Model('ridge')
    mp = pipeline_check(mp_params)
    assert isinstance(mp, Pipeline)

    mp_params = ModelPipeline('ridge')
    mp = pipeline_check(mp_params)
    assert isinstance(mp, Pipeline)


def test_get_default_pipeline_str():

    mp = pipeline_check('elastic_pipe')
    assert isinstance(mp, Pipeline)


def test_pipeline_check_compare():

    mp_params = Compare([Model('ridge'), Model('elastic')])
    mp = pipeline_check(mp_params, error_if_compare=False)

    assert isinstance(mp, CompareDict)
    assert len(mp) == 2

    mp_params = Compare([Option(Model('ridge'), 'ridge'),
                         Option(Model('elastic'), 'elastic')])
    mp = pipeline_check(mp_params, error_if_compare=False)

    assert isinstance(mp, CompareDict)
    assert len(mp) == 2

    pipe = mp["pipeline=ridge"]
    assert isinstance(pipe, Pipeline)

    pipe = mp["pipeline=elastic"]
    assert isinstance(pipe, Pipeline)


def test_pipeline_check_extra_args():

    mp_params = ModelPipeline(imputers=None,
                              model=Model('ridge'))
    mp = pipeline_check(mp_params)
    assert isinstance(mp, ModelPipeline)
    assert mp.imputers is None

    mp = pipeline_check(mp_params, imputers=Imputer('mean'),
                        ignore='ignore')
    assert mp.imputers is not None
    assert isinstance(mp, ModelPipeline)


def get_fake_dataset():

    fake = Dataset()

    fake['1'] = [1, 2, 3]
    fake['2'] = [4, 5, 6]
    fake['3'] = [7, 8, 9]
    fake = fake.set_role('3', 'target')
    fake._check_sr()

    return fake


def get_test_ps():

    return ProblemSpec(problem_type='default',
                       target='3',
                       scorer='default',
                       scope='all',
                       subjects='all',
                       n_jobs=2,
                       random_state=1)


def get_checked_ps():

    dataset = get_fake_dataset()

    p = get_test_ps()
    ps = problem_spec_check(p, dataset)

    return ps


def test_problem_spec_compare_fail():

    dataset = get_fake_dataset()
    p = ProblemSpec(scope=Compare(['all', 'float']))

    with pytest.raises(RuntimeError):
        problem_spec_check(p, dataset, error_if_compare=True)

    problem_spec_check(p, dataset, error_if_compare=False)


def test_problem_spec_compare():

    dataset = get_fake_dataset()
    dataset._check_sr()

    # Default names
    p = ProblemSpec(scope=Compare(['all', 'float']))
    ps = problem_spec_check(p, dataset, error_if_compare=False)

    assert isinstance(ps, CompareDict)
    assert len(ps) == 2

    assert ps["scope='all'"].scope == 'all'
    assert ps["scope='float'"].scope == 'float'
    assert ps[Options(scope='all')].scope == 'all'
    assert ps[Options(scope='float')].scope == 'float'

    # Custom names
    compare_scopes = Compare([Option('all', 'all'),
                              Option('float', '2')])
    p = ProblemSpec(scope=compare_scopes)
    ps = problem_spec_check(p, dataset, error_if_compare=False)
    assert ps["scope=all"].scope == 'all'
    assert ps["scope=2"].scope == 'float'


def test_problem_spec_multiple_compares():

    dataset = get_fake_dataset()
    dataset._check_sr()

    # Default names
    p = ProblemSpec(scope=Compare(['all', 'float']),
                    subjects=Compare(['all', 'train', 'test']))
    ps = problem_spec_check(p, dataset, error_if_compare=False)

    assert isinstance(ps, dict)
    assert len(ps) == 6

    assert ps["scope='all', subjects='all'"].scope == 'all'
    assert ps["scope='all', subjects='all'"].subjects == 'all'
    assert ps["scope='all', subjects='train'"].scope == 'all'
    assert ps["scope='all', subjects='train'"].subjects == 'train'
    assert ps["scope='float', subjects='test'"].scope == 'float'
    assert ps["scope='float', subjects='train'"].subjects == 'train'

    ft = ps[Options(scope='float', subjects='train')]
    assert ft.scope == 'float'
    assert ft.subjects == 'train'

    subset = ps["scope='float'"]
    assert isinstance(subset, CompareDict)
    assert len(subset) == 3

    with pytest.raises(KeyError):
        ps['nonsense']


def test_problem_spec_check():

    dataset = get_fake_dataset()
    dataset._check_sr()

    # Test some cases
    p = get_test_ps()

    ps = problem_spec_check(p, dataset)

    assert ps.problem_type == 'regression'
    assert ps.target == '3'
    assert ps.scorer != 'default'
    p.target = 9
    assert ps.target != 9
    assert ps.n_jobs == 2
    assert ps.random_state == 1

    # Test default case
    ps = problem_spec_check('default', dataset)

    assert ps.problem_type == 'regression'
    assert ps.target == '3'
    assert ps.scorer != 'default'

    p.target = '1'
    with pytest.raises(IndexError):
        ps = problem_spec_check(p, dataset)

    p.target = '4'
    with pytest.raises(IndexError):
        ps = problem_spec_check(p, dataset)

    p.target = 8
    with pytest.raises(IndexError):
        ps = problem_spec_check(p, dataset)


def test_preproc_preproc_param_search():

    ps = get_checked_ps()

    search = ParamSearch(search_type='RandomSearch',
                         cv='default',
                         n_iter=10,
                         scorer='default',
                         mp_context='loky',
                         n_jobs='default',
                         weight_scorer=True,
                         random_state='default',
                         dask_ip=None,
                         memmap_X=False,
                         search_only_params=None,
                         progress_loc=None)

    pipe = ModelPipeline(param_search=search)
    has_search = _preproc_param_search(pipe, ps)
    assert has_search is True

    search_d = pipe.param_search
    assert isinstance(search_d, dict)
    assert search_d['n_iter'] == 10
    assert search_d['search_type'] == 'RandomSearch'

    # These should be proc'ed since default
    assert search_d['n_jobs'] == 2
    assert search_d['random_state'] == 1
    assert search_d['scorer'] != 'default'
    assert search_d['weight_scorer'] is True
    assert isinstance(search_d['search_only_params'], dict)
    assert len(search_d['search_only_params']) == 0

    pipe = ModelPipeline(param_search=None)
    has_search = _preproc_param_search(pipe, ps)
    assert has_search is False
    assert pipe.param_search is None

    # Try non-default case
    search = ParamSearch(scorer='r2',
                         n_jobs=10,
                         random_state=9)
    pipe = ModelPipeline(param_search=search)
    has_search = _preproc_param_search(pipe, ps)
    search_d = pipe.param_search
    assert has_search is True
    assert search_d['n_jobs'] == 10
    assert search_d['random_state'] == 9
    assert callable(search_d['scorer'])


def test_preproc_pipeline():

    ps = get_checked_ps()
    data = get_fake_dataset()
    data._check_sr()

    # Test imputers first
    pipe = ModelPipeline(model='ridge', imputers='default')
    proc_pipe = _preproc_pipeline(pipe, ps, dataset=data)
    assert proc_pipe.imputers is None

    data.loc[0, '1'] = np.nan
    pipe = ModelPipeline(model='ridge', imputers='default')
    proc_pipe = _preproc_pipeline(pipe, ps, dataset=data)
    assert isinstance(proc_pipe.imputers[0], Imputer)
    assert isinstance(proc_pipe.imputers[1], Imputer)

    # Check CV case
    pipe = ModelPipeline(model='ridge', imputers='default', scalers=None,
                         param_search=ParamSearch(
                            search_type='DiscreteOnePlusOne'))
    proc_pipe = _preproc_pipeline(pipe, ps, dataset=data)
    assert isinstance(proc_pipe.param_search, dict)
    assert proc_pipe.param_search['search_type'] == 'DiscreteOnePlusOne'

    cv_obj = proc_pipe.param_search['cv']
    assert isinstance(cv_obj, BPtCV)
    assert cv_obj.cv_strategy.groups is None
    assert cv_obj.cv_strategy.stratify is None
    assert cv_obj.cv_strategy.train_only is None

    # Check another non default CV case - splits
    data = get_fake_dataset()
    data.copy_as_non_input('1', '4', inplace=True)
    data.copy_as_non_input('1', '5', inplace=True)

    # Remove category, and make sure dtype changes
    data = data.remove_scope('5', 'category')
    assert data['5'].dtype.name != 'category'

    pipe = ModelPipeline(param_search=ParamSearch(cv=CV(splits='4')))
    proc_pipe = _preproc_pipeline(pipe, ps, dataset=data)
    cv_obj = proc_pipe.param_search['cv']
    assert len(cv_obj.splits_vals) == 3
    assert cv_obj.splits_vals.nunique() == 3

    # Not a valid column error
    pipe = ModelPipeline(param_search=ParamSearch(cv=CV(splits='6')))
    with pytest.raises(KeyError):
        proc_pipe = _preproc_pipeline(pipe, ps, dataset=data)

    # Trigger role failure
    pipe = ModelPipeline(param_search=ParamSearch(cv=CV(splits='3')))
    with pytest.raises(RuntimeError):
        proc_pipe = _preproc_pipeline(pipe, ps, dataset=data)

    # Not category failure
    pipe = ModelPipeline(param_search=ParamSearch(cv=CV(splits='5')))
    with pytest.raises(RuntimeError):
        proc_pipe = _preproc_pipeline(pipe, ps, dataset=data)


def test_get_estimator_simple_case():

    ps = get_checked_ps()
    data = get_fake_dataset()

    pipe = ModelPipeline(model='ridge', imputers='default',
                         scalers=None, verbose=1)
    est = get_estimator(pipeline=pipe, dataset=data, problem_spec=ps)

    # Should be BPt pipeline output
    assert isinstance(est, BPtPipeline)

    # Make sure verbose arg propergates
    assert est.verbose == 1

    # Should just be model
    assert len(est.steps) == 1

    model_name = est.steps[0][0]
    assert isinstance(model_name, str)

    # Should be regression ridge, so make sure
    # this tests default ps steps too
    scope_model = est.steps[0][1]
    assert isinstance(scope_model, BPtModel)

    model = scope_model.estimator
    assert isinstance(model, Ridge)


def test_get_estimator_from_build_simple_case():

    ps = get_checked_ps()
    data = get_fake_dataset()

    pipe = ModelPipeline(model='ridge', imputers='default',
                         scalers=None, verbose=1)

    est = pipe.build(dataset=data, problem_spec=ps)

    # Should be BPt pipeline output
    assert isinstance(est, BPtPipeline)

    # Make sure verbose arg propergates
    assert est.verbose == 1

    # Should just be model
    assert len(est.steps) == 1

    model_name = est.steps[0][0]
    assert isinstance(model_name, str)

    # Should be regression ridge, so make sure
    # this tests default ps steps too
    scope_model = est.steps[0][1]
    assert isinstance(scope_model, BPtModel)

    model = scope_model.estimator
    assert isinstance(model, Ridge)


def test_get_estimator_with_ng_search():

    ps = get_checked_ps()
    data = get_fake_dataset()
    pipe = ModelPipeline(model=Model('ridge', params=1), scalers=None,
                         param_search=ParamSearch('RandomSearch'))
    search_est = get_estimator(pipeline=pipe, dataset=data,
                               problem_spec=ps)

    # Nevergrad cv
    assert isinstance(search_est, NevergradSearchCV)

    # Estimator should be pipeline, w/ ridge at last step
    est = search_est.estimator
    assert isinstance(est, BPtPipeline)

    scope_model = est.steps[-1][1]
    ridge = scope_model.estimator
    assert isinstance(ridge, Ridge)

    param_search = search_est.ps
    assert isinstance(param_search['cv'], BPtCV)
    assert param_search['search_type'] == 'RandomSearch'

    param_dists = search_est.param_distributions
    assert isinstance(param_dists, dict)
    assert len(param_dists) > 0


def test_get_estimator_n_jobs():

    ps = get_checked_ps()
    data = get_fake_dataset()
    pipe = ModelPipeline(model=Model('random forest'), scalers=None)
    est = get_estimator(pipeline=pipe, dataset=data,
                        problem_spec=ps)
    assert isinstance(est, BPtPipeline)

    scope_model = est.steps[0][1]
    assert isinstance(scope_model, BPtModel)

    model = scope_model.estimator
    assert isinstance(model, RandomForestRegressor)
    assert model.n_jobs == 2


def test_get_estimator_extra_params():

    ps = get_test_ps()
    data = get_fake_dataset()
    pipe = ModelPipeline(model=Model('ridge'), scalers=None)
    est = get_estimator(pipeline=pipe, dataset=data,
                        problem_spec=ps, model=Model('random forest'),
                        problem_type='binary')

    assert isinstance(est, BPtPipeline)

    scope_model = est.steps[0][1]
    assert isinstance(scope_model, BPtModel)

    model = scope_model.estimator
    assert isinstance(model, RandomForestClassifier)


def test_get_estimator_extra_params_pipeline():

    ps = get_test_ps()
    data = get_fake_dataset()
    pipe = Pipeline([Model('ridge')])

    # Using Pipeline so should ignore Model
    est = get_estimator(pipeline=pipe, dataset=data,
                        problem_spec=ps, model=Model('random forest'),
                        problem_type='regression')

    assert isinstance(est, BPtPipeline)

    scope_model = est.steps[0][1]
    assert isinstance(scope_model, BPtModel)

    model = scope_model.estimator
    assert isinstance(model, Ridge)


def test_get_estimator_n_jobs_ng():

    ps = get_checked_ps()
    data = get_fake_dataset()
    pipe = ModelPipeline(model=Model('random forest', params=1), scalers=None,
                         param_search=ParamSearch('RandomSearch'))
    search_est = get_estimator(pipeline=pipe, dataset=data,
                               problem_spec=ps)

    assert isinstance(search_est, NevergradSearchCV)
    est = search_est.estimator
    assert isinstance(est, BPtPipeline)

    scope_model = est.steps[0][1]
    assert isinstance(scope_model, BPtModel)

    model = scope_model.estimator
    assert isinstance(model, RandomForestRegressor)

    # Should be n_jobs 1 in model
    assert model.n_jobs == 1

    # and n_jobs 2 in nevergrad search cv
    assert search_est.n_jobs == 2


def test_get_estimator_n_jobs_ng_pipeline():

    ps = get_checked_ps()
    data = get_fake_dataset()

    pipe = Pipeline(steps=[Model('random forest', params=1)],
                    param_search=ParamSearch('RandomSearch'))

    search_est = get_estimator(pipeline=pipe, dataset=data,
                               problem_spec=ps)

    assert isinstance(search_est, NevergradSearchCV)
    est = search_est.estimator
    assert isinstance(est, BPtPipeline)

    scope_model = est.steps[0][1]
    assert isinstance(scope_model, BPtModel)

    model = scope_model.estimator
    assert isinstance(model, RandomForestRegressor)

    # Should be n_jobs 1 in model
    assert model.n_jobs == 1

    # and n_jobs 2 in nevergrad search cv
    assert search_est.n_jobs == 2


def test_get_estimator_with_scope():

    ps = get_checked_ps()
    data = get_fake_dataset()
    pipe = ModelPipeline(model=Model('ridge', scope='1'),
                         scalers=Scaler('robust', scope='float'))
    est = get_estimator(pipeline=pipe, dataset=data,
                        problem_spec=ps)

    assert len(est.steps) == 2

    scaler = est.steps[0][1]
    assert isinstance(scaler, ScopeTransformer)
    assert isinstance(scaler.estimator, RobustScaler)
    assert scaler.inds is Ellipsis

    model = est.steps[1][1]
    assert isinstance(model, BPtModel)
    assert isinstance(model.estimator, Ridge)
    assert model.inds == [0]


def test_get_binary_estimator_default_ps():

    data = get_fake_dataset()
    data = data.binarize('3', threshold=8)
    pipe = Model('random forest')
    est = get_estimator(pipeline=pipe, dataset=data)

    assert isinstance(est, BPtPipeline)
    assert isinstance(est.steps[0][1], BPtModel)
    assert isinstance(est.steps[0][1].estimator,  RandomForestClassifier)


def test_get_param_wrapped_model():

    ps = get_checked_ps()
    data = get_fake_dataset()
    pipe = Model('random forest', params=1,
                 param_search=ParamSearch('RandomSearch'))
    est = get_estimator(pipeline=pipe, dataset=data,
                        problem_spec=ps)

    assert isinstance(est, BPtPipeline)
    assert len(est.steps) == 1

    search_scope_est = est.steps[0][1]
    assert isinstance(search_scope_est, BPtModel)

    search_est = search_scope_est.estimator
    assert isinstance(search_est, NevergradSearchCV)

    param_search = search_est.ps
    assert isinstance(param_search['cv'], BPtCV)
    assert param_search['search_type'] == 'RandomSearch'

    e = search_est.estimator
    assert isinstance(e, RandomForestRegressor)
    assert e.random_state == 1

    param_dists = search_est.param_distributions
    assert isinstance(param_dists, dict)
    assert len(param_dists) > 0


def test_get_estimator_compare1():

    ps = get_checked_ps()
    data = get_fake_dataset()

    pipe = Pipeline([
        Model(obj=Compare(['random forest',
                           'ridge']))])

    est = get_estimator(pipeline=pipe, dataset=data,
                        problem_spec=ps)

    assert isinstance(est, CompareDict)
    assert isinstance(est["'random forest'"], BPtPipeline)
    assert isinstance(est["random forest"], BPtPipeline)


def test_get_estimator_compare_fail():

    ps = ProblemSpec(scope=Compare(['all', 'float']))
    data = get_fake_dataset()

    pipe = Pipeline([
        Model(obj=Compare(['random forest',
                           'ridge']))])

    with pytest.raises(RuntimeError):
        get_estimator(pipeline=pipe, dataset=data,
                      problem_spec=ps)


def test_get_estimator_compare_merge():

    ps = get_checked_ps()
    data = get_fake_dataset()

    pipe1 = Pipeline([
        Model(obj=Compare(['random forest',
                           'ridge']))])

    pipe2 = Pipeline([
        Model(obj=Compare(['elastic',
                           'ridge']))])

    pipe = Compare([Option(pipe1, 'pipe1'),
                    Option(pipe2, 'pipe2')])

    est = get_estimator(pipeline=pipe, dataset=data,
                        problem_spec=ps)

    assert isinstance(est, CompareDict)
    print(est)
    assert len(est) == 4

    # Make sure smart index work
    assert len(est['pipe1']) == 2
    assert len(est['pipe2']) == 2

    p1 = est[Options(pipeline='pipe1',
                     steps__0__obj='random forest')]
    assert isinstance(p1, BPtPipeline)
    assert isinstance(p1.steps[-1][1].estimator, RandomForestRegressor)
    assert p1.steps[-1][1].estimator.n_jobs == 2
    assert p1.steps[-1][1].estimator.random_state == 1

    p2 = est["pipeline=pipe1, steps__0__obj=ridge"]
    assert isinstance(p2, BPtPipeline)
    assert isinstance(p2.steps[-1][1].estimator, Ridge)
    assert p2.steps[-1][1].estimator.random_state == 1


def test_get_estimator_pipeline_with_custom_steps_base():

    ps = get_checked_ps()
    data = get_fake_dataset()

    trans = Transformer('one hot encoder', scope='all')
    model = Ridge()

    pipe = Pipeline(steps=[trans, model])
    est = get_estimator(pipeline=pipe, dataset=data, problem_spec=ps)

    assert isinstance(est, BPtPipeline)
    assert isinstance(est.steps[1][1], Ridge)


def test_get_estimator_pipeline_with_custom_steps_naming():

    ps = get_checked_ps()
    data = get_fake_dataset()

    scalers = [RobustScaler(), RobustScaler(), ('rs', RobustScaler())]
    model = Ridge()

    pipe = Pipeline(steps=scalers + [model])
    est = get_estimator(pipeline=pipe, dataset=data, problem_spec=ps)

    assert isinstance(est, BPtPipeline)
    assert isinstance(est.steps[-1][1], Ridge)

    r1 = est.steps[0][0]
    r2 = est.steps[1][0]
    r3 = est.steps[2][0]
    assert r1 != r2
    assert r1 != r3
    assert r2 != r3


def test_get_estimator_stacking_default():

    ps = get_checked_ps()
    data = get_fake_dataset()

    from ...default.pipelines import stacking_pipe

    # Just want to make sure it doesn't break during construction
    est = get_estimator(pipeline=stacking_pipe, dataset=data,
                        problem_spec=ps)
    assert len(est.steps) == 5

    # Test for breaking behavior because of duplicates, i.e. does
    # uniquify work.
    est = get_estimator(pipeline=stacking_pipe, dataset=data,
                        problem_spec=ps, problem_type='binary')
    assert len(est.steps) == 5


def test_nested_pipelines():

    ps = get_checked_ps()
    data = get_fake_dataset()

    pipe1 = Pipeline(steps=[Model('linear')])
    pipe2 = Pipeline(steps=[pipe1])

    est = get_estimator(pipeline=pipe2, dataset=data, problem_spec=ps)

    assert isinstance(est, BPtPipeline)

    # Make sure doesn't break on fit
    X = np.ones((20, 20))
    y = np.ones(20)
    est.fit(X, y)

    assert isinstance(est.steps[0][1], BPtPipeline)


def test_nested_pipelines_params():

    ps = get_checked_ps()
    data = get_fake_dataset()

    pipe1 = Pipeline(steps=[Model('ridge', params=1)])
    pipe2 = Pipeline(steps=[pipe1])

    est = get_estimator(pipeline=pipe2, dataset=data, problem_spec=ps)

    assert isinstance(est, BPtPipeline)

    # Make sure doesn't break on fit
    X = np.ones((20, 20))
    y = np.ones(20)
    est.fit(X, y)

    assert not isinstance(est.steps[0][1], BPtPipeline)
    assert isinstance(est.steps[0][1], BPtModel)
    assert isinstance(est.steps[0][1].estimator, BPtPipeline)
