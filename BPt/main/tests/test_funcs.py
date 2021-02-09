from ...pipeline.BPtPipeline import BPtPipeline
from ...pipeline.BPtSearchCV import NevergradSearchCV
from ...pipeline.ScopeObjs import ScopeTransformer, ScopeModel
from ..Params_Classes import (Model, Model_Pipeline, CV, Scaler,
                              Problem_Spec, Param_Search, Imputer)
from ..funcs import (model_pipeline_check, problem_spec_check, get_estimator,
                     _preproc_model_pipeline, _preproc_param_search)
from ..CV import BPtCV
from ...dataset.Dataset import Dataset
from nose.tools import assert_raises
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import RobustScaler
import numpy as np


def test_no_overlap_param_names():

    ps_params = set(Problem_Spec._get_param_names())
    pipe_params = set(Model_Pipeline._get_param_names())
    assert len(ps_params.intersection(pipe_params)) == 0


def test_model_pipeline_check():

    mp_params = Model_Pipeline(imputers=None,
                               model=Model('ridge'))
    mp = model_pipeline_check(mp_params)
    assert isinstance(mp, Model_Pipeline)

    mp_params = Model_Pipeline(imputers=None,
                               model='ridge')
    mp = model_pipeline_check(mp_params)
    assert isinstance(mp, Model_Pipeline)

    mp_params = Model('ridge')
    mp = model_pipeline_check(mp_params)
    assert isinstance(mp, Model_Pipeline)

    mp_params = Model_Pipeline('ridge')
    mp = model_pipeline_check(mp_params)
    assert isinstance(mp, Model_Pipeline)


def test_model_pipeline_check_extra_args():

    mp_params = Model_Pipeline(imputers=None,
                               model=Model('ridge'))
    mp = model_pipeline_check(mp_params)
    assert isinstance(mp, Model_Pipeline)
    assert mp.imputers is None

    mp = model_pipeline_check(mp_params, imputers=Imputer('mean'),
                              ignore='ignore')
    assert mp.imputers is not None
    assert isinstance(mp, Model_Pipeline)


def get_fake_dataset():

    fake = Dataset()

    fake['1'] = [1, 2, 3]
    fake['2'] = [4, 5, 6]
    fake['3'] = [7, 8, 9]
    fake = fake.set_role('3', 'target')

    return fake


def get_test_ps():

    return Problem_Spec(problem_type='default',
                        target='3',
                        scorer='default',
                        weight_scorer=False,
                        scope='all',
                        subjects='all',
                        n_jobs=2,
                        random_state=1)


def get_checked_ps():

    dataset = get_fake_dataset()

    p = get_test_ps()
    ps = problem_spec_check(p, dataset)

    return ps


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
    with assert_raises(IndexError):
        ps = problem_spec_check(p, dataset)

    p.target = '4'
    with assert_raises(IndexError):
        ps = problem_spec_check(p, dataset)

    p.target = 8
    with assert_raises(IndexError):
        ps = problem_spec_check(p, dataset)


def test_preproc_preproc_param_search():

    ps = get_checked_ps()

    search = Param_Search(search_type='RandomSearch',
                          cv='default',
                          n_iter=10,
                          scorer='default',
                          weight_scorer=True,
                          mp_context='loky',
                          n_jobs='default',
                          random_state='default',
                          dask_ip=None,
                          memmap_X=False,
                          search_only_params=None,
                          progress_loc=None)

    pipe = Model_Pipeline(param_search=search)
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

    pipe = Model_Pipeline(param_search=None)
    has_search = _preproc_param_search(pipe, ps)
    assert has_search is False
    assert pipe.param_search is None

    # Try non-default case
    search = Param_Search(scorer='r2',
                          n_jobs=10,
                          random_state=9)
    pipe = Model_Pipeline(param_search=search)
    has_search = _preproc_param_search(pipe, ps)
    search_d = pipe.param_search
    assert has_search is True
    assert search_d['n_jobs'] == 10
    assert search_d['random_state'] == 9
    assert callable(search_d['scorer'])


def test_preproc_model_pipeline():

    ps = get_checked_ps()
    data = get_fake_dataset()
    data._check_sr()

    # Test imputers first
    pipe = Model_Pipeline(model='ridge', imputers='default')
    proc_pipe = _preproc_model_pipeline(pipe, ps, dataset=data)
    assert proc_pipe.imputers is None

    data.loc[0, '1'] = np.nan
    pipe = Model_Pipeline(model='ridge', imputers='default')
    proc_pipe = _preproc_model_pipeline(pipe, ps, dataset=data)
    assert isinstance(proc_pipe.imputers[0], Imputer)
    assert isinstance(proc_pipe.imputers[1], Imputer)

    # Check CV case
    pipe = Model_Pipeline(model='ridge', imputers='default',
                          param_search=Param_Search(
                            search_type='DiscreteOnePlusOne'))
    proc_pipe = _preproc_model_pipeline(pipe, ps, dataset=data)
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

    # Removpe category, and make sure dtype changes
    data = data.remove_scope('5', 'category')
    assert data['5'].dtype.name != 'category'

    pipe = Model_Pipeline(param_search=Param_Search(cv=CV(splits='4')))
    proc_pipe = _preproc_model_pipeline(pipe, ps, dataset=data)
    cv_obj = proc_pipe.param_search['cv']
    assert len(cv_obj.splits_vals) == 3
    assert cv_obj.splits_vals.nunique() == 3

    # Not a valid column error
    pipe = Model_Pipeline(param_search=Param_Search(cv=CV(splits='6')))
    with assert_raises(KeyError):
        proc_pipe = _preproc_model_pipeline(pipe, ps, dataset=data)

    # Trigger role failure
    pipe = Model_Pipeline(param_search=Param_Search(cv=CV(splits='3')))
    with assert_raises(RuntimeError):
        proc_pipe = _preproc_model_pipeline(pipe, ps, dataset=data)

    # Not category failure
    pipe = Model_Pipeline(param_search=Param_Search(cv=CV(splits='5')))
    with assert_raises(RuntimeError):
        proc_pipe = _preproc_model_pipeline(pipe, ps, dataset=data)


def test_get_estimator_simple_case():

    ps = get_checked_ps()
    data = get_fake_dataset()

    pipe = Model_Pipeline(model='ridge', imputers='default', verbose=1)
    est = get_estimator(model_pipeline=pipe, dataset=data, problem_spec=ps)

    # Should be BPtpipeline output
    assert isinstance(est, BPtPipeline)

    # Make sure verbose arg propergates
    assert est.verbose == 1

    # Should just be model
    assert len(est.steps) == 1

    model_name = est.steps[0][0]
    assert isinstance(model_name, str)

    # Should be regression ridge, so make sure
    # this tests default ps steps too
    model = est.steps[0][1]
    assert isinstance(model, Ridge)


def test_get_estimator_with_ng_search():

    ps = get_checked_ps()
    data = get_fake_dataset()
    pipe = Model_Pipeline(model=Model('ridge', params=1),
                          param_search=Param_Search('RandomSearch'))
    search_est = get_estimator(model_pipeline=pipe, dataset=data,
                               problem_spec=ps)

    # Nevergrad cv
    assert isinstance(search_est, NevergradSearchCV)

    # Estimator should be pipeline, w/ ridge at last step
    est = search_est.estimator
    assert isinstance(est, BPtPipeline)
    assert isinstance(est.steps[-1][1], Ridge)

    param_search = search_est.ps
    assert isinstance(param_search['cv'], BPtCV)
    assert param_search['search_type'] == 'RandomSearch'

    param_dists = search_est.param_distributions
    assert isinstance(param_dists, dict)
    assert len(param_dists) > 0


def test_get_estimator_n_jobs():

    ps = get_checked_ps()
    data = get_fake_dataset()
    pipe = Model_Pipeline(model=Model('random forest'))
    est = get_estimator(model_pipeline=pipe, dataset=data,
                        problem_spec=ps)
    assert isinstance(est, BPtPipeline)

    model = est.steps[0][1]
    assert isinstance(model, RandomForestRegressor)
    assert model.n_jobs == 2


def test_get_estimator_extra_params():

    ps = get_test_ps()
    data = get_fake_dataset()
    pipe = Model_Pipeline(model=Model('ridge'))
    est = get_estimator(model_pipeline=pipe, dataset=data,
                        problem_spec=ps, model=Model('random forest'),
                        problem_type='binary')

    assert isinstance(est, BPtPipeline)
    model = est.steps[0][1]
    assert isinstance(model, RandomForestClassifier)


def test_get_estimator_n_jobs_ng():

    ps = get_checked_ps()
    data = get_fake_dataset()
    pipe = Model_Pipeline(model=Model('random forest', params=1),
                          param_search=Param_Search('RandomSearch'))
    search_est = get_estimator(model_pipeline=pipe, dataset=data,
                               problem_spec=ps)

    assert isinstance(search_est, NevergradSearchCV)
    est = search_est.estimator
    assert isinstance(est, BPtPipeline)

    model = est.steps[0][1]
    assert isinstance(model, RandomForestRegressor)

    # Should be n_jobs 1 in model
    assert model.n_jobs == 1

    # and n_jobs 2 in nevergrad search cv
    assert search_est.n_jobs == 2


def test_get_estimator_with_scope():

    ps = get_checked_ps()
    data = get_fake_dataset()
    pipe = Model_Pipeline(model=Model('ridge', scope='1'),
                          scalers=Scaler('robust', scope='float'))
    est = get_estimator(model_pipeline=pipe, dataset=data,
                        problem_spec=ps)

    assert len(est.steps) == 2

    scaler = est.steps[0][1]
    assert isinstance(scaler, ScopeTransformer)
    assert isinstance(scaler.estimator, RobustScaler)
    assert scaler.inds == [0, 1]

    model = est.steps[1][1]
    assert isinstance(model, ScopeModel)
    assert isinstance(model.estimator, Ridge)
    assert model.inds == [0]


def test_get_param_wrapped_model():

    ps = get_checked_ps()
    data = get_fake_dataset()
    pipe = Model('random forest', params=1,
                 param_search=Param_Search('RandomSearch'))
    est = get_estimator(model_pipeline=pipe, dataset=data,
                        problem_spec=ps)

    assert isinstance(est, BPtPipeline)
    assert len(est.steps) == 1

    search_est = est.steps[0][1]
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
