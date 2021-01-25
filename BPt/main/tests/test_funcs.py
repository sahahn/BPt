from ..Params_Classes import Model, Model_Pipeline, Problem_Spec
from ..funcs import model_pipeline_check, problem_spec_check
from ...dataset.Dataset import Dataset
from nose.tools import assert_raises


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


def get_fake_dataset():

    fake = Dataset()

    fake['1'] = [1, 2, 3]
    fake['2'] = [4, 5, 6]
    fake['3'] = [7, 8, 9]
    fake.set_role('3', 'target')

    return fake


def test_problem_spec_check():

    dataset = get_fake_dataset()

    # Test some cases
    p = Problem_Spec(problem_type='default',
                     target='3',
                     scorer='default',
                     weight_scorer=False,
                     scope='all',
                     subjects='all',
                     n_jobs=1,
                     random_state=1)

    ps = problem_spec_check(p, dataset)

    assert ps.problem_type == 'regression'
    assert ps.target == '3'
    assert ps.scorer != 'default'
    p.target = 9
    assert ps.target != 9
    assert ps.n_jobs == 1
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
