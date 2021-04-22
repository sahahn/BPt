from ..input import (ModelPipeline, Pipeline, Transformer,
                     Model, ParamSearch, FeatSelector)
from ..input_operations import Duplicate
from sklearn.linear_model import LinearRegression
from ...default.helpers import proc_str_input
import pytest


def test_pipeline_check_duplicate():

    t = Transformer('fake', scope=Duplicate(['1', '2']))
    pipe = Pipeline([t, Model('ridge')])

    assert len(pipe.steps) == 3
    assert pipe.steps[0].scope == '1'
    assert pipe.steps[0].obj == 'fake'
    assert pipe.steps[1].scope == '2'
    assert pipe.steps[1].obj == 'fake'


def test_pipeline_proc_input():

    assert proc_str_input('SOmeThing_reGressor') == 'something'

    with pytest.raises(RuntimeError):
        Pipeline(steps=[Model('SOmeThing_reGressor')])


def test_coarse_check_fail():

    with pytest.raises(RuntimeError):
        Model(obj='not real')

    with pytest.raises(RuntimeError):
        FeatSelector(obj='nope')

    with pytest.raises(RuntimeError):
        FeatSelector(obj='still fake')

    with pytest.raises(TypeError):
        FeatSelector()


def test_get_piece_params():

    t = Transformer('fake', extra='something')
    params = t.get_params()
    assert params['extra'] == 'something'

    t.set_params(extra=5)
    params2 = t.get_params()
    assert params2['extra'] == 5

    with pytest.raises(AttributeError):
        t.extra


def test_modelpipeline_check_duplicate():

    t = Transformer('fake', scope=Duplicate(['1', '2']))
    pipe = ModelPipeline(transformers=t, imputers=None,
                         scalers=None, model='ridge')

    steps = pipe._get_steps()
    assert len(steps) == 3
    assert steps[0].scope == '1'
    assert steps[0].obj == 'fake'
    assert steps[1].scope == '2'
    assert steps[1].obj == 'fake'


def test_pipeline_bad_input():

    with pytest.raises(TypeError):
        Pipeline(6)

    with pytest.raises(IndexError):
        Pipeline([])

    with pytest.raises(RuntimeError):
        Pipeline([Transformer('fake')])

    with pytest.raises(RuntimeError):
        Pipeline(['not real', Model('ridge')])

    with pytest.raises(RuntimeError):
        Pipeline([Model('ridge')], param_search='bad param str')


def test_pipeline_with_sklearn_native_input():
    '''Test some naming behavior w/ saved class attribute'''

    pipe = Pipeline(steps=[Transformer('fake'),
                    ('my_model', LinearRegression())])
    assert pipe.steps[1]._get_step()[0] == 'my_model'

    pipe2 = Pipeline(steps=[Transformer('fake'),
                     ('my_model', LinearRegression())])
    assert pipe2.steps[1]._get_step()[0] == 'my_model'

    del pipe
    pipe3 = Pipeline(steps=[Transformer('fake'),
                     ('my_model', LinearRegression())])
    assert pipe3.steps[1]._get_step()[0] == 'my_model'


def test_get_pipe():

    random_search = ParamSearch('RandomSearch', n_iter=60)
    u_feat = FeatSelector('univariate selection', params=2)
    svm = Model('svm', params=1)
    svm_search_pipe = Pipeline(steps=[u_feat, svm], param_search=random_search)
    Model(svm_search_pipe)
