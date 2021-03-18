from ..input import ModelPipeline, Pipeline, Transformer, Model
from ..input_operations import Duplicate
from sklearn.linear_model import LinearRegression
from nose.tools import assert_raises


def test_pipeline_check_duplicate():

    t = Transformer('fake', scope=Duplicate(['1', '2']))
    pipe = Pipeline([t, Model('ridge')])

    assert len(pipe.steps) == 3
    assert pipe.steps[0].scope == '1'
    assert pipe.steps[0].obj == 'fake'
    assert pipe.steps[1].scope == '2'
    assert pipe.steps[1].obj == 'fake'


def test_pipeline_proc_input():

    pipe = Pipeline(steps=[Model('SOmeThing_reGressor')])
    print(pipe)

    assert pipe.steps[0].obj == 'something'


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

    with assert_raises(TypeError):
        Pipeline(6)

    with assert_raises(IndexError):
        Pipeline([])

    with assert_raises(RuntimeError):
        Pipeline([Transformer('nonsense')])

    with assert_raises(RuntimeError):
        Pipeline(['not real', Model('ridge')])

    with assert_raises(RuntimeError):
        Pipeline([Model('ridge')], param_search='bad param str')


def test_pipeline_with_sklearn_native_input():
    '''Test some naming behavior w/ saved class attribute'''

    pipe = Pipeline(steps=[Transformer('nonsense'),
                    ('my_model', LinearRegression())])
    assert pipe.steps[1]._get_step()[0] == 'my_model'

    pipe2 = Pipeline(steps=[Transformer('nonsense'),
                     ('my_model', LinearRegression())])
    assert pipe2.steps[1]._get_step()[0] == 'my_model'

    del pipe
    pipe3 = Pipeline(steps=[Transformer('nonsense'),
                     ('my_model', LinearRegression())])
    assert pipe3.steps[1]._get_step()[0] == 'my_model'
