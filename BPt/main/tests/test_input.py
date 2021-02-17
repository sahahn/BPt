from ..input import ModelPipeline, Pipeline, Transformer, Model
from ..input_operations import Duplicate
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

    steps = pipe.get_steps()
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
        Pipeline(['fgdf', Model('ridge')])

    with assert_raises(RuntimeError):
        Pipeline([Model('ridge')], param_search='sdfsdf')
