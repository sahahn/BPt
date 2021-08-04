import pickle as pkl
from ..input import Model, Pipeline, ParamSearch
from ...pipeline.BPtPipeline import BPtPipeline
import warnings
import tempfile
import os


def test_default_model_build_float_scope():

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        model = Model('linear svm', params=1, scope='float')
        name_model, _ = model.build()
        _, sk_model = name_model

        assert sk_model.inds is Ellipsis


def test_default_model_build_scope_float_equal():

    model = Model('linear svm', params=1, scope='float')
    name_model, _ = model.build(scope='float')
    _, sk_model = name_model
    assert sk_model.inds is Ellipsis


def test_default_model_build_scope_all():

    model = Model('dt', params=1, scope='all')
    name_model, _ = model.build()
    _, sk_model = name_model
    assert sk_model.inds is Ellipsis


def test_default_pipeline_build():

    pipe = Pipeline([Model('rf', params=1)])
    est = pipe.build()

    assert isinstance(est, BPtPipeline)
    assert est.steps[0][1].inds is Ellipsis


def save_load_del(obj):

    temp_dr = tempfile.gettempdir()
    loc = os.path.join(temp_dr, 'temp.pkl')

    with open(loc, 'wb') as f:
        pkl.dump(obj, f)

    with open(loc, 'rb') as f:
        obj = pkl.load(f)

    os.remove(loc)


def test_pickle_pipe_with_params():

    pipe = Pipeline([Model('rf', params=1)])
    est = pipe.build()
    save_load_del(est)


def test_pickle_pipe_with_params_binary():

    pipe = Pipeline([Model('rf', params=1)])
    est = pipe.build(problem_type='binary')
    save_load_del(est)


def test_pickle_pipe_search():

    pipe = Pipeline([Model('rf', params=1)],
                    param_search=ParamSearch())
    est = pipe.build()
    save_load_del(est)


def test_pickle_pipe_search_binary():

    pipe = Pipeline([Model('rf', params=1)],
                    param_search=ParamSearch())
    est = pipe.build(problem_type='binary')
    save_load_del(est)
