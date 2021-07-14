import pickle as pkl
from ..input import Model, Pipeline
from ...pipeline.BPtPipeline import BPtPipeline
import warnings


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
