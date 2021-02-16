from ..input import ModelPipeline, Model, CV
from ...dataset.Dataset import Dataset
from ..funcs import evaluate, cross_val_score
from nose.tools import assert_raises
import pandas as pd
import numpy as np


def get_fake_dataset():

    fake = Dataset()

    fake['1'] = np.ones((20))
    fake['2'] = np.ones((20))
    fake['3'] = np.ones((20))
    fake = fake.set_role('3', 'target')

    return fake


def test_evaluate_match_cross_val_score():

    dt_pipe = ModelPipeline(model=Model('dt'))
    dataset = get_fake_dataset()
    evaluator = evaluate(model_pipeline=dt_pipe,
                         dataset=dataset,
                         problem_spec='default',
                         cv=5,
                         scorer='r2',
                         progress_bar=False,
                         store_preds=True,
                         store_estimators=True,
                         store_timing=True,
                         progress_loc=None,
                         problem_type='regression')

    cv_scores = cross_val_score(model_pipeline=dt_pipe,
                                dataset=dataset,
                                scorer='r2',
                                problem_spec='default',
                                problem_type='regression')

    assert np.sum(evaluator.scores['r2']) == np.sum(cv_scores)


def test_evaluate_regression_dt():

    dt_pipe = ModelPipeline(model=Model('dt'))
    dataset = get_fake_dataset()
    evaluator = evaluate(model_pipeline=dt_pipe,
                         dataset=dataset,
                         problem_spec='default',
                         cv=5,
                         progress_bar=False,
                         store_preds=True,
                         store_estimators=True,
                         store_timing=True,
                         progress_loc=None,
                         problem_type='regression')

    assert len(evaluator.scores) > 0
    assert len(evaluator.timing['fit_time']) > 0
    assert len(evaluator.timing['score_time']) > 0
    evaluator._estimators_check()

    fis_df = evaluator.get_fis()
    assert list(fis_df) == ['1', '2']
    assert len(fis_df) == 5
    assert len(evaluator.feature_importances_) == 2

    raw_fis = evaluator.get_feature_importances()
    assert len(raw_fis) == 5
    assert evaluator.coef_ is None

    assert isinstance(evaluator.preds, dict)
    assert len(evaluator.preds['predict']) == 5
    assert len(evaluator.preds['y_true']) == 5

    assert len(evaluator.train_subjs) == 5
    assert len(evaluator.val_subjs) == 5


def get_fake_category_dataset():

    fake = Dataset()

    base = np.ones((30))
    base[0:10] = 0
    base[10:20] = 2

    fake['1'] = base
    fake['2'] = base
    fake['3'] = base

    fake = fake.set_role('3', 'target')

    return fake


def test_evaluate_categorical_dt():

    dt_pipe = ModelPipeline(model=Model('dt'))
    dataset = get_fake_category_dataset()

    evaluator = evaluate(model_pipeline=dt_pipe,
                         dataset=dataset,
                         problem_spec='default',
                         cv=5,
                         progress_bar=False,
                         store_preds=True,
                         store_estimators=True,
                         store_timing=True,
                         progress_loc=None,
                         problem_type='categorical')

    assert len(evaluator.scores) > 0
    assert len(evaluator.timing['fit_time']) > 0
    assert len(evaluator.timing['score_time']) > 0
    evaluator._estimators_check()

    fis_df = evaluator.get_fis()
    assert list(fis_df) == ['1', '2']
    assert len(fis_df) == 5
    assert len(evaluator.feature_importances_) == 2

    raw_fis = evaluator.get_feature_importances()
    assert len(raw_fis) == 5
    assert evaluator.coef_ is None

    predict = evaluator.preds['predict']
    predict_proba = evaluator.preds['predict_proba']
    y_true = evaluator.preds['y_true']

    assert len(predict) == 5
    assert predict[0].shape == y_true[0].shape
    assert predict[0].shape[0] == predict_proba[0].shape[0]


def test_evaluate_categorical_linear():

    linear_pipe = ModelPipeline(model=Model('linear'))
    dataset = get_fake_category_dataset()

    evaluator = evaluate(model_pipeline=linear_pipe,
                         dataset=dataset,
                         problem_spec='default',
                         cv=5,
                         progress_bar=False,
                         store_preds=True,
                         store_estimators=True,
                         store_timing=True,
                         problem_type='categorical')

    assert len(evaluator.coef_) == len(evaluator.fis_)
    fis = evaluator.get_fis()
    first_fi = fis[0]
    assert isinstance(first_fi, pd.DataFrame)
    assert first_fi.shape == (5, 2)
    assert evaluator.feature_importances_ is None


def get_fake_binary_dataset():

    fake = Dataset()

    base = np.ones((20))
    base[:10] = 0

    fake['1'] = base
    fake['2'] = base
    fake['3'] = base

    fake = fake.set_role('3', 'target')

    return fake


def test_evaluate_binary():

    dt_pipe = ModelPipeline(model=Model('dt'))
    dataset = get_fake_binary_dataset()
    evaluator = evaluate(model_pipeline=dt_pipe,
                         dataset=dataset,
                         problem_spec='default',
                         cv=5,
                         progress_bar=False,
                         store_preds=True,
                         store_estimators=True,
                         store_timing=True,
                         progress_loc=None,
                         problem_type='binary')

    assert len(evaluator.scores) > 0
    assert len(evaluator.timing['fit_time']) > 0
    assert len(evaluator.timing['score_time']) > 0
    evaluator._estimators_check()

    fis_df = evaluator.get_fis()
    assert list(fis_df) == ['1', '2']
    assert len(fis_df) == 5
    assert len(evaluator.feature_importances_) == 2

    raw_fis = evaluator.get_feature_importances()
    assert len(raw_fis) == 5
    assert evaluator.coef_ is None

    predict = evaluator.preds['predict']
    predict_proba = evaluator.preds['predict_proba']
    y_true = evaluator.preds['y_true']

    assert len(predict) == 5
    assert predict[0].shape == y_true[0].shape
    assert predict[0].shape[0] == predict_proba[0].shape[0]


def test_evaluate_fail():

    dt_pipe = ModelPipeline(model=Model('dt'))
    dataset = get_fake_dataset()

    evaluator = evaluate(model_pipeline=dt_pipe,
                         dataset=dataset,
                         progress_bar=False,
                         store_estimators=False)

    with assert_raises(RuntimeError):
        evaluator._estimators_check()

    with assert_raises(RuntimeError):
        evaluator.permutation_importance(dataset)


def test_evaluate_cv():

    dt_pipe = ModelPipeline(model=Model('dt'))
    dataset = get_fake_binary_dataset()
    cv = CV(splits=3, n_repeats=2)
    evaluator = evaluate(model_pipeline=dt_pipe,
                         dataset=dataset,
                         problem_spec='default',
                         cv=cv,
                         progress_bar=False,
                         store_preds=True,
                         store_estimators=True,
                         store_timing=True,
                         progress_loc=None,
                         problem_type='binary')

    assert evaluator.n_repeats_ == 2
    assert evaluator.n_splits_ == 3
    assert len(evaluator.mean_scores) == 3
    assert len(evaluator.std_scores) == 6
