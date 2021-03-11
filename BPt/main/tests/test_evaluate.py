from BPt.main.BPtEvaluator import BPtEvaluator
from ...pipeline.BPtSearchCV import BPtGridSearchCV
from ..input import ModelPipeline, Model, CV, Pipeline, Scaler, ParamSearch
from ...dataset.Dataset import Dataset
from ...default.params.Params import Choice
from ..funcs import evaluate, cross_val_score, _sk_check_y
from nose.tools import assert_raises
import pandas as pd
import numpy as np
from ...extensions import LinearResidualizer
from ..compare import Compare, Option, CompareDict
from sklearn.tree import DecisionTreeClassifier


def get_fake_dataset():

    fake = Dataset()

    fake['1'] = np.ones((20))
    fake['2'] = np.ones((20))
    fake['3'] = np.ones((20))
    fake = fake.set_role('3', 'target')

    return fake


def test_sk_check_y():

    # Make sure passes
    y = pd.Series([1, 2, 3])
    _sk_check_y(y)

    # Fails
    y = pd.Series([1, 2, np.nan])
    with assert_raises(RuntimeError):
        _sk_check_y(y)


def test_evaluate_match_cross_val_score():

    dt_pipe = ModelPipeline(model=Model('dt'))
    dataset = get_fake_dataset()
    evaluator = evaluate(pipeline=dt_pipe,
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

    cv_scores = cross_val_score(pipeline=dt_pipe,
                                dataset=dataset,
                                scorer='r2',
                                problem_spec='default',
                                problem_type='regression')

    assert np.sum(evaluator.scores['r2']) == np.sum(cv_scores)


def test_evaluate_regression_dt():

    dt_pipe = ModelPipeline(model=Model('dt'))
    dataset = get_fake_dataset()
    evaluator = evaluate(pipeline=dt_pipe,
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

    assert len(evaluator.train_indices) == 5
    assert len(evaluator.val_indices) == 5


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

    evaluator = evaluate(pipeline=dt_pipe,
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

    evaluator = evaluate(pipeline=linear_pipe,
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
    evaluator = evaluate(pipeline=dt_pipe,
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

    evaluator = evaluate(pipeline=dt_pipe,
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
    evaluator = evaluate(pipeline=dt_pipe,
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


def test_evaluate_with_resid():

    dataset = get_fake_dataset()
    resid = LinearResidualizer(to_resid_df=dataset[['1']])
    pipe = Pipeline(steps=[Scaler(obj=resid, scope='all'), Model('dt')])

    evaluator = evaluate(pipeline=pipe,
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

    # Test matches / works with cross_val_score
    cv_scores = cross_val_score(pipeline=pipe,
                                dataset=dataset,
                                scorer='r2',
                                problem_spec='default',
                                problem_type='regression')
    assert np.sum(evaluator.scores['r2']) == np.sum(cv_scores)

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

    assert len(evaluator.train_indices) == 5
    assert len(evaluator.val_indices) == 5


def test_evaluate_with_resid_param_search():

    dataset = get_fake_dataset()
    resid = LinearResidualizer(to_resid_df=dataset[['1']])
    pipe = Pipeline(steps=[Scaler(obj=resid, scope='all'),
                           Model('dt', params=1)],
                    param_search=ParamSearch(n_iter=5))

    evaluator = evaluate(pipeline=pipe,
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

    # Test matches / works with cross_val_score
    cv_scores = cross_val_score(pipeline=pipe,
                                dataset=dataset,
                                scorer='r2',
                                problem_spec='default',
                                problem_type='regression')
    assert np.sum(evaluator.scores['r2']) == np.sum(cv_scores)

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

    assert len(evaluator.train_indices) == 5
    assert len(evaluator.val_indices) == 5


def test_evaluate_with_resid_grid_search():

    dataset = get_fake_dataset()
    resid = LinearResidualizer(to_resid_df=dataset[['1']])

    model = Model('dt', params={'criterion': Choice(['mse', 'friedman_mse'])})
    pipe = Pipeline(steps=[Scaler(obj=resid, scope='all'),
                           model], param_search=ParamSearch('grid'))

    evaluator = evaluate(pipeline=pipe,
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

    assert isinstance(evaluator.estimators[0], BPtGridSearchCV)

    # Test matches / works with cross_val_score
    cv_scores = cross_val_score(pipeline=pipe,
                                dataset=dataset,
                                scorer='r2',
                                problem_spec='default',
                                problem_type='regression')
    assert np.sum(evaluator.scores['r2']) == np.sum(cv_scores)

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

    assert len(evaluator.train_indices) == 5
    assert len(evaluator.val_indices) == 5


def test_evaluate_compare():

    dataset = get_fake_binary_dataset()

    pipe1 = ModelPipeline(model=Model('dt'))
    pipe2 = ModelPipeline(model=Model('linear'))

    pipe = Compare([Option(pipe1, 'pipe1'),
                    Option(pipe2, 'pipe2')])

    evaluator = evaluate(pipeline=pipe,
                         dataset=dataset,
                         progress_bar=False,
                         problem_type='binary')

    assert isinstance(evaluator, CompareDict)

    e1 = evaluator['pipe1']
    assert isinstance(e1, BPtEvaluator)
    e1_model = e1.estimators[0].steps[-1][1].estimator
    assert isinstance(e1_model, DecisionTreeClassifier)
