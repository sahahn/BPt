from .test_evaluate import get_fake_dataset
from ..input import (Model, Pipeline, Scaler, CV)
from ..funcs import evaluate
from ..BPtEvaluator import is_notebook
import pytest
import numpy as np


def test_is_notebook():

    assert not is_notebook()


def test_bpt_evaluator_compare_fail():

    pipe1 = Pipeline([Scaler('standard'), Model('linear')])
    pipe2 = Pipeline([Scaler('standard'), Model('dt')])

    dataset = get_fake_dataset()

    results1 = evaluate(pipeline=pipe1,
                        dataset=dataset,
                        progress_bar=False,
                        random_state=2,
                        cv=2)
    results2 = evaluate(pipeline=pipe2,
                        dataset=dataset,
                        progress_bar=False,
                        random_state=2,
                        cv=2)

    with pytest.raises(RuntimeError):
        results1.compare(results2)


def test_bpt_evaluator_compare():

    pipe1 = Pipeline([Scaler('standard'), Model('linear')])
    pipe2 = Pipeline([Model('dt')])

    dataset = get_fake_dataset()
    dataset['3'] = np.random.random(len(dataset))

    results1 = evaluate(pipeline=pipe1,
                        dataset=dataset,
                        progress_bar=False,
                        scorer=['neg_mean_squared_error',
                                'explained_variance'],
                        random_state=2,
                        cv=2)

    dataset['3'] = np.random.random(len(dataset))
    results2 = evaluate(pipeline=pipe2,
                        dataset=dataset,
                        progress_bar=False,
                        random_state=2,
                        scorer=['neg_mean_squared_error',
                                'explained_variance'],
                        cv=2)

    # Just in case, make sure dif results
    results2.mean_scores['explained_variance'] = .9
    results2.scores['explained_variance'] = [.8, 1]

    compare_df = results1.compare(results2)
    assert compare_df.shape == (2, 7)


def test_bpt_evaluator_compare_non_overlap_metric():

    pipe1 = Pipeline([Scaler('standard'), Model('linear')])
    pipe2 = Pipeline([Model('dt')])

    dataset = get_fake_dataset()
    dataset['3'] = np.random.random(len(dataset))

    results1 = evaluate(pipeline=pipe1,
                        dataset=dataset,
                        progress_bar=False,
                        scorer=['neg_mean_squared_error'],
                        random_state=2,
                        cv=2)

    # Just in case, make sure dif results
    dataset['3'] = np.random.random(len(dataset))
    results2 = evaluate(pipeline=pipe2,
                        dataset=dataset,
                        progress_bar=False,
                        random_state=2,
                        scorer=['neg_mean_squared_error',
                                'explained_variance'],
                        cv=2)

    compare_df = results1.compare(results2)
    assert compare_df.shape == (1, 7)


def test_bpt_evaluator_compare_non_overlap_cv1():

    pipe1 = Pipeline([Scaler('standard'), Model('linear')])
    pipe2 = Pipeline([Model('dt')])

    dataset = get_fake_dataset()
    dataset['3'] = np.random.random(len(dataset))

    results1 = evaluate(pipeline=pipe1,
                        dataset=dataset,
                        progress_bar=False,
                        scorer=['neg_mean_squared_error',
                                'explained_variance'],
                        random_state=2,
                        cv=2)

    # Just in case, make sure dif results
    dataset['3'] = np.random.random(len(dataset))
    results2 = evaluate(pipeline=pipe2,
                        dataset=dataset,
                        progress_bar=False,
                        random_state=2,
                        scorer=['neg_mean_squared_error',
                                'explained_variance'],
                        cv=3)
    results2.mean_scores['explained_variance'] = .9
    results2.scores['explained_variance'] = [.8, 1, .9]

    compare_df = results1.compare(results2)
    assert compare_df.shape == (2, 2)


def test_bpt_evaluator_compare_non_overlap_cv2():

    pipe1 = Pipeline([Scaler('standard'), Model('linear')])
    pipe2 = Pipeline([Model('dt')])

    dataset = get_fake_dataset()
    dataset['3'] = np.random.random(len(dataset))

    results1 = evaluate(pipeline=pipe1,
                        dataset=dataset,
                        progress_bar=False,
                        scorer=['neg_mean_squared_error',
                                'explained_variance'],
                        random_state=2,
                        cv=2)

    # Just in case, make sure dif results
    dataset['3'] = np.random.random(len(dataset))
    results2 = evaluate(pipeline=pipe2,
                        dataset=dataset,
                        progress_bar=False,
                        random_state=10,
                        scorer=['neg_mean_squared_error',
                                'explained_variance'],
                        cv=2)

    results2.mean_scores['explained_variance'] = .9
    results2.scores['explained_variance'] = [.8, 1]

    compare_df = results1.compare(results2)
    assert compare_df.shape == (2, 2)


def test_multiclass_get_preds_df():

    df = get_fake_dataset()
    df['3'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
    df.ordinalize('3', inplace=True)

    pipe = Pipeline([Model('linear')])

    results = evaluate(pipeline=pipe,
                       dataset=df,
                       progress_bar=False,
                       scorer='roc_auc_ovr',
                       cv=CV(splits=2))

    assert len(results.preds['predict']) == 2
    assert len(results.preds['predict_proba']) == 2

    assert len(results.preds['predict'][0]) == 10
    assert len(results.preds['predict_proba'][0]) == 10

    assert len(results.preds['predict_proba'][0][0]) == 3

    # Test get preds df
    r_df = results.get_preds_dfs()
    assert r_df[0].shape == (10, 8)
    assert r_df[0].shape == (10, 8)


def test_permutation_feature_importance():

    pipe = Pipeline([Scaler('standard'), Model('linear')])
    dataset = get_fake_dataset()
    results = evaluate(pipeline=pipe,
                       dataset=dataset,
                       progress_bar=False,
                       scorer='neg_mean_squared_error',
                       random_state=2,
                       cv=2)

    fis = results.permutation_importance(dataset, n_repeats=10)
    assert fis['importances_mean'].shape == (2, 2)
    assert fis['importances_std'].shape == (2, 2)
