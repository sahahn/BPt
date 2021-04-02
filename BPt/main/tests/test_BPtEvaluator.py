from .test_evaluate import get_fake_dataset
from ..input import (Model, Pipeline, Scaler)
from ..funcs import evaluate
from nose.tools import assert_raises
import numpy as np


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

    with assert_raises(RuntimeError):
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
