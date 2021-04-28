from .test_evaluate import get_fake_dataset
from ..input import (Model, Pipeline, Scaler, CV)
from ...dataset.Dataset import Dataset
from ..funcs import evaluate
from ..BPtEvaluator import is_notebook
import pytest
import numpy as np


def test_is_notebook():
    assert not is_notebook()


def test_bpt_evaluator_neg_verbose():

    dataset = get_fake_dataset()
    pipe = Pipeline(Model('dt'))

    evaluate(pipeline=pipe,
             dataset=dataset,
             progress_bar=False,
             random_state=2,
             cv=2,
             eval_verbose=-2)


def test_bpt_evaluator_repr():

    dataset = get_fake_dataset()
    pipe = Pipeline(Model('dt'))

    results = evaluate(pipeline=pipe,
                       dataset=dataset,
                       progress_bar=False,
                       cv=2)

    rr = repr(results)
    assert 'BPtEvaluator' in rr
    assert 'all_train_subjects' not in rr
    assert 'all_val_subjects' not in rr
    assert 'coef_' not in rr


def test_bpt_evaluator_repr2():

    dataset = get_fake_dataset()
    pipe = Pipeline(Model('linear'))

    results = evaluate(pipeline=pipe,
                       dataset=dataset,
                       progress_bar=False,
                       cv=2)

    rr = repr(results)
    assert 'coef_' in rr


def test_bpt_evaluator_store_preds_false():

    dataset = get_fake_dataset()
    pipe = Pipeline(Model('dt'))

    results = evaluate(pipeline=pipe,
                       dataset=dataset,
                       store_preds=False,
                       progress_bar=False,
                       cv=2)

    assert results.preds is None


def test_bpt_evaluator_progress_bars():

    dataset = get_fake_dataset()
    pipe = Pipeline(Model('dt'))

    # No repeats
    evaluate(pipeline=pipe,
             dataset=dataset,
             progress_bar=True,
             random_state=2,
             cv=CV(splits=2, n_repeats=1))

    # With repeats
    evaluate(pipeline=pipe,
             dataset=dataset,
             progress_bar=True,
             random_state=2,
             cv=CV(splits=2, n_repeats=2))


def test_bpt_evaluator_score():

    dataset = get_fake_dataset()
    pipe = Pipeline(Model('dt'))

    results = evaluate(pipeline=pipe,
                       dataset=dataset,
                       progress_bar=False,
                       random_state=2,
                       cv=2)

    # Make sure score attribute works
    first_scorer = list(results.mean_scores)[0]
    assert results.score == results.mean_scores[first_scorer]

    results = evaluate(pipeline=pipe,
                       dataset=dataset,
                       progress_bar=False,
                       random_state=2,
                       scorer='neg_mean_squared_error',
                       cv=2)

    assert results.mean_scores['neg_mean_squared_error'] == results.score


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


def test_subset_by_fail():

    data = np.array([[1, 1, 1, 1, 1, 1],
                     [2, 2, 2, 2, 2, 2],
                     [.1, .2, .3, .4, .5, .6],
                     [1, 1, 1, 2, 2, 2]])
    data = data.transpose((1, 0))

    data = Dataset(data=data,
                   columns=['1', '2', 't', 'grp'],
                   targets='t', non_inputs='grp')
    data = data.to_binary('grp')

    pipe = Pipeline([Scaler('standard'), Model('linear')])

    results = evaluate(pipeline=pipe,
                       dataset=data,
                       store_preds=False,
                       progress_bar=False,
                       random_state=2,
                       cv=2)

    with pytest.raises(RuntimeError):
        results.subset_by('grp', data)


def test_subset_by():

    data = np.array([[1, 1, 1, 1, 1, 1],
                     [2, 2, 2, 2, 2, 2],
                     [.1, .2, .3, .4, .5, .6],
                     [1, 1, 1, 2, 2, 2]])
    data = data.transpose((1, 0))

    data = Dataset(data=data,
                   columns=['1', '2', 't', 'grp'],
                   targets='t', non_inputs='grp')
    data = data.to_binary('grp')

    pipe = Pipeline([Scaler('standard'), Model('linear')])

    results = evaluate(pipeline=pipe,
                       dataset=data,
                       progress_bar=False,
                       random_state=2,
                       cv=2)
    subsets = results.subset_by('grp', data)

    g1 = subsets['1.0']
    g2 = subsets['2.0']

    assert len(g1.scores['explained_variance']) == 2
    assert len(g2.scores['explained_variance']) == 2
    assert len(g1.mean_scores) == 2
    assert len(g2.mean_scores) == 2

    assert len(g1.train_subjects) == 2
    assert len(g2.train_subjects) == 2
    assert len(g1.val_subjects) == 2
    assert len(g2.val_subjects) == 2

    assert len(g1.val_subjects[0].intersection(g2.val_subjects[0])) == 0
    assert len(g1.val_subjects[1].intersection(g2.val_subjects[1])) == 0

    g1_preds = g1.get_preds_dfs()
    g2_preds = g2.get_preds_dfs()

    assert list(g1_preds[0]) == list(g2_preds[0])
    assert list(g1_preds[1]) == list(g2_preds[1])


def test_subset_by_binary():

    data = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                     [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                     [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                     [1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2]])
    data = data.transpose((1, 0))

    data = Dataset(data=data,
                   columns=['1', '2', 't', 'grp'],
                   targets='t', non_inputs='grp')
    data = data.to_binary('grp')
    data = data.to_binary('t')

    pipe = Pipeline([Scaler('standard'), Model('linear')])

    results = evaluate(pipeline=pipe,
                       dataset=data,
                       progress_bar=False,
                       random_state=2,
                       scorer='roc_auc',
                       problem_type='binary',
                       cv=2)
    subsets = results.subset_by('grp', data)

    g1 = subsets['1']
    g2 = subsets['2']

    assert len(g1.scores['roc_auc']) == 2
    assert len(g2.scores['roc_auc']) == 2
    assert len(g1.mean_scores) == 1
    assert len(g2.mean_scores) == 1

    assert len(g1.train_subjects) == 2
    assert len(g2.train_subjects) == 2
    assert len(g1.val_subjects) == 2
    assert len(g2.val_subjects) == 2

    assert len(g1.val_subjects[0].intersection(g2.val_subjects[0])) == 0
    assert len(g1.val_subjects[1].intersection(g2.val_subjects[1])) == 0

    g1_preds = g1.get_preds_dfs()
    g2_preds = g2.get_preds_dfs()

    assert list(g1_preds[0]) == list(g2_preds[0])
    assert list(g1_preds[1]) == list(g2_preds[1])


def test_subset_by_categorical():

    data = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                     [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                      2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                     [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1,
                      1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1],
                     [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1,
                      1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 2, 1, 1, 1]]
                    )
    data = data.transpose((1, 0))

    data = Dataset(data=data,
                   columns=['1', '2', 't', 'grp'],
                   targets='t', non_inputs='grp')
    data = data.to_binary('grp')
    data = data.ordinalize('t')

    pipe = Pipeline([Scaler('standard'), Model('linear')])

    results = evaluate(pipeline=pipe,
                       dataset=data,
                       progress_bar=False,
                       random_state=1,
                       scorer='roc_auc_ovr',
                       problem_type='categorical',
                       cv=2)
    subsets = results.subset_by('grp', data)

    g1 = subsets['1']
    g2 = subsets['2']

    assert len(g1.scores['roc_auc_ovr']) == 2
    assert len(g2.scores['roc_auc_ovr']) == 2
    assert len(g1.mean_scores) == 1
    assert len(g2.mean_scores) == 1

    assert len(g1.train_subjects) == 2
    assert len(g2.train_subjects) == 2
    assert len(g1.val_subjects) == 2
    assert len(g2.val_subjects) == 2

    assert len(g1.val_subjects[0].intersection(g2.val_subjects[0])) == 0
    assert len(g1.val_subjects[1].intersection(g2.val_subjects[1])) == 0

    g1_preds = g1.get_preds_dfs()
    g2_preds = g2.get_preds_dfs()

    assert list(g1_preds[0]) == list(g2_preds[0])
    assert list(g1_preds[1]) == list(g2_preds[1])

    assert 'BPtEvaluatorSubset(grp=1)' in repr(g1)
    assert 'BPtEvaluatorSubset(grp=2)' in repr(g2)
