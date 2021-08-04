from BPt.pipeline.BPtPipeline import BPtPipeline
from BPt.main.BPtEvaluator import BPtEvaluator
from ...pipeline.BPtSearchCV import BPtGridSearchCV, NevergradSearchCV
from ..input import (Loader, ModelPipeline, Model, CV, Pipeline, Scaler,
                     ParamSearch, FeatSelector, Transformer)
from ..input_operations import Select
from ...dataset.Dataset import Dataset
from ...default.params.Params import Choice
from ..funcs import evaluate, cross_val_score, _sk_check_y
import pytest
import pandas as pd
import numpy as np
from ...extensions import LinearResidualizer
from ..compare import Compare, Option, CompareDict, Options
from sklearn.tree import DecisionTreeClassifier
from ...pipeline.BPtLoader import BPtLoader
import warnings
from ...pipeline.Selector import Selector
from ...pipeline.tests.helpers import get_fake_mapping, clean_fake_mapping


def get_fake_datafile_dataset(m, n):

    fm = get_fake_mapping(int(m*n))
    fm = {key: fm[key].loc for key in fm}
    data = Dataset(np.arange(int(m*n)).reshape((m, n)),
                   columns=['col1', 'col2'],
                   dtype='object')
    data = data.replace(fm)

    data.to_data_file(scope='all', inplace=True)
    data['t'] = np.random.RandomState(10).random(m)
    data.set_role('t', 'target', inplace=True)

    return data


def test_evaluate_with_loader():

    data = get_fake_datafile_dataset(50, 2)
    pipe = Pipeline([Loader('identity'), Model('dt')])

    results = evaluate(pipeline=pipe,
                       dataset=data,
                       progress_bar=False,
                       cv=3)

    assert len(results.feat_names[0]) == 8
    assert isinstance(results.estimators[0].steps[0][1], BPtLoader)

    # Clean up
    clean_fake_mapping(100)


def test_evaluate_with_loader_inverse_fis():

    data = get_fake_datafile_dataset(50, 2)
    pipe = Pipeline([Loader('identity'), Model('dt')])

    results = evaluate(pipeline=pipe,
                       dataset=data,
                       progress_bar=False,
                       cv=3)

    # Clean up
    clean_fake_mapping(100)

    inverse_fis = results.get_inverse_fis()

    assert len(inverse_fis) == 3
    assert inverse_fis[0].loc['col1'].shape == (2, 2)
    assert inverse_fis[0].loc['col2'].shape == (2, 2)


def test_evaluate_with_loader_inverse_fis_categorical():

    warnings.filterwarnings("ignore")

    data = get_fake_datafile_dataset(50, 2)
    pipe = Pipeline([Loader('identity'), Model('linear', tol=1)])

    data = data.k_bin(scope='t', n_bins=5)

    results = evaluate(pipeline=pipe,
                       dataset=data,
                       progress_bar=False,
                       cv=3,
                       random_state=3)
    # Clean up
    clean_fake_mapping(100)

    inverse_fis = results.get_inverse_fis()
    assert len(inverse_fis) == 3

    fold1 = inverse_fis[0]

    # One for each class
    assert len(fold1) == 5

    assert fold1[0].loc['col1'].shape == (2, 2)
    assert fold1[3].loc['col1'].shape == (2, 2)
    assert fold1[0].loc['col2'].shape == (2, 2)


def get_fake_dataset():

    fake = Dataset()

    fake['1'] = np.ones((20))
    fake['2'] = np.ones((20))
    fake['3'] = np.ones((20))
    fake = fake.set_role('3', 'target')

    return fake

def get_fake_dataset2():

    dataset = get_fake_dataset()

    dataset['ind'] = ['a', 'b', 'c', 'd', 'e', 'aa', 'bb', 'cc', 'dd', 'ee',
                      'a2', 'b2', 'c2', 'd2', 'e2', 'aa2',
                      'bb2', 'cc2', 'dd2', 'ee2']
    dataset = dataset.set_index('ind')

    return dataset

def test_sk_check_y():

    # Make sure passes
    y = pd.Series([1, 2, 3])
    _sk_check_y(y)

    # Fails
    y = pd.Series([1, 2, np.nan])
    with pytest.raises(RuntimeError):
        _sk_check_y(y)


def test_evaluate_match_cross_val_score():

    dt_pipe = ModelPipeline(model=Model('dt'))

    d1, d2 = get_fake_dataset(), get_fake_dataset2()
    for dataset in [d1, d2]:
    
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
    
    d1, d2 = get_fake_dataset(), get_fake_dataset2()
    for dataset in [d1, d2]:

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
        assert len(evaluator.timing['fit']) > 0
        assert len(evaluator.timing['score']) > 0
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

        assert len(evaluator.train_subjects) == 5
        assert len(evaluator.val_subjects) == 5


def test_regression_mean_fis():

    dt_pipe = ModelPipeline(model=Model('dt'))

    d1, d2 = get_fake_dataset(), get_fake_dataset2()
    for dataset in [d1, d2]:
        evaluator = evaluate(pipeline=dt_pipe,
                            dataset=dataset,
                            progress_bar=False,
                            problem_type='regression',
                            random_state=0)

        mean_fis = evaluator.get_fis(mean=True)
        assert len(mean_fis) == 0


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
    assert len(evaluator.timing['fit']) > 0
    assert len(evaluator.timing['score']) > 0
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


def test_evaluate_categorical_dt_mean_fis():

    dt_pipe = ModelPipeline(model=Model('linear'))
    dataset = get_fake_category_dataset()

    evaluator = evaluate(pipeline=dt_pipe,
                         dataset=dataset,
                         problem_spec='default',
                         progress_bar=False,
                         problem_type='categorical',
                         random_state=1,
                         cv=3)
    mean_fis = evaluator.get_fis(mean=True)
    assert len(mean_fis) == 3


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
    assert len(evaluator.timing['fit']) > 0
    assert len(evaluator.timing['score']) > 0
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

    with pytest.raises(RuntimeError):
        evaluator._estimators_check()

    with pytest.raises(RuntimeError):
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


def test_nan_targets():

    dt_pipe = ModelPipeline(model=Model('dt'))
    dataset = get_fake_binary_dataset()
    dataset.loc[0, '3'] = np.nan

    cv = CV(splits=3, n_repeats=1)
    evaluator = evaluate(pipeline=dt_pipe,
                         dataset=dataset,
                         cv=cv,
                         scorer='roc_auc',
                         progress_bar=False,
                         problem_type='binary')

    assert evaluator.n_repeats_ == 1
    assert evaluator.n_splits_ == 3

    for tr in evaluator.train_subjects:
        assert 0 not in tr

    # Should be in one here
    is_in = [0 in tr for tr in evaluator.all_train_subjects]
    assert any(is_in)

    assert len(evaluator.all_train_subjects) == len(evaluator.train_subjects)

    ati_len = len(sum([list(e) for e in evaluator.all_train_subjects], []))
    ti_len = len(sum([list(e) for e in evaluator.train_subjects], []))
    assert ati_len != ti_len

    avi_len = len(sum([list(e) for e in evaluator.all_val_subjects], []))
    vi_len = len(sum([list(e) for e in evaluator.val_subjects], []))
    assert avi_len != vi_len

    preds_dfs_dropt = evaluator.get_preds_dfs(drop_nan_targets=True)
    preds_dfs = evaluator.get_preds_dfs(drop_nan_targets=False)
    assert preds_dfs_dropt[0].shape != preds_dfs[0].shape


def test_only_fold_cv_param():

    dt_pipe = ModelPipeline(model=Model('dt'))
    dataset = get_fake_binary_dataset()
    cv = CV(splits=3, n_repeats=1, only_fold=0)
    evaluator = evaluate(pipeline=dt_pipe,
                         dataset=dataset,
                         cv=cv,
                         scorer='roc_auc',
                         progress_bar=False,
                         problem_type='binary')

    assert evaluator.n_repeats_ == 1
    assert evaluator.n_splits_ == 1
    assert len(evaluator.scores['roc_auc']) == 1
    assert len(evaluator.train_subjects) == 1


def test_multiple_only_fold_cv_param():

    dt_pipe = ModelPipeline(model=Model('dt'))
    dataset = get_fake_binary_dataset()
    cv = CV(splits=3, n_repeats=2, only_fold=[0, 4])
    evaluator = evaluate(pipeline=dt_pipe,
                         dataset=dataset,
                         cv=cv,
                         scorer='roc_auc',
                         progress_bar=False,
                         problem_type='binary')

    assert evaluator.n_repeats_ == 1
    assert evaluator.n_splits_ == 2
    assert len(evaluator.scores['roc_auc']) == 2
    assert len(evaluator.train_subjects) == 2


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
    assert len(evaluator.timing['fit']) > 0
    assert len(evaluator.timing['score']) > 0
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

    assert len(evaluator.train_subjects) == 5
    assert len(evaluator.val_subjects) == 5


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
    assert len(evaluator.timing['fit']) > 0
    assert len(evaluator.timing['score']) > 0
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

    assert len(evaluator.train_subjects) == 5
    assert len(evaluator.val_subjects) == 5


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
    assert len(evaluator.timing['fit']) > 0
    assert len(evaluator.timing['score']) > 0
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

    assert len(evaluator.train_subjects) == 5
    assert len(evaluator.val_subjects) == 5


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


def test_evaluate_compare_ps():

    dataset = get_fake_binary_dataset()

    pipe = ModelPipeline(model=Model('dt'))

    evaluator = evaluate(pipeline=pipe,
                         dataset=dataset,
                         scope=Compare(['1', '2']),
                         progress_bar=False)

    assert isinstance(evaluator, CompareDict)


def test_evaluate_extra_params_compare():

    dataset = get_fake_dataset()

    pipe = ModelPipeline(model=Model('dt', max_depth=Compare([1, 2, 3])))
    evaluator = evaluate(pipeline=pipe,
                         dataset=dataset,
                         progress_bar=False,
                         problem_type='regression')

    assert isinstance(evaluator, CompareDict)

    # Try different index methods
    e1 = evaluator['1']
    e2 = evaluator['model__max_depth=2']
    e3 = evaluator[Options(model__max_depth=3)]
    e4 = evaluator[Options(max_depth=3)]
    e5 = evaluator['max_depth=2']

    # Make sure actually set extra param
    es = [e1, e2, e3, e4, e5]
    for md, e in zip([1, 2, 3, 3, 2], es):
        assert e.estimators[0].steps[-1][1].estimator.max_depth == md


def test_evaluate_step_compare():

    dataset = get_fake_dataset()
    pipe = Pipeline(steps=[Compare([Model('dt'), Model('linear')])])

    evaluator = evaluate(pipeline=pipe,
                         dataset=dataset,
                         progress_bar=False,
                         problem_type='regression')
    assert isinstance(evaluator, CompareDict)


def test_evaluate_step_compare_mp():

    dataset = get_fake_dataset()
    pipe = ModelPipeline(model=Compare([Model('dt'), Model('linear')]))

    evaluator = evaluate(pipeline=pipe,
                         dataset=dataset,
                         progress_bar=False,
                         problem_type='regression')
    assert isinstance(evaluator, CompareDict)


def test_evaluate_step_compare2():

    dataset = get_fake_dataset()
    pipe = Pipeline(steps=[Compare([Scaler('standard'), Scaler('robust')]),
                           Model('dt')])

    evaluator = evaluate(pipeline=pipe,
                         dataset=dataset,
                         progress_bar=False,
                         problem_type='regression')
    assert isinstance(evaluator, CompareDict)


def test_evaluate_step_compare2_mp():

    dataset = get_fake_dataset()
    pipe = ModelPipeline(scalers=Compare([Scaler('standard'),
                                          Scaler('robust')]),
                         model=Model('dt'))

    evaluator = evaluate(pipeline=pipe,
                         dataset=dataset,
                         progress_bar=False,
                         problem_type='regression')
    assert isinstance(evaluator, CompareDict)


def test_evaluate_step_compare2_mp2():

    dataset = get_fake_dataset()
    pipe = ModelPipeline(scalers=[Scaler('standard'),
                                  Compare([Scaler('standard'),
                                           Scaler('robust')])],
                         model=Model('dt'))

    evaluator = evaluate(pipeline=pipe,
                         dataset=dataset,
                         progress_bar=False,
                         problem_type='regression')
    assert isinstance(evaluator, CompareDict)


def test_evaluate_with_compare_progress_bars():

    dataset = get_fake_dataset()
    pipe = ModelPipeline(scalers=[Scaler('standard'),
                                  Compare([Scaler('standard'),
                                           Scaler('robust')])],
                         model=Model('dt'))

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


def test_evaluator_get_X_transform_df():

    pipe = Pipeline([Scaler('standard'), Model('dt')])
    dataset = get_fake_dataset()
    evaluator = evaluate(pipeline=pipe,
                         dataset=dataset,
                         progress_bar=False,
                         cv=3)

    for fold in [0, 1, 2]:
        X_tr_trans = evaluator.get_X_transform_df(
            dataset, fold=fold, subjects='tr')
        assert X_tr_trans.sum().sum() == 0
        assert '1' in X_tr_trans
        assert '2' in X_tr_trans

        X_val_trans = evaluator.get_X_transform_df(
            dataset, fold=fold, subjects='val')
        assert X_val_trans.sum().sum() == 0
        assert '1' in X_val_trans
        assert '2' in X_val_trans

        assert len(X_tr_trans) + len(X_val_trans) == len(dataset)


def test_evaluate_pipeline_with_select():

    select_scaler = Select([Scaler('standard'), Scaler('robust')])
    select_model = Select([Model('linear'), Model('random forest')])

    pipe = Pipeline([select_scaler, select_model],
                    param_search=ParamSearch(n_iter=2))
    
    d1, d2 = get_fake_dataset(), get_fake_dataset2()
    for dataset in [d1, d2]:

        evaluator = evaluate(pipeline=pipe,
                            dataset=dataset,
                            progress_bar=False,
                            cv=3)

        assert isinstance(evaluator, BPtEvaluator)
        search_est = evaluator.estimators[0]
        assert isinstance(search_est, NevergradSearchCV)
        best_est = search_est.best_estimator_
        assert isinstance(best_est, BPtPipeline)
        step0 = best_est.steps[0]
        step1 = best_est.steps[1]
        assert isinstance(step0[1], Selector)
        assert isinstance(step1[1], Selector)
        assert step0[1].estimator_ == step0[1].estimators[step0[1].to_use][1]
        assert step1[1].estimator_ == step1[1].estimators[step1[1].to_use][1]


def test_select_nested_model_pipes_grid():

    p1 = Pipeline([Model('random forest')])
    p2 = Pipeline([Model('ridge')])

    select_pipe = Select([Model(p1), Model(p2)])

    pipe = Pipeline([select_pipe],
                    param_search=ParamSearch(search_type='grid'))
    
    d1, d2 = get_fake_dataset(), get_fake_dataset2()
    for dataset in [d1, d2]:
        evaluate(pipeline=pipe,
                dataset=dataset,
                progress_bar=False,
                cv=2)


def test_evaluate_modelpipeline_with_select():

    select_scaler = Select([Scaler('standard'), Scaler('robust')])
    select_model = Select([Model('linear'), Model('random forest')])

    pipe = ModelPipeline(scalers=select_scaler,
                         model=select_model,
                         param_search=ParamSearch(n_iter=2))

    d1, d2 = get_fake_dataset(), get_fake_dataset2()
    for dataset in [d1, d2]:

        evaluator = evaluate(pipeline=pipe,
                            dataset=dataset,
                            progress_bar=False,
                            cv=3)

        assert isinstance(evaluator, BPtEvaluator)
        search_est = evaluator.estimators[0]
        assert isinstance(search_est, NevergradSearchCV)
        best_est = search_est.best_estimator_
        assert isinstance(best_est, BPtPipeline)
        step0 = best_est.steps[0]
        step1 = best_est.steps[1]
        assert isinstance(step0[1], Selector)
        assert isinstance(step1[1], Selector)
        assert step0[1].estimator_ == step0[1].estimators[step0[1].to_use][1]
        assert step1[1].estimator_ == step1[1].estimators[step1[1].to_use][1]


def test_evaluate_pipeline_with_custom_selector():

    feat_selector = FeatSelector('selector', params=1)
    model = Model('linear')

    pipe = Pipeline(steps=[feat_selector, model],
                    param_search=ParamSearch(n_iter=2))

    d1, d2 = get_fake_dataset(), get_fake_dataset2()
    for dataset in [d1, d2]:

        evaluator = evaluate(pipeline=pipe,
                            dataset=dataset,
                            progress_bar=False,
                            cv=3)

        est1 = evaluator.estimators[0]
        assert isinstance(est1, NevergradSearchCV)

        b_est1 = est1.best_estimator_
        assert isinstance(b_est1, BPtPipeline)

        selector = b_est1[0]
        custom_sel = selector.estimator_

        from ...extensions.FeatSelectors import FeatureSelector as FS
        assert isinstance(custom_sel, FS)
        assert len(custom_sel.mask) == 2
        assert np.array_equal(custom_sel.mask, selector._get_support_mask())


def run_fs_checks(evaluator):

    est1 = evaluator.estimators[0]
    assert isinstance(est1, NevergradSearchCV)

    b_est1 = est1.best_estimator_
    assert isinstance(b_est1, BPtPipeline)

    ohe_tr = b_est1[0]
    assert ohe_tr.mapping_ == {0: 0, 1: 1}
    assert ohe_tr.out_mapping_ == {0: [0, 1], 1: 2}

    sel = b_est1[1]
    assert sel.mapping_ == {0: [0, 1], 1: 2}

    custom_sel = sel.estimator_
    assert custom_sel.mapping_ == {0: [0, 1], 1: 2}

    # Should be same as map to same spot
    msk = custom_sel.mask
    assert msk[0] == msk[1]

    if msk[0]:
        assert sel.out_mapping_[0] == 0
        assert sel.out_mapping_[1] == 1

        if msk[2]:
            assert sel.out_mapping_[2] == 2
            assert len(evaluator.feat_names[0]) == 3
        else:
            assert sel.out_mapping_[2] is None
            assert len(evaluator.feat_names[0]) == 2

    else:
        assert sel.out_mapping_[0] is None
        assert sel.out_mapping_[1] is None
        assert sel.out_mapping_[2] == 0
        assert len(evaluator.feat_names[0]) == 1

        if not msk[2]:
            raise AssertionError('Mask all False!')


def test_evaluate_pipeline_with_custom_selector_mapping():

    fake = Dataset()

    fake['1'] = np.ones((20))
    fake['1'][:10] = 0
    fake['2'] = np.ones((20))
    fake['3'] = np.ones((20))
    fake = fake.set_role('3', 'target')

    ohe = Transformer('one hot encoder', scope='1')
    feat_selector = FeatSelector('selector', params=1)
    model = Model('linear')

    pipe = Pipeline(steps=[ohe, feat_selector, model],
                    param_search=ParamSearch(n_iter=1))

    for rs in range(10):
        evaluator = evaluate(pipeline=pipe,
                             dataset=fake,
                             random_state=rs,
                             progress_bar=False,
                             cv=3)
        run_fs_checks(evaluator)


def test_evaluate_nan_targets():

    dataset = get_fake_dataset()
    dataset.loc[0, '3'] = np.nan
    dataset.loc[15, '3'] = np.nan

    pipe = Pipeline(Model('dt'))

    results = evaluate(pipeline=pipe,
                       dataset=dataset,
                       progress_bar=False,
                       cv=2)

    rr = repr(results)
    assert 'BPtEvaluator' in rr
    assert 'all_train_subjects' in rr
    assert 'all_val_subjects' in rr


def test_evaluate_nan_targets_named_index():

    dataset = get_fake_dataset2()
    dataset.loc['a', '3'] = np.nan
    dataset.loc['d', '3'] = np.nan

    pipe = Pipeline(Model('dt'))

    results = evaluate(pipeline=pipe,
                       dataset=dataset,
                       progress_bar=False,
                       cv=2)

    rr = repr(results)
    assert 'BPtEvaluator' in rr
    assert 'all_train_subjects' in rr
    assert 'all_val_subjects' in rr


def test_linear_svm_with_multiproc():

    search_cv = CV(splits=3, n_repeats=1)
    random_search = ParamSearch('RandomSearch', n_iter=16, cv=search_cv)
    linear_svm_search = Model('linear svm', params=1,
                              param_search=random_search)
    pipe = Pipeline(steps=[linear_svm_search])

    dataset = get_fake_dataset()

    _ = evaluate(pipeline=pipe,
                 dataset=dataset,
                 progress_bar=False,
                 cv=2, n_jobs=4)


def test_evaluate_cv_test():

    linear_pipe = ModelPipeline(model=Model('linear'))
    dataset = get_fake_category_dataset()
    dataset = dataset.set_test_split(.2, random_state=2)

    results = evaluate(pipeline=linear_pipe,
                       dataset=dataset,
                       problem_spec='default',
                       subjects='default',
                       cv='test',
                       problem_type='categorical')

    assert len(set(results.val_subjects[0]) - set(dataset.test_subjects)) == 0
    assert len(set(dataset.test_subjects) - set(results.val_subjects[0])) == 0


def test_evaluate_second_cv_test():

    dataset = get_fake_dataset()

    dataset = dataset.set_test_split(.2, random_state=2)

    pipe = Pipeline(Model('dt'))

    results = evaluate(pipeline=pipe,
                       dataset=dataset,
                       progress_bar=False,
                       subjects='all',
                       cv='test')

    assert len(set(results.val_subjects[0]) - set(dataset.test_subjects)) == 0
    assert len(set(dataset.test_subjects) - set(results.val_subjects[0])) == 0

def test_evaluate_second_cv_test_named_index():

    dataset = get_fake_dataset2()

    dataset = dataset.set_test_split(.2, random_state=2)

    pipe = Pipeline(Model('dt'))

    results = evaluate(pipeline=pipe,
                       dataset=dataset,
                       progress_bar=False,
                       subjects='all',
                       cv='test')

    assert len(set(results.val_subjects[0]) - set(dataset.test_subjects)) == 0
    assert len(set(dataset.test_subjects) - set(results.val_subjects[0])) == 0


