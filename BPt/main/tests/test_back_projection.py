from ...dataset.Dataset import Dataset
import numpy as np
import pandas as pd
import os
import tempfile
from ..funcs import pipeline_check, evaluate
from ..input import Loader, Ensemble, Model, CV

from sklearn.base import BaseEstimator, TransformerMixin

SZ = 20

class FakeLoader(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def transform(self, X):
        Xt = np.mean(X.reshape((2, -1)), axis=0)
        return Xt

    def inverse_transform(self, X_trans):
        c = np.concatenate([X_trans, X_trans]).reshape((2, -1))
        X = np.concatenate([c[0], c[1]])
        return X

def setup_dataset():

    # Init
    data = Dataset()

    # Add targets
    subjs = [f'subj{i}' for i in range(20)]
    data['r'] = pd.Series([.1, .2, .3, .4, .5, .6, .7, .8, .9, 1, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1],
                          index=subjs)
    data['b'] = pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                          index=subjs)
    data = data.set_role(['r', 'b'], 'target')
    data = data.to_binary(scope='b')

    # Gen and save fake data files
    files = {'d_files': []}
    temp_dr = tempfile.gettempdir()
    for i in range(20):
        save_loc = os.path.join(temp_dr, f'subj{i}.npy')
        x = np.arange(SZ) + np.random.random(SZ)
        np.save(save_loc, x)
        files['d_files'].append(save_loc)

    # Add to dataset
    data = data.add_data_files(files, file_to_subject='auto')

    return data

def get_pipe(pipe):

    pipe = pipeline_check(pipe)

    # Init loader from object
    loader = Loader(FakeLoader())

    # Add loader before rest of steps
    pipe.steps = [loader] + pipe.steps

    try:
        pipe.steps[-1].param_search.n_iter = 2
    except AttributeError:
        pipe.steps[-1].obj.param_search.n_iter = 2

    return pipe

def run_eval(pipe='elastic_pipe', target='t_regression', ensemble=None,
             scorer='default', cv=None):

    if ensemble is None:
        pipe = get_pipe(pipe)
    elif ensemble == 'voting':
        pipe = Ensemble('voting', models=[get_pipe(pipe) for _ in range(2)])
    elif ensemble == 'stacking':
        pipe = Ensemble('stacking', models=[get_pipe(pipe) for _ in range(2)],
                        base_model=Model('ridge'), cv=None)

    data = setup_dataset()

    return evaluate(pipeline=pipe, dataset=data,
                    target=target, mute_warnings=True,
                    progress_bar=False, cv=2, scorer=scorer)


def standard_check(results):


    fis = results.get_fis()

    assert fis.shape == (2, SZ//2)
    assert 'd_files_0' in list(fis)

    inverse_fis = results.get_inverse_fis()
    assert len(inverse_fis[0].loc['d_files']) == SZ
    assert len(inverse_fis[1].loc['d_files']) == SZ


def run_permutation(results):

    return results.permutation_importance(dataset=setup_dataset(),
                                          n_repeats=1,
                                          just_model=True,
                                          nested_model=True,
                                          return_as='dfs')['importances_mean']

def permutation_check(results):

    fis = run_permutation(results)

    assert fis.shape == (2, SZ//2)
    assert 'd_files_0' in list(fis)

    inverse_fis = results.get_inverse_fis(fis)
    assert len(inverse_fis[0].loc['d_files']) == SZ
    assert len(inverse_fis[1].loc['d_files']) == SZ

def test_basic_elastic_r():

    results = run_eval(pipe='elastic_pipe', target='r')
    standard_check(results)

def test_basic_elastic_b():

    results = run_eval(pipe='elastic_pipe', target='b')
    standard_check(results)

def test_basic_elastic_b_roc_auc():

    results = run_eval(pipe='elastic_pipe', target='b', scorer='roc_auc')
    standard_check(results)

def test_basic_rf_r():

    results = run_eval(pipe='rf_pipe', target='r')
    standard_check(results)

def test_basic_rf_b():

    results = run_eval(pipe='rf_pipe', target='b')
    standard_check(results)

def test_permutation_svm_r():

    results = run_eval(pipe='svm_pipe', target='r')
    standard_check(results)

def test_permutation_svm_r():

    results = run_eval(pipe='svm_pipe', target='r')
    permutation_check(results)

def test_permutation_svm_b():

    results = run_eval(pipe='svm_pipe', target='b')
    permutation_check(results)

def test_permutation_elastic_r():

    results = run_eval(pipe='elastic_pipe', target='r')
    permutation_check(results)

def test_permutation_elastic_b():

    results = run_eval(pipe='elastic_pipe', target='b')
    permutation_check(results)

def voting_base_check(results):

    # fis are concat for base
    fis = results.get_fis()
    assert fis.shape ==  (2, SZ)

    # Inverse should be same
    inverse_fis = results.get_inverse_fis()
    assert len(inverse_fis[0].loc['d_files']) == SZ
    assert len(inverse_fis[1].loc['d_files']) == SZ

    # Make sure permutations work too
    voting_permutation_check(results)

def voting_permutation_check(results):

    fis = run_permutation(results)
    assert fis.shape ==  (2, SZ)
    
    inverse_fis = results.get_inverse_fis(fis)
    assert len(inverse_fis[0].loc['d_files']) == SZ
    assert len(inverse_fis[1].loc['d_files']) == SZ

def test_voting_elastic_r():

    results = run_eval(pipe='elastic_pipe', target='r', ensemble='voting')
    voting_base_check(results)

def test_voting_elastic_b():

    results = run_eval(pipe='elastic_pipe', target='b', ensemble='voting')
    voting_base_check(results)

def test_voting_elastic_b_roc_auc():

    results = run_eval(pipe='elastic_pipe', target='b',
                       ensemble='voting', scorer='roc_auc')
    voting_base_check(results)

def test_voting_rf_r():

    results = run_eval(pipe='rf_pipe', target='r', ensemble='voting')
    voting_base_check(results)

def test_voting_rf_b():

    results = run_eval(pipe='rf_pipe', target='b', ensemble='voting')
    voting_base_check(results)

def test_voting_permutation_svm_r():
    
    results = run_eval(pipe='svm_pipe', target='r', ensemble='voting')
    voting_permutation_check(results)

def test_voting_permutation_svm_b():
    
    results = run_eval(pipe='svm_pipe', target='b', ensemble='voting')
    voting_permutation_check(results)

def fs_permutation_check(results):

    fis = run_permutation(results)
    inverse_fis = results.get_inverse_fis(fis)
    assert len(inverse_fis[0].loc['d_files']) == SZ
    assert len(inverse_fis[1].loc['d_files']) == SZ

def test_permutation_svm_fs_r():

    results = run_eval(pipe='svm_fs_pipe', target='r')
    fs_permutation_check(results)

def test_permutation_svm_fs_b():

    results = run_eval(pipe='svm_fs_pipe', target='b')
    fs_permutation_check(results)
    
def test_voting_permutation_svm_fs_r():
    
    results = run_eval(pipe='svm_fs_pipe', target='r', ensemble='voting')
    fs_permutation_check(results)

def test_voting_permutation_svm_fs_b():
    
    results = run_eval(pipe='svm_fs_pipe', target='b', ensemble='voting')
    fs_permutation_check(results)

def test_voting_permutation_svm_fs_b_roc_auc():
    
    results = run_eval(pipe='svm_fs_pipe', target='b',
                       ensemble='voting', scorer='roc_auc')
    fs_permutation_check(results)

def test_stacking_elastic_r():

    results = run_eval(pipe='elastic_pipe', target='r', ensemble='stacking')
    voting_base_check(results)

def test_stacking_elastic_r_custom_cv():

    results = run_eval(pipe='elastic_pipe', target='r', ensemble='stacking',
                       cv=CV(splits=3))
    voting_base_check(results)

def test_stacking_elastic_b():

    results = run_eval(pipe='elastic_pipe', target='b', ensemble='stacking')
    voting_base_check(results)

def test_stacking_elastic_b_roc_auc():

    results = run_eval(pipe='elastic_pipe', target='b',
                       ensemble='stacking', scorer='roc_auc')
    voting_base_check(results)

def test_stacking_rf_r():

    results = run_eval(pipe='rf_pipe', target='r', ensemble='stacking')
    voting_base_check(results)

def test_stacking_rf_b():

    results = run_eval(pipe='rf_pipe', target='b', ensemble='stacking')
    voting_base_check(results)

def test_stacking_permutation_svm_r():

    results = run_eval(pipe='svm_pipe', target='r', ensemble='stacking')
    voting_permutation_check(results)

def test_stacking_permutation_svm_b():

    results = run_eval(pipe='svm_pipe', target='b', ensemble='stacking')
    voting_permutation_check(results)

def test_stacking_permutation_svm_b_roc_auc():

    results = run_eval(pipe='svm_pipe', target='b',
                       ensemble='stacking', scorer='roc_auc')
    voting_permutation_check(results)


