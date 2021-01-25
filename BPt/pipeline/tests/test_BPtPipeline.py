from ..BPtPipeline import BPtPipeline
from .helpers import ToFixedTransformer, get_fake_mapping, clean_fake_mapping
from ..ScopeObjs import ScopeTransformer, ScopeModel
from ..BPtLoader import BPtLoader
from ...extensions import Identity
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import os
import tempfile
import nevergrad as ng
from .helpers import get_param_search
from ..BPtSearchCV import NevergradSearchCV
import shutil
from joblib import hash as joblib_hash


def test_BPtPipeline():

    # 'loaders', 'imputers',
    #  'scalers',
    #  'transformers',
    #  'feat_selectors', 'model']

    steps = []
    to_ones = ToFixedTransformer(to=1)
    st = ScopeTransformer(estimator=to_ones, inds=[1, 2])
    steps.append(('to_ones', st))

    model = ScopeModel(estimator=LinearRegression(), inds=[0, 1])
    steps.append(('model', model))

    names = [[], [], ['to_ones'], [], [], ['model']]
    pipe = BPtPipeline(steps=steps, names=names)

    X = np.zeros((3, 3))
    y = np.ones(3)

    pipe.fit(X, y)
    assert pipe['to_ones'].inds_ == [1, 2]

    # Should update so that next inds are 0, 2
    # as 1 -> 0, 2 -> 1, 0 -> 2, so 0,1 -> 2, 0, sorted = 0, 2
    assert pipe['model'].inds_ == [0, 2]
    assert len(pipe.mapping_) == 3
    assert pipe.mapping_[0] == 2
    assert pipe.mapping_[1] == 0
    assert pipe.mapping_[2] == 1

    # Make sure when re-fit resets mapping each time
    pipe.fit(X, y)
    assert pipe.mapping_[0] == 2
    assert pipe.mapping_[1] == 0
    assert pipe.mapping_[2] == 1

    # Test propegate n_jobs
    pipe.n_jobs = 2
    assert pipe['to_ones'].n_jobs == 2
    assert pipe['to_ones'].estimator.n_jobs == 2

    X_df = pd.DataFrame(X)
    X_trans = pipe.transform_df(X_df)
    assert X_trans[0].sum() == 0
    assert X_trans[1].sum() == 3
    assert X_trans[2].sum() == 3


def test_file_mapping_hash():

    # Make sure that regardless of DataFile position
    # in memory, that it hashes correctly.
    mapping = get_fake_mapping(10)
    h1 = joblib_hash(mapping)
    clean_fake_mapping(10)

    mapping = get_fake_mapping(10)
    h2 = joblib_hash(mapping)
    clean_fake_mapping(10)
    assert h1 == h2


def run_pipe_with_loader_ts(cache_fit_dr=None):

    steps = []

    # Loader - transform (5, 2) to (5, 8)
    # as each data_file contains np.zeros((2, 2))
    file_mapping = get_fake_mapping(100)
    loader = BPtLoader(estimator=Identity(),
                       inds=[0, 1],
                       file_mapping=file_mapping,
                       n_jobs=1,
                       fix_n_jobs=False,
                       cache_loc=None)
    steps.append(('loader', loader))

    # Add transformer to ones
    # input here should be (5, 8) of real val, original
    # inds of 0 should work on half
    to_ones = ToFixedTransformer(to=1)
    st = ScopeTransformer(estimator=to_ones, inds=[0])
    steps.append(('to_ones', st))

    # Add basic linear regression model
    # Original inds should work on all
    model = ScopeModel(estimator=LinearRegression(), inds=[0, 1])
    param_dists = {'estimator__fit_intercept': ng.p.Choice([True, False]),
                   'estimator__normalize':
                   ng.p.TransitionChoice([True, False])}
    search_model = NevergradSearchCV(estimator=model,
                                     param_search=get_param_search(),
                                     param_distributions=param_dists)

    steps.append(('model', search_model))

    # Create pipe
    names = [['loader'], [], ['to_ones'], [], [], ['model']]
    pipe = BPtPipeline(steps=steps, names=names,
                       cache_fit_dr=cache_fit_dr)

    X = np.arange(100).reshape((50, 2))
    y = np.ones(50)

    pipe.fit(X, y, train_data_index=np.arange(50))

    # Make sure fit worked correctly
    assert pipe[0].n_features_in_ == 2
    assert pipe[1].n_features_in_ == 8
    assert pipe[1].estimator_.n_features_in_ == 4
    assert len(pipe.mapping_[0]) == 4
    assert len(pipe.mapping_[1]) == 4
    assert 7 in pipe.mapping_[1]

    # Make sure reverse transform works
    X_df = pd.DataFrame(X)

    X_trans = pipe.transform_df(X_df, fs=False)
    assert X_trans.shape == (50, 8)
    assert X_trans.loc[4, '1_3'] == 9
    assert X_trans.loc[1, '1_2'] == 3
    assert X_trans.loc[4, '0_0'] == 1
    assert X_trans.loc[0, '0_0'] == 1

    # Make sure predict works,
    # seems safe to assume model
    # can learn to predict 1's
    # as all targets are 1's.
    # but may need to change?
    preds = pipe.predict(X)
    assert np.all(preds > .99)

    # Check bpt pipeline coef attribute
    assert np.array_equal(pipe[-1].best_estimator_.coef_,
                          pipe.coef_)

    # Clean fake file mapping
    clean_fake_mapping(100)

    return pipe


def test_pipeline_with_loader():

    # Base pipeline with loader tests
    run_pipe_with_loader_ts(cache_fit_dr=None)


def test_pipeline_fit_caching():

    # Run with cache fit dr
    cache_fit_dr =\
        os.path.join(tempfile.gettempdir(), 'test_cache')
    pipe = run_pipe_with_loader_ts(cache_fit_dr=cache_fit_dr)

    # Make sure computed hash + saved copy
    assert hasattr(pipe, 'hash_')
    assert os.path.exists(pipe._get_hash_loc())

    # Shouldn't load from cache
    assert not hasattr(pipe, 'loaded_')

    # Delete existing pipe
    del pipe

    # Run again to make sure loading from cache worked
    pipe = run_pipe_with_loader_ts(cache_fit_dr=cache_fit_dr)
    assert hasattr(pipe, 'hash_')
    assert pipe.loaded_ is True

    # Removed cached once done
    shutil.rmtree(cache_fit_dr)
