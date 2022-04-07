import numpy as np
import pandas as pd
from ..samplers import OverSampler
from ...dataset.dataset import Dataset
from ...main.input import Pipeline, Sampler, Model
from ...main.funcs import evaluate

def test_sampler():

    # Mock data
    groups_series = pd.Series([1, 2, 2, 3, 3, 3, 3, 3,],
                            index=['a', 'b', 'c', 'd',
                                   'e', 'f', 'g', 'h'])
    os = OverSampler(groups_series, random_state=2)

    X = np.ones((6, 2))
    X_test = np.ones((2, 2))

    # Make sure fit trans works
    X_fit_trans, resamp_inds, new_fit_index = os.fit_transform(X, fit_index=groups_series.index[:6])
    assert X_fit_trans.shape == (9, 2)
    assert X[resamp_inds].shape == (9, 2)
    assert len(new_fit_index) == 9
    assert all(np.unique(new_fit_index) == np.unique(groups_series.index[:6]))

    # Make sure all groups equal representation
    _, cnts = np.unique(groups_series.loc[new_fit_index], return_counts=True)
    assert len(np.unique(cnts)) == 1

    # Make sure test doesn't transform
    X_test_trans = os.transform(X_test, transform_index=groups_series.index[6:])
    assert X_test_trans.shape == (2, 2)

def test_in_evaluate():

    # Mock data
    df = Dataset([1, 2, 2, 3, 3, 3, 3, 3,], columns=['groups'],
                  index=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])

    df['1'] = np.random.random(size=8)
    df['2'] = np.random.random(size=8)
    df['3'] = np.random.random(size=8)

    df['t'] = np.random.random(size=8)
    df = df.set_role('groups', 'non input')
    df = df.set_role('t', 'target')

    os = OverSampler(df['groups'])
    pipe = Pipeline([Sampler(os), Model('linear')])

    results = evaluate(pipe, df, progress_bar=False, eval_verbose=0)

def test_in_evaluate2():

    # Mock data
    df = Dataset([1, 2, 2, 3, 3, 3, 3, 3,], columns=['groups'],
                  index=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])

    df['groups2'] = [1, 1, 1, 1, 1, 1, 1, 2]
    df['1'] = np.random.random(size=8)
    df['2'] = np.random.random(size=8)
    df['3'] = np.random.random(size=8)

    df['t'] = np.random.random(size=8)
    df = df.set_role(['groups', 'groups2'], 'non input')
    df = df.set_role('t', 'target')

    os = OverSampler()
    pipe = Pipeline([Sampler(os, ref_scope=['groups', 'groups2']),
                     Model('linear')])

    results = evaluate(pipe, df, progress_bar=False, eval_verbose=0)
    results.estimators[0].steps[0][1].estimator_.groups_vals_.unique()
