from ..transformer import BPtTransformerMV
from ..input import MVTransformer
from ....dataset.Dataset import Dataset
from ....main.input import Pipeline, Scaler, Model, ParamSearch
from ....main.funcs import get_estimator
import tempfile
import shutil
import os
import numpy as np
from mvlearn.embed import CCA


def get_X():

    X = np.array([[0., 0., 1., 0.1, -0.2],
                  [1., 0., 0., 0.9, 1.1],
                  [2., 2., 2., 6.2, 5.9],
                  [3., 5., 4., 11.9, 12.3]])

    return X


def basic_test():

    X = get_X()
    mv = BPtTransformerMV(estimator=CCA(multiview_output=False),
                          inds=[[0, 1, 2], [3, 4]])

    X_trans = mv.fit_transform(X)

    # Basic Checks
    assert X_trans.shape == (4, 1)
    assert mv.inds_ == [0, 1, 2, 3, 4]
    assert mv.view_inds_ == [[0, 1, 2], [3, 4]]
    assert mv.out_mapping_[0] == [0]
    assert mv.out_mapping_[4] == [0]
    assert len(mv.out_mapping_) == 5
    assert isinstance(mv.estimator_, CCA)
    assert mv.n_trans_feats_ == 1
    assert mv.n_features_in_ == 5

    assert len(mv.estimator_.means_) == 2
    assert len(mv.estimator_.means_[0] == 3)
    assert len(mv.estimator_.means_[1] == 2)

    assert len(mv.estimator_.loadings_) == 2
    assert len(mv.estimator_.loadings_[0]) == 3
    assert len(mv.estimator_.loadings_[1]) == 2


def test_with_cache():

    temp_dr = os.path.join(tempfile.gettempdir(), 'temp_dr')
    if os.path.exists(temp_dr):
        shutil.rmtree(temp_dr)

    X = get_X()
    mv = BPtTransformerMV(estimator=CCA(multiview_output=False),
                          inds=[[0, 1, 2], [3, 4]], cache_loc=temp_dr)

    # Fit once w/ caching
    mv.fit_transform(X)
    assert os.listdir(temp_dr) == ['joblib']

    # Fit again, should load from cached
    X_trans = mv.fit_transform(X)
    assert X_trans.shape == (4, 1)

    if os.path.exists(temp_dr):
        shutil.rmtree(temp_dr)

    assert not os.path.exists(temp_dr)


def test_with_mapping():

    X = get_X()
    mv = BPtTransformerMV(estimator=CCA(multiview_output=False),
                          inds=[[0, 1, 2], [3, 4]])

    # This mapping should ignore feat 1, adding it on after
    mapping = {0: 0, 1: 0, 2: 2, 3: 3, 4: 4}
    X_trans = mv.fit_transform(X, mapping=mapping)

    assert X_trans.shape == (4, 2)
    assert np.array_equal(X_trans[:, 1], X[:, 1])

    assert len(mv.estimator_.loadings_) == 2
    assert len(mv.estimator_.loadings_[0]) == 2
    assert len(mv.estimator_.loadings_[1]) == 2
    assert len(mv.estimator_.means_) == 2
    assert len(mv.estimator_.means_[0] == 2)
    assert len(mv.estimator_.means_[1] == 2)


def get_dataset():

    data = Dataset()
    data['0'] = [1, 0, 1, .3, .5, 7]
    data['1'] = [0, 0, 1, .1, -1, 1]
    data['2'] = [0, 1, 0, .2, -2, 2.2]
    data['3'] = [2, 2, 2, 0, 2, 0]
    data['4'] = [0, 1, 0, 4.3, 4.3, 10]
    data['5'] = [0, 0, 1, 0, 1, 0]

    data = data.add_scope(['0', '1', '2'], 'c1')
    data = data.add_scope(['3', '4'], 'c2')
    data = data.set_target('5')
    return data


def test_build():

    data = get_dataset()

    piece = MVTransformer(obj='cca', scopes=[['1', '2'], ['3', '4']])
    obj, params = piece.build(data)
    trans = obj[1]
    assert isinstance(trans, BPtTransformerMV)
    assert len(trans.inds) == 2
    assert trans.inds[0] == [1, 2]
    assert trans.inds[1] == [3, 4]

    cca = trans.estimator
    assert isinstance(cca, CCA)

    # Should be empty
    assert len(params) == 0

    # Try fit
    x_trans = trans.fit_transform(np.array(data[['0', '1', '2', '3', '4']]))
    assert x_trans.shape == (6, 2)
    assert np.array_equal(x_trans[:, 1], np.array(data['0']))


def test_basic_pipeline_integration():

    data = get_dataset()
    mv_trans = MVTransformer(obj='cca',
                             scopes=[['1', '2'], ['3', '4']],
                             n_components=2)

    pipe = Pipeline([Scaler('standard'), mv_trans, Model('linear')])
    est = get_estimator(pipe, data)

    assert len(est.steps) == 3
    assert isinstance(est.steps[1][1], BPtTransformerMV)
    assert isinstance(est.steps[1][1].estimator, CCA)

    # Get data
    X, y = data.get_Xy()

    # Fit
    est.fit(X, y)

    pred_y = est.predict(X)
    assert len(pred_y) == 6

    # Just make sure no errors scoring
    est.score(X, y)

    # Transform shape should be 2 comp + feat 0
    X_trans = est.transform(X)
    assert X_trans.shape == (6, 3)


def test_pipeline_with_search_integration():
    '''Want to make sure doesn't break in a multi-proc context.'''

    data = get_dataset()
    mv_trans = MVTransformer(obj='cca',
                             scopes=[['1', '2'], ['3', '4']],
                             n_components=2)

    param_search = ParamSearch(n_iter=2)
    pipe = Pipeline([Scaler('standard'), mv_trans, Model('dt', params=1)],
                    param_search=param_search)
    search_est = get_estimator(pipe, data, problem_type='regression', n_jobs=2)

    # Some assertions about estimator
    est = search_est.estimator
    assert len(est.steps) == 3
    assert isinstance(est.steps[1][1], BPtTransformerMV)
    assert isinstance(est.steps[1][1].estimator, CCA)

    # Get data
    X, y = data.get_Xy()

    # Fit
    search_est.fit(X, y)

    pred_y = search_est.predict(X)
    assert len(pred_y) == 6

    # Just make sure no errors scoring
    search_est.score(X, y)

    # Transform shape should be 2 comp + feat 0
    X_trans = search_est.transform(X)
    assert X_trans.shape == (6, 3)
