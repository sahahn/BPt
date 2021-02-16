from ..BPtSearchCV import (wrap_param_search, get_search_cv,
                           NevergradSearchCV, BPtGridSearchCV)
from ...default.params.Params import Choice
from .helpers import get_param_search


def test_wrap_param_search():

    param_search = get_param_search()
    model_obj = ('fake_name', 'fake_model')
    model_params = {'fake_name__param1': 1,
                    'fake_name__param2': Choice([1, 2]),
                    'other_model__param1': 1}

    search_obj, search_params =\
        wrap_param_search(param_search, model_obj, model_params)

    # Assert is name, exact name might change
    # but the original name should be present
    search_name = search_obj[0]
    assert isinstance(search_name, str)
    assert 'fake_name' in search_name

    search_model = search_obj[1]
    assert isinstance(search_model, NevergradSearchCV)
    assert(search_model.estimator == 'fake_model')

    # Should return other models params
    assert len(search_params) == 1
    assert search_params['other_model__param1'] == 1

    # Check search params
    params = search_model.param_distributions
    assert params['param1'] == 1
    assert len(params['param2']) == 2

    assert search_model.n_jobs == 2


def test_get_search_cv():

    param_search = get_param_search()
    estimator = 'fake_estimator'
    param_distributions = {'param1': 5}
    param_search['progress_loc'] = 'nowhere'

    search_model = get_search_cv(estimator, param_search,
                                 param_distributions)

    assert isinstance(search_model, NevergradSearchCV)
    assert(search_model.estimator == 'fake_estimator')
    assert search_model.n_jobs == 2
    assert search_model.param_distributions['param1'] == 5
    assert search_model.ps['progress_loc'] == 'nowhere'

    # Make sure get params works
    params = search_model.get_params()
    assert params['estimator'] == 'fake_estimator'
    assert params['ps']['n_jobs'] == 2
    assert params['ps']['random_state'] == 1


def test_get_grid_search():

    param_search = get_param_search()
    param_search['search_type'] = 'grid'
    estimator = 'fake_estimator'
    param_distributions = {'param1': 5}
    param_search['progress_loc'] = 'nowhere'

    search_model = get_search_cv(estimator, param_search,
                                 param_distributions)

    assert isinstance(search_model, BPtGridSearchCV)
