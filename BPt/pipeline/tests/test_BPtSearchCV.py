from ..BPtSearchCV import (wrap_param_search, get_search_cv,
                           NevergradSearchCV, BPtGridSearchCV)
from ...main.Params_Classes import Param_Search, CV
import nevergrad as ng
from ..helpers import is_ng


def get_param_search():

    param_search = Param_Search(search_type='RandomSearch',
                                splits=3,
                                n_repeats=1,
                                cv='default',
                                n_iter=10,
                                scorer='default',
                                weight_scorer=False,
                                mp_context='loky',
                                n_jobs='default',
                                dask_ip=None,
                                memmap_X=False,
                                search_only_params=None)

    param_search.set_random_state(1)
    param_search.set_n_jobs(2)
    param_search.set_scorer('regression')
    param_search.set_cv(CV())
    param_search.set_split_vals(None)

    return param_search


def test_wrap_param_search():

    param_search = get_param_search()
    model_obj = ('fake_name', 'fake_model')
    model_params = {'fake_name__param1': 1,
                    'fake_name__param2': ng.p.Choice([1, 2]),
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
    assert is_ng(params['param2'])

    assert search_model.n_jobs == 2


def test_get_search_cv():

    param_search = get_param_search()
    estimator = 'fake_estimator'
    param_distributions = {'param1': 5}
    progress_loc = 'nowhere'

    search_model = get_search_cv(estimator, param_search,
                                 param_distributions, progress_loc)

    assert isinstance(search_model, NevergradSearchCV)
    assert(search_model.estimator == 'fake_estimator')
    assert search_model.n_jobs == 2
    assert search_model.param_distributions['param1'] == 5
    assert search_model.progress_loc == progress_loc

    # Make sure get params works
    params = search_model.get_params()
    assert params['estimator'] == 'fake_estimator'
    assert params['param_search___n_jobs'] == 2
    assert params['param_search___random_state'] == 1


def test_get_grid_search():

    param_search = get_param_search()
    param_search.search_type = 'grid'
    estimator = 'fake_estimator'
    param_distributions = {'param1': 5}
    progress_loc = 'nowhere'

    search_model = get_search_cv(estimator, param_search,
                                 param_distributions, progress_loc)

    assert isinstance(search_model, BPtGridSearchCV)
