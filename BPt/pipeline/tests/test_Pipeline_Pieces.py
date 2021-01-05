from BPt.helpers.Data_Scopes import Data_Scopes
from ..Pipeline_Pieces import add_estimator_to_params, Feat_Selectors
from .helpers import get_fake_data_scopes


def test_add_estimator_to_params():

    params = {'s__1': 1, 'my_model__2': 2}
    returned = add_estimator_to_params(params)

    assert returned['s__estimator__1'] == 1
    assert returned['my_model__estimator__2'] == 2


def test_feature_selectors():

    data_scopes = get_fake_data_scopes()

    spec = {'problem_type': 'binary',
            'random_state': None,
            'n_jobs': 1,
             search_type}
    fs = Feat_Selectors(user_passed_objs={}, Data_Scopes=data_scopes)