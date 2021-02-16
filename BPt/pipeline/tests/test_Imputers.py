from pandas.core import base
from ...main.Params_Classes import Imputer, Model
from ..Pipeline_Pieces import Imputers
from ...dataset.Dataset import Dataset
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from ..ScopeObjs import ScopeTransformer


def test_iterative_imputer():

    spec = {'problem_type': 'binary',
            'random_state': 2,
            'n_jobs': 1,
            'scope': 'all'}

    dataset = Dataset()
    dataset['float col'] = [1, 2, 3, 4, 5, 6]
    dataset['cat'] = [1, 1, 2, 2, 3, 3]
    dataset['cat missing'] = [1, 1, np.nan, 2, np.nan, 3]
    dataset.add_scope(['cat', 'cat missing'], 'category', inplace=True)

    assert dataset.get_cols('category') == ['cat', 'cat missing']

    piece = Imputers(dataset=dataset, spec=spec, user_passed_objs={})

    input_piece = Imputer('iterative', scope='category',
                          base_model=Model('linear'),
                          base_model_type='default')

    objs, obj_params = piece.process([input_piece])

    assert len(obj_params) == 0

    base_trans = objs[0][1]
    assert isinstance(base_trans, ScopeTransformer)
    assert base_trans.inds == [1, 2]

    it_imputer = base_trans.estimator
    assert isinstance(it_imputer, IterativeImputer)
    assert it_imputer.random_state == 2

    # Should be categorical, so logistic regression
    base_estimator = it_imputer.estimator
    assert isinstance(base_estimator, LogisticRegression)


def test_iterative_imputer_default_float():

    spec = {'problem_type': 'binary',
            'random_state': 2,
            'n_jobs': 1,
            'scope': 'all'}

    dataset = Dataset()
    dataset['float col'] = [1, 2, 3, 4, 5, 6]
    dataset['cat'] = [1, 1, 2, 2, 3, 3]
    dataset['cat missing'] = [1, 1, np.nan, 2, np.nan, 3]
    dataset.add_scope(['cat', 'cat missing'], 'category', inplace=True)

    assert dataset.get_cols('category') == ['cat', 'cat missing']

    piece = Imputers(dataset=dataset, spec=spec, user_passed_objs={})

    input_piece = Imputer('iterative', scope='all',
                          base_model=Model('linear'),
                          base_model_type='default')

    objs, obj_params = piece.process([input_piece])

    assert len(obj_params) == 0

    base_trans = objs[0][1]
    assert isinstance(base_trans, ScopeTransformer)
    assert base_trans.inds == [0, 1, 2]

    it_imputer = base_trans.estimator
    assert isinstance(it_imputer, IterativeImputer)
    assert it_imputer.random_state == 2

    # Not all categorical, so should be linear regression
    base_estimator = it_imputer.estimator
    assert isinstance(base_estimator, LinearRegression)

    