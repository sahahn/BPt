from ..main.input import (Ensemble, FeatSelector, Scaler, Pipeline, Model,
                          ParamSearch, Transformer, Imputer)
from ..main.compare import Compare, Option


# Pieces
m_imputer = Imputer('mean', scope='float')
c_imputer = Imputer('median', scope='category')
r_scaler = Scaler('robust', scope='float')
ohe = Transformer('one hot encoder', scope='category')

random_search = ParamSearch('RandomSearch', n_iter=60)
elastic_search = Model('elastic', params=1, param_search=random_search)
lgbm_search = Model('lgbm', params=1, param_search=random_search)
ridge_search = Model('ridge', params=1, param_search=random_search)

u_feat = FeatSelector('univariate selection', params=2)
svm = Model('svm', params=1)
svm_search_pipe = Pipeline(steps=[u_feat, svm], param_search=random_search)
svm_search = Model(svm_search_pipe)

stacking = Ensemble('stacking', models=[elastic_search,
                                        ridge_search,
                                        svm_search,
                                        lgbm_search],
                    base_model=ridge_search,
                    n_jobs_type='models')

# Pre-defined pipelines
_base_steps = [m_imputer, c_imputer, r_scaler, ohe]

elastic_pipe = Pipeline(steps=_base_steps + [elastic_search])
ridge_pipe = Pipeline(steps=_base_steps + [ridge_search])
lgbm_pipe = Pipeline(steps=_base_steps + [lgbm_search])
svm_pipe = Pipeline(steps=_base_steps + [svm_search])
stacking_pipe = Pipeline(steps=_base_steps + [stacking])

compare_pipe = Compare([Option(elastic_pipe, name='elastic'),
                        Option(ridge_pipe, name='ridge'),
                        Option(svm_pipe, name='svm'),
                        Option(lgbm_pipe, name='lgbm')])

pieces = {'m_imputer': m_imputer,
          'c_imputer': c_imputer,
          'r_scaler': r_scaler,
          'ohe': ohe,
          'random_search': random_search,
          'elastic_search': elastic_search,
          'lgbm_search': lgbm_search,
          'u_feat': u_feat,
          'svm': svm,
          'svm_search_pipe': svm_search_pipe,
          'svm_search': svm_search,
          'ridge_search': ridge_search,
          'stacking': stacking}

pipelines = {'elastic_pipe': elastic_pipe,
             'ridge_pipe': ridge_pipe,
             'lgbm_pipe': lgbm_pipe,
             'svm_pipe': svm_pipe,
             'stacking_pipe': stacking_pipe,
             'compare_pipe': compare_pipe}

pieces_keys = list(pieces)
pipelines_keys = list(pipelines)
