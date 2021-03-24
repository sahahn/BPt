from ..main.input import (Ensemble, FeatSelector, Scaler, Pipeline, Model,
                          ParamSearch, Transformer, Imputer)

pieces = ['m_imputer', 'c_imputer', 'r_scaler',
          'ohe', 'random_search', 'elastic_search',
          'lgbm_search', 'u_feat', 'svm', 'svm_search_pipe',
          'svm_search', 'ridge_search', 'stacking']

pipelines = ['elastic_pipe', 'ridge_pipe', 'lgbm_pipe',
             'svm_pipe', 'stacking_pipe']

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
                    base_model=ridge_search)

# Pre-defined pipelines
_base_steps = [m_imputer, c_imputer, r_scaler, ohe]

elastic_pipe = Pipeline(steps=_base_steps + [elastic_search])
ridge_pipe = Pipeline(steps=_base_steps + [ridge_search])
lgbm_pipe = Pipeline(steps=_base_steps + [lgbm_search])
svm_pipe = Pipeline(steps=_base_steps + [svm_search])
stacking_pipe = Pipeline(steps=_base_steps + [stacking])
