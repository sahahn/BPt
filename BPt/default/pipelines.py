from ..main.input import (Scaler, Pipeline, Model,
                          ParamSearch, Transformer, Imputer)

m_imputer = Imputer('mean', scope='float')
c_imputer = Imputer('median', scope='category')
r_scaler = Scaler('robust', scope='float')
ohe = Transformer('one hot encoder', scope='category')

base = [m_imputer, c_imputer, r_scaler, ohe]

elastic_search = Model('elastic', params=1,
                       param_search=ParamSearch('RandomSearch', n_iter=60))

elastic_pipe = Pipeline(steps=base + [elastic_search])
