from .Params import Scalar

P = {}

# Transformers
P['pca var search'] =\
        {'n_components': Scalar(init=.75, lower=.1, upper=.99),
         'svd_solver': 'full'}

P['ohe'] =\
        {'sparse': False,
         'handle_unknown': 'ignore'}

P['dummy code'] =\
        {'sparse': False,
         'drop': 'first',
         'handle_unknown': 'error'}
