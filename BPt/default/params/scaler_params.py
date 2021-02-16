from .Params import TransitionChoice
import numpy as np

P = {}

# Scalers
P['base standard'] = {'with_mean': True,
                      'with_std': True}

P['base minmax'] = {'feature_range': (0, 1)}

P['base robust'] = {'quantile_range': (5, 95)}

P['base winsorize'] = {'quantile_range': (1, 99)}

P['robust gs'] =\
        {'quantile_range':
         TransitionChoice([(x, 100-x) for x in np.arange(1, 40)])}

P['winsorize gs'] =\
        {'quantile_range':
         TransitionChoice([(x, 100-x) for x in np.arange(1, 40)])}

P['base yeo'] = {'method': 'yeo-johnson',
                 'standardize': True}

P['base boxcox'] = {'method': 'box-cox',
                    'standardize': True}

P['base quant norm'] = {'output_distribution': 'normal'}

P['base quant uniform'] = {'output_distribution': 'uniform'}
