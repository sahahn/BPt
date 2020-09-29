'''
Let's store some global-ish pre-set variables in here
'''

SCOPES = set(['float', 'data', 'data files',
              'float covars', 'fc', 'all',
              'n', 'cat', 'categorical', 'covars'])

ORDERED_NAMES = ['loaders', 'imputers', 'scalers',
                 'transformers', 'feat_selectors', 'model']


def is_f2b(d_type):

    valid_names = ['f2b', 'float_to_bin', 'float to bin',
                   'f2c', 'float_to_cat', 'float to cat',
                   'ftb', 'ftc', 'float_to_categorical',
                   'float to categorical', 'float_to_c',
                   'float to c']

    return d_type in valid_names
