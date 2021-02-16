
P = {}

# Imputers
P['mean imp'] = {'strategy': 'mean'}
P['median imp'] = {'strategy': 'median'}
P['most freq imp'] = {'strategy': 'most_frequent'}
P['constant imp'] = {'strategy': 'constant'}
P['iterative imp'] = {'initial_strategy': 'mean',
                      'skip_complete': True}
