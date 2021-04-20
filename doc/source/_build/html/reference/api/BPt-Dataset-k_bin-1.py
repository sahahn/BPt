import BPt as bp
data = bp.Dataset([.1, .2, .3, .4, .5, .6, .7, .8, .9],
                  columns=['feat'])

# Apply k_bin, not in place, then plot
data.k_bin('feat', n_bins=3, strategy='uniform').plot('feat')

# Apply with dif params
data.k_bin('feat', n_bins=5, strategy='uniform').plot('feat')