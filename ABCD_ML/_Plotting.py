"""
_Plotting.py
====================================
Main class extension file for the some plotting functionality.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def Show_Targets_Dist(self):
    '''This method displays some summary statistics about
    the loaded targets, as well as plots the distibution.
    For now it only supports loaded binary and regression target types.
    '''

    assert self.targets.shape[1] == 1, \
        "This function is not yet avaliable for categorical targets"

    print(self.targets.describe())
    vals = self.targets[self.targets_key]
    print()
    print('Num. of unique vals:', len(np.unique(vals)))

    sns.distplot(self.targets)
    plt.title('Target distribution')
    plt.show()
