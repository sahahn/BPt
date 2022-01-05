from ..loaders import Identity
import numpy as np
import tempfile
import os
import pytest
import warnings


def test_identity():

    i = Identity()
    x = np.ones((10, 10))
    assert i.fit_transform(x).shape == (100, )


def test_single_connectivity_measure():

    try:
        from ..loaders import SingleConnectivityMeasure
    except ImportError:
        return

    scm = SingleConnectivityMeasure()
    X_trans = scm.fit_transform(np.ones((5, 10)))
    assert X_trans.shape == (10, 10)

    scm = SingleConnectivityMeasure(vectorize=True, discard_diagonal=True)
    X_trans = scm.fit_transform(np.ones((5, 10)))
    assert X_trans.shape == (45, )

    scm = SingleConnectivityMeasure(vectorize=True)
    X_trans = scm.fit_transform(np.ones((5, 10)))
    assert X_trans.shape == (55, )


def test_threshold_network_measures_abs():

    try:
        from ..loaders import ThresholdNetworkMeasures
    except:
        return

    nm =\
        ThresholdNetworkMeasures(threshold=0.2,
                                 threshold_type='abs',
                                 threshold_method='value',
                                 to_compute='avg_degree')

    X = np.array([[.1, .2, .3, -.5],
                  [.2, 0, -.5, 0],
                  [.3, -.5, 0, 0],
                  [-.5, 0, 0, 0]])

    # Fit
    nm.fit(X)
    assert nm._feat_names[0] == 'avg_degree'
    assert len(nm._feat_names) == 1

    # Threshold should stay fixed
    nm._threshold_check(X)
    assert nm.threshold == .2

    thresh_X = nm._apply_threshold(X)
    assert len(np.unique(thresh_X)) == 2

    # Should just drop .1 edge
    assert np.sum(thresh_X) == 8

    import networkx as nx
    G = nx.from_numpy_array(thresh_X)

    degrees = [n[1] for n in G.degree()]
    X_trans = nm.transform(X)
    assert len(X_trans) == 1
    assert np.mean(degrees) == X_trans[0]


def test_threshold_network_measures_neg():

    try:
        from ..loaders import ThresholdNetworkMeasures
    except:
        return

    nm =\
        ThresholdNetworkMeasures(threshold=-.2,
                                 threshold_type='neg',
                                 threshold_method='value')

    X = np.array([[.1, .2, .3, -.5],
                  [.2, 0, -.5, 0],
                  [.3, -.5, 0, 0],
                  [-.5, 0, 0, 0]])

    thresh_X = nm._apply_threshold(X)
    assert np.sum(thresh_X) == 4
    assert thresh_X[0][-1] == 1
    assert thresh_X[0][1] == 0


def test_threshold_network_measures_pos():

    try:
        from ..loaders import ThresholdNetworkMeasures
    except:
        return

    nm =\
        ThresholdNetworkMeasures(threshold=.2,
                                 threshold_type='pos',
                                 threshold_method='value')

    X = np.array([[.1, .2, .3, -.5],
                  [.2, 0, -.5, 0],
                  [.3, -.5, 0, 0],
                  [-.5, 0, 0, 0]])

    thresh_X = nm._apply_threshold(X)
    assert np.sum(thresh_X) == 4
    assert thresh_X[0][-1] == 0
    assert thresh_X[0][1] == 1


def test_threshold_network_measures_density():

    try:
        from ..loaders import ThresholdNetworkMeasures
    except:
        return

    nm =\
        ThresholdNetworkMeasures(threshold=.2,
                                 threshold_type='abs',
                                 threshold_method='density')

    X = np.array([[0, .2, .3, -.4],
                  [.2, 0, -.5, 0],
                  [.3, -.5, 0, 0],
                  [-.4, 0, 0, 0]])

    thresh_X = nm._apply_threshold(X)
   
    # 20% of 16 possible edges
    # rounded up is 4
    assert np.sum(thresh_X) == 4

def test_threshold_network_measures_density_pos():

    try:
        from ..loaders import ThresholdNetworkMeasures
    except:
        return

    nm =\
        ThresholdNetworkMeasures(threshold=.2,
                                 threshold_type='pos',
                                 threshold_method='density')

    X = np.array([[0, .2, .3, -.4],
                  [.2, 0, -.5, 0],
                  [0, -.5, 0, 0],
                  [-.4, 0, 0, 0]])

    thresh_X = nm._apply_threshold(X)
   
    # 20% of 16 possible edges
    # rounded up is 4
    assert np.sum(thresh_X) == 3

def test_threshold_network_measures_density_neg():

    try:
        from ..loaders import ThresholdNetworkMeasures
    except:
        return

    nm =\
        ThresholdNetworkMeasures(threshold=.2,
                                 threshold_type='neg',
                                 threshold_method='density')

    X = np.array([[0, .2, .3, -.4],
                  [.2, 0, -.5, 0],
                  [.3, 0, 0, 0],
                  [-.4, 0, 0, 0]])

    thresh_X = nm._apply_threshold(X)
   
    # 20% of 16 possible edges
    # rounded up is 4
    assert np.sum(thresh_X) == 3
