from ..loaders import (SurfMaps, SurfLabels, Identity,
                       load_surf, SingleConnectivityMeasure,
                       ThresholdNetworkMeasures)
import numpy as np
import tempfile
import os
import pytest
import warnings


def test_identity():

    i = Identity()
    x = np.ones((10, 10))
    assert i.fit_transform(x).shape == (100, )


def test_load_surf_from_np():

    tmp = os.path.join(tempfile.gettempdir(), 'fake.npy')
    np.save(tmp, np.ones(10))

    loaded = load_surf(tmp)
    assert loaded.shape == (10,)
    os.remove(tmp)


def test_load_surf_None():

    loaded = load_surf(None)
    assert loaded is None


def test_load_surf_bad():

    with pytest.raises(FileNotFoundError):
        load_surf('not_real.npy')


def test_load_nilearn():

    try:
        import nibabel as nib
    except ImportError:
        return

    temp_loc = os.path.join(tempfile.gettempdir(), 'temp.nii')
    obj = nib.Nifti1Image(np.ones((100, 1, 1, 1)), np.eye(4))
    nib.save(obj, temp_loc)

    warnings.simplefilter('ignore')
    loaded = load_surf(temp_loc)
    assert loaded.shape == (100,)


def test_load_surf_array():

    loaded = load_surf(np.ones(5))
    assert np.sum(loaded) == 5


def test_surf_labels():

    sl = SurfLabels(labels=np.array([1, 1, 2, 2]))
    data = np.array([5, 5, 10, 10])

    Xt = sl.fit_transform(data)
    assert np.array_equal(Xt, np.array([5, 10]))

    reverse_trans = sl.inverse_transform(Xt)
    assert np.array_equal(reverse_trans, np.array([5, 5, 10, 10]))


def test_surf_labels2():

    sl = SurfLabels(labels=np.array([0, 2, 2, 2, 3, 3]))
    data = np.array([1, 2, 3, 4, 5, 6])

    Xt = sl.fit_transform(data)
    assert np.array_equal(Xt, np.array([3, 5.5]))

    reverse_trans = sl.inverse_transform(Xt)
    assert np.array_equal(reverse_trans, np.array([0, 3, 3, 3, 5.5, 5.5]))


def test_surf_labels_mask():

    labels = np.array([1, 1, 2, 2])
    data = np.array([1, 2, 3, 4])

    mask = np.array([1, 0, 0, 0])
    Xt = SurfLabels(labels=labels, mask=mask,
                    strategy='mean').fit_transform(data)

    assert np.array_equal(Xt, np.array([2, 3.5]))


def test_bad_surf_labels_mask():

    labels = np.array([1, 1, 2, 2])
    data = np.array([1, 2, 3, 4])
    mask = np.array([1, 0, 0, 0, 0])

    with pytest.raises(RuntimeError):
        SurfLabels(labels=labels, mask=mask,
                   strategy='mean').fit_transform(data)


def test_surf_labels_dif_strats():

    labels = np.array([1, 1, 2, 2])
    data = np.array([1, 2, 3, 4])

    Xt = SurfLabels(labels=labels,
                    strategy='median').fit_transform(data)
    assert np.array_equal(Xt, np.array([1.5, 3.5]))

    Xt = SurfLabels(labels=labels,
                    strategy='sum').fit_transform(data)
    assert np.array_equal(Xt, np.array([3, 7]))

    Xt = SurfLabels(labels=labels,
                    strategy='min').fit_transform(data)
    assert np.array_equal(Xt, np.array([1, 3]))

    Xt = SurfLabels(labels=labels,
                    strategy='max').fit_transform(data)
    assert np.array_equal(Xt, np.array([2, 4]))

    Xt = SurfLabels(labels=labels,
                    strategy='std').fit_transform(data)
    assert np.array_equal(Xt, np.array([.5, .5]))

    Xt = SurfLabels(labels=labels,
                    strategy='var').fit_transform(data)
    assert np.array_equal(Xt, np.array([.25, .25]))


def test_surf_data_2d():

    labels = np.array([1, 2, 1, 2])
    data = np.array([[1, 2, 3], [2, 3, 4],
                     [1, 2, 3], [2, 3, 4]])

    sl = SurfLabels(labels=labels)
    Xt = sl.fit_transform(data)
    assert np.array_equal(Xt, np.array([1, 2, 3, 2, 3, 4]))

    reverse_trans = sl.inverse_transform(Xt)
    assert np.array_equal(reverse_trans, np.array([[1, 2, 3],
                                                   [2, 3, 4],
                                                   [1, 2, 3],
                                                   [2, 3, 4]]))


def test_surf_data_2d_no_vectorize():

    labels = np.array([1, 2, 1, 2])
    data = np.array([[1, 2, 3], [2, 3, 4],
                     [1, 2, 3], [2, 3, 4]])

    sl = SurfLabels(labels=labels, vectorize=False)
    Xt = sl.fit_transform(data)
    assert np.array_equal(Xt, np.array([[1, 2, 3], [2, 3, 4]]))

    reverse_trans = sl.inverse_transform(Xt)
    assert np.array_equal(reverse_trans, np.array([[1, 2, 3],
                                                   [2, 3, 4],
                                                   [1, 2, 3],
                                                   [2, 3, 4]]))


def test_surf_labels_2d_fail():

    labels = np.array([[1, 1, 1], [2, 2, 2], [1, 1, 1], [2, 2, 2]])
    data = np.array([[1, 2, 3], [2, 3, 4], [1, 2, 3], [2, 3, 4]])

    sl = SurfLabels(labels=labels, vectorize=False)

    with pytest.raises(RuntimeError):
        sl.fit_transform(data)


def test_surf_maps_ls():

    maps = np.array([[0, 1],
                     [0, 2],
                     [1, 0],
                     [1, 0]])

    data = np.array([1.0, 1.0, 5.0, 5.0])
    data_dtype = data.dtype.name

    sm = SurfMaps(maps=maps, strategy='ls')
    Xt = sm.fit_transform(data)

    assert data_dtype == Xt.dtype.name
    np.testing.assert_almost_equal(Xt, np.array([5, .6]), decimal=6)

    # Reverse transform
    r_trans = sm.inverse_transform(Xt)
    np.array_equal(r_trans, np.array([.6, 1.2, 5, 5]))


def test_surf_maps_average():

    maps = np.array([[1, 0],
                     [1, 0],
                     [0, 1],
                     [0, 2]])

    data = np.array([1.0, 3.0, 1.0, 2.0])
    data_dtype = data.dtype.name

    sm = SurfMaps(maps=maps, strategy='average')
    Xt = sm.fit_transform(data)

    assert data_dtype == Xt.dtype.name
    np.array_equal(Xt, np.array([2, 1.66666667]))


def test_surf_maps_average_fail():

    maps = np.array([[1, 0],
                     [1, 0],
                     [0, 1],
                     [0, 2]])

    data = np.array([1.0, 3.0, 1.0, 2.0])

    sm = SurfMaps(maps=maps, strategy='average')
    Xt = sm.fit_transform(data)

    with pytest.raises(RuntimeError):
        sm.inverse_transform(Xt)


def test_single_connectivity_measure():

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
    G = nx.from_numpy_matrix(thresh_X)

    degrees = [n[1] for n in G.degree()]
    X_trans = nm.transform(X)
    assert len(X_trans) == 1
    assert np.mean(degrees) == X_trans[0]


def test_threshold_network_measures_neg():

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

    nm =\
        ThresholdNetworkMeasures(threshold=.2,
                                 threshold_type='abs',
                                 threshold_method='density')

    X = np.array([[.1, .2, .3, -.4],
                  [.2, 0, -.5, 0],
                  [.3, -.5, 0, 0],
                  [-.4, 0, 0, 0]])

    thresh_X = nm._apply_threshold(X)
    assert np.sum(thresh_X) == 2
