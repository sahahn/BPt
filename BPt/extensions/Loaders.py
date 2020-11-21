from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import warnings
import networkx as nx
from scipy.linalg import lstsq


class Identity(BaseEstimator, TransformerMixin):

    def __init__(self):
        '''This loader will simply flatten the input object, if not already.
        This loader is used to for example pass along loaded surfaces
        or volumes to a PCA or other simmilar transformation.'''
        pass

    def fit(self, X, y=None):
        pass

    def fit_transform(self, X, y=None):

        return self.transform(X)

    def transform(self, X):

        return X.flatten()


def load_surf(surf):
    '''Helper function to load a surface within BPt, w/ appropriate
    checks for important'''

    # If str, assume file path
    if isinstance(surf, str):

        # Only try to load with numpy if str ends with numpy ext
        if surf.endswith('.npy') or surf.endswith('.npz'):
            try:
                surf = np.load(surf)
                return surf
            except ValueError:
                pass

        # If not numpy, or numpy failed, try nilearn load surf data
        try:
            from nilearn.surface import load_surf_data
        except ImportError:
            raise ImportError('nilearn does not appear to be installed! ' +
                              'Install with "pip install nilearn", to load ' +
                              'surfaces from a file path.')

        surf = load_surf_data(surf)
        return surf

    # Keep as None for None
    elif surf is None:
        return None

    # If a Parc object
    elif hasattr(surf, 'get_parc'):
        return surf.get_parc(copy=True)

    # Otherwise assume either valid array-like passed
    else:
        return np.array(surf).copy()


class SurfLabels(BaseEstimator, TransformerMixin):

    def __init__(self, labels,
                 background_label=0,
                 mask=None,
                 strategy='mean',
                 vectorize=True):
        '''This class functions simmilar to NiftiLabelsMasker from nilearn,
        but instead is for surfaces (though it could work on a cifti
        image too).

        Parameters
        ----------
        labels : str or array-like
            This should represent an array, of the same size as the data
            dimension, as a mask
            with unique integer values for each ROI. You can also pass a str
            location in which
            to load in this array (though the saved file must be loadable by
            either numpy.load, or
            if not a numpy array, will try and load with
            nilearn.surface.load_surf_data(), which you
            will need nilearn installed to use.)

        background_labels : int, array-like of int, optional
            This parameter determines which label, if any,
            in the corresponding
            passed labels, should be treated as 'background'
            and therefore no ROI
            calculated for that value or values.
            You may pass either a single interger
            value, an array-like of integer values.

            If not background label is desired, just pass
            a label which doesn't exist in any of the data,
            e.g., -100.

            ::

                default = 0

        mask : None, str or array-like, optional
            This parameter allows you to optional pass a mask of values in
            which to not calculate ROI values for.
            This can be passed as a str or
            array-like of values (just like labels),
            and should be comprised of
            a boolean array (or 1's and 0's),
            where a value of 1 means that value
            will be ignored (set to background label)
            should be kept, and a value of 0,
            for that value should be masked away.
            This array should have the same length as the passed `labels`.

            ::

                default = None

        strategy: specific str, custom_func, optional
            This parameter dictates the function to be applied
            to each data's ROI's
            individually, e.g., mean to calculate the mean by ROI.

            If a str is passed, it must correspond to one of the below preset
            options:

            - 'mean'
                Calculate the mean with np.mean

            - 'sum'
                Calculate the sum with np.sum

            - 'min' or 'minimum
                Calculate the min value with np.min

            - 'max' or 'maximum
                Calculate the max value with np.max

            - 'std' or 'standard_deviation'
                Calculate the standard deviation with np.std

            - 'var' or  'variance'
                Calculate the variance with np.var

            If a custom function is passed, it must accept two arguments,
            custom_func(X_i, axis=data_dim), X_i, where X_i is a subjects data
            array where that subjects data corresponds to
            labels == some class i, and can potentially
            be either a 1D array or 2D array, and an axis argument
            to specify which axis is
            the data dimension (e.g., if calculating for a time-series
            [n_timepoints, data_dim], then data_dim = 1,
            if calculating for say stacked contrasts where
            [data_dim, n_contrasts], data_dim = 0, and lastly for a 1D
            array, data_dim is also 0.

            ::

                default = 'mean'

        vectorize : bool, optional
            If the returned array should be flattened to 1D. E.g., if the
            last step in a set of loader steps this should be True, if before
            a different step it may make sense to set to False.

            ::

                default = True
        '''

        self.labels = labels
        self.background_label = background_label
        self.mask = mask
        self.strategy = strategy
        self.vectorize = vectorize

    def fit(self, X, y=None):

        # Load mask if any
        self.mask_ = load_surf(self.mask)

        # Load labels
        self.labels_ = load_surf(self.labels)

        # Mask labels if mask
        if self.mask_ is not None:

            # Raise error if wrong shapes
            if len(self.mask_) != len(self.labels_):
                raise RuntimeError('length of mask must have '
                                   'the same length / shape as '
                                   'the length of labels!')
            self.labels_[self.mask_.astype(bool)] = self.background_label

        # X can either be a 1D surface, or a 2D surface
        # (e.g. - for timeseries or stacked contrasts)
        if len(X.shape) > 2:
            raise RuntimeError('X can be at most 2D.')

        if len(self.labels_) not in X.shape:
            raise RuntimeError('Size of labels not found in X. '
                               'Make sure your data is in the same '
                               'space as the labels you are using!')

        # Proc self.background_label, if int turn to np array
        if isinstance(self.background_label, int):
            self.background_label_ = np.array([self.background_label])

        # Otherwise, if already list-like, just cast to np array
        else:
            self.background_label_ = np.array(self.background_label)

        # Set the _non_bkg_unique as the valid labels to get ROIs for
        self.non_bkg_unique_ = np.setdiff1d(np.unique(self.labels_),
                                            self.background_label_)

        # Proc strategy if need be
        strats = {'mean': np.mean,
                  'median': np.median,
                  'sum': np.sum,
                  'minimum': np.min,
                  'min': np.min,
                  'maximum': np.max,
                  'max': np.max,
                  'standard_deviation': np.std,
                  'std': np.std,
                  'variance': np.var,
                  'var': np.var}

        if self.strategy in strats:
            self.strategy_ = strats[self.strategy]
        else:
            self.strategy_ = self.strategy

        return self

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def _check_fitted(self):
        if not hasattr(self, "labels_"):
            raise ValueError('It seems that SurfLabels has not been fitted. '
                             'You must call fit() before calling transform()')

    def transform(self, X):
        ''' If X has the both the same dimensions, raise warning'''

        self._check_fitted()

        if len(X.shape) == 2 and (X.shape[0] == X.shape[1]):
            warnings.warn('X was passed with the same length',
                          ' in each dimension, ',
                          'Assuming that axis=0 is the data dimension',
                          ' w/ vertex values')

        # The data dimension is just the dimension with
        # the same len as the label
        self.data_dim_ = X.shape.index(len(self.labels_))
        self.X_shape_ = X.shape

        # Get the ROI value for each label
        X_trans = []
        for i in self.non_bkg_unique_:

            if self.data_dim_ == 0:
                X_i = X[self.labels_ == i]
            else:
                X_i = X[:, self.labels_ == i]

            X_trans.append(self.strategy_(X_i, axis=self.data_dim_))

        if self.data_dim_ == 1:
            X_trans = np.stack(X_trans, axis=1)
        else:
            X_trans = np.array(X_trans)

        # Return based on vectorizes
        if not self.vectorize:
            return X_trans

        # Save original shape if vectorize called,
        # used for reverse transform
        self.original_shape_ = X_trans.shape
        return X_trans.flatten()

    def inverse_transform(self, X_trans):

        # Reverse the vectorize
        if self.vectorize:
            X_trans = X_trans.reshape(self.original_shape_)

        X = np.zeros(self.X_shape_, dtype=X_trans.dtype, order='C')

        if self.data_dim_ == 1:
            X_trans = np.rollaxis(X_trans, -1)
            X = np.rollaxis(X, -1)

        for i, label in enumerate(self.non_bkg_unique_):
            X[self.labels_ == label] = X_trans[i]

        if self.data_dim_ == 1:
            X = np.rollaxis(X, -1)

        return X


class SurfMaps(BaseEstimator, TransformerMixin):

    def __init__(self, maps, strategy='auto', mask=None, vectorize=True):
        '''Simmilar to NiftiMapsMasker from nilearn
        but for surfaces, and designed to work with BPt Loader.

        This object calculates the signal for each of the
        passed maps as extracted from the input during fit,
        and returns for each map a value.

        maps : str or array-like, optional
            This parameter represents the maps in which
            to apply to each surface, where the shape of
            the passed maps should be (# of vertex, # of maps)
            or in other words, the size of the data array in the first
            dimension and the number of maps
            (i.e., the number of outputted ROIs from fit)
            as the second dimension.

            You may pass maps as either an array-like,
            or the str file location of a numpy or other
            valid surface file format array in which to load.

        strategy : {'auto', 'ls', 'average'}, optional
            The stratgey in which the maps are used to extract
            signal. If 'ls' is selected, which stands for
            least squares, the least-squares solution will
            be used for each region.

            Alternatively, if 'average' is passed, then
            the weighted average value for each map
            will be computed.

            By default 'auto' will be selected,
            which will use 'average' if the passed
            maps contain only positive weights, and
            'ls' in the case that there are
            any negative values in the passed maps.

            Otherwise, you can set a specific strategy.
            In deciding which method to use,
            consider an example. Let's say the
            fit data X, and maps are

            ::

                data = np.array([1, 1, 5, 5])
                maps = np.array([[0, 0],
                                 [0, 0],
                                 [1, -1],
                                 [1, -1]])

            In this case, the 'ls' method would
            yield region signals [2.5, -2.5], whereas
            the weighted 'average' method, would yield
            [5, 5], notably ignoring the negative weights.
            This highlights an important limitation to the
            weighted averaged method, as it does not
            handle negative values well.

            On the other hand, consider changing the maps
            weights to

            ::

                data = np.array([1, 1, 5, 5])
                maps = np.array([[0, 1],
                                 [0, 2],
                                 [1, 0],
                                 [1, 0]])

                ls_sol = [5. , 0.6]
                average_sol = [5, 1]

            In this case, we can see that the weighted
            average gives a maybe more intuative summary
            of the regions. In general, it depends on
            what signal you are trying to summarize, and
            how you are trying to summarize it.

        mask : None, str or array-like, optional
            This parameter allows you to optional pass a mask of values in
            which to not calculate ROI values for.
            This can be passed as a str or
            array-like of values (just like maps),
            and should be comprised of
            a boolean array (or 1's and 0's),
            where a value of 1 means that value
            will be ignored (set to 0)
            should be kept, and a value of 0,
            for that value should be masked away.
            This array should have the same length as the passed `maps`.
            Specifically, where the shape of maps is (size, n_maps),
            the shape of mask should be (size).

            ::

                default = None

        vectorize : bool, optional
            If the returned array should be flattened to 1D. E.g., if this is
            the last step in a set of loader steps this should be True.
            Also note, if the surface data it is being applied to is 1D,
            then the output will be 1D regardless of this parameter.

            ::

                default = True
        '''

        self.maps = maps
        self.strategy = strategy
        self.mask = mask
        self.vectorize = vectorize

    def fit(self, X, y=None):

        # Load mask if any
        self.mask_ = load_surf(self.mask)

        # Load maps
        self.maps_ = load_surf(self.maps)

        # Make the maps if passed mask set to 0 in those spots
        if self.mask_ is not None:

            # Raise error if wrong shapes
            if len(self.mask_) != self.maps_.shape[0]:
                raise RuntimeError('length of mask must have '
                                   'the same length / shape as '
                                   'the first dimension of passed '
                                   'maps!')
            self.maps_[self.mask_.astype(bool)] = 0

        # X can either be a 1D surface, or a 2D surface
        # (e.g. - for timeseries or stacked contrasts)
        if len(X.shape) > 2:
            raise RuntimeError('X can be at most 2D.')

        # Check to make sure dimension of data is correct
        if self.maps_.shape[0] not in X.shape:
            raise RuntimeError('Size of labels not found in X. '
                               'Make sure your data is in the same '
                               'space as the labels you are using!')

        # Make sure strategy exists
        if self.strategy not in ['auto', 'ls', 'average']:
            raise RuntimeError('strategy must be '
                               '"ls", "average" or "auto"!')

        return self

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def _check_fitted(self):
        if not hasattr(self, "maps_"):
            raise ValueError('It seems that SurfMaps has not been fitted. '
                             'You must call fit() before calling transform()')

    def transform(self, X):

        self._check_fitted()

        if len(X.shape) == 2 and (X.shape[0] == X.shape[1]):
            warnings.warn('X was passed with the same length',
                          ' in each dimension, ',
                          'Assuming that axis=0 is the data dimension',
                          ' w/ vertex values')

        # The data dimension is just the dimension with
        # the first dimension of maps
        self.data_dim_ = X.shape.index(self.maps_.shape[0])
        self.X_shape_ = X.shape

        # If data in second dimension, transpose
        if self.data_dim_ == 1:
            X = X.T

        # Set strategy if auto
        strategy = self.strategy
        if strategy == 'auto':
            strategy = 'ls'
            if np.all(self.maps_ >= 0):
                strategy = 'average'

        # Run the correct transform based on strategy
        if strategy == 'ls':
            X_trans = self._transform_ls(X)
        elif strategy == 'average':
            X_trans = self._transform_average(X)
        else:
            X_trans = None

        # Always return as shape of extra data if any by
        # number of maps, with number of maps as last dimension

        # Return based on vectorize
        if not self.vectorize:
            return X_trans

        # Save original transformed output shape if vectorize
        self.original_shape_ = X_trans.shape
        return X_trans.flatten()

    def _transform_ls(self, X):
        '''X should be data points, or data points x stacked.'''

        X_trans = lstsq(self.maps_, X)[0]
        return X_trans.T

    def _transform_average(self, X):
        '''X should be data points, or data points x stacked.'''

        X_trans = []

        # For each map - take weighted average
        for m in range(self.maps_.shape[1]):

            try:
                X_trans.append(np.average(X, axis=0, weights=self.maps_[:, m]))
            except ZeroDivisionError:
                pass

        return np.array(X_trans).T

    def inverse_transform(self, X_trans):

        # Reverse the vectorize, if needed
        if self.vectorize:
            X_trans = X_trans.reshape(self.original_shape_)

        # For ls should be something like
        # np.dot(region_signals, maps.T)

        # For inverse transform weighted average,
        # make sure to include check for sum(weights) == 0
        # and to fill in that spot with zeros

        raise RuntimeError('Not Implemented Yet')


# Create wrapper for nilearn connectivity measure to make it
# work with 1 subject
try:
    from nilearn.connectome import ConnectivityMeasure

    class Connectivity(ConnectivityMeasure):

        def proc_X(self, X):

            if not isinstance(X, list):
                if len(np.shape(X)) == 2:
                    return [X]

            return X

        def fit(self, X, y=None):
            return super().fit(self.proc_X(X), y)

        def fit_transform(self, X, y=None):
            return super().fit_transform(self.proc_X(X), y)

        def transform(self, X):
            return super().transform(self.proc_X(X))

except ImportError:
    pass


class Networks(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=.2, threshold_method='abs',
                 to_compute='avg_degree'):

        self.threshold = threshold
        self.threshold_method = threshold_method
        self.to_compute = to_compute

    def fit(self, X, y=None):
        '''X is a 2d correlation matrix'''

        if isinstance(self.to_compute, str):
            self.to_compute = [self.to_compute]

        try:
            import networkx
        except ImportError:
            raise ImportError(
                'To use this class, make sure you have networkx installed!')
        return self

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def _apply_threshold(self, X):

        if self.threshold_method == 'abs':
            return np.where(np.abs(X) >= self.threshold, 1, 0)
        elif self.threshold_method == 'pos':
            return np.where(X[0] >= self.threshold, 1, 0)
        elif self.threshold_method == 'neg':
            return np.where(X[0] <= self.threshold, 1, 0)
        elif self.threshold_method == 'density':
            top_n = round((len(np.triu(X).flatten())-len(X))/2*self.threshold)
            thres = sorted(np.triu(X).flatten(), reverse=True)[top_n]
            return np.where(X >= thres, 1, 0)

    def _threshold_check(self, X):

        while np.sum(self._apply_threshold(X)) == 0:
            print('warning setting threshold lower', self.threshold)
            self.threshold -= .01

    def transform(self, X):

        # Squeeze X
        X = np.squeeze(X)

        # Make sure the specified threshold doesn't break everything
        self._threshold_check(X)

        # Apply threshold
        X = self._apply_threshold(X)
        G = nx.from_numpy_matrix(X)

        func_dict = {'avg_cluster': nx.average_clustering,
                     'assortativity': nx.degree_assortativity_coefficient,
                     'global_eff': nx.global_efficiency,
                     'local_eff': nx.local_efficiency,
                     'transitivity': nx.transitivity,
                     'avg_eigenvector_centrality':
                     self._avg_eigenvector_centrality,
                     'avg_closeness_centrality':
                     self._avg_closeness_centrality,
                     'avg_degree': self._avg_degree,
                     'avg_triangles': self._avg_triangles,
                     'avg_pagerank': self._avg_pagerank,
                     'avg_betweenness_centrality':
                     self._avg_betweenness_centrality,
                     'avg_information_centrality':
                     self._avg_information_centrality,
                     'avg_shortest_path_length':
                     nx.average_shortest_path_length
                     }

        X_trans = []
        for compute in self.to_compute:
            X_trans += [func_dict[compute](G)]
        return np.array(X_trans)

    # Local
    def _avg_degree(self, G):
        avg_degree = np.mean([i[1] for i in nx.degree(G)])
        return avg_degree

    def _avg_triangles(self, G):
        avg_triangles = np.mean([nx.triangles(G)[i] for i in nx.triangles(G)])
        return avg_triangles

    def _avg_eigenvector_centrality(self, G):
        avg_eigenvector_centrality =\
            np.mean([nx.eigenvector_centrality_numpy(G)[
                     i] for i in nx.eigenvector_centrality_numpy(G)])
        return avg_eigenvector_centrality

    def _avg_closeness_centrality(self, G):
        avg_closeness_centrality = np.mean(
            [nx.closeness_centrality(G)[i] for i in nx.closeness_centrality(G)])
        return avg_closeness_centrality

    def _avg_betweenness_centrality(self, G):
        avg_betweenness_centrality = np.mean(
            [nx.betweenness_centrality(G)[i]
             for i in nx.betweenness_centrality(G)])
        return avg_betweenness_centrality

    def _avg_information_centrality(self, G):
        avg_information_centrality = np.mean(
            [nx.information_centrality(G)[i]
             for i in nx.information_centrality(G)])
        return avg_information_centrality

    def _avg_pagerank(self, G):
        avg_pagerank = np.mean([nx.pagerank(G)[i] for i in nx.pagerank(G)])
        return avg_pagerank
