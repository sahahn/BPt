from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import warnings
import networkx as nx
from scipy.linalg import lstsq


class Identity(BaseEstimator, TransformerMixin):
    '''This loader will simply flatten the input object, if not already.
    This loader is used to for example pass along loaded surfaces
    or volumes to a PCA or other similar transformation.

    This object is designed to be used with input class :class:`BPt.Loader`
    for operating on single subjects at a time.
    '''

    def __init__(self):
        pass

    def fit(self, X, y=None):
        '''Fit accepts anything for X,
        and doesn't do anything except save the
        original shape of X.

        Parameters
        ----------
        X : numpy.array
            numpy.array with any shape, for one subject.

        y : numpy.array, optional
            This parameter is skipped.

            ::

                default = None

        '''
        self.X_shape_ = X.shape
        return self

    def fit_transform(self, X, y=None):
        '''Calls fit then transform, and returns the transformed output.

        Parameters
        ----------
        X : numpy.array
            numpy.array with any shape, for one subject.

        y : numpy.array, optional
            This parameter is skipped, it exists
            for compatibility.

            ::

                default = None

        '''
        return self.fit(X).transform(X)

    def transform(self, X):
        '''Transform simply returns a flattened version of the passed
        X, making it compatible with downstream classifiers.

        Parameters
        ----------
        X : numpy.array
            numpy.array with any shape, for one subject.

        Returns
        ---------
        X_trans : numpy.array
            1D flattened array for this subject.

        '''
        return X.flatten()

    def inverse_transform(self, X_trans):
        '''Inverse transform, i.e., un-flatten.

        Parameters
        ----------
        X_trans : numpy.array
            1D transformed numpy.array.

        Returns
        ---------
        X : numpy.array
            Data in original shape.

        '''

        return X_trans.reshape(self.X_shape_)


def load_surf(surf):
    '''Helper function to load a surface within BPt, w/ appropriate
    checks for important'''

    # If str, assume file path
    if isinstance(surf, str):

        # Only try to load with numpy if str ends with numpy ext
        if surf.endswith('.npy') or surf.endswith('.npz'):
            surf = np.load(surf)
            return surf

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
    '''Extract signals from non-overlapping labels.

    This class functions similar to
    :class:`nilearn.input_data.NiftiLabelsMasker`, except
    it is designed to work for surface, cifti or any arbitrary
    1 or 2D numpy arrays.

    Parameters
    -----------
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
            Calculate the mean with :func:`numpy.mean`

        - 'sum'
            Calculate the sum with :func:`numpy.sum`

        - 'min' or 'minimum
            Calculate the min value with :func:`numpy.min`

        - 'max' or 'maximum
            Calculate the max value with :func:`numpy.max`

        - 'std' or 'standard_deviation'
            Calculate the standard deviation with :func:`numpy.std`

        - 'var' or  'variance'
            Calculate the variance with :func:`numpy.var`

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

    See Also
    -----------
    SurfMaps : For extracting non-static / probabilistic parcellations.
    nilearn.input_data.NiftiLabelsMasker : For working with volumetric data.

    '''

    def __init__(self, labels,
                 background_label=0,
                 mask=None,
                 strategy='mean',
                 vectorize=True):

        self.labels = labels
        self.background_label = background_label
        self.mask = mask
        self.strategy = strategy
        self.vectorize = vectorize

    def fit(self, X, y=None):
        '''Fit this object according
        the passed subjects data, X.

        Parameters
        ----------
        X : numpy.array
            numpy.array as either a 1D array,
            or a 2D array, where one dimension
            is matches the shape of the past
            labels and the other represents
            different time-points or modalities.

        y : numpy.array, optional
            This parameter is skipped.

            ::

                default = None
        '''

        # Load mask if any
        self.mask_ = load_surf(self.mask)

        # Load labels
        self.labels_ = load_surf(self.labels)

        if len(self.labels_.shape) > 1:
            raise RuntimeError('The passed labels array must be flat, ' +
                               ' i.e., a single dimension')

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
        '''Fit, then transform this object.

        Parameters
        ----------
        X : numpy.array
            numpy.array as either a 1D array,
            or a 2D array, where one dimension
            is matches the shape of the past
            labels and the other represents
            different time-points or modalities.

        y : numpy.array, optional
            This parameter is skipped.

            ::

                default = None

        Returns
        ---------
        X_trans : numpy.array
            The transformed data, either as 1D
            array is passed 1D data, or 2D if passed
            vectorize=False and originally 2D data.
        '''
        return self.fit(X, y).transform(X)

    def _check_fitted(self):
        if not hasattr(self, "labels_"):
            raise ValueError('It seems that SurfLabels has not been fitted. '
                             'You must call fit() before calling transform()')

    def transform(self, X):
        '''Transform this the passed data.

        If X has the both the same dimension's, raise warning.

        Parameters
        ----------
        X : numpy.array
            numpy.array as either a 1D array,
            or a 2D array, where one dimension
            is matches the shape of the past
            labels and the other represents
            different time-points or modalities.

        y : numpy.array, optional
            This parameter is skipped.

            ::

                default = None

        Returns
        ---------
        X_trans : numpy.array
            The transformed data, either as 1D
            array is passed 1D data, or 2D if passed
            vectorize=False and originally 2D data.
        '''

        self._check_fitted()

        if len(X.shape) == 2 and (X.shape[0] == X.shape[1]):
            warnings.warn('X was passed with the same length' +
                          ' in each dimension, ' +
                          'Assuming that axis=0 is the data dimension' +
                          ' w/ vertex values')

        if len(X.shape) > 2:
            raise RuntimeError('The shape of X can be at most 2 dimensions.')

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
        '''Reverse the original transformation.

        Parameters
        ----------
        X_trans : numpy.array
            Data with the same number of outputted
            features as data transformed
            by this object, e.g., the original
            transformed data or corresponding
            feature importances.

        Returns
        --------
        X : numpy.array
            The reverse transformed data
            passed back in its original space.

        '''

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
    '''Extract signals from overlapping labels.

    This class functions similar to
    :class:`nilearn.input_data.NiftiMapsMasker`, except
    it is designed to work for surface, cifti or any arbitrary
    1 or 2D numpy arrays.

    This object calculates the signal for each of the
    passed maps as extracted from the input during fit,
    and returns for each map a value.

    Parameters
    -----------
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
        The strategy in which the maps are used to extract
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
        average gives a maybe more intuitive summary
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

    See Also
    -----------
    SurfLabels : For extracting static / non-probabilistic parcellations.
    nilearn.input_data.NiftiMapsMasker : For volumetric nifti data.

    Examples
    ----------
    First let's define an example set of probabilistic maps, we
    will assume there are just 5 features in our data, and we will
    define 6 total maps.

    .. ipython:: python

        import numpy as np
        from BPt.extensions import SurfMaps

        # This should have shape number of features x number of maps!
        prob_maps = np.array([[3, 1, 1, 1, 1, 1],
                              [1, 3, 1, 1, 1, 1],
                              [1, 1, 3, 1, 1, 1],
                              [1, 1, 1, 3, 1, 1],
                              [1, 1, 1, 1, 3, 1]])
        prob_maps.shape

    Next we can define some input data to use with these maps.

    .. ipython:: python

        data1 = np.arange(5, dtype='float')
        data1
        data2 = np.ones(5, dtype='float')
        data2

    Now let's define the actual object and use it to transform the data.

    .. ipython:: python

        sm = SurfMaps(maps=prob_maps)
        sm.fit_transform(data1)
        sm.fit_transform(data2)

    Okay so what is going on when we transform this data? Basically we are
    just taking weighted averages for each one of the defined maps. We could
    also explicitly change the strategy from 'auto' to 'ls' which would
    take the least squares solution instead.

    .. ipython:: python

        sm = SurfMaps(maps=prob_maps, strategy='ls')
        data_trans = sm.fit_transform(data1)
        data_trans

    While a little less intuitive, the least squares solution allows
    us to reverse the feature transformation (although not always exactly)

    .. ipython:: python

        sm.inverse_transform(data_trans)

    This can be useful in the say the case of converting back downstream
    calculated feature importance to the original data space.
    '''

    def __init__(self, maps, strategy='auto', mask=None, vectorize=True):
        self.maps = maps
        self.strategy = strategy
        self.mask = mask
        self.vectorize = vectorize

    def fit(self, X, y=None):
        '''Fit this object according
        the passed subjects data, X.

        Parameters
        ----------
        X : numpy.array
            numpy.array as either a 1D array,
            or a 2D array, where one dimension
            is matches the shape of the past
            labels and the other represents
            different time-points or modalities.

        y : numpy.array, optional
            This parameter is skipped.

            ::

                default = None
        '''

        # Load mask if any
        self.mask_ = load_surf(self.mask)

        # Load maps
        self.maps_ = load_surf(self.maps)

        # Save dtype
        self.dtype_ = X.dtype

        # Warn if non-float
        if 'float' not in self.dtype_.name:
            warnings.warn('The original datatype is non-float, ' +
                          'this may lead to rounding errors! ' +
                          'Pass data as type float to ensure ' +
                          'the results of transform are not truncated.')

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
        '''Fit, then transform this object.

        Parameters
        ----------
        X : numpy.array
            numpy.array as either a 1D array,
            or a 2D array, where one dimension
            is matches the shape of the past
            labels and the other represents
            different time-points or modalities.

        y : numpy.array, optional
            This parameter is skipped.

            ::

                default = None

        Returns
        ---------
        X_trans : numpy.array
            The transformed data, either as 1D
            array is passed 1D data, or 2D if passed
            vectorize=False and originally 2D data.
        '''

        return self.fit(X, y).transform(X)

    def _check_fitted(self):
        if not hasattr(self, "maps_"):
            raise ValueError('It seems that SurfMaps has not been fitted. '
                             'You must call fit() before calling transform()')

    def transform(self, X):
        '''Transform this the passed data.

        If X has the both the same dimension's, raise warning.

        Parameters
        ----------
        X : numpy.array
            numpy.array as either a 1D array,
            or a 2D array, where one dimension
            is matches the shape of the past
            labels and the other represents
            different time-points or modalities.

        y : numpy.array, optional
            This parameter is skipped.

            ::

                default = None

        Returns
        ---------
        X_trans : numpy.array
            The transformed data, either as 1D
            array is passed 1D data, or 2D if passed
            vectorize=False and originally 2D data.
        '''

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
        self.strategy_ = self.strategy
        if self.strategy_ == 'auto':
            self.strategy_ = 'ls'
            if np.all(self.maps_ >= 0):
                self.strategy_ = 'average'

        # Run the correct transform based on strategy
        if self.strategy_ == 'ls':
            X_trans = self._transform_ls(X)
        elif self.strategy_ == 'average':
            X_trans = self._transform_average(X)
        else:
            X_trans = None

        # Convert back to original dtype
        X_trans = X_trans.astype(self.dtype_)

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
        '''Reverse the original transformation.

        Parameters
        ----------
        X_trans : numpy.array
            Data with the same number of outputted
            features as data transformed
            by this object, e.g., the original
            transformed data or corresponding
            feature importances.

        Returns
        --------
        X : numpy.array
            The reverse transformed data
            passed back in its original space.

        '''

        # Reverse the vectorize, if needed
        if self.vectorize:
            X_trans = X_trans.reshape(self.original_shape_)

        if self.strategy_ == 'ls':
            return np.dot(X_trans, self.maps_.T)

        elif self.strategy_ == 'average':
            raise RuntimeError('Cannot calculate reverse of average.')


def proc_X(X):

    if not isinstance(X, list):
        if len(np.shape(X)) == 2:
            return [X]

    return X


def proc_X_trans(X_trans, vectorize):

    if X_trans.shape[0] == 1:
        X_trans = X_trans.reshape(X_trans.shape[1:])

    if vectorize:
        X_trans = X_trans.flatten()

    return X_trans


# Create wrapper for nilearn connectivity measure to make it
# work with 1 subject
try:
    from nilearn.connectome import ConnectivityMeasure

    class SingleConnectivityMeasure(ConnectivityMeasure):
        '''| See :class:`nilearn.connectome.ConnectivityMeasure`.
          This class is just a wrapper to let this object work when passed
          a single connectivity matrix.

        | This class requires extra dependency nilearn to be installed.
        '''

        def fit(self, X, y=None):
            return super().fit(proc_X(X), y)

        def fit_transform(self, X, y=None):
            X_trans = super().fit_transform(proc_X(X), y)
            return proc_X_trans(X_trans, self.vectorize)

        def transform(self, X):
            X_trans = super().transform(proc_X(X))
            return proc_X_trans(X_trans, self.vectorize)

except ImportError:
    pass


def avg(func):

    def mean_func(G):
        return np.mean(func(G).values())

    return mean_func


class ThresholdNetworkMeasures(BaseEstimator, TransformerMixin):
    '''This class is designed for thresholding and then extracting network
    measures from an input correlation matrix.

    Parameters
    -----------
    threshold : float, optional
        A floating point threshold between 0 and 1.
        This represents the threshold at which a connection
        in the passed data needs to be in order for it to
        be set as an edge. The type of threshold_method
        also changes how this threshold behaves.

        ::

            default = .2

    threshold_type : {'abs', 'pos', 'neg'}
        The type of thresholding, e.g., should the threshold
        be applied to:

        - 'abs'
            The absolute value of the connections

        - 'pos'
            Only consider edges as passed the threshold if >= self.threshold

        - 'neg'
            Only consider edges as passed the if <= self.threshold

        ::

            default = 'abs'

    threshold_method : {'value', 'density'}, optional
        The method for thresholding. The two
        defined options are either to define an edge
        strictly by value, e.g., if threshold_type is 'abs',
        and threshold is .2, then any connection greater than or
        equal to .2 will be set as an edge.

        Alternatively, you may specify that the threshold be
        treated as a density. What this means is that if the threshold
        is set to for example .2, then the top 20% of edges by weight
        will be set as an edge, regardless of the actual value of the edge.

        Note: The passed percentage will be considered
        out of all of the the total possible edges in the adjacency matrix.
        That includes both the upper and lower repeated triangles
        if non-directional edges.

        ::

            default = 'value'

    to_compute : valid_measure or list of, optional
        | Either a single str representing a network
            measure to compute, or a list of valid
            measures. You may also pass any custom function
            which accepts one argument G, and returns
            a value.

        | The following global measures are currently implemented
            as options:
        |


        - 'avg_cluster':
            :func:`networkx.algorithms.cluster.average_clustering`
        - 'assortativity':
            :func:`networkx.algorithms.assortativity.degree_assortativity_coefficient`
        - 'global_eff':
            :func:`networkx.algorithms.efficiency_measures.global_efficiency`
        - 'local_eff':
            :func:`networkx.algorithms.efficiency_measures.local_efficiency`
        - 'sigma':
            :func:`networkx.algorithms.smallworld.sigma`
        - 'omega`:
            :func:`networkx.algorithms.smallworld.omega`
        - 'transitivity':
            :func:`networkx.algorithms.cluster.transitivity`

        |
        | You may also select from one of the following
            averages of local measures:
        |

        - 'avg_eigenvector_centrality':
            :func:`networkx.algorithms.centrality.eigenvector_centrality_numpy`
        - 'avg_closeness_centrality':
            :func:`networkx.algorithms.centrality.closeness_centrality`
        - 'avg_degree':
            Average graph degree.
        - 'avg_triangles':
            :func:`networkx.algorithms.cluster.triangles`
        - 'avg_pagerank':
            :func:`networkx.algorithms.link_analysis.pagerank_alg.pagerank`
        - 'avg_betweenness_centrality':
            :func:`networkx.algorithms.centrality.betweenness_centrality`
        - 'avg_information_centrality':
            :func:`networkx.algorithms.centrality.information_centrality`
        - 'avg_shortest_path_length':
            :func:`networkx.algorithms.shortest_paths.generic.average_shortest_path_length`


        ::

            default = 'avg_degree'

    '''

    # @TODO DIGRAPH CASE??

    def __init__(self, threshold=.2,
                 threshold_type='abs',
                 threshold_method='value',
                 to_compute='avg_degree'):

        self.threshold = threshold
        self.threshold_type = threshold_type
        self.threshold_method = threshold_method
        self.to_compute = to_compute

    @property
    def feat_names_(self):
        '''The list of feature names returned
        by this objects transform function. This property
        is special in that it can interact with :class:`BPt.Loader`,
        passing along feature name information.
        '''
        return self._feat_names

    @feat_names_.setter
    def feat_names_(self, feat_names):
        self._feat_names = feat_names

    def fit(self, X, y=None):
        '''X is a 2d correlation matrix'''

        if isinstance(self.to_compute, str):
            self.to_compute = [self.to_compute]

        try:
            import networkx
            networkx
        except ImportError:
            raise ImportError(
                'To use this class, make sure you have networkx installed!')

        # The dictionary of valid options
        self._func_dict = {
            'avg_cluster': nx.average_clustering,
            'assortativity': nx.degree_assortativity_coefficient,
            'global_eff': nx.global_efficiency,
            'local_eff': nx.local_efficiency,
            'sigma': nx.sigma,
            'omega': nx.omega,
            'transitivity': nx.transitivity,
            'avg_eigenvector_centrality': avg(nx.eigenvector_centrality_numpy),
            'avg_closeness_centrality': avg(nx.closeness_centrality),
            'avg_degree': self._avg_degree,
            'avg_triangles': avg(nx.triangles),
            'avg_pagerank': avg(nx.pagerank),
            'avg_betweenness_centrality': avg(nx.betweenness_centrality),
            'avg_information_centrality': avg(nx.information_centrality),
            'avg_shortest_path_length': nx.average_shortest_path_length
        }

        # Compute the feat names to return once here.
        self.feat_names_ = []
        for compute in self.to_compute:
            if compute not in self._func_dict:
                self.feat_names_.append(compute.__name__)
            else:
                self.feat_names_.append(compute)

        return self

    def fit_transform(self, X, y=None):
        '''Fit, then transform a passed 2D numpy correlation matrix.

        Parameters
        ----------
        X : numpy array
            A 2D numpy array representing an input correlation
            matrix.

        Returns
        ---------
        X_trans : numpy array
            Returns a flat array of length number of
            measures in parameter to_compute, representing
            the calculated network statistics.
        '''

        return self.fit(X, y).transform(X)

    def _apply_threshold(self, X):

        # Process threshold type on copy of X
        X_t = X.copy()
        if self.threshold_type == 'abs':
            X_t = np.abs(X_t)

        # If Value
        if self.threshold_method == 'value':
            if self.threshold_type == 'neg':
                return np.where(X_t <= self.threshold, 1, 0)
            return np.where(X_t >= self.threshold, 1, 0)

        elif self.threshold_method == 'density':

            top_n = round(len(X_t) * self.threshold) - 1

            # If less than 0, set to 0
            if top_n < 0:
                top_n = 0

            reverse = False if self.threshold_type == 'neg' else True
            thresh = sorted(np.triu(X_t).flatten(), reverse=reverse)[top_n]
            return np.where(X_t >= thresh, 1, 0)

        raise RuntimeError(str(self.threshold_method) + ' not a valid.')

    def _threshold_check(self, X):

        while np.sum(self._apply_threshold(X)) == 0:
            warnings.warn('Setting threshold lower than: ' +
                          str(self.threshold) + '. As, otherwise no edges ' +
                          'will be set. New threshold = ' +
                          str(self.threshold - .01))
            self.threshold -= .01

    def transform(self, X):
        '''Transform a passed 2D numpy correlation matrix.

        Parameters
        ----------
        X : numpy array
            A 2D numpy array representing an input correlation
            matrix.

        Returns
        ---------
        X_trans : numpy array
            Returns a flat array of length number of
            measures in parameter to_compute, representing
            the calculated network statistics.
        '''

        # Make sure the specified threshold doesn't break everything
        self._threshold_check(X)

        # Apply threshold
        X = self._apply_threshold(X)
        G = nx.from_numpy_matrix(X)

        X_trans = []

        for compute in self.to_compute:
            if compute not in self._func_dict:
                X_trans += [compute(G)]
            else:
                X_trans += [self._func_dict[compute](G)]
        return np.array(X_trans)

    def _avg_degree(self, G):
        avg_degree = np.mean([i[1] for i in nx.degree(G)])
        return avg_degree
