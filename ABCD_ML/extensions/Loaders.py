from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import warnings


class Identity(BaseEstimator, TransformerMixin):

    def __init__(self):
        '''This loader simply flatten the input array and passed it along'''
        pass

    def fit(self, X, y=None):
        pass

    def fit_transform(self, X, y=None):
        
        return self.transform(X)

    def transform(self, X):
        
        return X.flatten()


def load_surf(surf):
    '''Helper function to load a surface within ABCD_ML, w/ appropriate
    checks for important'''

    # If str, assume file path
    if isinstance(surf, str):

        try:
            surf = np.load(surf)
            return surf
        except ValueError:
            pass

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

    # Otherwise assume either valid array-like passed or a Parc object
    else:

        try:
            return surf.get_parc(copy=True)
        except AttributeError:
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

        background_labels : int, array-like of int or None, optional
            This parameter determines which label, if any, in the corresponding
            passed labels, should be treated as 'background' and therefore no ROI
            calculated for that value or values. You may pass either a single interger
            value, an array-like of integer values, or None, to calculate ROIs for everything.

            (default = 0)

        mask : None, str or array-like, optional
            This parameter allows you to optional pass a mask of values in
            which to not calculate ROI values for. This can be passed as a str or
            array-like of values (just like labels), and should be comprised of
            a boolean array (or 1's and 0's), where a value of 1 means that value
            should be kept, and a value of 0, for that value should be masked away.
            This array should have the same length as the passed `labels`.

            (default = None)

        strategy: specific str, custom_func, optional
            This parameter dictates the function to be applied to each data's ROI's
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
            custom_func(X_i, axis=data_dim), X_i, where X_i is a subjects data array
            where that subjects data corresponds to labels == some class i, and can potentially
            be either a 1D array or 2D array, and an axis argument to specify which axis is
            the data dimension (e.g., if calculating for a time-series [n_timepoints, data_dim], then data_dim = 1,
            if calculating for say stacked contrasts where [data_dim, n_contrasts], data_dim = 0, and lastly for a 1D
            array, data_dim is also 0.

            (default = 'mean')

        vectorize : bool, optional
            If the returned array should be flattened to 1D. E.g., if the
            last step in a set of loader steps this should be True, if before
            a different step it may make sense to set to False.

            (default = True)
        '''
        
        self.labels = labels
        self.background_label = background_label
        self.mask = mask
        self.strategy = strategy
        self.vectorize = vectorize
        
        self.strats = {'mean': np.mean,
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
        
    def fit(self, X, y=None):
        
        # Load mask if any
        self.mask_ = load_surf(self.mask)
        
        # Load labels
        self.labels_ = load_surf(self.labels)
        
        # Mask labels if mask
        if self.mask_ is not None:
            self.mask_ = self.mask_.astype(bool)
            self.labels_[self.mask_] = self.background_label
        
        # X can either be a 1D surface, or a 2D surface (e.g. - for timeseries or stacked contrasts)
        if len(X.shape) > 2:
            raise RuntimeError('X can be at most 2D.')
        
        if len(self.labels_) not in X.shape:
            raise RuntimeError('Size of labels not found in X. '+ \
                               'Make sure your data is in the same standard space ' + \
                               'as the labels you are using!')
            
        # Proc self.background_label, if int turn to np array
        if isinstance(self.background_label, int):
            self._background_label = np.array([self.background_label])

        # If None...
        elif self.background_label is None:
            self._background_label = np.array([])
        
        # Otherwise, if already list-like, just cast to np array
        else:
            self._background_label = np.array(self.background_label)
        
        
            
        # Set the _non_bkg_unique as the valid labels to get ROIs for
        self._non_bkg_unique = np.setdiff1d(np.unique(self.labels_),
                                            self._background_label)
            
        # Proc strategy if need be
        if self.strategy in self.strats:
            self.strategy = self.strats[self.strategy]

        return self
            
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
        
    def _check_fitted(self):
        if not hasattr(self, "labels_"):
            raise ValueError('It seems that SurfLabels has not been fitted. '
                             'You must call fit() before calling transform()')
            
    def transform(self, X):
        ''' If X has the both the same dimensions, raise warning'''
        
        if len(X.shape) == 2 and (X.shape[0] == X.shape[1]):
            warnings.warn('X was passed with the same length in each dimension, '
                          'Assuming that axis=0 is that data dimension w/ vertex values')
        
        # The data dimension is just the dimension with the same len as the label
        self.data_dim_ = X.shape.index(len(self.labels_))
        self.X_shape_ = X.shape
        
        # Get the ROI value for each label
        X_trans = []
        for i in self._non_bkg_unique:

            if self.data_dim_ == 0:
                X_i = X[self.labels_ == i]
            else:
                X_i = X[:, self.labels_ == i]

            X_trans.append(self.strategy(X_i, axis=self.data_dim_))

        if self.data_dim_ == 1:
            X_trans = np.stack(X_trans, axis=1)
        else:
            X_trans = np.array(X_trans)

        # Return based on vectorizes
        if not self.vectorize:
            return X_trans
        
        self.o_shape_ = X_trans.shape
        return X_trans.flatten()

    def reverse_transform(self, X_trans):
        
        # Reverse the vectorize
        if self.vectorize:
            X_trans = X_trans.reshape(self.o_shape_)
            
        X = np.zeros(self.X_shape_,
                     dtype=X_trans.dtype, order='C')
        
        if self.data_dim_ == 1:
            X = np.rollaxis(X, -1)
            X_trans = np.rollaxis(X_trans, -1)

        for i, label in enumerate(self._non_bkg_unique):
            X[self.labels_ == label] = X_trans[i]
            
        if self.data_dim_ == 1:
            X = np.rollaxis(X, -1)
                
        return X


# Create wrapper for nilearn connectivity measure to make it work with 1 subject
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

