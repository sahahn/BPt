from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

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
            raise ImportError('nilearn does not appear to be installed!' + \
                                'Install with "pip install nilearn", to load ' + \
                                'surfaces from a file path.')
        
        surf = load_surf_data(surf)
        return surf

    # Keep as None for None
    elif surf is None:
        return None

    # Otherwise assume valid array-like passed
    else:

        # Cast to numpy array
        surf = np.array(surf)
        
        # Return copy to be safe
        return surf.copy()


class SurfLabels(BaseEstimator, TransformerMixin):
    
    def __init__(self, labels,
                 background_label=0,
                 mask=None,
                 strategy='mean'):
        '''This class functions simmilar to NiftiLabelsMasker from nilearn,
        but instead is for surfaces (though it could likely work on a cifti image too).
        
        Parameters
        ----------
        labels : str or array-like
            This should represent an array, of the same size as the data dimension, as a mask
            with unique integer values for each ROI. You can also pass a str location in which
            to load in this array (though the saved file must be loadable by either numpy.load, or
            if not a numpy array, will try and load with nilearn.surface.load_surf_data, which you
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
        '''
        
        self.labels = labels
        self.background_label = background_label
        self.mask = mask
        self.strategy = strategy
        
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
        
        # The data dimension is just the dimension with the same len as the label
        data_dim = X.shape.index(len(self.labels_))
        
        # Get the ROI value for each label
        X_trans = []
        for i in self._non_bkg_unique:

            if data_dim == 0:
                X_i = X[self.labels_ == i]
            else:
                X_i = X[:, self.labels_ == i]

            X_trans.append(self.strategy(X_i, axis=data_dim))

        # Return a 1D array!
        return np.array(X_trans).flatten()