from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import warnings


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
        
        if isinstance(X, list):
            self.n_subjects_ = len(X)
            self.X_shape_ = X[0].shape
        else:
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
        if isinstance(X, list):
            return np.array([x.flatten() for x in X])
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

    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
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


def get_loader_pipe(parc, pipe='elastic_pipe', obj_params=None, **loader_params):
    '''TODO write doc  -  then add to docs.'''

    from ..main.funcs import pipeline_check
    from ..main.input import Loader

    if obj_params is None:
        obj_params = {}

    try:
        
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            from nilearn.input_data  import NiftiLabelsMasker, NiftiMapsMasker

        from neurotools.transform import SurfLabels, SurfMaps
        from neurotools.loading import load
        
    except ImportError:
        raise ImportError('neurotools must be installed!')

    # Apply pipeline check
    pipe = pipeline_check(pipe)

    # Get dimensionality of parcellation
    parc_dims = len(load(parc).shape)

    # Get correct object based off 
    if parc_dims == 1:
        obj = SurfLabels(labels=parc, **obj_params)
    elif parc_dims == 2:
        obj = SurfMaps(maps=parc, **obj_params)
    elif parc_dims == 3:
        obj = NiftiLabelsMasker(labels_img=parc, **obj_params)
    elif parc_dims == 4:
        obj = NiftiMapsMasker(maps_img=parc, **obj_params)

    # Init loader from object
    loader = Loader(obj, **loader_params)

    # Add loader before rest of steps
    pipe.steps = [loader] + pipe.steps

    return pipe

    

    

    


