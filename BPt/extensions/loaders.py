from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import warnings
import networkx as nx


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

        If 'density', then this value represents the
        percent of edges to keep, out of all possible edges.

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

        The passed percentage will be considered
        out of all the possible edges. This will be used to
        select a threshold value, rounding up if needed, then
        all edges above or equal to the threshold will be kept
        (positive or abs case) or in neg case, all edges less than or equal.

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
            'avg_cluster': (nx.average_clustering, False),
            'assortativity': (nx.degree_assortativity_coefficient, False),
            'global_eff': (nx.global_efficiency, False),
            'local_eff': (nx.local_efficiency, False),
            'sigma': (nx.sigma, False),
            'omega': (nx.omega, False),
            'transitivity': (nx.transitivity, False),
            'avg_eigenvector_centrality': (nx.eigenvector_centrality_numpy),
            'avg_closeness_centrality': (nx.closeness_centrality),
            'avg_degree': (self._avg_degree, False),
            'avg_triangles': (nx.triangles, True),
            'avg_pagerank': (nx.pagerank, True),
            'avg_betweenness_centrality': (nx.betweenness_centrality, True),
            'avg_information_centrality': (nx.information_centrality, True),
            'avg_shortest_path_length': (nx.average_shortest_path_length, False)
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

            # Rounded up
            top_n = round(X_t.shape[0] * X_t.shape[1] * self.threshold) - 1

            # If less than 0, set to 0
            if top_n < 0:
                top_n = 0

            # If neg, sort differently
            reverse = False if self.threshold_type == 'neg' else True
            thresh = sorted(X_t.flatten(), reverse=reverse)[top_n]

            # Neg and pos case
            if self.threshold_type == 'neg':
                return np.where(X_t <= thresh, 1, 0)
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

        # Before trying from numpy array call squeeze
        G = nx.from_numpy_array(np.squeeze(X))

        X_trans = []

        for compute in self.to_compute:
            if compute not in self._func_dict:
                X_trans += [compute(G)]
            else:
                X_trans += [self._compute(G, self._func_dict[compute])]
        return np.array(X_trans)

    def _avg_degree(self, G):
        avg_degree = np.mean([i[1] for i in nx.degree(G)])
        return avg_degree

    def _compute(self, G, func_avg):

        func, avg = func_avg[0], func_avg[1]

        if avg:
            return np.mean(func(G).values())
        
        return func(G)


def get_loader_pipe(parc, pipe='elastic_pipe', obj_params=None, **loader_params):
    '''TODO write doc  -  then add to docs.'''

    from ..main.funcs import pipeline_check
    from ..main.input import Loader

    if obj_params is None:
        obj_params = {}

    try:
        from neurotools.transform import SurfLabels, SurfMaps
        from neurotools.loading import load
        from nilearn.input_data  import NiftiLabelsMasker, NiftiMapsMasker
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

    

    

    


