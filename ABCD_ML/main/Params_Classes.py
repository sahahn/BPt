from copy import deepcopy
from sklearn.base import BaseEstimator
import pandas as pd
from ..helpers.ML_Helpers import conv_to_list, proc_input

from ..helpers.VARS import ORDERED_NAMES
from ..main.Input_Tools import is_duplicate, is_pipe, is_select, is_special

def proc_all(base_obj):

    if base_obj is None:
        return

    proc_name(base_obj, 'obj')
    proc_name(base_obj, 'base_model')
    proc_name(base_obj, 'models')
    proc_name(base_obj, 'metric')

def proc_name(base_obj, name):

    if hasattr(base_obj, name):
        obj = getattr(base_obj, name)

        if isinstance(obj, list):
            for o in obj:
                proc_all(o)

        elif hasattr(obj, 'obj'):
            proc_all(obj)

        elif isinstance(obj, str):
            setattr(base_obj, name, proc_input(obj))

class Params(BaseEstimator):

    def set_values(self, args):

        for key in args:
            if key != 'self' and hasattr(self, key):

                try:
                    if isinstance(args[key], str):
                        if args[key] != 'default':
                            setattr(self, key, args[key])
                    else:
                        setattr(self, key, args[key])

                # If value error, set to key
                except ValueError:
                    setattr(self, key, args[key])

    def copy(self):
        return deepcopy(self)

class Check():

    def check_args(self):

        if hasattr(self, 'obj'):
            self._check_obj()

        if hasattr(self, 'params'):
            self._check_params()

        if hasattr(self, 'obj') and hasattr(self, 'params'):
            self._check_obj_and_params()

        if hasattr(self, 'scope'):
            self._check_scope()

        if hasattr(self, 'extra_params'):
            self._check_extra_params()

        if hasattr(self, 'base_model'):
            self._check_base_model()

        try:
            self.check_extra_args()
        except AttributeError:
            pass

    def _check_obj(self):

        obj = getattr(self, 'obj')

        if obj is None:
            raise IOError('passed obj cannot be None, to ignore the object itself',
                          'set it within Model_Pipeline to None, not obj here.')

        if isinstance(obj, list) and not is_pipe(obj):
            raise IOError('You may only pass a list of objs with special input',
                          'Pipe()')

    def _check_params(self):

        params = getattr(self, 'params')

        if params is None:
            setattr(self, 'params', 0)

    def _check_obj_and_params(self):

        obj = getattr(self, 'obj')
        params = getattr(self, 'params')

        if is_pipe(obj):
            
            if not isinstance(params, list):

                if params == 0:
                    new_params = [0 for i in range(len(obj))]
                    setattr(self, 'params', new_params)
                else:
                    raise IOError('obj is passed as Pipe, but params was not. Make sure',
                                  'params is either a list or Pipe with the same length as',
                                  'self.obj')
  
            elif len(params) != len(obj):

                if params == [0]:
                    new_params = [0 for i in range(len(obj))]
                    setattr(self, 'params', new_params)
                else:
                    raise IOError('obj is passed as Pipe, Make sure',
                                'params is either a list or Pipe with the same length as',
                                'self.obj')

    def _check_scope(self):

        pass
        # Nothing to check as of right now
        # scope = getattr(self, 'scope')
        
    def _check_extra_params(self):

        extra_params = getattr(self, 'extra_params')
        if extra_params is None:
            setattr(self, 'extra_params', {})

        elif not isinstance(extra_params, dict):
            raise IOError('extra params must be a dict!')

    def _check_base_model(self):

        base_model = getattr(self, 'base_model')

        # None is okay
        if base_model is None:
            return

        if not hasattr(base_model, '_is_model'):
            raise IOError('base_model must be either None or a valid Model / Ensemble',
                          'set of wrapeper params!')

class Piece(Params, Check):
    pass

class Loader(Piece):

    def __init__(self, obj, params=0, scope='data files', extra_params=None):
        ''' Loader refers to transformations which operate on loaded Data_Files.
        (See :func:`Load_Data_Files`).
        They in essence take in saved file locations, and after some series
        of transformations pass on compatible features. Notably loaders 
        define operations which are computed on single files indepedently.

        Parameters
        ----------
        obj : str, custom obj or :class:`Pipe`
            `obj` selects the base loader object to use, this can be either a str corresponding to
            one of the preset loaders found at :ref:`Loaders`. 
            Beyond pre-defined loaders, users can 
            pass in custom objects (they just need to have a defined fit_transform function
            which when passed the already loaded file, will return a 1D representation of that subjects
            features.

            obj can also be passed as a :class:`Pipe`. See :class:`Pipe`'s documentation to
            learn more on how this works, and why you might want to use it.

            See :ref:`Pipeline Objects` to read more about pipeline objects in general.

            For example, the 'identity' loader will load in saved data at the stored file
            location, lets say they are 2d numpy arrays, and will return a flattened version
            of the saved arrays, with each data point as a feature. A more practical example
            might constitute loading in say 3D neuroimaging data, and passing on features as
            extracted by ROI.

        params : int, str or dict of :ref:`params<Params>`, optional
            `params` determines optionally if the distribution of hyper-parameters to
            potentially search over for this loader. Preset param distributions are
            listed for each choice of obj at :ref:`Loaders`, and you can read more on
            how params work more generally at :ref:`Params`.

            If obj is passed as :class:`Pipe`, see :class:`Pipe` for an example on how different
            corresponding params can be passed to each piece individually.

            ::

                default = 0

        scope : :ref:`valid scope<Scopes>`, optional
            `scope` determines on which subset of features the specified loader
            should transform. See :ref:`Scopes` for more information on how scopes can
            be specified.

            You will likely want to use either custom key based scopes, or the 
            'data files' preset scope, as something like 'covars' won't make much sense,
            when atleast for now, you cannot even load Covars data files.

            ::

                default = 'data files'

        extra_params : :ref`extra params dict<Extra Params>`, optional

            See :ref:`Extra Params`

            ::

                default = None
        '''

        self.obj = obj
        self.params = params
        self.scope = scope
        self.extra_params = extra_params

        self.check_args()

class Imputer(Piece):

    def __init__(self, obj, params=0, scope='float',
                 base_model=None, extra_params=None):
        ''' 
        Parameters
        ----------
        '''

        self.obj = obj
        self.params = params
        self.scope = scope
        self.base_model = base_model
        self.extra_params = extra_params

        self.check_args()

class Scaler(Piece):

    def __init__(self, obj, params=0, scope='float', extra_params=None):
        ''' 
        Parameters
        ----------
        '''

        self.obj = obj
        self.params = params
        self.scope = scope
        self.extra_params = extra_params

        self.check_args()

class Transformer(Piece):

    def __init__(self, obj, params=0, scope='float', extra_params=None):
        ''' 
        Parameters
        ----------
        '''

        self.obj = obj
        self.params = params
        self.scope = scope
        self.extra_params = extra_params

        self.check_args()

class Sampler(Piece):

    def __init__(self, obj, params=0, sample_on='targets', extra_params=None):
        ''' 
        Parameters
        ----------
        '''

        self.obj = obj
        self.params = params
        self.sample_on = sample_on
        self.extra_params = extra_params

        self.check_args()

    def add_strat_u_name(self, func):
        self.sample_on = func(self.sample_on)

class Feat_Selector(Piece):

    def __init__(self, obj, params=0, base_model=None, extra_params=None):
        ''' 
        Parameters
        ----------
        '''

        self.obj = obj
        self.params = params
        self.base_model = base_model
        self.extra_params = extra_params

        self.check_args()

class Model(Piece):

    def __init__(self, obj, params=0, extra_params=None):
        ''' 

        Parameters
        ----------

        model : str or list of str
        Each string refers to a type of model to train.
        If a list of strings is passed then an ensemble model
        will be created over all individual models.

          view the docs at :ref:`Models`

        If 'default', and not already defined, set to 'linear'
        (default = 'default')

        model_params : int, str, or list of
        Each `model` has atleast one default parameter distribution
        saved with it.

        This parameter is used to select between different
        distributions to be used with different search types,
        when `search_type` == None, `model_params` is automatically
        set to default 0.

        This parameter can be selected with either an integer index
        (zero based), or the str name for a given `model`.
        Likewise with `model`, if passed list input, this means
        a list was passed to `model` and the indices should correspond.

        The different parameter distributions avaliable for each
        `model`, can be shown by calling :func:`Show_Models`
        or on the docs at :ref:`Models`

        If 'default', and not already defined, set to 0
        (default = 'default')
        '''

        self.obj = obj
        self.params = params
        self.extra_params = extra_params
        self._is_model = True

        self.check_args()

class Ensemble(Piece):

    def __init__(self, obj, models, params=0, needs_split=False,
                 des_split=.2, single_estimator=False, cv=3,
                 extra_params=None):
        ''' The Ensemble object is valid base :class:`Model_Pipeline` piece, designed
        to be passed as input to the `model` parameter of :class:`Model_Pipeline`, or
        to its own models parameters.

        This class is used to create a variety ensembled models, typically based on
        :class:`Model` pieces.

        Parameters
        ----------


        '''

        self.obj = obj
        self.models = models
        self.params = params
        self.needs_split = needs_split
        self.des_split = des_split
        self.single_estimator = single_estimator
        self.cv = cv
        self.extra_params = extra_params
        self._is_model = True

        self.check_args()

    def check_extra_args(self):

        if isinstance(self.models, list):
            for model in self.models:
                if not hasattr(model, '_is_model'):
                    raise IOError('All models must be valid Model/Ensemble param wrapped!')

        else:
            if not hasattr(self.models, '_is_model'):
                raise IOError('Passed model in models must be a valid Model/Ensemble, i.e., param wrapped.')

class Param_Search(Params):

    def __init__(self, search_type='RandomSearch',
                 splits=3, n_iter=10, metric='default',
                 weight_metric=False):
        
        self.search_type = search_type
        self.splits = splits
        self.n_iter = n_iter
        self.metric = metric
        self.weight_metric = weight_metric

        self._splits_vals = None
        self._n_jobs = 1

    def set_split_vals(self, vals):
        self._splits_vals = vals

    def check_args(self):

        if isinstance(self.metric, list):
            raise IOError('metric within Param Search cannot be list-like')
        if isinstance(self.weight_metric, list):
            raise IOError('weight_metric within Param Search cannot be list-like')

class Feat_Importance(Params):

    def __init__(self, obj, metric='default'):
        # Likely want other params here!

        self.obj = obj
        self.metric = metric

class Drop_Strat():

    def __init__(self):

        self.obj = None
        self.params = 0

    def check_args(self):
        return

    def proc_input(self):
        return

class Model_Pipeline(Params):

    def __init__(self, loaders=None, imputers='default',
                 scalers='default', transformers=None,
                 samplers=None, feat_selectors=None,
                 model='default', param_search=None,
                 feat_importances='default',
                 cache=None):
        ''' Model_Pipeline is defined as essentially a wrapper around
        all of the explicit modelling pipeline parameters. This object is
        used as input to 
        :func:`Evaluate <ABCD_ML.ABCD_ML.Evaluate>` and :func:`Test <ABCD_ML.ABCD_ML.Test>`

        The ordering of the parameters listed below defines the pre-set
        order in which these Pipeline pieces are composed
        (params up to model, param_search and feat_importances are not ordered pipeline pieces).
        For more flexibility, one can always use custom defined objects, or even pass custom defined
        pipelines directly to model (i.e., in the case where you have a specific pipeline you want to use
        already defined, but say just want to use the loaders from ABCD_ML).

        Parameters
        ----------
        loaders : :class:`Loader`, list of or None, optional
            Each :class:`Loader` refers to transformations which operate on loaded Data_Files
            (See :func:`Load_Data_Files <ABCD_ML.ABCD_ML.Load_Data_Files>`). See :class:`Loader`
            explcitly for more information on how to create a valid object, with relevant params and scope.

            In the case that a list of Loaders is passed to loaders, if a native
            python list, then passed loaders will be applied sequentially (likely each
            passed loader given a seperate scope, as the output from one loader cannot be input
            to another- note to create actual sequential loader steps, look into using the
            :class:`Pipe` wrapper
            argument when creating a single :class:`Loader` obj).

            Passed loaders can also be wrapped in a
            :class:`Select` wrapper, e.g., as either

            .. code-block::

                # Just passing select
                loaders = Select([Loader(...), Loader(...)])
                
                # Or nested
                loaders = [Loader(...), Select([Loader(...), Loader(...)])]

            In this way, most of the pipeline objects can accept lists, or nested lists with
            param wrapped, not just loaders!

            .. code-block::

                (default = None)

        imputers : :class:`Imputer`, list of or None, optional
            If there is any missing data (NaN's) that have been kept
            within data or covars, then an imputation strategy must be
            defined! This param controls what kind of imputation strategy to use.

            Each :class:`Imputer` contains information around which imputation
            strategy to use, what scope it is applied to (in this case only 'float' vs. 'cat'),
            and other relevant base parameters (i.e., a base model if an iterative imputer is selected).

            In the case that a list of :class:`Imputer` are passed,
            they will be applied sequentially, though note that unless custom scopes
            are employed, at most passing only an imputer for float data and
            an imputer for categorical data makes sense.
            You may also use input wrapper, like :class:`Select`.

            In the case that no NaN data is passed, but imputers is not None, it will simply be
            set to None.

            ::

                (default = [Imputer('mean', scope='float'),
                            Imputer('median', scope='cat')])

        scalers : :class:`Scaler`, list of or None, optional
            Each :class:`Scaler` refers to any potential data scaling where a
            transformation on the data (without access to the target variable) is
            computed, and the number of features or data points does not change.
            Each :class:`Scaler` object contains information about the base object, what
            scope it should be applied to, and saved param distributions if relevant.

            As with other pipeline params, scalers can accept a list of :class:`Scaler` objects,
            in order to apply sequential transformations 
            (or again in the case where each object has a seperate scope,
            these are essentially two different streams of transformations,
            vs. when two Scalers with the same scope are passed, the output from one
            is passed as input to the next). Likewise, you may also use valid input wrappers,
            e.g., :class:`Select`.

            ::

                (default = Scaler('standard'))

        transformers : :class:`Transformer`, list of or None, optional
            Each :class:`Transformer` defines a type of transformation to
            the data that changes the number of features in perhaps non-deterministic
            or not simply removal (i.e., different from feat_selectors), for example
            applying a PCA, where both the number of features change, but
            also the new features do not 1:1 correspond to the original
            features. See :class:`Transformer` for more information.

            Transformers can be composed sequentially with list or special
            input type wrappers, the same as other objects.

            ::

                (default = None)

        samplers : :class:`Sampler`, list of or None, optional
            Each :class:`Sampler` refers to an optional type
            of data point resampling in which to preform, i.e., 
            in attempt to correct for a class imbalance. See the
            base :class:`Sampler` object for more information on
            what different sampler options and restrictions are.

            If passed a list, the sampling will be applied sequentially.

            ::

                (default = None)

        feat_selectors : :class:`Feat_Selector`, list of or None, optional
            Each :class:`Feat_Selector` refers to an optional feature selection stage
            of the Pipeline. See :class:`Feat_Selector` for specific options.

            Input can be composed in a list, to apply feature selection sequentially,
            or with special Input Type wrapper, e.g., :class:`Select`.

        model : :class:`Model`, :class:`Ensemble`, optional
            model accepts one input of type
            :class:`Model` or :class:`Ensemble`. Though,
            while it cannot accept a list (i.e., no sequential behavior), you
            may still pass Input Type wrapper like :class:`Select` to perform
            model selection via param search.

            See :class:`Model` for more information on how to specify a single
            model to ABCD_ML, and :class:`Ensemble` for information on how
            to build an ensemble of models.

            Note: You must have provide a model, there is no option for None. Instead
            default behavior is to use a simple linear/logistic regression, which will likely
            not be a good choice.
            
            ::

                (default = Model('linear'))

        param_search : :class:`Param_Search` or None, optional
            :class:`Param_Search` can be provided in order to speficy a corresponding
            hyperparameter search for the provided pipeline pieces. When defining each
            piece, you may set hyperparameter distributions for that piece. If param search
            is None, these distribution will be essentially ignored, but if :class:`Param_Search`
            is passed here, then they will be used along with the strategy defined in the passed
            :class:`Param_Search` to conduct a nested hyper-param search.

            Note: If using input wrapper types like :class:`Select`, then a param search
            must be passed!

            ::

                (default = None)

        feat_importances : :class:`Feat_Importance` list of or None, optional
            Provide here either a single, or list of :class:`Feat_Importance` param objects
            in which to specify what importance values, and with what settings should be computed.
            See the base :class:`Feat_Importance` object for more information on how to specify
            these objects. 

            In this case of a passed list, all passed Feat_Importances will attempt to be
            computed.

            ::

                (default = Feat_Importance('base'))

        cache : str or None, optional
            The base scikit-learn Pipeline, upon which the ABCD_ML Pipeline extends,
            allows for the caching of fitted transformers - which in this context means all
            steps except for the model. If this behavior is desired
            (in the cases where a non-model step take a long time to fit),
            then a str indicating a directory where
            the cache should be stored can be passed to cache. If this directory
            does not aready exist, it will be created.

            Note: cache_dr's are not automatically removed, and while different calls
            to Evaluate or Test may benefit from overlapping cached steps - the size of the
            cache can also grow pretty quickly, so you may need to manually monitor the size
            of the cache and perform manual deletions when it grows too big depending on your
            storage resources.

            ::

                (default = None)

        '''

        

        self.loaders = loaders
        
        if imputers == 'default':
            imputers = [Imputer('mean', scope='float'),
                        Imputer('median', scope='cat')]
        self.imputers = imputers

        if scalers == 'default':
            scalers = Scaler('standard')
        self.scalers = scalers

        self.transformers = transformers
        self.samplers = samplers

        # Special place holder case for drop strat, for compat.
        self._drop_strat = Drop_Strat()

        self.feat_selectors = feat_selectors

        if model == 'default':
            model = Model('linear')
        self.model = model

        self.param_search = param_search

        if feat_importances == 'default':
            feat_importances = Feat_Importance('base')
        self.feat_importances = feat_importances

        self.cache = cache

        self._n_jobs = 1

        # Perform all preproc on input which can be run
        # more then once, these are essentially checks on the input
        self._proc_checks()

    def _proc_all_pieces(self, func):
        '''Helper to proc all pieces with a function
        that accepts a list of pieces'''

        for param_name in ORDERED_NAMES:

            if param_name != '_drop_strat':
                params = getattr(self, param_name)

                if params is None:
                    new_params = params

                elif isinstance(params, list):
                    new_params = func(params)

                # If not list-
                else:
                    new_params = func([params])
                    if isinstance(new_params, list) and len(new_params) == 1:
                        new_params = new_params[0]

                setattr(self, param_name, new_params)

    def _proc_duplicates(self, params):

        if isinstance(params, list):
            new_params = deepcopy(params)
            del new_params[:]

            if len(params) == 1:
                return self._proc_duplicates(params[0])

            for p in params:
                new_params.append(self._proc_duplicates(p))

            return new_params

        elif hasattr(params, 'scope'):
            scopes = getattr(params, 'scope')
            new_params = []

            if is_duplicate(scopes):
                for scope in scopes:
                    new = deepcopy(params)
                    new.scope = scope
                    new_params.append(new)

                return new_params

        return params

    def _proc_input(self, params):

        for p in params:
            if isinstance(p, list):
                self._proc_input(p)
            else:
                proc_all(p)

        return params

    def _check_args(self, params):

        for p in params:
            if isinstance(p, list):
                self._check_args(p)
            else:
                p.check_args()

        return params

    def check_imputer(self, data):

        # If no NaN, no imputer
        if not pd.isnull(data).any().any():
            self.imputer = None

    def check_samplers(self, func):
        if self.samplers is not None:
            if isinstance(self.samplers, list):
                for s in self.samplers:
                    s.add_strat_u_name(func)
            else:
                self.samplers.add_strat_u_name(func)

    def preproc(self, n_jobs):

        # Base proc checks
        self._proc_checks()

        # Store n_jobs
        self.set_n_jobs(n_jobs)
        
    def _proc_checks(self):

        # Check for duplicate scopes
        self._proc_all_pieces(self._proc_duplicates)

        # Proc input
        self._proc_all_pieces(self._proc_input)
        proc_all(self.param_search)
        proc_all(self.feat_importances)

        # Double check input args in case something changed
        self._proc_all_pieces(self._check_args)

        # Proc param search if not None
        if self.param_search is not None:
            self.param_search.check_args()

    def set_n_jobs(self, n_jobs):

        # If no param search, each objs base n_jobs is
        # the passed n_jobs
        if self.param_search is None:
            self._n_jobs = n_jobs

        # Otherwise, base jobs are 1, and the search_n_jobs
        # are set to passed n_jobs
        else:
            self._n_jobs = 1
            self.param_search._n_jobs = n_jobs

    def get_ordered_pipeline_params(self):

        # Conv all to list & return in order as a deep copy
        return deepcopy([conv_to_list(getattr(self, piece_name))
                         for piece_name in ORDERED_NAMES])

    def params_print(self, params, indent, _print=print, end='\n'):

        if not isinstance(params, list):
            _print(' ' * indent, params, sep='', end=end)
            return

        elif len(params) == 1:
            self.params_print(params[0], indent, _print=print)
            return

        elif not is_special(params):
            
            _print('[', sep='', end='')
            self._p_stack.append(']')
            self.params_print(params[0], indent, _print=_print, end='')
            _print(',', sep='')
            indent += 1

        elif is_select(params):

            _print('Select([', sep='', end='')
            self._p_stack.append('])')
            self.params_print(params[0], indent, _print=_print, end='')
            _print(',', sep='')
            indent += 9

        for param in params[1:-1]:
            self.params_print(param, indent, _print=_print, end='')
            _print(',', sep='')
    
        self.params_print(params[-1], indent, _print=_print, end='')
        _print(self._p_stack.pop(), end=end)

        return

    def print_all(self, _print=print):

        self._p_stack = []
        
        _print('Model_Pipeline')
        _print('--------------')

        pipeline_params = self.get_ordered_pipeline_params()
        for name, params in zip(ORDERED_NAMES, pipeline_params):

            if name == '_drop_strat':
                pass

            elif params is None:
                pass

            else:
                _print(name + '=\\')
                self.params_print(params, 0, _print=_print)
                _print()

        _print('param_search=\\')
        _print(self.param_search)
        _print()
        _print('feat_importances=\\')
        _print(self.feat_importances)
        _print()

        if self.cache is not None:
            _print('cache =', self.cache)
        _print()

class Problem_Spec(Params):

    def __init__(self, problem_type='regression',
                 target=0, metric='default', weight_metric=False,
                 scope='all', subjects='all',
                 n_jobs='default',
                 random_state='default'):
        '''Problem Spec is defined as an object of params encapsulating the set of
        parameters shared by modelling class functions
        :func:`Evaluate <ABCD_ML.ABCD_ML.Evaluate>` and :func:`Test <ABCD_ML.ABCD_ML.Test>`

        Parameters
        ----------
        problem_type : str, optional

            - 'regression'
                For ML on float/continuous target data.

            - 'binary'
                For ML on binary target data.

            - 'categorical'
                For ML on categorical target data, as multiclass.

            (default = 'regression')

        target : int or str, optional
            The loaded target in which to use during modelling.
            This can be the int index (as assigned by order loaded in,
            e.g., first target loaded is 0, then the next is 1),
            or the name of the target column.
            If only one target is loaded, just leave as default of 0.

            (default = 0)

        metric : str or list, optional
            Indicator str for which metric(s) to use when calculating
            average validation score in Evaluate, or Test set score in Test.

            A list of str's can be passed as well, in this case, scores for
            all of the requested metrics will be calculated and returned.

            Note: If using a Param_Search, the Param_Search object has a
            metric parameter as well. This metric describes the metric optimized
            in a parameter search. In the case that this metric is left as 'default',
            the the first metric passed here will be used in Param_Search!

            For a full list of supported metrics please view the docs at :ref:`Metrics`

            If left as 'default', assign a reasonable metric based on the
            passed problem type.

            - 'regression'  : 'r2'
            - 'binary'      : 'macro roc auc'
            - 'categorical' : 'matthews'

            (default = 'default')

        weight_metric : bool, list of, optional
            If True, then the metric of interest will be weighted within
            each repeated fold by the number of subjects in that validation set.
            This parameter only typically makes sense for custom split behavior where 
            validation folds may end up with differing sizes.
            When default CV schemes are employed, there is likely no point in
            applying this weighting, as the validation folds will have simmilar sizes.

            If you are passing mutiple metrics, then you can also pass a 
            list of values for weight_metric, with each value set as boolean True or False,
            specifying if the corresponding metric by index should be weighted or not.

            (default = False)

        scope : key str or Scope obj, optional
            This parameter allows the user to optionally
            run an expiriment with just a subset of the loaded features
            / columns.

            See :ref:`Scopes` for a more detailed explained / guide on how scopes
            are defined and used within ABCD_ML.

        subjects : str, array-like or Value_Subset, optional
            This parameter allows the user to optionally run Evaluate or Test
            with just a subset of the loaded subjects. It is notably distinct
            from the `train_subjects`, and `test_subjects` parameters directly
            avaliable to Evaluate and Test, as those parameters typically refer
            to train/test splits. Specifically, any value specified for this 
            subjects parameter will be applied AFTER selecting the relevant train
            or test subset.

            One use case for this parameter might be specifying subjects of just one
            sex, where you would still want the same training set for example, but just want
            to test sex specific models.

            If set to 'all' (as is by default), all avaliable subjects will be
            used.

            `subjects` can accept either a specific array of subjects,
            or even a loc of a text file (formatted one subject per line) in
            which to read from.

            A special wrapper, Value_Subset, can also be used to specify more specific,
            specifically value specific, subsets of subjects to use.
            See :class:`Value_Subset` for how this input wrapper can be used.
            
            (default = 'all')

        n_jobs : int, or 'default'
            n_jobs are employed witin the context of a call to Evaluate or Test. 
            If left as default, the class wide ABCD_ML value will be used.

            In general, the way n_jobs are propegated to the different pipeline
            pieces on the backend is that, if there is a parameter search, the base
            ML pipeline will all be set to use 1 job, and the n_jobs budget will be used
            to train pipelines in parellel to explore different params. Otherwise, if no
            param search, n_jobs will be used for each piece individually, though some
            might not support it.

            (default = 'default')

        random_state : int, RandomState instance, None or 'default', optional
            Random state, either as int for a specific seed, or if None then
            the random seed is set by np.random.

            If 'default', use the saved class value.
            ( Defined in :class:`ABCD_ML <ABCD_ML.ABCD_ML>`)
            
            (default = 'default')
        '''

        self.problem_type = problem_type
        self.target = target

        if metric == 'default':
            default_metrics = {'regression': 'r2',
                               'binary': 'macro roc auc',
                               'categorical': 'matthews'}
            metric = default_metrics[self.problem_type]

        self.metric = metric
        self.weight_metric = weight_metric
        self.scope = scope
        self.subjects = subjects

        self.n_jobs = n_jobs
        self.random_state = random_state

        self._final_subjects = None

        self._proc_checks()

    def _proc_checks(self):
        proc_all(self)
        

    def set_final_subjects(self, final_subjects):
        self._final_subjects = final_subjects

    def get_model_spec(self):

        return {'problem_type': self.problem_type,
                'random_state': self.random_state}

    def print_all(self, _print=print):

        _print('Problem_Spec')
        _print('------------')
        _print('problem_type =', self.problem_type)
        _print('target =', self.target)
        _print('metric =', self.metric)
        _print('weight_metric =', self.weight_metric)
        _print('scope =', self.scope)
        _print('subjects =', self.subjects)
        _print('len(subjects) =', len(self._final_subjects),
               '(before overlap w/ train/test subjects)')
        _print('n_jobs', self.n_jobs)
        _print('random_state', self.random_state)
        _print()
