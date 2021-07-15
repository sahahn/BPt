from .compare import Compare, Option
from copy import deepcopy, copy
from sklearn.base import BaseEstimator
from ..default.helpers import proc_input, coarse_any_obj_check
from ..util import conv_to_list, BPtInputMixIn

from ..main.input_operations import (Pipe, Select,
                                     ValueSubset, Duplicate)
from ..default.options.scorers import process_scorer
from .CV import get_bpt_cv
from ..pipeline.constructors import (LoaderConstructor, ImputerConstructor,
                                     ScalerConstructor, TransformerConstructor,
                                     FeatSelectorConstructor, ModelConstructor,
                                     CustomConstructor)

import warnings
from pandas.util._decorators import doc
from sklearn.utils.estimator_checks import check_estimator
from sklearn.pipeline import _name_estimators


def proc_all(base_obj):

    if base_obj is None:
        return

    proc_name(base_obj, 'obj')
    proc_name(base_obj, 'base_model')
    proc_name(base_obj, 'models')
    proc_name(base_obj, 'scorer')
    proc_name(base_obj, 'target_scaler')


def proc_name(base_obj, name):

    if hasattr(base_obj, name):
        obj = getattr(base_obj, name)

        if isinstance(obj, (list, tuple)):
            for o in obj:
                proc_all(o)

        elif hasattr(obj, 'obj'):
            proc_all(obj)

        elif isinstance(obj, str):
            setattr(base_obj, name, proc_input(obj))


def uniquify(obj):

    for p_str in ['obj', 'base_model', 'models', 'target_scaler']:

        if hasattr(obj, p_str):
            p = getattr(obj, p_str)

            # Recursive check first
            if isinstance(p, list):
                [uniquify(pp) for pp in p]
            elif hasattr(p, '_uniquify'):
                p._uniquify()

            # Then set with copy
            setattr(obj, p_str, copy(p))


class Params(BaseEstimator):

    def copy(self):
        '''This method returns a deepcopy of the base object.'''
        return deepcopy(self)

    def get_params(self, deep=True):

        params = super().get_params(deep=deep)

        if 'extra_params' in params and deep:
            extra_params = params.pop('extra_params')
            params.update(extra_params)

        return params

    def set_params(self, **params):

        # If no extra params
        if not hasattr(self, 'extra_params'):
            return super().set_params(**params)

        # Possible extra params case
        # Get base, no extra param keys
        base_keys = super().get_params().keys()

        # Set any base params
        base_params = {key: params[key] for key in params
                       if key in base_keys}
        super().set_params(**base_params)

        # Calculate any extra params to set
        extra_params = {key: params[key] for key in params
                        if key not in base_keys}

        # Update extra params
        self.extra_params.update(extra_params)

        return self


class Check():

    def _check_args(self):
        '''This method is used to ensure arguments are valid.
        It called automatically at the end of init.'''

        if hasattr(self, 'obj'):
            self._check_obj()

        if hasattr(self, 'params'):
            self._check_params()

        if hasattr(self, 'obj') and hasattr(self, 'params'):
            self._check_obj_and_params()

        if hasattr(self, 'base_model'):
            self._check_base_model()

        try:
            self._check_extra_args()
        except AttributeError:
            pass

    def _check_obj(self):

        obj = getattr(self, 'obj')

        if obj is None:
            raise RuntimeError('Passed obj cannot be None')

        if isinstance(obj, list) and not isinstance(obj, Pipe):
            raise RuntimeError('You may not pass a list of objs')

        # If str, then run a coarse general check
        # that won't handle if the object is of the wrong
        # problem type or for the wrong piece
        # but should catch typos
        if isinstance(obj, str):
            coarse_any_obj_check(obj)

    def _check_params(self):

        params = getattr(self, 'params')

        if params is None:
            setattr(self, 'params', 0)

    def _check_obj_and_params(self):

        obj = getattr(self, 'obj')
        params = getattr(self, 'params')

        if isinstance(obj, Pipe):

            if not isinstance(params, list):

                if params == 0:
                    new_params = [0 for i in range(len(obj))]
                    setattr(self, 'params', new_params)
                else:
                    raise IOError('obj is passed as Pipe, but params was not.',
                                  'Make sure',
                                  'params is either a list or Pipe ',
                                  'with the same length as',
                                  'self.obj')

            elif len(params) != len(obj):

                if params == [0]:
                    new_params = [0 for i in range(len(obj))]
                    setattr(self, 'params', new_params)
                else:
                    raise IOError('obj is passed as Pipe, Make sure',
                                  'params is either a list or Pipe with the ',
                                  'same length as',
                                  'self.obj')

    def _check_base_model(self):

        base_model = getattr(self, 'base_model')

        # None is okay
        if base_model is None:
            return

        # Assume select is okay too
        if isinstance(base_model, Select):
            return

        if not isinstance(base_model, Model):
            raise IOError('base_model must be either None or a valid '
                          'Model / Ensemble ',
                          'set of wrapper params!')


class Piece(Params, Check):
    '''This is the base piece in which :ref`pipeline_objects` inherit from. This
    class should not be used directly.'''

    def build(self, dataset='default', problem_spec='default',
              **problem_spec_params):
        '''This method is used to convert a single pipeline piece into
        the base sklearn style object used in the pipeline. This method
        is mostly used to investigate pieces and is not necessarily
        designed to produce independently usable pieces.

        For now this method will not work when the base
        obj is a custom object.

        Parameters
        -----------
        dataset : :class:`Dataset` or 'default', optional
            The Dataset in which the pipeline piece should be initialized
            according to. For example, pipeline's can include Scopes,
            these need a reference Dataset.

            If left as default will initialize and use
            an instance of a FakeDataset class, which will
            work fine for initializing pipeline objects
            with scope of 'all', but should be used with caution
            when elements of the pipeline use non 'all' scopes.
            In these cases a warning will be issued.

            It is advisable to use this
            build function only for viewing objects. If using
            the build function instead for eventual modelling it
            is important to pass the correct :class:`Dataset` in
            the case that any of the pipeline pieces are at all
            dependant on the structure of the input data.

            Note: If problem type is not defined in problem_spec
            and Dataset is left as default, then a problem type of
            'regression' will be used.

            ::

                default = 'default'

        problem_spec : :class:`ProblemSpec` or 'default', optional
            This parameter accepts an instance of the
            params class :class:`ProblemSpec`.
            The ProblemSpec is essentially a wrapper
            around commonly used
            parameters needs to define the context
            the model pipeline should be evaluated in.
            It includes parameters like problem_type, scorer, n_jobs,
            random_state, etc...

            See :class:`ProblemSpec` for more information
            and for how to create an instance of this object.

            If left as 'default', then will initialize a
            ProblemSpec with default params.

            ::

                default = "default"

        problem_spec_params : :class:`ProblemSpec` params, optional
            You may also pass any valid problem spec argument-value pairs here,
            in order to override a value in the passed :class:`ProblemSpec`.
            Overriding params should be passed in kwargs style, for example:

            ::

                func(..., problem_type='binary')

        Returns
        -------
        estimator : sklearn compatible estimator
            Returns the BPt-style sklearn compatible estimator
            version of this piece as converted to internally
            when building the pipeline

        params : dict
            Returns a dictionary with any parameter distributions
            associated with this object, for example
            this can be used to check what exactly
            pre-existing parameter distributions point
            to.

        Examples
        --------

        Given a dataset and pipeline piece (this can be any
        of the valid :ref:`api.pipeline_pieces` not just :class:`Model`
        as used here).

        .. ipython:: python

            import BPt as bp

            dataset = bp.Dataset()
            dataset['col1'] = [1, 2, 3]
            dataset['col2'] = [3, 4, 5]
            dataset.set_role('col2', 'target', inplace=True)
            dataset

            piece = bp.Model('ridge', params=1)
            piece

        We can call build from the piece

        .. ipython:: python

            estimator, params = piece.build(dataset)
            estimator
            params

        '''

        # If default use FakeDataset
        if isinstance(dataset, str) and (dataset == 'default'):
            from ..dataset.fake_dataset import FakeDataset
            dataset = FakeDataset()

        from ..main.funcs import problem_spec_check

        # Get proc'ed problem spec
        ps = problem_spec_check(problem_spec, dataset=dataset,
                                **problem_spec_params)

        # Make sure dataset up to date
        dataset._check_sr()

        # Get constructor
        constructor =\
            self._constructor(spec=ps._get_spec(),
                              dataset=dataset, user_passed_objs={})

        # Return the objs and params
        objs, params = constructor.process(self)
        return objs[0], params

    def _get_param_names(self):

        param_names = super()._get_param_names()

        if hasattr(self, 'extra_params'):
            ep = getattr(self, 'extra_params')
            if isinstance(ep, dict) and len(ep) > 0:
                param_names += ['extra_params']

        return param_names

    def _uniquify(self):
        uniquify(self)


_piece_docs = {}

_piece_docs[
    "params"
] = """params : int, str or dict of :ref:`params<Params>`, optional
        | The parameter `params` can be used to set an associated distribution
            of hyper-parameters, fixed parameters or combination of.

        | Preset parameter options can be found distributions are
            listed for each choice of params with the corresponding
            obj at :ref:`Pipeline Options<pipeline_options>`.

        | More information on how this parameter
            works can be found at :ref:`Params`.

        ::

            default = 0
"""

_piece_docs[
    "scope"
] = """scope : :ref:`Scope`, optional
        | The `scope` parameter determines the subset of
            features / columns in which this object
            should operate on within the created pipeline. For example,
            by specifying scope = 'float', then
            this object will only operate on columns with scope
            float.

        | See :ref:`Scope` for more information on
            how scopes can be specified.
"""

_piece_docs[
    "extra_params"
] = """extra_params : :ref:`extra_params`
        | You may pass additional kwargs style arguments
            for this piece as :ref:`extra_params`. Any values
            passed here will be used to try and set that value
            in the requested obj.

        | Any parameter value pairs
            specified here will take priority over any
            set via params. For example, lets say in
            the object we are initializing, 'fake obj' it has a parameter
            called size, and we want it fixed as 10, we can specify that with:

        ::

            (obj='fake obj', ..., size=10)

        See :ref:`extra_params` for more information.
"""

_base_docs = _piece_docs.copy()

_piece_docs[
    "cache_loc"
] = """cache_loc : str, Path or None, optional
        | This parameter can optionally be set to a
            str or path representing the location in which
            this object will be cached after fitting.
            To skip this option, keep as the default argument of None.

        | If set, the python library joblib is used
            to cache a copy after fitting and in the case
            that a cached copy already exists will load from
            that copy instead of re-fitting the base object.

        ::

            default = None

    """

_bp_docs = _piece_docs.copy()

_piece_docs[
    "param_search"
] = """param_search : :class:`ParamSearch`, None, optional
        | This parameter optionally specifies that this object
          should be nested with a hyper-parameter search.

        | If passed an instance of :class:`ParamSearch`, the
          underlying object, or components of the underlying
          object (if a pipeline) must have atleast one valid
          hyper-parameter distribution to search over.

        | If left as None, the default, then no hyper-parameter
          search will be performed.

        ::

            default = None

"""

_piece_docs[
    "target_scaler"
] = """target_scaler : :class:`Scaler`, None, optional
        | Can optionally pass an instance of
            :class:`Scaler` here to have properly
            nested target scaling / reverse scaling (before scoring)
            applied.

        .. warning::

            This parameter is still experimental.
            It has not been fully tested in
            complicated nesting cases, e.g., if Model is
            wrapping a nested Pipeline, this param will
            likely break.

        ::

            default = None

"""


@doc(**_base_docs)
class Loader(Piece):
    ''' Loader refers to transformations which operate on :ref:`data_files`.
    They in essence take in saved file locations and after some series
    of specified transformations pass on compatible features.

    The Loader object can operate in two ways. Either
    the Loader can define operations which are computed on
    single files independently, or load and pass on data
    to the defined `obj` as a list, where each element of
    the list is a subject's data. See parameter behav.

    Parameters
    ----------
    obj : str, custom obj or :class:`Pipe`
        | `obj` selects the base loader object to use, this can be either
          a str corresponding to
          one of the preset loaders found at :ref:`Loaders`.
          Beyond pre-defined loaders, users can
          pass in custom objects as long as they have functions
          corresponding to the correct behavior.

        | `obj` can also be passed as a :class:`Pipe`.
          See :class:`Pipe`'s documentation to
          learn more on how this works, and why you might want to use it.
          See :ref:`Pipeline Objects<pipeline_objects>` to read more
          about pipeline objects in general.

        | For example, the 'identity' loader will load in saved data at
          the stored file location, lets say they are 2d numpy arrays,
          and will return a flattened version
          of the saved arrays, with each data point as a feature.
          A more practical example
          might constitute loading in say 3D neuroimaging data,
          and passing on features as extracted by ROI.

    behav : 'single' or 'all', optional
        | The Loader object can operate under two different
          behaviors, corresponding to operations which can
          be done for each subject's Data File independently ('single')
          and operations which must be done using information
          from all train subject's Data Files ('all').

        | 'single' is the default behavior, if requested
          then the Loader will load each subject's Data File
          seperately and apply the passed `obj` fit_transform.
          The benefit of this method in contrast to 'all' is
          that only one subject's full raw data needs to be
          loaded at once, whereas with all, you must have enough
          avaliable memory to load all of the current training or
          validation subject's raw data at one. Likewise 'single'
          allows for caching fit_transform operations for each
          individual subject (which can then be more flexibly re-used).

        | Behavior 'all' is designed to work with base objects
          that accept a list of length n_subjects to their fit_transform
          function, where each element of the list will be that subject's
          loaded Data File. This behavior requires loading all data
          into memory, but allows for using information from the rest
          of the group split. For example we would need to set Loader
          to 'all' if we wanted to use
          :class:`nilearn.connectome.ConnectivityMeasure`
          with parameter kind = "tangent" as this transformer requires
          information from the rest of the loaded subjects when training.
          On the otherhand, if we used kind = "correlation",
          then we could use either behavior
          'all' or 'single' since "correlation" can be computed for each
          subject individually.

        ::

            default = 'single'

    {params}

    {scope}
        ::

            default = 'data file'

    cache_loc : str, Path or None, optional
        | Optional location in which to if set, the Loader transform
          function will be cached for each subject. These cached
          transformations can then be loaded for each subject when
          they appear again in later folds.

        .. warning ::

            If behav = 'all', then this parameter is currently
            skipped!

        Set to None, to ignore.

        ::

            default = None

    fix_n_wrapper_jobs : int or False, optional
        Typically this parameter is left as default, but
        in special cases you may want to set this. It controls
        the number of jobs fixed for the Loading Wrapper.
        This parameter can be used to set that value.

        ::

            default = False

    {extra_params}

    Notes
    --------
    If obj is passed as :class:`Pipe`, see :class:`Pipe`
    for an example on how different
    corresponding params can be passed
    to each piece individually.

    See Also
    ---------
    Dataset.add_data_files : For adding data files to :class:`Dataset`.
    Pipe : An input helper class used with this object.

    Examples
    -----------
    A basic example is shown below:

    .. ipython:: python

        import BPt as bp
        loader = bp.Loader(obj='identity')
        loader

    This specifies that the :class:`BPt.extensions.Identity` loader be used
    (which just loads and flattens files).

    '''

    _constructor = LoaderConstructor

    def __init__(self, obj, behav='single', params=0, scope='data file',
                 cache_loc=None,
                 fix_n_wrapper_jobs=False, **extra_params):

        self.obj = obj

        # Make sure valid behav param
        if behav not in ['single', 'all']:
            raise RuntimeError('behav must be either "single" or "all".')

        self.behav = behav
        self.params = params
        self.scope = scope
        self.cache_loc = cache_loc
        self.extra_params = extra_params
        self.fix_n_wrapper_jobs = fix_n_wrapper_jobs

        self._check_args()


@doc(**_bp_docs)
class Imputer(Piece):
    '''This input object is used to specify imputation steps
    for a :class:`Pipeline`.

    If there is any missing data (NaN's), then an imputation strategy
    is likely necessary (with some expections, i.e., a final model which
    can accept NaN values directly). This object allows for defining
    an imputation strategy. In general, you should need at most two
    Imputers, one for all `float` type data and one for all
    categorical data. If there is no missing data, this piece will be skipped.

    Parameters
    ----------
    obj : str
        `obj` selects the base imputation strategy to use.
        See :ref:`Imputers` for all avaliable options.
        Notably, if 'iterative' is passed,
        then a base model must also be passed!

        See :ref:`Pipeline Objects<pipeline_objects>` to read more about
        pipeline objects in general.

    {params}

    {scope}
        | The main options that make sense for imputer are
            one for `float` data and one for `category` datatypes.
            Though in some cases other choices may make sense.

        | Note: If using iterative imputation you may want to carefully
            consider the scope passed. For example, while it may be beneficial
            to impute categorical and float features seperately, i.e., with
            different base_model_type's
            (categorical for categorical and regression for float), you must
            also consider that in predicting the missing values under
            this setup, the categorical imputer would not have access to
            to the float features and vice versa.
            In this way, you may want to either just
            treat all features as float, or
            instead of imputing categorical features, load missing
            values as a separate category - and then set the scope
            here to be 'all', such that the iterative imputer has
            access to all features.

        ::

            default = 'all'

    {cache_loc}

    base_model : :class:`Model`, :class:`Ensemble` or None, optional
        If 'iterative' is passed to obj, then a base_model is required in
        order to perform iterative imputation! The base model can be
        any valid :class:`Model` or :class:`Ensemble`

        ::

            default = None

    base_model_type : 'default' or Problem Type, optional
        In setting a base imputer model, it may be desirable to
        have this model have a different 'problem type', then your
        over-arching problem. For example, if performing iterative
        imputation on categorical features only, you will likely
        want to use a categorical predictor - but for imputing on
        float-type features, you will want to use a 'regression' type
        base model.

        Choices are 'binary', 'regression', 'categorical' or 'default'.
        If 'default', then the following behavior will be applied:
        If all columns within the passed scope of this Imputer object
        have scope / data type 'category', then the problem
        type for the base model will be set to 'categorical'.
        In all other cases, the problem type will be set to 'regression'.

        ::

            default = 'default'


    {extra_params}

    '''

    _constructor = ImputerConstructor

    def __init__(self, obj, params=0, scope='all',
                 cache_loc=None, base_model=None, base_model_type='default',
                 **extra_params):

        self.obj = obj
        self.params = params
        self.scope = scope
        self.cache_loc = cache_loc
        self.base_model = deepcopy(base_model)
        self.base_model_type = base_model_type
        self.extra_params = extra_params

        self._check_args()


@doc(**_bp_docs)
class Scaler(Piece):
    '''The Scaler piece refers to
    a piece in the :class:`Pipeline` or :class:`ModelPipeline`,
    which is responsible for performing any sort of scaling or
    transformation on the data
    which doesn't require the target variable, and doesn't
    change the number of data points or features.
    These are typically transformations like feature scaling.

    Parameters
    ----------
    obj : str or custom obj
        | `obj` if passed a str selects a scaler from the preset defined
            scalers, See :ref:`Scalers`
            for all avaliable options. If passing a custom object, it must
            be a sklearn compatible
            transformer, and further must not require the target variable,
            not change the number of data
            points or features.
        |
        | See :ref:`Pipeline Objects<pipeline_objects>` to read more about
            pipeline objects in general.

    {params}

    {scope}
        ::

            default = 'float'

    {cache_loc}

    {extra_params}

    '''

    _constructor = ScalerConstructor

    def __init__(self, obj, params=0, scope='float',
                 cache_loc=None, **extra_params):

        self.obj = obj
        self.params = params
        self.scope = scope
        self.cache_loc = cache_loc
        self.extra_params = extra_params

        self._check_args()


@doc(**_bp_docs)
class Transformer(Piece):
    ''' The Transformer is base optional component of the
    :class:`Pipeline` or :class:`ModelPipeline` classes.
    Transformers define any type of transformation to the loaded
    data which may change the number
    of features in a non-simple way (i.e., conceptually distinct
    from :class:`FeatSelector`, where you
    know in advance the transformation is just selecting a subset
    of existing features). These are
    transformations like applying Principle Component Analysis,
    or on the fly One Hot Encoding.

    Parameters
    ----------
    obj : str or custom_obj
        `obj` if passed a str selects from the avaliable class
        defined options for
        transformer as found at :ref:`Transformers`.

        If a custom object is passed as `obj`, it must be a
        sklearn api compatible
        transformer (i.e., have fit, transform, get_params
        and set_params methods, and further be
        cloneable via sklearn's clone function).

        See :ref:`Pipeline Objects<pipeline_objects>` to read more about
        pipeline objects in general.

    {params}

    {scope}

        | It may in some cases be useful to consider the use of
            :class:`Duplicate` here.

        ::

            default = 'data'

    {cache_loc}

    {extra_params}
    '''

    _constructor = TransformerConstructor

    def __init__(self, obj, params=0, scope='data', cache_loc=None,
                 **extra_params):

        self.obj = obj
        self.params = params
        self.scope = scope
        self.cache_loc = cache_loc
        self.extra_params = extra_params

        self._check_args()


@doc(**_bp_docs)
class FeatSelector(Piece):
    ''' The FeatSelector class is a base piece of
    :class:`ModelPipeline` or :class:`Pipeline`, which is designed
    to preform feature selection.

    Parameters
    ----------
    obj : str or custom_obj
        | The `obj` parameter selects which
            feature selection strategy to use.

        | See :ref:`Feat Selectors` for all avaliable preset options
            and :ref:`Pipeline Objects<pipeline_objects>` to read more
            about pipeline objects in general.

        | Notably, if 'rfe' (recursive feature elimination) is passed, then a
            base model must also be passed!

    {params}

    {scope}
        ::

            default = 'all'

    {cache_loc}

    base_model : :class:`Model`, :class:`Ensemble` or None, optional
        | If 'rfe' is passed to obj, then a base_model is required in
            order to perform recursive feature elimination.

        | The base model can be any valid :class:`Model` or :class:`Ensemble`.

        ::

            default = None

    {extra_params}

    See Also
    ---------
    Scaler : For pieces which don't change the number of features.
    Transformer : For pieces which change the number of features
                  in different ways.

    '''

    _constructor = FeatSelectorConstructor

    def __init__(self, obj, params=0, scope='all',
                 cache_loc=None, base_model=None, **extra_params):

        self.obj = obj
        self.params = params
        self.scope = scope
        self.cache_loc = cache_loc
        self.base_model = deepcopy(base_model)
        self.base_model_type = None
        self.extra_params = extra_params
        self._check_args()


@doc(**_piece_docs)
class Model(Piece):
    '''The Model class represents a
    base component of the :class:`Pipeline`
    / :class:`ModelPipeline`. This component
    is namely the estimator responsible for making
    predictions or classifications.

    obj : str or custom_obj
        | The passed object should be either
            a preset str indicator found at :ref:`Models`,
            or from a custom passed user model (compatible w/ sklearn api).

        .. note::

            The passed parameter should be either a single
            str or a custom object, not a list-like of either.
            In the case that an ensemble of :class:`Model` is
            needed, see :class:`Ensemble`.

        | See :ref:`Pipeline Objects<pipeline_objects>` to
            read more about pipeline objects in general.

    {params}

    {scope}
        ::

            default = 'all'

    {param_search}

    {target_scaler}

    {cache_loc}

    {extra_params}

    See Also
    ---------
    Ensemble : For an ensemble of Models.

    '''

    _constructor = ModelConstructor

    def __init__(self, obj, params=0, scope='all',
                 param_search=None, target_scaler=None,
                 cache_loc=None, **extra_params):

        self.obj = obj
        self.params = params
        self.scope = scope
        self.cache_loc = cache_loc
        self.param_search = param_search
        self.target_scaler = target_scaler
        self.extra_params = extra_params

        self._check_args()


@doc(**_piece_docs)
class Ensemble(Model):
    '''The Ensemble object is a
    :class:`ModelPipeline` (or :class:`Pipeline`) piece,
    designed to be passed as an estimator, the same
    as :class:`Model`. This class is used to create
    a variety of different ensemble based estimators.

    Parameters
    ----------
    obj : str
        | Each str passed to ensemble refers to a type of ensemble to train,
            based on also the passed input to the `models` parameter,
            and also the additional parameters passed when
            initializing this object.

        | See :ref:`Ensemble Types` to see all
            avaliable options for ensembles.

        .. warning::

            Passing custom objects here, while technically possible,
            is not currently full supported.
            That said, there are just certain assumptions that
            the custom object must meet in order to work, specifically,
            they should have similar input params to other similar existing
            ensembles, e.g., in the case the `single_estimator` is False
            and `needs_split` is also False, then the passed object needs
            to be able to accept an input parameter `estimators`,
            which accepts a list of (str, estimator) tuples.
            Whereas if needs_split is still False,
            but single_estimator is True, then the passed object needs
            to support an init param of `base_estimator`,
            which accepts a single estimator.

    models : :class:`Model`, :class:`Ensemble` or list of
        | The `models` parameter is designed to accept any single model-like
            pipeline parameter object, i.e.,
            :class:`Model` or even another :class:`Ensemble`.
            The passed pieces here will be used along with the
            requested ensemble object to
            create the requested ensemble.

        | See :class:`Model` for how to create a valid base model(s)
            to pass as input here.

    {params}

    {scope}
        ::

            default = 'all'

    {param_search}

    {target_scaler}

    base_model : :class:`Model`, None, optional
        | In the case that an ensemble method which has
            the parameter `final_estimator` (not base model),
            for example in the case of stacking,
            then you may pass a Model type
            object here to be used as that final estimator.

        | Otherwise, by default this will be left as None,
            and if the requested ensemble has the final_estimator
            parameter, then it will pass None to the object
            (which is typically for setting the default).

        ::

            default = None

    cv : :class:`CV` or None, optional
        | Used for passing custom nested internal
            CV split behavior to
            ensembles which employ splits, e.g., stacking.

        | The passed input can be either an instance
            of :class:`CV` or can be any valid
            scikit-learn style cv, e.g., the integer 5.

        ::

            default = None

    single_estimator : bool, optional
        | The parameter `single_estimator` is used to let the
            Ensemble object know if the passed `models`
            should be a single estimator, or in other words
            if the base ensemble object is expecting the input
            as just one estimator.

        | This parameter is used for ensemble types
            that requires an init param `base_estimator`.
            In the case that multiple models
            are passed to `models`, but `single_estimator` is True,
            then the models will automatically
            be wrapped in a voting ensemble,
            thus creating one single estimator.

        ::

            default = False

    n_jobs_type : 'ensemble' or 'models', optional
        | Valid options are either 'ensemble' or 'models'.

        | This parameter controls how the total n_jobs are distributed, if
            'ensemble', then the n_jobs will be used all in the ensemble object
            and every instance within the sub-models set to n_jobs = 1.
            Alternatively, if passed 'models', then the ensemble
            object will not be multi-processed, i.e.,
            will be set to n_jobs = 1, and the n_jobs will
            be distributed to each base model.

        | For example, if you are training a stacking regressor
            with n_jobs = 16, and you have 16+ models, then 'ensemble'
            is likely a good choice here. If instead you have only
            3 base models, and one or more of those 3 could benefit from
            a higher n_jobs, then setting n_jobs_type to 'models' might
            give a speed-up.

        ::

            default = 'ensemble'

    {extra_params}

    '''

    _constructor = ModelConstructor

    # @TODO add cache_loc here? Do the wrapper's support it?

    def __init__(self, obj, models,
                 params=0, scope='all',
                 param_search=None,
                 target_scaler=None,
                 base_model=None,
                 cv=None,
                 single_estimator=False,
                 n_jobs_type='ensemble',
                 **extra_params):

        self.obj = obj

        # Force passed models if not a list, into a list
        if not isinstance(models, list):
            models = [models]
        if isinstance(models, Select):
            models = [models]

        self.models = models
        self.params = params
        self.scope = scope
        self.param_search = param_search
        self.target_scaler = target_scaler
        self.base_model = base_model
        self.cv = cv
        self.single_estimator = single_estimator
        self.n_jobs_type = n_jobs_type
        self.extra_params = extra_params

        self._check_args()

    def _check_extra_args(self):

        if isinstance(self.models, list):
            for model in self.models:
                if not isinstance(model, Model):
                    raise IOError(
                        'All models must be valid Model/Ensemble !')

        else:
            if not isinstance(self.models, Model):
                raise IOError(
                    'Passed model in models must be a valid Model/Ensemble.')


class Custom(Piece):
    '''Wrapper around pieces passed
    as valid sklearn estimator.'''

    _constructor = CustomConstructor

    def __init__(self, step):

        # Save step
        self.step = step

    def _get_step(self):

        # If already tuple return as is
        if isinstance(self.step, tuple):
            step = self.step

        # Otherwise, get as tuple
        else:
            step = _name_estimators([self.step])[0]

        # Change name if needed
        name, obj = step[0], step[1]

        # Return step
        return (name, obj)

    def __repr__(self):
        return self.step.__repr__()

    def __str__(self):
        return self.step.__str__()


class ParamSearch(Params):
    '''ParamSearch is special input object designed to be
    used with :class:`ModelPipeline` or :class:`Pipeline` that
    is used in order to define a hyperparameter search strategy.

    | When passed to :class:`Pipeline`, its search strategy is
      applied in the context of any set :ref:`Params` within the base pieces.
      Specifically, there must be at least one parameter search
      somewhere in the object ParamSearch is passed!
      All backend hyper-parameter searches make use of the
      <https://github.com/facebookresearch/nevergrad>`_ library.

    Parameters
    ----------
    search_type : str, optional
        | The type of nevergrad hyper-parameter search to conduct. See
            :ref:`Search Types<search_type_options>` for all avaliable options.

        | You may pass 'grid' here in addition to the
            supported nevergrad searches. This will use sklearn's
            GridSearch. Note in this case some of the other parameters
            are ignored, these are: weight_scorer, mp_context, dask_ip,
            memmap_X, search_only_params

        ::

            default = 'RandomSearch'

    cv : :class:`CV` or 'default', optional
        | The hyper-parameter search works by internally
            evaluating each combination of parameters. In
            order to internally evaluate a set of parameters,
            there must be some type of cross-validation defined.
            This parameter is used to represent the choice of
            cross-validation to use. The set of parameters which
            achieves the highest average score across the folds
            defined here will be selected.

        | Passed input should be an instance of :class:`CV`.
            If left as the custom str 'default', then a
            :class:`CV` object will be initialized and used
            with just the default parameters
            (which is a once repeated 3-fold cross-validation).

        ::

            default = 'default'

    n_iter : int, optional
        | This parameter represents the number of different
            hyper-parameters to try. It can also be thought of
            as the budget given to the underlying search algorithm.

        | How well a hyper-parameter search works
            (i.e., the quality of the chosen parameters) and how long
            it takes to run, are quite dependent on both this parameter
            and the passed cv strategy. In general, if too few
            choices are provided the algorithm will likely not select high
            performing hyperparameters, and alternatively
            if too high a value/budget is set, then you may find
            overfit/non-generalize hyper-parameter choices.

        | Other factors which might influence the 'right' number of
            `n_iter` to specify are:

            - `search_type`
                Depending on the underlying search type, it may
                take a bigger or smaller budget
                on average to find a good set of hyper-parameters

            - The dimension of the underlying search space
                If you are only optimizing a few, say 2,
                underlying parameter distributions,
                this will require a far smaller budget then say a really
                high dimensional search space.

            - The CV strategy
                The CV strategy defined via `cv`
                may make it easier or harder to overfit when searching
                for hyper-parameters, thus conceptually
                a good choice of cross-validation strategy can
                serve to increase the number `n_iter` you can use
                before overfitting or alternatively a bad choice may limit it.

            - Number of data points / subjects
                Along with CV strategy, the number of data points/subjects
                will greatly influence
                how quickly you overfit, and therefore a
                good choice of `n_iter`.

        ::

            default = 10

    scorer : str or 'default', optional
        | In order for a set of hyper-parameters to be evaluated,
            a single scorer must be defined. In the case
            that multiple scorer's are passed here, the first
            one will be used and the others ignored.

        | For a full list of supported scorers please view the
            scikit-learn docs at:
            https://scikit-learn.org/stable/modules/model_evaluation.html#the-scoring-parameter-defining-model-evaluation-rules

        .. note::

            If selecting a custom (i.e., anything but 'default'),
            be careful to make sure to select an appropriate scorer for
            the underlying problem type.

        | If left as 'default', a reasonable scorer based on the
            problem type is used.

        - 'regression'  : 'explained_variance'
        - 'binary'      : 'matthews'
        - 'categorical' : 'matthews'

        ::

            default = 'default'

    weight_scorer : bool or 'default', optional
        | The `weight_scorer` parameter allows for
            optionally weighting the scores by
            the number of subjects within each validation fold.
            The mean score is then if set to True, a weighted average
            instead.

        | This parameter is typically only useful in the case
            where the folds vary dramatically by size (e.g., in
            the case of a leave-out-group cross-validation, where
            the groups vary in size).

        ::

            default = False

    mp_context : str, optional
        | When a hyper-parameter search is launched there are different
            ways through python that the multi-processing can be launched
            (assuming n_jobs > 1). Depending on the system running the code,
            some options may be more reliable than others.

        | Valid options are:

            - 'loky': Create and use the python library loky backend.

            - 'fork': Python default fork mp_context

            - 'forkserver': Python default forkserver mp_context

            - 'spawn': Python default spawn mp_context

        | New as of version 1.3.6+, the 'loky' backend will be used,
            which is quite reliable across a number of systems and
            shouldn't really need to be changed.

        ::

            default = 'loky'

    n_jobs : int or 'default', optional
        | This parameter can be set in the case that
            a specific number of jobs (i.e., processors)
            should be used to run this parameter search.

        | If left as the default value of 'default'
            then the choice of n_jobs will be inherited
            through the context in which whatever
            object this search is associated with is
            used. This is typically a good choice,
            and this value can be left as 'default'.

        ::

            default = 'default'

    random_state : int, None or 'default', optional
        | If left as 'default', the random_state
            as set through an associated :class:`ProblemSpec`
            will be used. Otherwise, you may specify a
            specific value here. Either an integer representing
            a fixed random state or None to specify a
            new random state each time.

        ::

            default = 'default'


    dask_ip : str or None, optional
        If None, the default, then ignore this parameter.
        Otherwise, this parameter represents experimental Dask support.
        In this case the passed parameter should be a string representing
        the ip of a created dask cluster. A dask Client object will then
        be created and passed this ip in order to connect to the cluster.

        .. warning::

            This functionality is still experimental, and
            will not work if the underlying search_type is 'grid'.

        Built in to using dask to evaluate each combination of parameters
        is pre-scattering the data to each cluster node.

        ::

            default = None

    memmap_X : bool, optional
        | When passing large memory arrays in each parameter search,
            it can be useful as a memory reduction technique to pass
            numpy memmap'ed arrays. This solves an issue where
            the loky backend will not properly pass too large arrays.

        .. warning::

            This option can slow down code a large amount, and typically
            should not be used unless out of memory errors are encountered.
            This option will be skipped if the underlying
            search_type is 'grid', or is using `dask_ip` also.
            In the case of using `dask_ip` large data will be
            pre-scattered instead anyway.

        ::

            default = False

    search_only_params : dict or None, optional
        | In some rare cases, it may be the case that you want
            to specify that certain parameters be passed only during
            the nested parameter searches. A dict of parameters
            can be passed here to accomplish that. For example,
            if passing:

            ::

                search_only_params = {'svm classifier__probability': False}

        | Assuming that the default / selecting parameter for this svm
            classifier for probability is True by default, then only when
            exploring nested hyper-parameter options will probability be set
            to False, but when fitting the final model with the best
            parameters found from the search, it will revert back to
            the default (i.e., in this case probability = True).

        .. note::

            This may be difficult for non-advanced users to
            use, as you must pass parameters exactly as how
            they are represented internally. Using build
            on the piece of interest may be helpful in figuring
            out what this looks like for a specific piece and parameter.

        | To ignore this parameter / option.
            simply keep the default value of None.

        ::

            default = None

    verbose : int, optional
        Controls the verbosity of the search, where
        the higher value set, the more messages will be printed.
        By default no verbosity(i.e., verbose=0) will be used.

        ::

            default = 0

    progress_loc : None or str, optional
        This is an optional parameter. If set
        to non-None, then it should be passed a
        str representing the location of a text file
        to append to after every completed parameter.

        ::

            default = None
    '''

    def __init__(self, search_type='RandomSearch', cv='default',
                 n_iter=10, scorer='default', weight_scorer=False,
                 mp_context='loky', n_jobs='default',
                 random_state='default',
                 dask_ip=None, memmap_X=False,
                 search_only_params=None, verbose=0, progress_loc=None):
        self.search_type = search_type

        if cv == 'default':
            cv = CV()
        self.cv = cv

        # @TODO allow passing single arguments, e.g. splits
        # as param and have it set within cv

        self.n_iter = n_iter
        self.scorer = scorer
        self.weight_scorer = weight_scorer
        self.mp_context = mp_context
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.dask_ip = dask_ip
        self.memmap_X = memmap_X
        self.verbose = verbose
        self.progress_loc = progress_loc
        self.search_only_params = search_only_params

        self._check_args()

    def _as_dict(self, ps):

        params = self.get_params()

        if self.random_state == 'default':
            params['random_state'] = ps.random_state

        if self.n_jobs == 'default':
            params['n_jobs'] = ps.n_jobs

        params['scorer'] = process_scorer(self.scorer,
                                          ps.problem_type)

        if self.search_only_params is None:
            params['search_only_params'] = {}

        return params

    def _check_args(self):

        if isinstance(self.scorer, list):
            raise IOError('scorer within Param Search cannot be list-like')
        if isinstance(self.weight_scorer, list):
            raise IOError(
                'weight_scorer within Param Search cannot be list-like')


def check_if_sklearn_step(step):

    # Skip def. not valid cases
    if isinstance(step, (Piece, Pipeline, BPtInputMixIn, str)):
        return False

    # If Tuple
    if isinstance(step, tuple):
        est = step[1]

    # Otherwise assume directly as obj
    else:
        est = step

    # Use sklearn base check
    try:
        check_estimator(est)
    except AttributeError:
        return False

    return True


def add_nested_deep_params(params):
    '''When passed shallow params, get
    nested deep params w/ possibility for
    list.'''

    for param_key in list(params):
        value = params[param_key]

        # If parameter value is list
        if isinstance(value, list):
            for i, step in enumerate(value):

                # Compare case
                if isinstance(step, Compare):
                    params[param_key + '__' + str(i)] = step
                    continue

                # Skip non objects
                if not hasattr(step, 'get_params'):
                    continue

                # Get step params
                step_params = step.get_params(deep=True)

                # Add parameter
                for key in step_params:
                    new_key = param_key + '__' + str(i) + '__' + key
                    params[new_key] = step_params[key]

        # Otherwise, check for nested
        else:

            # Skip non objects
            if not hasattr(value, 'get_params'):
                continue

            # Get value params if object
            value_params = value.get_params(deep=True)

            for key in value_params:
                params[param_key + '__' + key] = value_params[key]


_pipeline_docs = {}
_pipeline_docs['param_search'] = _piece_docs['param_search']

_pipeline_docs['cache_loc'] = """cache_loc : Path str or None, optional
        Optional parameter specifying a directory
        in which full BPt pipeline's should
        be cached after fitting. This should be
        either left as None, or passed a str representing
        a directory in which cached fitted pipeline should be saved.

        ::

            default = None
"""

_pipeline_docs['verbose'] = """verbose : bool, optional
        If True, then print statements about
        the current progress of the pipeline during fitting.

        Note: If in a multi-processed context, where pipelines
        are being fit on different threads, verbose output may be
        messy (i.e., overlapping messages from different threads).

        ::

            default = 0
"""


@doc(**_pipeline_docs)
class Pipeline(Params):
    '''This class is used to create flexible BPt style pipeline's.

    See :class:`ModelPipeline` for an alternate version of this class
    which enforces a strict ordering on how pipeline pieces can be set, and
    also includes a number of useful default behaviors.

    Parameters
    -----------
    steps : list of :ref:`Pipeline Objects<pipeline_objects>`
        | Input here is a list of :ref:`Pipeline Objects<pipeline_objects>`
            or custom valid sklearn-compliant objects / pipeline steps.
            These can be in any order as long as there is only
            one :class:`Model` or :class:`Ensemble`
            and it is at the end of the list, i.e., the last step.
            This constraint excludes any nested models.

        | See below for example usage.

    {param_search}

    {cache_loc}

    {verbose}

    Notes
    -------
    This class differs from :class:`sklearn.pipeline.Pipeline`
    most drastically in that this class is not itself directly a
    sklearn-complaint estimator (i.e., an object with fit and predict methods).
    Instead, this object represents a flexible set of input pieces,
    that can all vary depending on the eventual :class:`Dataset`
    and :class:`ProblemSpec` they are used in the context of.
    This means that instances of this class can be easily
    re-used across different data and setups, for example
    with different underlying problem_types (running a binary and then
    a regression version).

    Examples
    ----------
    The base behavior is to use all valid Pipeline objects,
    for example:

    .. ipython:: python

        pipe = bp.Pipeline(steps=[bp.Imputer('mean'),
                                  bp.Scaler('robust'),
                                  bp.Model('elastic')])
        pipe

    | This would creates a pipeline with mean imputation, robust scaling
      and an elastic net, all using the BPt style custom objects.

    | This object can also work with :class:`sklearn.pipeline.Pipeline`
      style steps. Or a mix of BPt style and sklearn style, for example:

    .. ipython:: python

        from sklearn.linear_model import Ridge

        pipe = bp.Pipeline(steps=[bp.Imputer('mean'),
                                  bp.Scaler('robust'),
                                  ('ridge regression', Ridge())])
        pipe

    You may also pass sklearn objects directly instead of as
    a tuple, i.e., in the :func:`sklearn.pipeline.make_pipeline`
    input style. For example:

    .. ipython:: python

        from sklearn.linear_model import Ridge
        pipe = bp.Pipeline(steps=[Ridge()])
        pipe

    .. note::

        Passing objects as sklearn-style
        ensures they have essentially a scope of 'all'
        and no associated hyper-parameter distributions.

    '''

    def __init__(self, steps, param_search=None,
                 cache_loc=None, verbose=False):
        self.steps = steps
        self.param_search = param_search
        self.cache_loc = cache_loc
        self.verbose = verbose

        self._proc_checks()

    def _proc_checks(self):

        # Check if steps passed as single model
        self._check_steps_is_model()

        # Check for any custom passed steps
        self._check_for_custom()

        # Check last step
        self._check_last_step()

        # Validate rest of input
        self._validate_input()

        # Proc duplicates, then flatten
        self.steps = self._proc_duplicates(self.steps)
        self._flatten_steps()

        # Uniquify
        self._uniquify()

        # Proc input
        self.steps = self._proc_input(self.steps)

    def _check_steps_is_model(self):

        # If passed steps as just one step
        # wrap in list, though has to be potentially
        # valid model step
        if isinstance(self.steps, self._model_like):
            self.steps = [self.steps]

    def _check_for_custom(self):

        for i, step in enumerate(self.steps):

            # If is custom estimator, wrap in Custom
            if check_if_sklearn_step(step):
                self.steps[i] = Custom(step)

    def _check_last_step(self):

        if not isinstance(self.steps, list):
            raise TypeError('steps must be a list!')

        if isinstance(self.steps, Select):
            raise TypeError('steps cannot be Select')

        if len(self.steps) == 0:
            raise IndexError('steps cannot be empty!')

        # If last step is str, conv to Model
        if isinstance(self.steps[-1], str):
            self.steps[-1] = Model(self.steps[-1])

        # Make sure model
        self._check_is_model(self.steps[-1])

    def _check_is_model(self, step):

        # If bpt mixin, extra cases
        if isinstance(step, BPtInputMixIn):

            if isinstance(step, Compare):
                for option in step.options:
                    assert isinstance(option, Option)

            elif isinstance(step, Select):
                [self._check_is_model(o) for o in step]

            # Note: Pipe case not used for model
            else:
                raise RuntimeError(repr(step) + ' is not a valid Model')

        # If last step isn't model - or custom, raise error
        if not isinstance(self.steps[-1], self._model_like):
            raise RuntimeError(repr(step) + ' is not a valid Model')

    def _validate_input(self):
        '''Make sure all steps are Pieces, and param
        search is valid if not None'''

        # Validate steps
        for step in self.steps:

            if not isinstance(step, self._step_like):
                raise RuntimeError('passed step:' + repr(step) +
                                   ' is not a valid Pipeline Piece / '
                                   'input wrapper')

            step._check_args()

        # Validate param search
        if self.param_search is not None:
            if not isinstance(self.param_search, ParamSearch):
                raise RuntimeError(repr(self.param_search) + 'is not a'
                                   ' ParamSearch.')

            # Check input args in case anything changed
            self.param_search._check_args()

    def _uniquify(self):

        # First call recursively
        for step in self.steps:
            if isinstance(step, self._step_like):
                step._uniquify()

        # Then set steps with copy of steps
        self.steps = [copy(step) for step in self.steps]

    def _check_args(self):

        for step in self.steps:
            step._check_args()

    def _flatten_steps(self):

        # If just one element cast to list and end
        if not isinstance(self.steps, list):
            self.steps = [self.steps]
            return
        if isinstance(self.steps, Select):
            self.steps = [self.steps]
            return

        steps = []
        for step in self.steps:
            if isinstance(step, list) and not isinstance(step, BPtInputMixIn):
                for sub_step in step:
                    steps.append(sub_step)
            else:
                steps.append(step)

        self.steps = steps

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

            if isinstance(scopes, Duplicate):
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

    def _get_steps(self):
        return self.steps

    def _check_imputers(self, is_na):

        # If no NaN set to none
        if not is_na:
            self.imputers = None

    def get_params(self, deep=True):

        # Base behav get top level
        params = super().get_params(deep=False)

        # Then proc if deep
        if deep:
            add_nested_deep_params(params)

        return params

    def set_params(self, **params):

        for key in params:
            value = params[key]

            # Get last key
            split_key = key.split('__')
            last_key = split_key[-1]

            obj = self
            for key_piece in split_key[:-1]:

                # If int
                try:
                    indx = int(key_piece)
                    obj = obj[indx]

                # If parameter
                except ValueError:
                    obj = getattr(obj, key_piece)

            # Set value via set params if avaliable / not self
            if obj != self and hasattr(obj, 'set_params'):
                obj.set_params(**{last_key: value})
                return self

            # Check if last key is int
            try:
                indx = int(last_key)
                obj[indx] = value

            except ValueError:
                setattr(obj, last_key, value)

            return self

    def build(self, dataset='default', problem_spec='default',
              **problem_spec_params):
        '''This method generates a sklearn compliant :ref:`estimator<develop>`
        version of the current :class:`Pipeline` with respect to a passed
        dataset and :class:`Dataset` and :class:`ProblemSpec`.

        This method calls :func:`get_estimator` with pipeline set as
        itself.

        Parameters
        -----------
        dataset : :class:`Dataset` or 'default', optional
            The Dataset in which the pipeline should be initialized
            according to. For example, pipeline's can include scopes,
            which require a reference dataset.

            If left as default will initialize and use
            an instance of a FakeDataset class, which will
            work fine for initializing pipeline objects
            with scope of 'all', but should be used with caution
            when elements of the pipeline use non 'all' scopes.
            In these cases a warning will be issued.

            It is advisable to use this
            build function only for viewing pipelines. If using
            the build function instead for eventual modelling it
            is important to pass the correct :class:`Dataset` in
            the case that any of the pipeline pieces are at all
            dependant on the structure of the input data.

            Note: If problem type is not defined in problem_spec
            and Dataset is left as default, then a problem type of
            'regression' will be used.

            ::

                default = 'default'

        problem_spec : :class:`ProblemSpec` or 'default', optional
            This parameter accepts an instance of the
            params class :class:`ProblemSpec`.
            The ProblemSpec is essentially a wrapper
            around commonly used
            parameters needs to define the context
            the model pipeline should be evaluated in.
            It includes parameters like problem_type, scorer, n_jobs,
            random_state, etc...

            See :class:`ProblemSpec` for more information
            and for how to create an instance of this object.

            If left as 'default', then will initialize a
            ProblemSpec with default params.

            ::

                default = "default"

        problem_spec_params : :class:`ProblemSpec` params, optional
            You may also pass any valid problem spec argument-value pairs here,
            in order to override a value in the passed :class:`ProblemSpec`.
            Overriding params should be passed in kwargs style, for example:

            ::

                func(..., problem_type='binary')

        Returns
        -------
        estimator : sklearn compatible estimator
            Returns the BPt-style sklearn compatible estimator
            version of this piece as converted to internally
            when building the pipeline

        params : dict
            Returns a dictionary with any parameter distributions
            associated with this object, for example
            this can be used to check what exactly
            pre-existing parameter distributions point
            to.
        '''

        from .funcs import get_estimator

        return get_estimator(pipeline=self, dataset=dataset,
                             problem_spec=problem_spec,
                             **problem_spec_params)

    @property
    def _model_like(self):
        # Compare, Select and Pipe are all BPtInputMixIn
        # though they can store model_like.
        return (Model, Pipeline, Custom, Compare, Select, Pipe)

    @property
    def _step_like(self):
        # Compare, Select and Pipe are all BPtInputMixIn
        # though they can store piece.
        return (Piece, Pipeline, Compare, Select, Pipe)


@doc(**_pipeline_docs)
class ModelPipeline(Pipeline):
    '''The ModelPipeline class is used to create BPtPipeline's.
    The ModelPipeline differs from :class:`Pipeline`
    in that it enforces a simplification on the ordering of
    pieces, representing the typical order in which they might appear.
    See :class:`Pipeline` for a more flexible version of this class
    which does not enforce any ordering.

    The order enforced, which follows the order of the input arguments, is:

    1. loaders,
    2. imputers
    3. scalers
    4. transformers
    5. feat_selectors
    6. model

    For each parameter, which the exception of model,
    you may pass either one instance of that piece
    or a list of that piece. In the case that
    a list is passed, then it will be treated as
    a sequential set of steps / transformation
    where the output from each element of the
    list if passed on to the next as input.

    Parameters
    ----------
    loaders : :class:`Loader` or list of, optional
        Each :class:`Loader` refers to transformations
        which operate on loaded Data_Files
        See :class:`Loader`.

        You may wish to consider using the
        :class:`Pipe` input class
        when creating a single :class:`Loader` obj

        Passed loaders can also be wrapped in a
        :class:`Select` wrapper, e.g., as either

        ::

            # Just passing select
            loaders = Select([Loader(...), Loader(...)])

            # Or nested
            loaders = [Loader(...), Select([Loader(...), Loader(...)])]

        In this way, most of the pipeline objects can accept lists,
        or nested lists with param wrapped, not just loaders!

        ::

            default = None

    imputers : :class:`Imputer`, list of or None, optional
        If there is any missing data (NaN's) that have been kept
        within the input data, then an imputation strategy should likely
        be defined. This param controls what kind of
        imputation strategy to use.

        See :class:`Imputer`.

        You may also pass a value of 'default' here, which in
        the case that any NaN data is present within the training
        set, the set of two imputers:

        ::

            'default' == [Imputer('mean', scope='float'),
                          Imputer('median', scope='category')]

        Will be used. Otherwise, if there is no NaN present
        in the input data, no Imputer will be used.

        ::

            default = 'default'

    scalers : :class:`Scaler`, list of or None, optional
        Each :class:`Scaler` refers to any potential data scaling where a
        transformation on the data
        (without access to the target variable) is
        computed, and the number of features or
        data points does not change.

        See :class:`Scaler`.

        By keeping the default behavior, with parameter, 'default',
        standard scaling is applied,
        which sets each feature to have a std of 1 and mean 0.

        ::

            default = 'default'

    transformers : :class:`Transformer`, list of or None, optional
        Each :class:`Transformer` defines a type of transformation to
        the data that changes the number of
        features in perhaps non-deterministic
        or not simply removal
        (i.e., different from feat_selectors), for example
        applying a PCA, where both the number of features change, but
        also the new features do not 1:1 correspond to the original
        features. See :class:`Transformer` for more information.

        ::

            default = None

    feat_selectors : :class:`FeatSelector`, list of or None, optional
        Each :class:`FeatSelector` refers to an optional
        feature selection stage of the Pipeline.

        See :class:`FeatSelector`.

        ::

            default = None

    model : :class:`Model`, :class:`Ensemble`, optional
        This parameter accepts one input of type
        :class:`Model` or :class:`Ensemble`. Though,
        while it cannot accept a list (i.e., no sequential behavior), you
        may still pass Input Type wrapper like :class:`Select` to perform
        model selection via param search.

        See :class:`Model` for more information on how to specify a single
        model to BPt, and :class:`Ensemble` for information on how
        to build an ensemble of models.

        This parameter cannot be None. In the default
        case of passing 'default', a ridge regression
        is used.

        ::

            default =  'default'

    {param_search}

    {cache_loc}

    {verbose}

    '''

    @property
    def fixed_piece_order(self):
        '''ModelPipeline has a fixed order in which
        pieces are constructed, this is:

            1. loaders,
            2. imputers
            3. scalers
            4. transformers
            5. feat_selectors
            6. model

        '''
        return ['loaders', 'imputers', 'scalers',
                'transformers', 'feat_selectors', 'model']

    def __init__(self,
                 loaders=None, imputers='default',
                 scalers='default', transformers=None,
                 feat_selectors=None,
                 model='default',
                 param_search=None,
                 cache_loc=None,
                 verbose=False):

        if isinstance(loaders, str):
            loaders = Loader(loaders)
        self.loaders = loaders

        if imputers == 'default':
            imputers = Imputer('default')
        elif isinstance(imputers, str):
            imputers = Imputer(imputers)
        self.imputers = imputers

        if isinstance(scalers, str):
            if scalers == 'default':
                scalers = Scaler('standard')
                if verbose > 0:
                    print('Passed default scalers, setting to:',
                          scalers)
            else:
                scalers = Scaler(scalers)

        self.scalers = scalers

        if isinstance(transformers, str):
            transformers = Transformer(transformers)
        self.transformers = transformers

        if isinstance(feat_selectors, str):
            feat_selectors = FeatSelector(feat_selectors)
        self.feat_selectors = feat_selectors

        if model == 'default':
            model = Model('ridge')
            if verbose > 0:
                print('Passed default model, setting to:', model)

        elif isinstance(model, str):
            model = Model(model)
        self.model = model

        self.param_search = param_search
        self.cache_loc = cache_loc
        self.verbose = verbose

        # Perform all preproc on input which can be run
        # more then once, these are essentially checks on the input
        self._proc_checks()

    def _proc_all_pieces(self, func):
        '''Helper to proc all pieces with a function
        that accepts a list of pieces'''

        for param_name in self.fixed_piece_order:
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

    def _uniquify(self):

        for param_name in self.fixed_piece_order:
            params = getattr(self, param_name)

            # Call recursively
            if isinstance(params, list):
                [p._uniquify() for p in params]
            elif hasattr(params, '_uniquify'):
                params._uniquify()

            # Then set params with copy
            setattr(self, param_name, copy(params))

    def _check_args(self, params):

        for p in params:
            if isinstance(p, list):
                self._check_args(p)
            else:
                p._check_args()

        return params

    def _check_imputers(self, is_na):

        # If no NaN set to none
        if not is_na:
            self.imputers = None

        elif isinstance(self.imputers, Imputer):
            if self.imputers.obj == 'default':
                self.imputers = [Imputer('mean', scope='float'),
                                 Imputer('median', scope='category')]

                if self.verbose > 0:
                    print('Default passed, setting imputers to:',
                          self.imputers)

        return

    def _proc_checks(self):

        try:
            self.model = self.models
            print('Passed models, set as model')
        except AttributeError:
            pass

        try:
            self.transformers = self.transformer
            print('Passed transformer, set as transformers')
        except AttributeError:
            pass

        try:
            self.imputers = self.imputer
            print('Passed imputer, set as imputers')
        except AttributeError:
            pass

        try:
            self.feat_selectors = self.feat_selector
            print('Passed feat_selector, set as feat_selectors')
        except AttributeError:
            pass

        try:
            self.scalers = self.scaler
            print('Passed scaler, set as scalers')
        except AttributeError:
            pass

        try:
            self.loaders = self.loader
            print('Passed loader, set as loaders')
        except AttributeError:
            pass

        to_check = ['ensemble', 'scorer', 'metric',
                    'feature_selector', 'feature_selectors']
        for p in to_check:
            if hasattr(self, p):
                print('Warning: ModelPipeline user set param', p,
                      ' was set,',
                      'but will have no effect as it is not',
                      ' a valid parameter!')

        # Uniquify
        self._uniquify()

        # Check for duplicate scopes
        self._proc_all_pieces(self._proc_duplicates)

        # Proc input
        self._proc_all_pieces(self._proc_input)
        proc_all(self.param_search)

        # Double check input args in case something changed
        self._proc_all_pieces(self._check_args)

        # Proc param search if not None
        if self.param_search is not None:
            self.param_search._check_args()

    def _get_steps(self):

        # Return as a flat list in order
        params = []
        for piece_name in self.fixed_piece_order:

            piece = getattr(self, piece_name)
            if piece is None:
                continue

            params += conv_to_list(piece)

        return params

    def _get_params_by_step(self):

        return [conv_to_list(getattr(self, piece_name))
                for piece_name in self.fixed_piece_order]

    def _get_indent(self, indent):

        if indent is None:
            return ''
        else:
            ind = 0
            for s in self._p_stack:
                if s == ']':
                    ind += 1
                elif s == '])':
                    ind += 8

            return ' ' * ind

    def _params_print(self, params, indent, _print=print, end='\n'):

        if not isinstance(params, list):
            _print(self._get_indent(indent), params, sep='', end=end)
            return

        elif len(params) == 1:
            self._params_print(params[0], 0, _print=_print)
            return

        elif not isinstance(params, BPtInputMixIn):

            _print(self._get_indent(indent), '[', sep='', end='')
            self._p_stack.append(']')
            self._params_print(params[0], None, _print=_print, end='')
            _print(',', sep='')

        elif isinstance(params, Select):
            _print(self._get_indent(indent), 'Select([', sep='', end='')
            self._p_stack.append('])')
            self._params_print(params[0], None, _print=_print, end='')
            _print(',', sep='')

        for param in params[1:-1]:
            self._params_print(param, 0, _print=_print, end='')
            _print(',', sep='')

        self._params_print(params[-1], 0, _print=_print, end='')
        _print(self._p_stack.pop(), end=end)

        return

    def print_all(self, _print=print):
        '''This method can be used to print a formatted
        representation of this object.'''

        self._p_stack = []

        _print('ModelPipeline')
        _print('-------------')

        pipeline_params = self._get_params_by_step()
        for name, params in zip(self.fixed_piece_order, pipeline_params):

            if params is not None:
                _print(name + '=\\')
                self._params_print(params, 0, _print=_print)
                _print()

        _print('param_search=\\')
        _print(self.param_search)
        _print()


class ProblemSpec(Params):
    '''Problem Spec is defined as an object encapsulating the set of
    parameters used by different :ref:`Evaluation Functions<api.evaluate>`

    Parameters
    ----------
    target : int or str, optional
        The target variable to predict, where the target variable
        is a loaded column within the :class:`Dataset` eventually used
        that has been set to :ref:`Role` target.

        This parameter can be passed either as the name of the
        column, or by passing an integer index. If passed
        an interger index (default = 0), then all loaded target
        variables will be sorted in alphabetical order and that
        index used to select the target to model.

        ::

            default = 0

    scorer : str or list, optional
        Indicator str for which scorer(s) to use when calculating
        validation scores in the context of different
        :ref:`Evaluation Functions<api.evaluate>`.

        A list of str's can be passed as well, in this case, scores for
        all of the requested scorers will be calculated and returned.
        In some cases though, for example :func:`cross_val_score` only
        one scorer can be used, and if passed a list here, the first
        element of the list will be used.

        Note: If using a nested :class:`ParamSearch`, this object
        has its own separate scorer param.

        For a full list of the base sklearn supported scorers please view the
        scikit-learn docs at:
        https://scikit-learn.org/stable/modules/model_evaluation.html#the-scoring-parameter-defining-model-evaluation-rules

        You can also view the BPt reference to these options at :ref:`Scorers`.

        If left as 'default', reasonable scorers will be assigned based
        on the underlying problem type.

        - 'regression'  : ['explained_variance', 'neg_mean_squared_error']
        - 'binary'      : ['matthews', 'roc_auc', 'balanced_accuracy']
        - 'categorical' : ['matthews', 'roc_auc_ovr', 'balanced_accuracy']

        ::

            default = 'default'

    scope : :ref:`Scope`, optional
        This parameter allows for specifying that
        only subset of columns be used in what modelling
        this ProblemSpec is passed to.

        See :ref:`Scope` for a more detailed explained / guide
        on how scopes are defined and used within BPt.

        ::

            default = 'all'

    subjects : :ref:`Subjects`, optional
        This parameter allows for specifying that
        the current experiment be run with only a subset of
        the current subjects.

        A common use of this parameter is to
        specify the reserved keyword 'train' to
        specify that only the training subjects should be used.

        If set to 'default', special behavior will be
        used where if a train/test split is defined then subjects
        will be set to 'train' by default (unless cv='test', then subjects
        will be set to 'all'). If a train/test split is not defined,
        then subjects will be set to 'all'.

        See :ref:`Subjects` for more information
        of the different accepted BPt subject style inputs.

        ::

            default = 'default'

    problem_type : str or 'default', optional
        This parameter controls what type of machine learning
        should be conducted. As either a regression, or classification
        where 'categorical' represents a special case of
        binary classification, where typically a binary
        classifier is trained on each class.

        - 'default'
            Determine the problem type based on how
            the requested target variable is loaded.

        - 'regression', 'f' or 'float'
            For ML on float/continuous target data.

        - 'binary' or 'b'
            For ML on binary target data.

        - 'categorical' or 'c'
            For ML on categorical target data, as multiclass.

        This can almost always be left as default.

        ::

            default = 'default'

    n_jobs : int
        n_jobs are employed within the context
        of a call to Evaluate or Test.

        In general, the way n_jobs are propagated to the different pipeline
        pieces on the backend is that,
        if there is a parameter search, the base
        ML pipeline will all be set to use 1 job,
        and the n_jobs budget will be used
        to train pipelines in parallel to explore different params.
        Otherwise, if no param search,
        n_jobs will be used for each piece individually,
        though some might not support it.

        ::

            default = 1

    random_state : int, RandomState instance or None, optional
        Random state, either as int for a specific seed, or if None then
        the random seed is set by np.random.

        This parameter is used to ensure replicability
        of experiments (wherever possible!).
        In some cases even with a random seed, depending on
        the pipeline pieces being used,
        if any have a component that occassionally yields
        different results, even with the same
        random seed, e.g., some model optimizations,
        then you might still not get
        exact replicability.

        .. note::

            There are some good arguments to be made for not using a
            fixed seed in some cases. See:
            https://scikit-learn.org/stable/common_pitfalls.html#controlling-randomness


        ::

            default = 1

    base_dtype : numpy dtype
        The dataset is cast to a numpy array of float.
        This parameter can be used to change the default
        behavior, e.g., if more resolution or less is needed.

        ::

            default = 'float32'

    '''
    def __init__(self, target=0, scorer='default',
                 scope='all', subjects='default',
                 problem_type='default',
                 n_jobs=1, random_state=1,
                 base_dtype='float32'):

        self.problem_type = problem_type
        self.target = target
        self.scorer = scorer
        self.scope = scope
        self.subjects = subjects
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.base_dtype = base_dtype

        self._checked = False

        self._proc_checks()

    def _proc_checks(self):
        proc_all(self)

    def print_all(self, _print=print):
        '''This method can be used to print a formatted
        representation of this object.'''

        _print('ProblemSpec')
        _print('------------')
        _print('problem_type =', self.problem_type)
        _print('target =', self.target)
        _print('scorer =', self.scorer)
        _print('scope =', self.scope)

        if isinstance(self.subjects, ValueSubset):
            _print('subjects =', self.subjects)
        elif len(self.subjects) < 50:
            _print('subjects =', self.subjects)

        _print('n_jobs =', self.n_jobs)
        _print('random_state =', self.random_state)
        _print()

    def _get_spec(self):

        return {'n_jobs': self.n_jobs,
                'random_state': self.random_state,
                'problem_type': self.problem_type,
                'scope': self.scope}


class CVStrategy(Params):
    '''This objects is used to encapsulate a set of parameters
    representing a cross-validation strategy.

    Parameters
    -----------
    groups : str or None, optional
        The str should refer to the column key in which
        to preserve groups by during any CV splits.
        To create a combination of unique values, use
        :func:`Dataset.add_unique_overlap`.

        Note: the passed column must be of type category as well.

        ::

            default = None

    stratify : str or None, optional
        The str input should refer to a loaded non input
        variable which if of type category. It will assign it
        as a value to preserve
        distribution of groups by during any during any CV splits.

        To create a combination of unique values, use
        :func:`Dataset.add_unique_overlap`.

        Note: the passed column must be of type category as well.

        Any target_cols passed must be categorical
        or binary, and cannot be
        float. Though you may consider creating a binary / k bin copy
        of a float / cont. type target variable.

        ::

            default = None

    train_only_subjects : None or :ref:`subjects`, optional
        If passed  any valid :ref:`subjects` style input here,
        these subjects will be condiered train only and will
        be assigned to every training fold, and never to
        a testing or validation fold.

        ::

            default = None
    '''

    # @TODO add support for test_only ?
    def __init__(self, groups=None, stratify=None,
                 train_only_subjects=None):

        self.groups = groups
        self.stratify = stratify
        self.train_only_subjects = train_only_subjects

        if groups is not None and stratify is not None:
            raise RuntimeError('Warning: BPt does not currently '
                               'support groups and'
                               'stratify together!'
                               'Please use only one.')

        if not isinstance(groups, str) and groups is not None:
            raise RuntimeError('Passing non str for groups is depreciated. '
                               ' Instead, first use the function'
                               ' add_unique_overlap(cols=..., new_cols=...) to'
                               ' create explicitly the specified new values.')

        if not isinstance(stratify, str) and stratify is not None:
            raise RuntimeError('Passing non str for stratify is depreciated. '
                               ' Instead, first use the function'
                               ' add_unique_overlap(cols=..., new_cols=...) to'
                               ' create explicitly the specified new values.')


class CV(Params):
    ''' This object is used to define a BPt style custom
    CV strategy, e.g., as KFold

    Parameters
    ----------
    splits : int, float, str or list of str, optional
        `splits` allows you to specify the base of what CV strategy
        should be used.

        Specifically, options for split are:

        - int
            The number of k-fold splits to conduct. (E.g., 3 for
            3-fold CV split to be conducted at
            every hyper-param evaluation).

        - float
            Must be 0 < `splits` < 1, and defines a
            single train-test like split,
            with `splits` % of the current training data size used
            as a validation set.

        - str
            If a str is passed, then it must correspond to a
            loaded categorical non input variable. In
            this case, a leave-out-group CV will be used according
            to the value of the variable.

        Note that the parameter n_repeats is designed to work with
        any of these choices.

        ::

            default = 3

    n_repeats : int, optional
        The number of times to repeat the defined strategy
        as defined in `splits`.

        For example, if `n_repeats` is set to 2, and `splits` is 3,
        then a twice repeated 3-fold CV will be performed

        ::

            default = 1

    cv_strategy : None or :class:`CVStrategy`, optional
        Optional cv_strategy to employ for calculating splits.
        If passed None, use no strategy.

        See :class:`CVStrategy`

        Can also pass valid cv_strategy args seperately.
        If any passed, they will override any values
        set in the pass cv_strategy if any.

        ::

            default = None

    random_state : 'context', int or None, optional
        The fixed random seed in which this
        CV object should adhere to.

        If left as default value of 'context', then
        the random state will be set based on the
        context of where it is called, i.e., typically
        the random_state set in :class:`ProblemSpec`.

        ::

            default = 'context'

    only_fold : int, list of int, or None, optional
        This parameter specifies if special subset of the requested
        CV folds should be used. If kept as None, normal all
        fold behavior will be used. Otherwise, if passed as an int,
        then that int must represent a valid cv fold, e.g.,

        ::

            only_fold = 0

        Would run the CV but only the 1st fold. Likewise if
        a list is passed,

        ::

            only_fold = [0, 2]

        Then only the first and 3rd folds will be run.
        This parameter is useful in cases where the base
        experiment is too computationally intensive, and
        it is desired to run a complete CV but in smaller
        chunks.

        .. warning::

            When used with n_repeats > 1, only_fold
            will index folds from the repeats, e.g.,
            split=2, n_repeats=2, only_fold can be 0, 1, 2, or 3,
            but functionally for say computing summary scores, like std across
            repeats, n_repeats will be treated as 1.

            For example if passed with the setup above only_fold=[0, 1, 2],
            then progress bars and summary stats will still show n_repeats=1.

        ::

            default = None

    cv_strategy_kwargs : kwargs, optional
        If any additional parameters are passed in kwargs style,
        e.g., ::

            splits = 3

        Then they will try to be set in the base cv_strategy.

    '''

    def __init__(self, splits=3, n_repeats=1,
                 cv_strategy=None, random_state='context',
                 only_fold=None,
                 **cv_strategy_kwargs):

        self.splits = splits
        self.n_repeats = n_repeats
        self.cv_strategy = cv_strategy
        self.random_state = random_state
        self.only_fold = only_fold

        if self.cv_strategy is None:
            self.cv_strategy = CVStrategy()

        self.cv_strategy.set_params(**cv_strategy_kwargs)

    def _apply_dataset(self, dataset):
        return get_bpt_cv(self, dataset)

# Depreciations
#############################


def depreciate(func):

    # Get current and new names
    name = str(func.__name__)
    new_name = str(func.new_class.__name__)

    warn_message = name + ' is depreciated and in a future version will'
    warn_message += ' be removed. Use the new class '
    warn_message += new_name + ' to mute this warning.'

    class DepreciatedClass():

        __doc__ = warn_message

        def __new__(cls, *args, **kwargs):

            # Warn on creation
            warnings.warn(warn_message)

            # Return new class instead of this one
            return func.new_class(*args, **kwargs)

    return DepreciatedClass


@depreciate
class Feat_Selector():
    new_class = FeatSelector


@depreciate
class Param_Search():
    new_class = ParamSearch


@depreciate
class Model_Pipeline():
    new_class = ModelPipeline


@depreciate
class Problem_Spec():
    new_class = ProblemSpec
